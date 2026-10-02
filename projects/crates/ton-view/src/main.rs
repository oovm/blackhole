//! 黑洞模拟器 CLI：渲染单帧 PNG（直接看画面，不靠哈希）。

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use image::{ImageBuffer, Rgba};
use ton_simulation::{capture_convergence_samples, RenderMode, Scene};
use ton_types::{BlackHole, Camera, DiskModelKind, IntegratorKind, Vec3};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let out = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("blackhole.png"));

    let width = parse_u32_arg(&args, "--width", 800);
    let height = parse_u32_arg(&args, "--height", 450);
    let mass = parse_f64_arg(&args, "--mass", 1.0);
    let spin = parse_f64_arg(&args, "--spin", 0.0);
    let integrator = parse_integrator(&args);
    let disk_model = parse_disk_model(&args);
    let ucf_tone = args.iter().any(|a| a == "--ucf-tone");
    let ucf_vulkan = args.iter().any(|a| a == "--ucf-vulkan");
    let bloom = args.iter().any(|a| a == "--bloom");
    let strict = args.iter().any(|a| a == "--strict");
    let convergence = args.iter().any(|a| a == "--convergence");
    let overlay_shadow = args.iter().any(|a| a == "--overlay-shadow");
    let hdr_linear = args.iter().any(|a| a == "--hdr-linear");
    let camera = parse_camera(&args, mass, spin);
    let render_mode = parse_render_mode(&args);
    let step_size = parse_f64_arg(&args, "--step-size", 0.08);
    let max_steps = parse_optional_u32_arg(&args, "--max-steps");

    let bh = if spin.abs() > f64::EPSILON {
        BlackHole::kerr(mass, spin)
    } else {
        BlackHole::schwarzschild(mass)
    };

    let mut builder = Scene::builder(bh)
        .with_integrator(integrator)
        .with_disk_model(disk_model)
        .with_camera(camera)
        .with_resolution(width, height)
        .with_render_mode(render_mode)
        .with_step_size(step_size)
        .with_post_bloom(bloom)
        .with_overlay_analytic_shadow(overlay_shadow);

    if let Some(steps) = max_steps {
        builder = builder.with_max_steps(steps);
    }

    if disk_model != DiskModelKind::None {
        builder = builder.with_default_disk();
    }

    let scene = builder.build();
    let trace = scene.trace;

    if convergence {
        run_convergence_report(bh, camera, width, height, integrator);
        return ExitCode::SUCCESS;
    }

    eprintln!(
        "TON view: mode={}, integrator={}, disk={}, camera={}, {}x{} → {}",
        render_mode_label(render_mode),
        integrator_label(integrator),
        disk_model_label(disk_model),
        camera_label(&args),
        width,
        height,
        out.display()
    );
    eprintln!(
        "  M={:.3} r_s={:.3} b_c={:.3} shadowθ@10M={:.4} rad step={:.4} max_steps={} escape_r={:.1}",
        bh.mass,
        bh.event_horizon_radius(),
        bh.critical_impact_parameter(),
        bh.shadow_angular_radius(10.0 * bh.mass),
        trace.step_size,
        trace.max_steps,
        trace.escape_radius
    );

    let (fb, failure_pct) = if ucf_vulkan {
        if render_mode != RenderMode::Beauty {
            eprintln!("--ucf-vulkan only supports Beauty mode");
            return ExitCode::from(4);
        }
        if spin.abs() > f64::EPSILON {
            eprintln!("--ucf-vulkan Schwarzschild path does not support Kerr yet");
            return ExitCode::from(4);
        }
        let frame = match scene.render_frame_ucf_vulkan() {
            Ok(frame) => frame,
            Err(error) => {
                eprintln!("UCF Vulkan Beauty failed without fallback: {error}");
                return ExitCode::from(3);
            }
        };
        (frame, 0.0)
    } else if render_mode != RenderMode::Beauty {
        let output = scene.render_frame_with_histogram();
        if let Some(hist) = output.termination_histogram {
            eprintln!("{}", hist);
            (output.framebuffer, hist.trace_failure_percent())
        } else {
            (output.framebuffer, 0.0)
        }
    } else if ucf_tone || bloom {
        let output = scene.render_frame_with_histogram();
        let failure_pct = output
            .termination_histogram
            .as_ref()
            .map(|h| h.trace_failure_percent())
            .unwrap_or(0.0);
        if let Some(hist) = &output.termination_histogram {
            eprintln!("{}", hist);
        }
        let (frame, report) = scene
            .render_frame_ucf_cpu()
            .expect("UCF CPU render failed");
        eprintln!("  UCF ops: {}", report.executed_ops.join(" → "));
        (frame, failure_pct)
    } else {
        let output = scene.render_frame_with_histogram();
        let failure_pct = output
            .termination_histogram
            .as_ref()
            .map(|h| h.trace_failure_percent())
            .unwrap_or(0.0);
        if let Some(hist) = &output.termination_histogram {
            eprintln!("{}", hist);
        }
        (output.framebuffer, failure_pct)
    };

    if failure_pct > 0.0 {
        eprintln!("  trace failures: {:.2}%", failure_pct);
    }
    if strict && failure_pct > 0.1 {
        eprintln!("strict: trace failure rate {:.2}% exceeds 0.1% threshold", failure_pct);
        return ExitCode::from(2);
    }

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }

    let rgba = if hdr_linear {
        fb.to_rgba8_linear()
    } else {
        fb.to_rgba8()
    };
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, rgba).expect("buffer size mismatch");
    img.save(&out).expect("failed to write PNG");

    eprintln!("done.");
    ExitCode::SUCCESS
}

fn equatorial_far_camera(bh: BlackHole) -> Camera {
    let m = bh.mass;
    let dist = 55.0 * m;
    Camera::look_at(
        Vec3::new(dist, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0),
        std::f64::consts::FRAC_PI_4,
    )
}

fn parse_camera(args: &[String], mass: f64, spin: f64) -> Camera {
    let bh = if spin.abs() > f64::EPSILON {
        BlackHole::kerr(mass, spin)
    } else {
        BlackHole::schwarzschild(mass)
    };
    match args
        .iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
    {
        Some("equatorial") => equatorial_far_camera(bh),
        _ => Camera::cinematic(bh),
    }
}

fn parse_integrator(args: &[String]) -> IntegratorKind {
    match args
        .iter()
        .position(|a| a == "--integrator")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
    {
        Some("schwarzschild") => IntegratorKind::SchwarzschildOrbit,
        Some("sem-classical") => IntegratorKind::SemiClassicalRayMarch,
        Some("geodesic3d") => IntegratorKind::Geodesic3d,
        Some("kerr") => IntegratorKind::KerrEquatorial,
        _ => IntegratorKind::Geodesic3d,
    }
}

fn parse_render_mode(args: &[String]) -> RenderMode {
    match args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
    {
        Some("termination") => RenderMode::TerminationMap,
        Some("capture") => RenderMode::CaptureSilhouette,
        Some("failure") => RenderMode::FailureMap,
        _ => RenderMode::Beauty,
    }
}

fn render_mode_label(mode: RenderMode) -> &'static str {
    match mode {
        RenderMode::Beauty => "beauty",
        RenderMode::TerminationMap => "termination",
        RenderMode::CaptureSilhouette => "capture",
        RenderMode::FailureMap => "failure",
    }
}

fn parse_disk_model(args: &[String]) -> DiskModelKind {
    match args
        .iter()
        .position(|a| a == "--disk")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
    {
        Some("fbm") => DiskModelKind::ProceduralFbm,
        Some("solid") => DiskModelKind::SolidUniform,
        Some("none") => DiskModelKind::None,
        _ => DiskModelKind::NovikovThorneThin,
    }
}

fn integrator_label(k: IntegratorKind) -> &'static str {
    match k {
        IntegratorKind::SchwarzschildOrbit => "schwarzschild",
        IntegratorKind::SemiClassicalRayMarch => "sem-classical",
        IntegratorKind::Geodesic3d => "geodesic3d",
        IntegratorKind::KerrEquatorial => "kerr",
    }
}

fn disk_model_label(k: DiskModelKind) -> &'static str {
    match k {
        DiskModelKind::None => "none",
        DiskModelKind::NovikovThorneThin => "novikov",
        DiskModelKind::ProceduralFbm => "fbm",
        DiskModelKind::SolidUniform => "solid",
    }
}

fn run_convergence_report(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    integrator: IntegratorKind,
) {
    eprintln!("capture convergence (no disk, no post):");
    let samples = capture_convergence_samples(
        bh,
        camera,
        width,
        height,
        integrator,
        &[0.08, 0.04, 0.02, 0.01],
    );
    eprintln!(
        "  {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}",
        "step",
        "max_steps",
        "captured",
        "escaped",
        "failed",
        "boundary_r",
        "b_analytic",
        "agree",
        "on_axis"
    );
    for s in samples {
        eprintln!(
            "  {:8.4} {:8} {:7.2}% {:7.2}% {:7.2}% {:10.5} {:10.5} {:7.2}% {:8.5}",
            s.step_size,
            s.max_steps,
            s.captured_fraction * 100.0,
            s.escaped_fraction * 100.0,
            s.failure_fraction * 100.0,
            s.mean_boundary_radius,
            s.schwarzschild_boundary_radius_ndc,
            s.capture_agreement * 100.0,
            s.on_axis_shadow_radius_ndc
        );
    }
}

fn camera_label(args: &[String]) -> &'static str {
    match args
        .iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
    {
        Some("equatorial") => "equatorial",
        _ => "cinematic",
    }
}

fn parse_u32_arg(args: &[String], flag: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_optional_u32_arg(args: &[String], flag: &str) -> Option<u32> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_f64_arg(args: &[String], flag: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

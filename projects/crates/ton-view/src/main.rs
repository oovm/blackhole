//! 黑洞模拟器 CLI：渲染单帧 PNG（直接看画面，不靠哈希）。

use std::env;
use std::path::PathBuf;

use image::{ImageBuffer, Rgba};
use ton_simulation::Scene;
use ton_types::{BlackHole, Camera, DiskModelKind, IntegratorKind, Vec3};

fn main() {
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
    let bloom = args.iter().any(|a| a == "--bloom");
    let camera = parse_camera(&args, mass, spin);

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
        .with_post_bloom(bloom);

    if disk_model != DiskModelKind::None {
        builder = builder.with_default_disk();
    }

    let scene = builder.build();

    eprintln!(
        "TON view: integrator={}, disk={}, camera={}, {}x{} → {}",
        integrator_label(integrator),
        disk_model_label(disk_model),
        camera_label(&args),
        width,
        height,
        out.display()
    );
    eprintln!(
        "  M={:.3} r_s={:.3} b_c={:.3} shadowθ@10M={:.4} rad",
        bh.mass,
        bh.event_horizon_radius(),
        bh.critical_impact_parameter(),
        bh.shadow_angular_radius(10.0 * bh.mass)
    );

    let fb = if ucf_tone || bloom {
        let (frame, report) = scene
            .render_frame_ucf_cpu()
            .expect("UCF CPU render failed");
        eprintln!("  UCF ops: {}", report.executed_ops.join(" → "));
        frame
    } else {
        scene.render_frame()
    };

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }

    let rgba = fb.to_rgba8();
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, rgba).expect("buffer size mismatch");
    img.save(&out).expect("failed to write PNG");

    eprintln!("done.");
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

fn parse_disk_model(args: &[String]) -> DiskModelKind {
    match args
        .iter()
        .position(|a| a == "--disk")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
    {
        Some("fbm") => DiskModelKind::ProceduralFbm,
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

fn parse_f64_arg(args: &[String], flag: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

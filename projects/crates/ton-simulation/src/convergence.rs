//! 捕获轮廓步长收敛验收（无盘、无星空、无后处理）。

use ton_geodesic::{
    schwarzschild_geodesic_analytic_captured, trace_ray_diagnostic, RayTermination, TraceConfig,
};
use ton_renderer::{
    on_axis_shadow_radius_ndc, schwarzschild_boundary_radius_ndc,
};
use ton_types::{BlackHole, Camera, IntegratorKind, Vec3};

fn is_captured_termination(t: RayTermination) -> bool {
    matches!(t, RayTermination::Captured | RayTermination::OriginInsideHorizon)
}

/// 单档步长的捕获统计。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureStepSample {
    pub step_size: f64,
    pub max_steps: u32,
    pub captured_fraction: f64,
    pub escaped_fraction: f64,
    pub failure_fraction: f64,
    /// 数值捕获/逃逸边界平均 NDC 半径。
    pub mean_boundary_radius: f64,
    /// 史瓦西冲击参数解析边界平均 NDC 半径（倾角相机适用）。
    pub schwarzschild_boundary_radius_ndc: f64,
    /// 轴对称小角近似半径（对照列）。
    pub on_axis_shadow_radius_ndc: f64,
    /// 数值 vs 解析捕获分类一致率。
    pub capture_agreement: f64,
}

/// 无盘捕获轮廓：在多个步长下统计终止分布与边界半径。
pub fn capture_convergence_samples(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    integrator: IntegratorKind,
    step_sizes: &[f64],
) -> Vec<CaptureStepSample> {
    step_sizes
        .iter()
        .map(|&step| {
            let trace = TraceConfig::with_step_size(step);
            sample_capture_at_step(bh, camera, width, height, integrator, trace)
        })
        .collect()
}

fn sample_capture_at_step(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    integrator: IntegratorKind,
    trace: TraceConfig,
) -> CaptureStepSample {
    let mut captured = 0u32;
    let mut escaped = 0u32;
    let mut failed = 0u32;
    let mut boundary_r_sum = 0.0;
    let mut boundary_n = 0u32;
    let mut agree = 0u32;
    let mut terminations = vec![RayTermination::Escaped; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let dir = pixel_ray_dir(x, y, width, height, camera);
            let diag = trace_ray_diagnostic(integrator, bh, camera.position, dir, trace);
            let idx = y as usize * width as usize + x as usize;
            terminations[idx] = diag.termination;
            let num_captured = is_captured_termination(diag.termination);
            let ana_captured =
                schwarzschild_geodesic_analytic_captured(bh, camera.position, dir);
            if num_captured == ana_captured {
                agree += 1;
            }
            match diag.termination {
                RayTermination::Captured => captured += 1,
                RayTermination::Escaped => escaped += 1,
                RayTermination::StepBudgetExceeded
                | RayTermination::NonFiniteState
                | RayTermination::InitFailed
                | RayTermination::DomainError => failed += 1,
                _ => {}
            }
        }
    }

    let aspect = width as f64 / height as f64;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y as usize * width as usize + x as usize;
            if !is_captured_termination(terminations[idx]) {
                continue;
            }
            let neighbors = [
                terminations[idx - 1],
                terminations[idx + 1],
                terminations[idx - width as usize],
                terminations[idx + width as usize],
            ];
            if neighbors.iter().any(|t| !is_captured_termination(*t)) {
                let u = ((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * aspect;
                let v = 1.0 - (y as f64 + 0.5) / height as f64 * 2.0;
                boundary_r_sum += (u * u + v * v).sqrt();
                boundary_n += 1;
            }
        }
    }

    let total = (width * height) as f64;
    let observer_distance = camera.position.length();
    CaptureStepSample {
        step_size: trace.step_size,
        max_steps: trace.max_steps,
        captured_fraction: captured as f64 / total,
        escaped_fraction: escaped as f64 / total,
        failure_fraction: failed as f64 / total,
        mean_boundary_radius: if boundary_n > 0 {
            boundary_r_sum / boundary_n as f64
        } else {
            0.0
        },
        schwarzschild_boundary_radius_ndc: schwarzschild_boundary_radius_ndc(
            bh,
            camera,
            width,
            height,
        ),
        on_axis_shadow_radius_ndc: on_axis_shadow_radius_ndc(bh, observer_distance, camera.fov_y),
        capture_agreement: agree as f64 / total,
    }
}

fn pixel_ray_dir(x: u32, y: u32, w: u32, h: u32, cam: Camera) -> Vec3 {
    let u = ((x as f64 + 0.5) / w as f64) * 2.0 - 1.0;
    let v = 1.0 - ((y as f64 + 0.5) / h as f64) * 2.0;
    let aspect = w as f64 / h as f64;
    cam.ray_direction(u * aspect, v)
}

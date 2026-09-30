//! 捕获轮廓步长收敛冒烟：失败率为零，边界半径随步长变化有界。

use std::collections::BTreeMap;

use ton_geodesic::{trace_ray_diagnostic, RayTermination, TraceConfig};
use ton_simulation::capture_convergence_samples;
use ton_types::{BlackHole, Camera, IntegratorKind, Vec3};

fn pixel_ray_dir(x: u32, y: u32, w: u32, h: u32, cam: Camera) -> Vec3 {
    let u = (x as f64 / w as f64) * 2.0 - 1.0;
    let v = 1.0 - (y as f64 / h as f64) * 2.0;
    let aspect = w as f64 / h as f64;
    cam.ray_direction(u * aspect, v)
}

#[test]
#[ignore = "diagnostic: termination histogram"]
fn debug_capture_termination_histogram() {
    let bh = BlackHole::schwarzschild(1.0);
    let cam = Camera::cinematic(bh);
    let trace = TraceConfig::with_step_size(0.08);
    let mut hist = BTreeMap::<&'static str, u32>::new();
    for y in 0..72u32 {
        for x in 0..128u32 {
            let dir = pixel_ray_dir(x, y, 128, 72, cam);
            let d = trace_ray_diagnostic(IntegratorKind::Geodesic3d, bh, cam.position, dir, trace);
            let key = match d.termination {
                RayTermination::Captured => "Captured",
                RayTermination::Escaped => "Escaped",
                RayTermination::StepBudgetExceeded => "StepBudgetExceeded",
                RayTermination::NonFiniteState => "NonFiniteState",
                RayTermination::InitFailed => "InitFailed",
                RayTermination::DomainError => "DomainError",
                RayTermination::OriginInsideHorizon => "OriginInsideHorizon",
                other => {
                    eprintln!("other termination {:?}", other);
                    "Other"
                }
            };
            hist.entry(key).and_modify(|c| *c += 1).or_insert(1);
        }
    }
    for (k, v) in hist {
        eprintln!("{}: {}", k, v);
    }
}

#[test]
fn capture_silhouette_has_zero_failures_at_scaled_budget() {
    let bh = BlackHole::schwarzschild(1.0);
    let cam = Camera::cinematic(bh);
    let samples = capture_convergence_samples(
        bh,
        cam,
        96,
        54,
        IntegratorKind::Geodesic3d,
        &[0.08, 0.04, 0.02],
    );
    for s in &samples {
        assert!(
            s.failure_fraction < 1e-6,
            "step {} failed {:.2}%",
            s.step_size,
            s.failure_fraction * 100.0
        );
    }
}

#[test]
fn capture_boundary_radius_stable_across_step_halving() {
    let bh = BlackHole::schwarzschild(1.0);
    let cam = Camera::cinematic(bh);
    let samples = capture_convergence_samples(
        bh,
        cam,
        128,
        72,
        IntegratorKind::Geodesic3d,
        &[0.08, 0.04, 0.02],
    );
    let r0 = samples[0].mean_boundary_radius;
    let r2 = samples[2].mean_boundary_radius;
    assert!(r0 > 0.05 && r2 > 0.05, "expected visible shadow boundary");
    let drift = (r0 - r2).abs() / r0.max(1e-6);
    assert!(
        drift < 0.08,
        "boundary radius drift {:.1}% exceeds 8% (r0={:.4}, r2={:.4})",
        drift * 100.0,
        r0,
        r2
    );
    let analytic = samples[0].schwarzschild_boundary_radius_ndc;
    let boundary_drift = (r0 - analytic).abs() / analytic.max(1e-6);
    assert!(
        boundary_drift < 0.12,
        "numerical boundary {:.4} vs analytic {:.4} drift {:.1}%",
        r0,
        analytic,
        boundary_drift * 100.0
    );
    assert!(
        samples[0].capture_agreement > 0.95,
        "capture agreement {:.1}% below baseline",
        samples[0].capture_agreement * 100.0
    );
}

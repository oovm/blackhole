//! 捕获轮廓步长收敛冒烟：失败率为零，边界半径随步长变化有界。

use ton_simulation::capture_convergence_samples;
use ton_types::{BlackHole, Camera, IntegratorKind};

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
    let analytic = samples[0].analytic_shadow_radius_ndc;
    let analytic_drift = (r0 - analytic).abs() / analytic.max(1e-6);
    assert!(
        analytic_drift < 0.35,
        "numerical boundary {:.4} vs analytic {:.4} drift {:.1}% (integrator still converging)",
        r0,
        analytic,
        analytic_drift * 100.0
    );
}

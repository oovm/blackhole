//! UCF 测地线程序与 Rust 参考实现对照。

use ton_geodesic::{trace_geodesic_3d_diagnostic, RayTermination, TraceConfig};
use ton_metric::{
    schwarzschild_geodesic_analytic_captured, schwarzschild_geodesic_impact_parameter,
};
use ton_types::{BlackHole, Vec3};
use ucf_program::{geodesic_capture_classify, geodesic_radial_infall_steps, CpuInterpreter, Value};

#[test]
fn ucf_capture_classify_matches_schwarzschild_analytic() {
    let bh = BlackHole::schwarzschild(1.0);
    let prog = geodesic_capture_classify();
    let mut cpu = CpuInterpreter::new();
    let bc = bh.critical_impact_parameter();
    cpu.with_uniform(1, Value::F64(bc));

    let samples = [
        (Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        (Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.12, 0.0)),
        (Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        (Vec3::new(15.0, 2.0, 1.0), Vec3::new(-0.9, 0.1, 0.05)),
    ];

    for (origin, dir) in samples {
        let d = dir.normalize();
        let b = schwarzschild_geodesic_impact_parameter(bh.mass, origin, d)
            .expect("impact parameter");
        cpu.with_uniform(0, Value::F64(b));
        let ucf = cpu.run(&prog).expect("run").value.as_f64().expect("f64");
        let rust = if schwarzschild_geodesic_analytic_captured(bh, origin, dir) {
            1.0
        } else {
            0.0
        };
        assert!(
            (ucf - rust).abs() < 1e-9,
            "b={b:.4} ucf={ucf} rust={rust}"
        );
    }
}

#[test]
fn ucf_radial_infall_captures_before_geodesic3d_budget() {
    let bh = BlackHole::schwarzschild(1.0);
    let origin = Vec3::new(10.0, 0.0, 0.0);
    let dir = Vec3::new(-1.0, 0.0, 0.0);
    let step = 0.08;
    let diag = trace_geodesic_3d_diagnostic(
        bh,
        origin,
        dir,
        TraceConfig::with_step_size(step),
    );
    assert_eq!(diag.termination, RayTermination::Captured);

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(origin.length()));
    cpu.with_uniform(1, Value::F64(bh.event_horizon_radius()));
    cpu.with_uniform(2, Value::F64(step));
    let ucf_steps = cpu
        .run(&geodesic_radial_infall_steps())
        .expect("run")
        .value
        .as_f64()
        .expect("f64");
    assert!(ucf_steps > 0.0);
    assert!(
        (ucf_steps - 100.0).abs() < 1e-6,
        "expected 100 Euler steps, got {ucf_steps}"
    );
    assert!(diag.steps > 0);
}

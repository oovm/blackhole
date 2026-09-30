//! `ton-geodesic` 单元测试。

mod sem_classical_tests {
    use ton_geodesic::{trace_sem_classical, RayHit, TraceConfig};
    use ton_types::{BlackHole, Vec3};

    #[test]
    fn sem_classical_outward_ray_escapes_from_far_away() {
        let bh = BlackHole::schwarzschild(1.0);
        let hit = trace_sem_classical(
            bh,
            Vec3::new(500.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            TraceConfig {
                escape_radius: 600.0,
                ..TraceConfig::default()
            },
        )
        .unwrap();
        assert!(matches!(hit, RayHit::Escaped { .. }));
    }
}

mod dispatch_tests {
    use ton_geodesic::{trace_ray, RayHit, TraceConfig};
    use ton_types::{BlackHole, IntegratorKind, TonError, Vec3};

    #[test]
    fn dispatch_routes_to_sem_classical() {
        let bh = BlackHole::schwarzschild(1.0);
        let hit = trace_ray(
            IntegratorKind::SemiClassicalRayMarch,
            bh,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        )
        .unwrap();
        assert!(matches!(hit, RayHit::Captured { .. }));
    }

    #[test]
    fn dispatch_routes_to_geodesic_3d() {
        let bh = BlackHole::schwarzschild(1.0);
        let hit = trace_ray(
            IntegratorKind::Geodesic3d,
            bh,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        )
        .unwrap();
        assert!(matches!(hit, RayHit::Captured { .. }));
    }

    #[test]
    fn dispatch_routes_to_kerr_equatorial() {
        let bh = BlackHole::kerr(1.0, 0.5);
        let hit = trace_ray(
            IntegratorKind::KerrEquatorial,
            bh,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        )
        .unwrap();
        assert!(matches!(hit, RayHit::Captured { .. }));
    }
}

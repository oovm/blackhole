//! 小尺寸 CPU golden：每种算法族独立校验和，不互相替代。

use ton_sim::Scene;
use ton_types::{BlackHole, DiskModelKind, IntegratorKind};

fn fnv1a64(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    let mut hash = 0xcbf29ce484222325u64;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[test]
fn golden_schwarzschild_orbit_novikov_48x27() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::SchwarzschildOrbit)
        .with_disk_model(DiskModelKind::NovikovThorneThin)
        .with_resolution(48, 27)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let hash = fnv1a64(&rgba);
    assert_eq!(
        hash,
        968992088206601693,
        "update golden if intentional algorithm change"
    );
}

#[test]
fn golden_sem_classical_fbm_48x27() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::SemiClassicalRayMarch)
        .with_disk_model(DiskModelKind::ProceduralFbm)
        .with_resolution(48, 27)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let hash = fnv1a64(&rgba);
    assert_eq!(
        hash,
        857101654930588111,
        "update golden if intentional algorithm change"
    );
}

#[test]
fn golden_geodesic_3d_novikov_48x27() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::Geodesic3d)
        .with_disk_model(DiskModelKind::NovikovThorneThin)
        .with_resolution(48, 27)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let hash = fnv1a64(&rgba);
    assert_eq!(
        hash,
        1281810340172712011,
        "update golden if intentional algorithm change"
    );
}

#[test]
fn golden_kerr_equatorial_novikov_48x27() {
    let scene = Scene::builder(BlackHole::kerr(1.0, 0.5))
        .with_default_disk()
        .with_integrator(IntegratorKind::KerrEquatorial)
        .with_disk_model(DiskModelKind::NovikovThorneThin)
        .with_resolution(48, 27)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let hash = fnv1a64(&rgba);
    assert_eq!(
        hash,
        968992088206601693,
        "update golden if intentional algorithm change"
    );
}

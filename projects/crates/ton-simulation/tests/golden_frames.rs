//! 小尺寸 CPU 渲染冒烟：看像素分布，不靠 FNV 哈希。

use ton_simulation::Scene;
use ton_types::{BlackHole, DiskModelKind, IntegratorKind};

fn count_pixels(rgba: &[u8], pred: impl Fn(u8, u8, u8) -> bool) -> usize {
    rgba.chunks(4)
        .filter(|px| pred(px[0], px[1], px[2]))
        .count()
}

#[test]
fn geodesic3d_cinematic_has_disk_and_shadow() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::Geodesic3d)
        .with_disk_model(DiskModelKind::NovikovThorneThin)
        .with_resolution(96, 54)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let bright = count_pixels(&rgba, |r, g, b| (r as u16 + g as u16 + b as u16) > 120);
    let dark = count_pixels(&rgba, |r, g, b| (r as u16 + g as u16 + b as u16) < 12);
    assert!(bright > 200, "expected bright disk pixels, got {bright}");
    assert!(dark > 80, "expected shadow pixels, got {dark}");
}

#[test]
fn schwarzschild_equatorial_still_renders_disk() {
    use ton_types::{Camera, Vec3};
    let bh = BlackHole::schwarzschild(1.0);
    let m = bh.mass;
    let cam = Camera::look_at(
        Vec3::new(55.0 * m, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0),
        std::f64::consts::FRAC_PI_4,
    );
    let scene = Scene::builder(bh)
        .with_default_disk()
        .with_camera(cam)
        .with_integrator(IntegratorKind::SchwarzschildOrbit)
        .with_disk_model(DiskModelKind::NovikovThorneThin)
        .with_resolution(64, 36)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let bright = count_pixels(&rgba, |r, g, b| (r as u16 + g as u16 + b as u16) > 80);
    assert!(bright > 100, "expected equatorial disk band, got {bright}");
}

#[test]
fn ucf_cpu_path_runs_full_chain() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::Geodesic3d)
        .with_post_bloom(true)
        .with_resolution(32, 24)
        .build();
    let (frame, report) = scene.render_frame_ucf_cpu().expect("ucf cpu");
    assert_eq!(report.executed_ops.len(), 7);
    assert!(frame.to_rgba8().iter().any(|&b| b > 0));
}

#[test]
fn solid_uniform_disk_renders_opaque_band() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::Geodesic3d)
        .with_disk_model(DiskModelKind::SolidUniform)
        .with_resolution(96, 54)
        .build();
    let output = scene.render_frame_with_histogram();
    let hdr_disk = output
        .framebuffer
        .pixels
        .iter()
        .filter(|p| p.r > 1.5 && p.g > 0.8 && p.b < 1.2)
        .count();
    let rgba = output.framebuffer.to_rgba8();
    let warm = count_pixels(&rgba, |r, g, b| r > g && g > b && r > 80);
    let dark = count_pixels(&rgba, |r, g, b| (r as u16 + g as u16 + b as u16) < 15);
    assert!(hdr_disk > 150, "expected linear HDR disk hits, got {hdr_disk}");
    assert!(warm > 100, "expected warm disk after tone map, got {warm}");
    assert!(dark > 50, "expected shadow pixels, got {dark}");
}

#[test]
fn sem_classical_fbm_has_variation() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::SemiClassicalRayMarch)
        .with_disk_model(DiskModelKind::ProceduralFbm)
        .with_resolution(64, 36)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let unique_g = rgba.chunks(4).map(|p| p[1]).collect::<std::collections::HashSet<_>>().len();
    assert!(unique_g > 4, "fbm disk should vary in green channel");
}

//! 纯色盘透镜结构冒烟：上下半屏均应有盘像（前景 + 透镜抬升）。

use ton_simulation::Scene;
use ton_types::{BlackHole, DiskModelKind, IntegratorKind};

fn count_warm_in_band(rgba: &[u8], width: u32, y0: u32, y1: u32) -> usize {
    rgba.chunks(4)
        .enumerate()
        .filter(|(i, px)| {
            let y = *i as u32 / width;
            y >= y0
                && y < y1
                && px[0] > 70
                && px[0] as u16 > px[1] as u16
                && px[1] as u16 > px[2] as u16
        })
        .count()
}

#[test]
fn solid_disk_appears_in_both_vertical_halves() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::Geodesic3d)
        .with_disk_model(DiskModelKind::SolidUniform)
        .with_resolution(128, 72)
        .build();
    let rgba = scene.render_frame().to_rgba8();
    let h = scene.height;
    let mid = h / 2;
    let top = count_warm_in_band(&rgba, scene.width, 0, mid);
    let bottom = count_warm_in_band(&rgba, scene.width, mid, h);
    assert!(
        top > 40 && bottom > 40,
        "expected lensed disk in both halves (top={top}, bottom={bottom})"
    );
}

#[test]
fn solid_disk_linear_hdr_has_high_energy_pixels() {
    let scene = Scene::builder(BlackHole::schwarzschild(1.0))
        .with_default_disk()
        .with_integrator(IntegratorKind::Geodesic3d)
        .with_disk_model(DiskModelKind::SolidUniform)
        .with_resolution(96, 54)
        .build();
    let fb = scene.render_frame();
    let hot = fb
        .pixels
        .iter()
        .filter(|p| p.r > 2.0 && p.g > 1.0)
        .count();
    assert!(hot > 100, "expected linear HDR disk energy, got {hot} hot pixels");
}

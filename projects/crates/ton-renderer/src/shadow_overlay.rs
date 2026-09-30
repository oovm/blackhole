//! 解析阴影视角半径参考环（捕获轮廓验收用）。

use ton_types::{BlackHole, Camera, Rgb};

use crate::framebuffer::Framebuffer;

/// 观察距离处的阴影在半视场归一化平面上的角半径（与 `Camera::ray_direction` 一致）。
pub fn analytic_shadow_radius_ndc(bh: BlackHole, observer_distance: f64, fov_y: f64) -> f64 {
    let theta = bh.shadow_angular_radius(observer_distance);
    theta / (fov_y * 0.5).tan()
}

/// 在诊断帧上叠加解析阴影参考环（绿色，1px 宽）。
pub fn draw_analytic_shadow_ring(fb: &mut Framebuffer, bh: BlackHole, camera: Camera) {
    let w = fb.width;
    let h = fb.height;
    if w == 0 || h == 0 {
        return;
    }
    let dist = camera.position.length();
    let r_ndc = analytic_shadow_radius_ndc(bh, dist, camera.fov_y);
    let aspect = w as f64 / h as f64;
    let ring = Rgb::new(0.15, 0.85, 0.35);
    let half_width = 1.25 / w.min(h) as f64;
    for y in 0..h {
        for x in 0..w {
            let u = (x as f64 / w as f64 * 2.0 - 1.0) * aspect;
            let v = 1.0 - y as f64 / h as f64 * 2.0;
            let r = (u * u + v * v).sqrt();
            if (r - r_ndc).abs() <= half_width {
                fb.set(x, y, ring);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_shadow_radius_is_positive() {
        let bh = BlackHole::schwarzschild(1.0);
        let r = analytic_shadow_radius_ndc(bh, 10.0, 0.85);
        assert!(r > 0.5 && r < 2.0);
    }
}

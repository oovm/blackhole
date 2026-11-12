//! 解析阴影参考（史瓦西冲击参数轮廓，适用于倾角相机）。

use ton_geodesic::schwarzschild_geodesic_analytic_captured;
use ton_types::{BlackHole, Camera, Rgb, Vec3};

use crate::framebuffer::Framebuffer;

/// 轴对称小角近似阴影视角半径（仅作对照，倾角相机会低估）。
pub fn on_axis_shadow_radius_ndc(bh: BlackHole, observer_distance: f64, fov_y: f64) -> f64 {
    let theta = bh.shadow_angular_radius(observer_distance);
    theta / (fov_y * 0.5).tan()
}

/// 与 `Camera::ray_direction` 一致的像素射线方向。
pub fn pixel_ray_dir(x: u32, y: u32, w: u32, h: u32, cam: Camera) -> Vec3 {
    let u = ((x as f64 + 0.5) / w as f64) * 2.0 - 1.0;
    let v = 1.0 - ((y as f64 + 0.5) / h as f64) * 2.0;
    let aspect = w as f64 / h as f64;
    cam.ray_direction(u * aspect, v)
}

/// 在像素网格上统计解析捕获/逃逸边界的平均 NDC 半径（与数值收敛同算法）。
pub fn schwarzschild_boundary_radius_ndc(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
) -> f64 {
    let aspect = width as f64 / height as f64;
    let mut boundary_r_sum = 0.0;
    let mut boundary_n = 0u32;
    let w = width as usize;
    let h = height as usize;
    let mut captured = vec![false; w * h];
    for y in 0..height {
        for x in 0..width {
            let dir = pixel_ray_dir(x, y, width, height, camera);
            captured[y as usize * w + x as usize] =
                schwarzschild_geodesic_analytic_captured(bh, camera.position, dir);
        }
    }
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y as usize * w + x as usize;
            if !captured[idx] {
                continue;
            }
            let neighbors = [
                captured[idx - 1],
                captured[idx + 1],
                captured[idx - w],
                captured[idx + w],
            ];
            if neighbors.iter().any(|c| !*c) {
                let u = ((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * aspect;
                let v = 1.0 - (y as f64 + 0.5) / height as f64 * 2.0;
                boundary_r_sum += (u * u + v * v).sqrt();
                boundary_n += 1;
            }
        }
    }
    if boundary_n == 0 {
        0.0
    } else {
        boundary_r_sum / boundary_n as f64
    }
}

/// 数值与解析捕获分类一致率（0..1）。
pub fn capture_classification_agreement(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    numerical_captured: &[bool],
) -> f64 {
    if numerical_captured.len() != (width * height) as usize {
        return 0.0;
    }
    let mut agree = 0u32;
    for y in 0..height {
        for x in 0..width {
            let idx = y as usize * width as usize + x as usize;
            let dir = pixel_ray_dir(x, y, width, height, camera);
            let analytic = schwarzschild_geodesic_analytic_captured(bh, camera.position, dir);
            if numerical_captured[idx] == analytic {
                agree += 1;
            }
        }
    }
    agree as f64 / numerical_captured.len() as f64
}

/// 在诊断帧上叠加史瓦西冲击参数阴影边界（绿色，1px 宽）。
pub fn draw_analytic_shadow_ring(fb: &mut Framebuffer, bh: BlackHole, camera: Camera) {
    let w = fb.width;
    let h = fb.height;
    if w == 0 || h == 0 {
        return;
    }
    let aspect = w as f64 / h as f64;
    let mut captured = vec![false; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let dir = pixel_ray_dir(x, y, w, h, camera);
            captured[y as usize * w as usize + x as usize] =
                schwarzschild_geodesic_analytic_captured(bh, camera.position, dir);
        }
    }
    let ring = Rgb::new(0.15, 0.85, 0.35);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y as usize * w as usize + x as usize;
            if !captured[idx] {
                continue;
            }
            let edge = [
                captured[idx - 1],
                captured[idx + 1],
                captured[idx - w as usize],
                captured[idx + w as usize],
            ];
            if edge.iter().any(|c| !*c) {
                fb.set(x, y, ring);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::schwarzschild_impact_parameter;

    #[test]
    fn on_axis_radius_matches_small_angle_formula() {
        let bh = BlackHole::schwarzschild(1.0);
        let r = on_axis_shadow_radius_ndc(bh, 10.0, 0.85);
        assert!(r > 0.5 && r < 2.0);
    }

    #[test]
    fn impact_parameter_captured_matches_bc() {
        let bh = BlackHole::schwarzschild(1.0);
        let origin = Vec3::new(10.0, 0.0, 0.0);
        let head_on = Vec3::new(-1.0, 0.0, 0.0);
        assert!(schwarzschild_geodesic_analytic_captured(bh, origin, head_on));
        let grazing = Vec3::new(-1.0, 0.0, 0.01).normalize();
        let b = schwarzschild_impact_parameter(origin, grazing);
        assert!(b < bh.critical_impact_parameter() + 0.1);
    }

    #[test]
    fn cinematic_boundary_radius_matches_on_axis_schwarzschild() {
        let bh = BlackHole::schwarzschild(1.0);
        let cam = Camera::cinematic(bh);
        let schwarzschild = schwarzschild_boundary_radius_ndc(bh, cam, 128, 72);
        let on_axis = on_axis_shadow_radius_ndc(bh, cam.position.length(), cam.fov_y);
        assert!(schwarzschild > 0.4 && schwarzschild < 1.6);
        assert!((schwarzschild - on_axis).abs() < 0.05);
    }
}

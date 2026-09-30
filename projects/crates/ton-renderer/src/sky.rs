use ton_types::{Rgb, Vec3};

/// 深蓝渐变背景 + 立方体面网格高斯星点（无阈值椒盐噪声）。
pub fn sky_color(direction: Vec3) -> Rgb {
    let d = direction.normalize();
    let t = (d.y * 0.5 + 0.5).clamp(0.0, 1.0);
    let base = Rgb::new(0.02 * t, 0.03 * t, 0.08 + 0.12 * t);
    let stars = star_brightness(d);
    base.add(Rgb::WHITE.scale(stars * 0.55))
}

fn cell_hash(a: i32, b: i32, face: i32, salt: i32) -> f64 {
    let n = ((a as f64) * 127.1 + (b as f64) * 311.7 + (face as f64) * 74.7 + (salt as f64) * 19.3)
        .sin()
        * 43758.5453;
    n - n.floor()
}

/// 在立方体面 UV 网格上放置稀疏星点，邻域高斯叠加保证方向微扰时亮度连续。
fn star_brightness(d: Vec3) -> f64 {
    let ax = d.x.abs();
    let ay = d.y.abs();
    let az = d.z.abs();
    let (face, u, v) = if ax >= ay && ax >= az {
        if d.x >= 0.0 {
            (0, d.y / d.x, d.z / d.x)
        } else {
            (1, -d.y / d.x, d.z / d.x)
        }
    } else if ay >= az {
        if d.y >= 0.0 {
            (2, d.x / d.y, d.z / d.y)
        } else {
            (3, -d.x / d.y, d.z / d.y)
        }
    } else if d.z >= 0.0 {
        (4, d.x / d.z, d.y / d.z)
    } else {
        (5, -d.x / d.z, d.y / d.z)
    };

    let cells = 96.0;
    let u = (u * 0.5 + 0.5) * cells;
    let v = (v * 0.5 + 0.5) * cells;
    let iu = u.floor() as i32;
    let iv = v.floor() as i32;

    let mut brightness = 0.0;
    for dv in -1..=1 {
        for du in -1..=1 {
            let cx = iu + du;
            let cy = iv + dv;
            let seed = cell_hash(cx, cy, face, 0);
            if seed < 0.018 {
                continue;
            }
            let su = cx as f64 + 0.12 + 0.76 * cell_hash(cx, cy, face, 1);
            let sv = cy as f64 + 0.31 + 0.76 * cell_hash(cx, cy, face, 2);
            let du = u - su;
            let dv = v - sv;
            let r2 = du * du + dv * dv;
            let intensity = 0.25 + 0.75 * cell_hash(cx, cy, face, 3);
            brightness += (-r2 * 140.0).exp() * intensity;
        }
    }
    brightness.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_field_is_deterministic() {
        let d = Vec3::new(0.2, 0.7, -0.3).normalize();
        let a = sky_color(d);
        let b = sky_color(d);
        assert_eq!(a, b);
    }

    #[test]
    fn star_field_is_smooth_under_small_perturbation() {
        let d0 = Vec3::new(0.15, 0.62, -0.41).normalize();
        let d1 = Vec3::new(0.1503, 0.6198, -0.4101).normalize();
        let c0 = sky_color(d0);
        let c1 = sky_color(d1);
        let delta = (c0.r - c1.r).abs() + (c0.g - c1.g).abs() + (c0.b - c1.b).abs();
        assert!(
            delta < 0.08,
            "salt noise would flip abruptly, delta={delta}"
        );
    }

    #[test]
    fn background_has_finite_brightness() {
        let d = Vec3::new(0.0, 1.0, 0.0);
        let c = sky_color(d);
        assert!(c.r.is_finite() && c.g.is_finite() && c.b.is_finite());
        assert!(c.r >= 0.0 && c.g >= 0.0 && c.b >= 0.0);
        assert!(c.r <= 1.0 && c.g <= 1.0 && c.b <= 1.0);
    }
}

use crate::{BlackHole, Vec3};

/// 远场欧氏近似冲击参数 `b ≈ |**r** × **k**|`（**k** 为单位方向）。
///
/// 倾角相机与有限半径观测点请用 `ton_metric::schwarzschild_geodesic_impact_parameter`（`b = L/E`）。
pub fn schwarzschild_impact_parameter(origin: Vec3, direction: Vec3) -> f64 {
    let d = direction.normalize();
    origin.cross(d).length()
}

/// 解析捕获判定：`b < b_c` 或起点已在视界内。
pub fn schwarzschild_analytic_captured(bh: BlackHole, origin: Vec3, direction: Vec3) -> bool {
    if !origin.length().is_finite() {
        return false;
    }
    if origin.length() <= bh.event_horizon_radius() {
        return true;
    }
    schwarzschild_impact_parameter(origin, direction) < bh.critical_impact_parameter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_on_ray_from_far_observer_is_captured() {
        let bh = BlackHole::schwarzschild(1.0);
        let origin = Vec3::new(20.0, 0.0, 0.0);
        let dir = Vec3::new(-1.0, 0.0, 0.0);
        assert!(schwarzschild_analytic_captured(bh, origin, dir));
    }

    #[test]
    fn wide_miss_ray_escapes() {
        let bh = BlackHole::schwarzschild(1.0);
        let origin = Vec3::new(20.0, 0.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        assert!(!schwarzschild_analytic_captured(bh, origin, dir));
    }
}

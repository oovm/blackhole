//! 静止观测者与赤道圆轨道发射体之间的独立频移参考。

use ton_types::{TonError, TonResult};
use crate::SchwarzschildCartesianNullState;

/// 相机向场景追踪的空间速度与实际到达观测者的光子动量相反。
/// 返回 `nu_observer / nu_emitter`，观测者位于有限半径且保持静止。
/// 只支持稳定赤道圆轨道，不裁剪频移或覆盖光子状态。
pub fn schwarzschild_disk_frequency_shift(
    mass: f64, observer_radius: f64, reverse_ray: SchwarzschildCartesianNullState,
) -> TonResult<f64> {
    let radius = reverse_ray.position.length();
    if !mass.is_finite() || mass <= 0.0 || !observer_radius.is_finite() || observer_radius <= 2.0 * mass
        || !reverse_ray.position.is_finite() || !reverse_ray.velocity.is_finite()
        || !radius.is_finite() || radius < 6.0 * mass
        || reverse_ray.position.z.abs() > 1.0e-4 * mass
        || !reverse_ray.energy.is_finite() || reverse_ray.energy <= 0.0 {
        return Err(TonError::Domain("invalid circular disk frequency input".into()));
    }
    let omega = (mass / radius.powi(3)).sqrt();
    let emitter_time = (1.0 - 3.0 * mass / radius).sqrt().recip();
    let reverse_angular = reverse_ray.angular_momentum().z;
    let emitter_frequency = emitter_time * (reverse_ray.energy + omega * reverse_angular);
    let observer_frequency = reverse_ray.energy / (1.0 - 2.0 * mass / observer_radius).sqrt();
    let shift = observer_frequency / emitter_frequency;
    if !emitter_frequency.is_finite() || emitter_frequency <= 0.0 || !shift.is_finite() || shift <= 0.0 {
        return Err(TonError::Domain("nonpositive circular disk frequency".into()));
    }
    Ok(shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::Vec3;

    #[test]
    fn circular_disk_frequency_matches_analytic_sign_and_observer_lapse() {
        let state = SchwarzschildCartesianNullState { t: 0.0, position: Vec3::new(8.0, 0.0, 0.0),
            energy: 1.0, velocity: Vec3::new(0.0, 0.0, 1.0) };
        let transverse = schwarzschild_disk_frequency_shift(1.0, 24.0, state).unwrap();
        assert!((transverse - ((1.0f64 - 3.0 / 8.0) / (1.0 - 2.0 / 24.0)).sqrt()).abs() < 1.0e-14);
        let receding = schwarzschild_disk_frequency_shift(1.0, 24.0,
            SchwarzschildCartesianNullState { velocity: Vec3::new(0.0, 0.5, 1.0), ..state }).unwrap();
        let approaching = schwarzschild_disk_frequency_shift(1.0, 24.0,
            SchwarzschildCartesianNullState { velocity: Vec3::new(0.0, -0.5, 1.0), ..state }).unwrap();
        assert!(receding < transverse && approaching > transverse);
        let distant = schwarzschild_disk_frequency_shift(1.0, 1.0e12, state).unwrap();
        assert!(distant < transverse);
        assert!(schwarzschild_disk_frequency_shift(1.0, 2.0, state).is_err());
        assert!(schwarzschild_disk_frequency_shift(1.0, 24.0,
            SchwarzschildCartesianNullState { position: Vec3::new(5.9, 0.0, 0.0), ..state }).is_err());
    }
}

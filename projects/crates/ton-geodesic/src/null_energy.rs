//! 零测地线的守恒能量降阶参考，避免积分近视界发散的时间四速度。

use ton_types::{TonError, TonResult};
use crate::SphericalState;

/// 降阶状态，`energy = (1 - 2M/r) u^t` 是常量而非时间四速度。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildNullEnergyState {
    /// 坐标时间。
    pub t: f64,
    /// 半径。
    pub r: f64,
    /// 极角。
    pub theta: f64,
    /// 方位角。
    pub phi: f64,
    /// 守恒能量。
    pub energy: f64,
    /// 径向速度。
    pub ur: f64,
    /// 极角速度。
    pub utheta: f64,
    /// 方位角速度。
    pub uphi: f64,
}

impl SchwarzschildNullEnergyState {
    /// 只接受有限、视界外、未来指向且满足零约束的初值。
    pub fn from_spherical(mass: f64, state: SphericalState) -> TonResult<Self> {
        let values = [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi];
        if !mass.is_finite() || mass <= 0.0 || state.r <= 2.0 * mass || state.ut <= 0.0
            || values.iter().any(|value| !value.is_finite())
        { return Err(TonError::Domain("invalid null energy state".into())); }
        let f = (state.r - 2.0 * mass) / state.r;
        let energy = f * state.ut;
        let spatial = state.ur.powi(2) + f * state.r.powi(2) *
            (state.utheta.powi(2) + state.theta.sin().powi(2) * state.uphi.powi(2));
        let residual = (spatial - energy.powi(2)).abs() / (spatial.abs() + energy.powi(2));
        if !residual.is_finite() || residual > 1.0e-6 {
            return Err(TonError::Domain("null energy reduction requires a null initial state".into()));
        }
        Ok(Self { t: state.t, r: state.r, theta: state.theta, phi: state.phi, energy,
            ur: state.ur, utheta: state.utheta, uphi: state.uphi })
    }

    /// 重建时间四速度用于与完整度规参考对照，不回写投影后的速度。
    pub fn to_spherical(self, mass: f64) -> SphericalState {
        SphericalState { t: self.t, r: self.r, theta: self.theta, phi: self.phi,
            ut: self.energy / ((self.r - 2.0 * mass) / self.r),
            ur: self.ur, utheta: self.utheta, uphi: self.uphi }
    }
}

fn derivative(mass: f64, state: [f64; 8]) -> [f64; 8] {
    let [_, radius, theta, _, energy, ur, utheta, uphi] = state;
    let f = (radius - 2.0 * mass) / radius;
    let (sin, cos) = theta.sin_cos();
    let angular = utheta.powi(2) + sin.powi(2) * uphi.powi(2);
    [energy / f, ur, utheta, uphi, 0.0,
        (radius - 3.0 * mass) * angular,
        -2.0 * ur * utheta / radius + sin * cos * uphi.powi(2),
        -2.0 * ur * uphi / radius - 2.0 * cos / sin * utheta * uphi]
}

/// 独立 RK4 单步。由零约束消去径向方程中的相消项，能量导数恒为零。
///
/// 本函数没有速度归一化或守恒量投影。它只适用于零测地线，不替代通用
/// 时类测地线方程。调用方仍须误差控制、事件检查与坐标定义域检查。
pub fn schwarzschild_null_energy_rk4_step(
    mass: f64, state: SchwarzschildNullEnergyState, step: f64,
) -> SchwarzschildNullEnergyState {
    let initial = [state.t, state.r, state.theta, state.phi, state.energy, state.ur, state.utheta, state.uphi];
    let add = |slope: [f64; 8], scale: f64| std::array::from_fn(|index| initial[index] + scale * slope[index]);
    let k1 = derivative(mass, initial);
    let k2 = derivative(mass, add(k1, step * 0.5));
    let k3 = derivative(mass, add(k2, step * 0.5));
    let k4 = derivative(mass, add(k3, step));
    let next: [f64; 8] = std::array::from_fn(|index|
        initial[index] + step * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0);
    SchwarzschildNullEnergyState { t: next[0], r: next[1], theta: next[2], phi: next[3], energy: next[4], ur: next[5], utheta: next[6], uphi: next[7] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schwarzschild_geodesic_initial_state, schwarzschild_geodesic_rk4_step};
    use ton_types::Vec3;

    #[test]
    fn reduced_null_step_matches_full_metric_reference() {
        let initial = schwarzschild_geodesic_initial_state(1.0, Vec3::new(12.0, 3.0, 5.0), Vec3::new(-1.0, 0.2, -0.1)).unwrap();
        let reduced = SchwarzschildNullEnergyState::from_spherical(1.0, initial).unwrap();
        let actual = schwarzschild_null_energy_rk4_step(1.0, reduced, 0.01);
        let expected = schwarzschild_geodesic_rk4_step(1.0, initial, 0.01);
        let actual = actual.to_spherical(1.0);
        for (actual, expected) in [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi]
            .into_iter().zip([expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi])
        { assert!((actual - expected).abs() < 1.0e-10); }
    }

    #[test]
    fn near_horizon_radial_infall_preserves_energy_and_affine_radius() {
        let initial = schwarzschild_geodesic_initial_state(1.0, Vec3::new(2.01, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)).unwrap();
        let reduced = SchwarzschildNullEnergyState::from_spherical(1.0, initial).unwrap();
        let actual = schwarzschild_null_energy_rk4_step(1.0, reduced, 0.002);
        assert_eq!(actual.energy, reduced.energy);
        assert_eq!(actual.ur, reduced.ur);
        assert!((actual.r - (reduced.r + 0.002 * reduced.ur)).abs() < 1.0e-14);
        let non_null = SphericalState { ut: initial.ut * 1.1, ..initial };
        assert!(SchwarzschildNullEnergyState::from_spherical(1.0, non_null).is_err());
    }
}

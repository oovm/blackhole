//! 面积半径笛卡尔坐标中的独立零测地线参考，不存在角坐标轴奇异性。

use ton_types::{TonError, TonResult, Vec3};
use crate::{SchwarzschildNullEnergyState, SphericalState};

/// 空间坐标是 `r n`，并非平直时空坐标，速度按仿射参数定义。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildCartesianNullState {
    /// Schwarzschild 坐标时间。
    pub t: f64,
    /// 面积半径笛卡尔位置。
    pub position: Vec3,
    /// 守恒能量 `E = f u^t`。
    pub energy: f64,
    /// 仿射空间速度，不归一化。
    pub velocity: Vec3,
}

impl SchwarzschildCartesianNullState {
    /// 从相机的欧氏方向定义仿射初速，零约束决定未来指向能量。
    /// 不经过球坐标，因此相机位于坐标轴时也有定义。
    pub fn from_ray(mass: f64, position: Vec3, direction: Vec3) -> TonResult<Self> {
        if !valid_position(mass, position) || !direction.is_finite()
            || !direction.length().is_finite() || direction.length() <= f64::EPSILON {
            return Err(TonError::Domain("invalid Cartesian null ray".into()));
        }
        let velocity = direction.normalize();
        let radius = position.length();
        let radial = position.dot(velocity) / radius;
        let tangent = velocity.sub(position.scale(radial / radius));
        let energy = (radial * radial + ((radius - 2.0 * mass) / radius) * tangent.dot(tangent)).sqrt();
        Ok(Self { t: 0.0, position, energy, velocity })
    }

    /// 转换有效零初态，不修改或归一化四速度。
    pub fn from_spherical(mass: f64, state: SphericalState) -> TonResult<Self> {
        let reduced = SchwarzschildNullEnergyState::from_spherical(mass, state)?;
        let (sin_theta, cos_theta) = state.theta.sin_cos();
        let (sin_phi, cos_phi) = state.phi.sin_cos();
        let radial = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        let theta = Vec3::new(cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta);
        let phi = Vec3::new(-sin_phi, cos_phi, 0.0);
        Ok(Self { t: state.t, position: radial.scale(state.r), energy: reduced.energy,
            velocity: radial.scale(state.ur).add(theta.scale(state.r * state.utheta))
                .add(phi.scale(state.r * sin_theta * state.uphi)) })
    }

    /// 当前角动量向量，不用于投影或积分强制修正。
    pub fn angular_momentum(self) -> Vec3 { self.position.cross(self.velocity) }

    /// 零约束相对残差，在外部坐标轴上同样有效。
    pub fn null_residual(self, mass: f64) -> f64 {
        let radius = self.position.length();
        let radial = self.position.dot(self.velocity) / radius;
        let angular = self.angular_momentum();
        let spatial = radial * radial + ((radius - 2.0 * mass) / radius)
            * angular.dot(angular) / (radius * radius);
        let temporal = self.energy * self.energy;
        (spatial - temporal).abs() / (spatial.abs() + temporal.abs())
    }

    pub(crate) fn components(self) -> [f64; 8] {
        [self.t, self.position.x, self.position.y, self.position.z, self.energy,
            self.velocity.x, self.velocity.y, self.velocity.z]
    }

    fn from_components(values: [f64; 8]) -> Self {
        Self { t: values[0], position: Vec3::new(values[1], values[2], values[3]),
            energy: values[4], velocity: Vec3::new(values[5], values[6], values[7]) }
    }
}

fn valid_position(mass: f64, position: Vec3) -> bool {
    mass.is_finite() && mass > 0.0 && position.is_finite() && position.length().is_finite()
        && position.length() > 2.0 * mass
}

fn derivative(mass: f64, values: [f64; 8]) -> TonResult<[f64; 8]> {
    let state = SchwarzschildCartesianNullState::from_components(values);
    if !valid_position(mass, state.position) || values.iter().any(|value| !value.is_finite()) || state.energy <= 0.0 {
        return Err(TonError::Domain("Cartesian null RK4 stage outside coordinate domain".into()));
    }
    let radius = state.position.length();
    let angular = state.angular_momentum();
    let coefficient = -3.0 * mass * angular.dot(angular) / radius.powi(5);
    let acceleration = state.position.scale(coefficient);
    Ok([state.energy / ((radius - 2.0 * mass) / radius),
        state.velocity.x, state.velocity.y, state.velocity.z, 0.0,
        acceleration.x, acceleration.y, acceleration.z])
}

/// 独立 RK4 单步，所有中间状态必须在视界外，失败不修改输入。
///
/// 球坐标零约束给出 `r''=(r-3M)|n'|²`，角方程抵消笛卡尔切向加速度，
/// 因而 `x''=-3M|x×v|²x/r⁵`。这里用当前状态计算角动量，不固定或投影它。
pub fn schwarzschild_cartesian_null_rk4_step(
    mass: f64, state: SchwarzschildCartesianNullState, step: f64,
) -> TonResult<SchwarzschildCartesianNullState> {
    if !step.is_finite() || step <= 0.0 {
        return Err(TonError::Domain("invalid Cartesian null RK4 step".into()));
    }
    let initial = state.components();
    let add = |slope: [f64; 8], scale: f64| std::array::from_fn(|index| initial[index] + scale * slope[index]);
    let k1 = derivative(mass, initial)?;
    let k2 = derivative(mass, add(k1, step * 0.5))?;
    let k3 = derivative(mass, add(k2, step * 0.5))?;
    let k4 = derivative(mass, add(k3, step))?;
    let next = std::array::from_fn(|index| initial[index]
        + step * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0);
    derivative(mass, next)?;
    Ok(SchwarzschildCartesianNullState::from_components(next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schwarzschild_geodesic_initial_state, schwarzschild_geodesic_rk4_step};

    #[test]
    fn cartesian_null_step_matches_full_metric_reference() {
        for position in [Vec3::new(12.0, 3.0, 5.0), Vec3::new(5.0, -2.0, -10.0)] {
            let direction = Vec3::new(-1.0, 0.2, -0.1);
            let spherical = schwarzschild_geodesic_initial_state(1.0, position, direction).unwrap();
            let state = SchwarzschildCartesianNullState::from_ray(1.0, position, direction).unwrap();
            let converted = SchwarzschildCartesianNullState::from_spherical(1.0, spherical).unwrap();
            assert!(state.position.sub(converted.position).length() < 1.0e-13);
            assert!(state.velocity.sub(converted.velocity).length() < 1.0e-13);
            assert!((state.energy - converted.energy).abs() < 1.0e-13);
            let actual = schwarzschild_cartesian_null_rk4_step(1.0, state, 0.01).unwrap();
            let expected = SchwarzschildCartesianNullState::from_spherical(1.0,
                schwarzschild_geodesic_rk4_step(1.0, spherical, 0.01)).unwrap();
            assert!(actual.position.sub(expected.position).length() < 1.0e-10);
            assert!(actual.velocity.sub(expected.velocity).length() < 1.0e-10);
            assert!((actual.t - expected.t).abs() < 1.0e-10);
        }
    }

    #[test]
    fn cartesian_radial_axis_infall_has_exact_affine_motion() {
        let initial = SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 2.01), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let result = schwarzschild_cartesian_null_rk4_step(1.0, initial, 0.002).unwrap();
        assert_eq!(result.velocity, initial.velocity);
        assert_eq!(result.energy, initial.energy);
        assert!((result.position.z - 2.008).abs() < 1.0e-14);
        assert_eq!(result.null_residual(1.0), 0.0);
        assert!(schwarzschild_cartesian_null_rk4_step(1.0, initial, 0.02).is_err());
        assert!(SchwarzschildCartesianNullState::from_ray(1.0, initial.position, Vec3::ZERO).is_err());
    }

    #[test]
    fn cartesian_ray_crosses_coordinate_axis_without_singular_state() {
        let initial = SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(-0.05, 0.0, 10.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let result = schwarzschild_cartesian_null_rk4_step(1.0, initial, 0.1).unwrap();
        assert!(result.position.x > 0.0 && result.position.y == 0.0);
        assert!(result.null_residual(1.0) < 1.0e-11);
        assert!(result.angular_momentum().sub(initial.angular_momentum()).length() < 1.0e-10);
    }

    #[test]
    fn cartesian_problem_rays_escape_with_converging_invariants() {
        let camera = ton_types::Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        for index in [1712usize, 7503] {
            let u = (((index % 128) as f64 + 0.5) / 128.0 * 2.0 - 1.0) * 128.0 / 72.0;
            let v = 1.0 - ((index / 128) as f64 + 0.5) / 72.0 * 2.0;
            let initial = SchwarzschildCartesianNullState::from_ray(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
            let angular = initial.angular_momentum();
            let angular_squared = angular.dot(angular);
            let mut errors = Vec::new();
            for step in [0.08, 0.04] {
                let mut current = initial;
                let mut maximum_null = 0.0f64;
                let mut maximum_angular = 0.0f64;
                let mut escaped = false;
                for _ in 0..8192 {
                    current = schwarzschild_cartesian_null_rk4_step(1.0, current, step).unwrap();
                    let momentum = current.angular_momentum();
                    maximum_null = maximum_null.max(current.null_residual(1.0));
                    maximum_angular = maximum_angular.max((momentum.dot(momentum) / angular_squared - 1.0).abs());
                    if current.position.length() >= 80.0 && current.position.dot(current.velocity) > 0.0 { escaped = true; break; }
                }
                assert!(escaped);
                assert!(maximum_null < 1.0e-7 && maximum_angular < 1.0e-7);
                errors.push(maximum_null.max(maximum_angular));
            }
            assert!(errors[1] < errors[0] * 0.2, "ray={index}, errors={errors:?}");
        }
    }
}

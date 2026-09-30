//! 史瓦西度规下零测地线 Christoffel 右端项（坐标基 t, r, θ, φ）。

use ton_types::{BlackHole, Vec3};

use crate::spherical::SphericalCoords;

/// 由守恒量 `b = L/E` 计算冲击参数（与积分器初值一致，优于远场欧氏 `|r×k|`）。
pub fn schwarzschild_geodesic_impact_parameter(
    mass: f64,
    position: Vec3,
    direction: Vec3,
) -> Option<f64> {
    let u = schwarzschild_null_four_velocity(mass, position, direction)?;
    let sph = SphericalCoords::from_cartesian(position);
    let r = sph.r;
    let f = 1.0 - 2.0 * mass / r;
    if f <= 0.0 {
        return None;
    }
    let sin_t = sph.theta.sin();
    let energy = f * u[0];
    let angular = r * r * sin_t * sin_t * u[3];
    if energy <= 0.0 || !energy.is_finite() {
        return None;
    }
    Some(angular.abs() / energy)
}

/// 史瓦西解析捕获：`b < b_c` 或起点在视界内。
pub fn schwarzschild_geodesic_analytic_captured(
    bh: BlackHole,
    position: Vec3,
    direction: Vec3,
) -> bool {
    if position.length() <= bh.event_horizon_radius() {
        return true;
    }
    match schwarzschild_geodesic_impact_parameter(bh.mass, position, direction) {
        Some(b) => b < bh.critical_impact_parameter(),
        None => false,
    }
}

/// 由笛卡尔位置与方向构造零测地线四速度 `(u^t, u^r, u^θ, u^φ)`。
pub fn schwarzschild_null_four_velocity(
    mass: f64,
    position: Vec3,
    direction: Vec3,
) -> Option<[f64; 4]> {
    let sph = SphericalCoords::from_cartesian(position);
    let r = sph.r;
    if r <= f64::EPSILON {
        return None;
    }
    let sin_t = sph.theta.sin().max(1.0e-12);
    let f = 1.0 - 2.0 * mass / r;
    if f <= 0.0 {
        return None;
    }
    let dir = direction.normalize();
    let e_r = position.scale(1.0 / r);
    let e_theta = Vec3::new(
        sph.theta.cos() * sph.phi.cos(),
        sph.theta.cos() * sph.phi.sin(),
        -sph.theta.sin(),
    );
    let e_phi = Vec3::new(-sph.phi.sin(), sph.phi.cos(), 0.0);
    let ur = dir.dot(e_r);
    let utheta = dir.dot(e_theta) / r;
    let uphi = dir.dot(e_phi) / (r * sin_t);
    let spatial = (1.0 / f) * ur * ur + r * r * utheta * utheta + r * r * sin_t * sin_t * uphi * uphi;
    if spatial <= 0.0 || !spatial.is_finite() {
        return None;
    }
    let ut = spatial.sqrt() / f.sqrt();
    Some([ut, ur, utheta, uphi])
}

/// `du^μ/dλ = -Γ^μ_{αβ} u^α u^β`（史瓦西对角度规解析 Christoffel）。
pub fn schwarzschild_geodesic_accel(mass: f64, r: f64, theta: f64, u: [f64; 4]) -> [f64; 4] {
    let f = 1.0 - 2.0 * mass / r;
    let fp = 2.0 * mass / (r * r);
    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let sin2 = sin_t * sin_t;
    let (ut, ur, utheta, uphi) = (u[0], u[1], u[2], u[3]);

    let gamma_t_tr = fp / (2.0 * f);
    let du_t = -2.0 * gamma_t_tr * ut * ur;

    let gamma_r_tt = f * fp / 2.0;
    let gamma_r_rr = -fp / (2.0 * f);
    let gamma_r_tt_theta = -f * r;
    let gamma_r_pp = -f * r * sin2;
    let du_r = -(gamma_r_tt * ut * ut
        + gamma_r_rr * ur * ur
        + gamma_r_tt_theta * utheta * utheta
        + gamma_r_pp * uphi * uphi);

    let inv_r = 1.0 / r;
    let gamma_theta_rtheta = inv_r;
    let gamma_theta_pp = -sin_t * cos_t;
    let du_theta = -(2.0 * gamma_theta_rtheta * ur * utheta + gamma_theta_pp * uphi * uphi);

    let cot = if sin_t.abs() > 1.0e-12 {
        cos_t / sin_t
    } else {
        0.0
    };
    let gamma_phi_rphi = inv_r;
    let gamma_phi_thetaphi = cot;
    let du_phi = -(2.0 * gamma_phi_rphi * ur * uphi + 2.0 * gamma_phi_thetaphi * utheta * uphi);

    [du_t, du_r, du_theta, du_phi]
}

/// 四速度 `(u^r, u^θ, u^φ)` 在球坐标下对应的笛卡尔切向量（未归一化）。
pub fn spherical_velocity_to_cartesian(
    r: f64,
    theta: f64,
    phi: f64,
    ur: f64,
    utheta: f64,
    uphi: f64,
) -> Vec3 {
    let sin_t = theta.sin();
    let e_r = Vec3::new(
        sin_t * phi.cos(),
        sin_t * phi.sin(),
        theta.cos(),
    );
    let e_theta = Vec3::new(
        theta.cos() * phi.cos(),
        theta.cos() * phi.sin(),
        -theta.sin(),
    );
    let e_phi = Vec3::new(-phi.sin(), phi.cos(), 0.0);
    e_r
        .scale(ur)
        .add(e_theta.scale(r * utheta))
        .add(e_phi.scale(r * sin_t * uphi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::BlackHole;

    #[test]
    fn geodesic_impact_parameter_matches_euclidean_at_far_radius() {
        let bh = BlackHole::schwarzschild(1.0);
        let pos = Vec3::new(10.0, 0.0, 0.0);
        let dir = Vec3::new(-1.0, 0.15, 0.05).normalize();
        let b_geo = schwarzschild_geodesic_impact_parameter(bh.mass, pos, dir).expect("b");
        let b_euc = pos.cross(dir).length();
        assert!((b_geo - b_euc).abs() / b_geo < 0.08, "geo={b_geo} euc={b_euc}");
    }

    #[test]
    fn null_velocity_satisfies_quadratic_constraint() {
        let bh = BlackHole::schwarzschild(1.0);
        let pos = Vec3::new(10.0, 0.0, 0.0);
        let dir = Vec3::new(-1.0, 0.0, 0.0);
        let u = schwarzschild_null_four_velocity(bh.mass, pos, dir).expect("null init");
        let r = 10.0;
        let f = 1.0 - 2.0 * bh.mass / r;
        let quad = -f * u[0] * u[0]
            + (1.0 / f) * u[1] * u[1]
            + r * r * u[2] * u[2]
            + r * r * u[3] * u[3];
        assert!(quad.abs() < 1.0e-9, "null constraint violated: {quad}");
    }
}

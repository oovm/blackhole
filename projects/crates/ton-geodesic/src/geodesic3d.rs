//! 史瓦西 3D 零测地线（Christoffel + RK4，独立于赤道轨道方程）。

use ton_metric::{
    schwarzschild_geodesic_accel, schwarzschild_null_four_velocity,
    spherical_velocity_to_cartesian, SphericalCoords,
};
use ton_types::{BlackHole, Rgb, TonError, TonResult, Vec3};

use crate::radiance::RayRadiance;

use crate::ray::RayHit;
use crate::trace::TraceConfig;

#[derive(Debug, Clone, Copy)]
struct GeodesicState {
    t: f64,
    r: f64,
    theta: f64,
    phi: f64,
    u: [f64; 4],
}

impl GeodesicState {
    fn cartesian(&self) -> Vec3 {
        SphericalCoords {
            r: self.r,
            theta: self.theta,
            phi: self.phi,
        }
        .to_cartesian()
    }

    fn tangent_cartesian(&self) -> Vec3 {
        spherical_velocity_to_cartesian(self.r, self.theta, self.phi, self.u[1], self.u[2], self.u[3])
    }
}

fn state_deriv(mass: f64, state: GeodesicState) -> ([f64; 4], [f64; 4]) {
    let accel = schwarzschild_geodesic_accel(mass, state.r, state.theta, state.u);
    (state.u, accel)
}

fn state_add(state: GeodesicState, deriv: ([f64; 4], [f64; 4]), scale: f64) -> GeodesicState {
    GeodesicState {
        t: state.t + deriv.0[0] * scale,
        r: state.r + deriv.0[1] * scale,
        theta: state.theta + deriv.0[2] * scale,
        phi: state.phi + deriv.0[3] * scale,
        u: [
            state.u[0] + deriv.1[0] * scale,
            state.u[1] + deriv.1[1] * scale,
            state.u[2] + deriv.1[2] * scale,
            state.u[3] + deriv.1[3] * scale,
        ],
    }
}

fn rk4_geodesic_step(mass: f64, state: GeodesicState, dlambda: f64) -> GeodesicState {
    let k1 = state_deriv(mass, state);
    let k2 = state_deriv(mass, state_add(state, k1, dlambda * 0.5));
    let k3 = state_deriv(mass, state_add(state, k2, dlambda * 0.5));
    let k4 = state_deriv(mass, state_add(state, k3, dlambda));
    let blend = |a: [f64; 4], b: [f64; 4], c: [f64; 4], d: [f64; 4]| {
        [
            (a[0] + 2.0 * b[0] + 2.0 * c[0] + d[0]) / 6.0,
            (a[1] + 2.0 * b[1] + 2.0 * c[1] + d[1]) / 6.0,
            (a[2] + 2.0 * b[2] + 2.0 * c[2] + d[2]) / 6.0,
            (a[3] + 2.0 * b[3] + 2.0 * c[3] + d[3]) / 6.0,
        ]
    };
    let coord = blend(k1.0, k2.0, k3.0, k4.0);
    let accel = blend(k1.1, k2.1, k3.1, k4.1);
    state_add(state, (coord, accel), dlambda)
}

fn disk_crossing(prev: &GeodesicState, next: &GeodesicState, inner: f64, outer: f64, rs: f64) -> Option<f64> {
    let z0 = prev.cartesian().z;
    let z1 = next.cartesian().z;
    if z0.signum() == z1.signum() {
        return None;
    }
    let t = z0.abs() / (z0.abs() + z1.abs());
    let hit = prev.cartesian().add(next.cartesian().sub(prev.cartesian()).scale(t));
    let r = hit.length();
    if r > rs && r >= inner && r <= outer {
        Some(r)
    } else {
        None
    }
}

/// 完整 3D 零测地线：Christoffel RK4，与 `SchwarzschildOrbit` 独立 golden。
pub fn trace_geodesic_3d(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
) -> TonResult<RayHit> {
    let mass = bh.mass;
    if !mass.is_finite() || mass <= 0.0 {
        return Err(TonError::Domain("mass must be finite and positive".into()));
    }
    let rs = bh.event_horizon_radius();
    if origin.length() <= rs {
        return Err(TonError::Captured);
    }
    let u0 = schwarzschild_null_four_velocity(mass, origin, direction)
        .ok_or_else(|| TonError::Domain("failed to initialize null four-velocity".into()))?;
    let sph = SphericalCoords::from_cartesian(origin);
    let mut state = GeodesicState {
        t: 0.0,
        r: sph.r,
        theta: sph.theta,
        phi: sph.phi,
        u: u0,
    };
    let dlambda = config.step_size;
    for step in 0..config.max_steps {
        if state.r <= rs {
            return Ok(RayHit::Captured {
                position: state.cartesian(),
            });
        }
        if state.r >= config.escape_radius {
            return Ok(RayHit::Escaped {
                direction: state.tangent_cartesian().normalize(),
            });
        }
        if !state.r.is_finite() || state.u.iter().any(|v| !v.is_finite()) {
            return Err(TonError::Domain("non-finite geodesic state".into()));
        }
        let prev = state;
        state = rk4_geodesic_step(mass, prev, dlambda);
        if step > 0 {
            if let (Some(inner), Some(outer)) = (config.disk_inner, config.disk_outer) {
                if let Some(radius) = disk_crossing(&prev, &state, inner, outer, rs) {
                    return Ok(RayHit::Disk {
                        position: state.cartesian(),
                        radius,
                    });
                }
            }
        }
    }
    Err(TonError::Domain("max steps exceeded".into()))
}

/// 3D 测地线 + 薄盘多次穿越 HDR 累积；`emit_disk` 返回局部发射，`sky` 在逃逸时乘透射率叠加。
pub fn trace_geodesic_3d_radiance<F, S>(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
    mut emit_disk: F,
    sky: S,
) -> TonResult<RayRadiance>
where
    F: FnMut(f64, Vec3) -> Rgb,
    S: Fn(Vec3) -> Rgb,
{
    const MAX_DISK_HITS: u32 = 8;
    let mass = bh.mass;
    if !mass.is_finite() || mass <= 0.0 {
        return Err(TonError::Domain("mass must be finite and positive".into()));
    }
    let rs = bh.event_horizon_radius();
    if origin.length() <= rs {
        return Ok(RayRadiance::BLACK);
    }
    let u0 = schwarzschild_null_four_velocity(mass, origin, direction)
        .ok_or_else(|| TonError::Domain("failed to initialize null four-velocity".into()))?;
    let sph = SphericalCoords::from_cartesian(origin);
    let mut state = GeodesicState {
        t: 0.0,
        r: sph.r,
        theta: sph.theta,
        phi: sph.phi,
        u: u0,
    };
    let dlambda = config.step_size;
    let mut radiance = Rgb::BLACK;
    let mut transmittance = 1.0;
    let mut disk_hits = 0u32;

    for step in 0..config.max_steps {
        if state.r <= rs {
            return Ok(RayRadiance::new(radiance));
        }
        if state.r >= config.escape_radius {
            let sky_dir = state.tangent_cartesian().normalize();
            return Ok(RayRadiance::new(
                radiance.add(sky(sky_dir).scale(transmittance)),
            ));
        }
        if !state.r.is_finite() || state.u.iter().any(|v| !v.is_finite()) {
            return Err(TonError::Domain("non-finite geodesic state".into()));
        }
        let prev = state;
        state = rk4_geodesic_step(mass, prev, dlambda);
        if step > 0 {
            if let (Some(inner), Some(outer)) = (config.disk_inner, config.disk_outer) {
                if disk_hits < MAX_DISK_HITS {
                    if let Some(radius) = disk_crossing(&prev, &state, inner, outer, rs) {
                        let hit_pos = prev
                            .cartesian()
                            .add(state.cartesian().sub(prev.cartesian()).scale(0.5));
                        let local = emit_disk(radius, hit_pos);
                        radiance = radiance.add(local.scale(transmittance));
                        let absorption = local.r.max(local.g).max(local.b) * 0.12;
                        transmittance *= (1.0 - absorption).clamp(0.05, 1.0);
                        disk_hits += 1;
                    }
                }
            }
        }
    }
    Err(TonError::Domain("max steps exceeded".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::TonError;

    #[test]
    fn radial_inward_is_captured() {
        let bh = BlackHole::schwarzschild(1.0);
        let hit = trace_geodesic_3d(
            bh,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        )
        .unwrap();
        assert!(matches!(hit, RayHit::Captured { .. }));
    }

    #[test]
    fn rejects_non_positive_mass() {
        let result = trace_geodesic_3d(
            BlackHole::schwarzschild(0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        );
        assert!(matches!(result, Err(TonError::Domain(_))));
    }
}

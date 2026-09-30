//! 克尔赤道平面 null 轨道积分（`u(φ)` RK4，自旋修正 RHS）。

use ton_metric::SphericalCoords;
use ton_types::{BlackHole, TonError, TonResult, Vec3};

use crate::orbit::{kerr_rk4_step, OrbitState};
use crate::ray::RayHit;
use crate::trace::TraceConfig;

fn spherical_to_cartesian(r: f64, phi: f64) -> Vec3 {
    SphericalCoords {
        r,
        theta: std::f64::consts::FRAC_PI_2,
        phi,
    }
    .to_cartesian()
}

/// 克尔赤道近似：与 `SchwarzschildOrbit` 同族接口，独立 golden。
pub fn trace_kerr_equatorial(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
) -> TonResult<RayHit> {
    let mass = bh.mass;
    if !mass.is_finite() || mass <= 0.0 {
        return Err(TonError::Domain("mass must be finite and positive".into()));
    }
    if bh.spin.abs() >= mass {
        return Err(TonError::Domain("spin must satisfy |a| < M".into()));
    }
    let rs = bh.event_horizon_radius();
    let start = SphericalCoords::from_cartesian(origin);
    if start.r <= rs {
        return Err(TonError::Captured);
    }
    let dir = direction.normalize();
    let r0 = start.r;
    let phi0 = start.phi;
    let radial = origin.scale(1.0 / r0);
    let radial_velocity = dir.dot(radial);
    let angular_momentum = origin.cross(dir).z.abs();
    if angular_momentum <= 1.0e-9 {
        if radial_velocity < 0.0 {
            return Ok(RayHit::Captured {
                position: origin,
            });
        }
        return Ok(RayHit::Escaped { direction: dir });
    }
    let du_dphi = -radial_velocity / angular_momentum;
    let mut state = OrbitState::new(r0, du_dphi, phi0);
    for step in 0..config.max_steps {
        let r = state.radius();
        if r <= rs {
            return Ok(RayHit::Captured {
                position: spherical_to_cartesian(r, state.phi),
            });
        }
        if step > 0 {
            if let (Some(inner), Some(outer)) = (config.disk_inner, config.disk_outer) {
                if r > rs && r >= inner && r <= outer {
                    return Ok(RayHit::Disk {
                        position: spherical_to_cartesian(r, state.phi),
                        radius: r,
                    });
                }
            }
        }
        if r >= config.escape_radius {
            return Ok(RayHit::Escaped {
                direction: Vec3::new(state.phi.cos(), state.phi.sin(), 0.0).normalize(),
            });
        }
        state = kerr_rk4_step(mass, bh.spin, state, config.dphi);
        if state.u <= 0.0 || !state.u.is_finite() {
            return Err(TonError::Escaped);
        }
    }
    Err(TonError::Domain("max steps exceeded".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::trace_schwarzschild;
    use ton_types::TonError;

    #[test]
    fn zero_spin_matches_schwarzschild_orbit() {
        let bh = BlackHole::kerr(1.0, 0.0);
        let origin = Vec3::new(10.0, 0.0, 0.0);
        let direction = Vec3::new(-1.0, 0.0, 0.0);
        let cfg = TraceConfig::default();
        let kerr = trace_kerr_equatorial(bh, origin, direction, cfg).unwrap();
        let schw = trace_schwarzschild(bh, origin, direction, cfg).unwrap();
        assert_eq!(std::mem::discriminant(&kerr), std::mem::discriminant(&schw));
    }

    #[test]
    fn rejects_excessive_spin() {
        let bh = BlackHole::kerr(1.0, 1.0);
        let result = trace_kerr_equatorial(
            bh,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        );
        assert!(matches!(result, Err(TonError::Domain(_))));
    }
}

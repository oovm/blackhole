use ton_metric::SphericalCoords;
use ton_types::{BlackHole, TonError, TonResult, Vec3};

use crate::orbit::{rk4_step, OrbitState};
use crate::ray::RayHit;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceConfig {
    pub max_steps: u32,
    pub dphi: f64,
    /// 半经典 Ray March 步长（几何单位）。
    pub step_size: f64,
    /// 薄盘内缘（几何单位）；与 `disk_outer` 成对使用。
    pub disk_inner: Option<f64>,
    /// 薄盘外缘。
    pub disk_outer: Option<f64>,
    pub escape_radius: f64,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            max_steps: 4096,
            dphi: 0.02,
            step_size: 0.2,
            disk_inner: None,
            disk_outer: None,
            escape_radius: 1.0e4,
        }
    }
}

pub fn trace_schwarzschild(
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
        // 跳过 step 0：相机可能在盘内，不得把起点当成撞盘。
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
        state = rk4_step(mass, state, config.dphi);
        if state.u <= 0.0 || !state.u.is_finite() {
            return Err(TonError::Escaped);
        }
    }
    Err(TonError::Domain("max steps exceeded".into()))
}

fn spherical_to_cartesian(r: f64, phi: f64) -> Vec3 {
    SphericalCoords {
        r,
        theta: std::f64::consts::FRAC_PI_2,
        phi,
    }
    .to_cartesian()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::TonError;

    #[test]
    fn radial_rays_skip_azimuthal_integrator() {
        let bh = BlackHole::schwarzschild(1.0);
        let inward = trace_schwarzschild(
            bh,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        )
        .unwrap();
        assert!(matches!(inward, RayHit::Captured { .. }));

        let outward = trace_schwarzschild(
            bh,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            TraceConfig::default(),
        )
        .unwrap();
        assert!(matches!(outward, RayHit::Escaped { .. }));
    }

    #[test]
    fn rejects_non_positive_mass() {
        let result = trace_schwarzschild(
            BlackHole::schwarzschild(0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        );
        assert!(matches!(result, Err(TonError::Domain(_))));
    }
}

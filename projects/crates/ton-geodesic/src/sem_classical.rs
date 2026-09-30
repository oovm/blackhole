//! 半经典 Ray March 积分器（3D，RK4 速度归一化）。

use ton_metric::semi_classical_acceleration;
use ton_types::{BlackHole, TonError, TonResult, Vec3};

use crate::ray::RayHit;
use crate::trace::TraceConfig;

struct RayState {
    pos: Vec3,
    vel: Vec3,
}

fn rk4_velocity_step(bh: BlackHole, pos: Vec3, vel: Vec3, dt: f64) -> (Vec3, Vec3) {
    let accel = |p: Vec3| semi_classical_acceleration(bh, p);
    let k1_v = accel(pos);
    let k1_p = vel;
    let pos2 = pos.add(k1_p.scale(dt * 0.5));
    let vel2 = vel.add(k1_v.scale(dt * 0.5)).normalize();
    let k2_v = accel(pos2);
    let k2_p = vel2;
    let pos3 = pos.add(k2_p.scale(dt * 0.5));
    let vel3 = vel.add(k2_v.scale(dt * 0.5)).normalize();
    let k3_v = accel(pos3);
    let k3_p = vel3;
    let pos4 = pos.add(k3_p.scale(dt));
    let vel4 = vel.add(k3_v.scale(dt)).normalize();
    let k4_v = accel(pos4);
    let k4_p = vel4;
    let new_vel = vel
        .add(k1_v.add(k2_v.scale(2.0)).add(k3_v.scale(2.0)).add(k4_v).scale(dt / 6.0))
        .normalize();
    let new_pos = pos
        .add(
            k1_p
                .add(k2_p.scale(2.0))
                .add(k3_p.scale(2.0))
                .add(k4_p)
                .scale(dt / 6.0),
        );
    (new_pos, new_vel)
}

fn disk_crossing(
    prev: &RayState,
    next: &RayState,
    inner: f64,
    outer: f64,
    rs: f64,
) -> Option<f64> {
    let z0 = prev.pos.z;
    let z1 = next.pos.z;
    if z0.signum() == z1.signum() {
        return None;
    }
    let t = z0.abs() / (z0.abs() + z1.abs());
    let hit = prev.pos.add(next.pos.sub(prev.pos).scale(t));
    let r = hit.length();
    if r > rs && r >= inner && r <= outer {
        Some(r)
    } else {
        None
    }
}

/// 半经典 Ray March：独立于 `SchwarzschildOrbit` 的 golden 路径。
pub fn trace_sem_classical(
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
    let mut state = RayState {
        pos: origin,
        vel: direction.normalize(),
    };
    let dt = config.step_size;
    for step in 0..config.max_steps {
        let r = state.pos.length();
        if r <= rs {
            return Ok(RayHit::Captured {
                position: state.pos,
            });
        }
        if r >= config.escape_radius {
            return Ok(RayHit::Escaped {
                direction: state.vel,
            });
        }
        let prev = state;
        let (new_pos, new_vel) = rk4_velocity_step(bh, prev.pos, prev.vel, dt);
        state = RayState {
            pos: new_pos,
            vel: new_vel,
        };
        if !state.pos.is_finite() || !state.vel.is_finite() {
            return Err(TonError::Domain("non-finite ray state".into()));
        }
        if step > 0 {
            if let (Some(inner), Some(outer)) = (config.disk_inner, config.disk_outer) {
                if let Some(radius) = disk_crossing(&prev, &state, inner, outer, rs) {
                    return Ok(RayHit::Disk {
                        position: state.pos,
                        radius,
                    });
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
    fn inward_radial_is_captured() {
        let bh = BlackHole::schwarzschild(1.0);
        let hit = trace_sem_classical(
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
        let result = trace_sem_classical(
            BlackHole::schwarzschild(0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            TraceConfig::default(),
        );
        assert!(matches!(result, Err(TonError::Domain(_))));
    }
}

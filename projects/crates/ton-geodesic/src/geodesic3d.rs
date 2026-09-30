//! 史瓦西 3D 零测地线（Christoffel + RK4，独立于赤道轨道方程）。

use ton_metric::{
    schwarzschild_geodesic_accel, schwarzschild_null_four_velocity,
    spherical_velocity_to_cartesian, SphericalCoords,
};
use ton_types::{BlackHole, DiskFrame, Rgb, TonError, TonResult, Vec3};

use crate::radiance::{DiskEmission, RadianceTraceOutcome, RayRadiance};
use crate::ray::RayHit;
use crate::termination::{RayTermination, TraceDiagnostic};
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

fn null_residual(mass: f64, state: &GeodesicState) -> f64 {
    let r = state.r;
    let theta = state.theta;
    let f = 1.0 - 2.0 * mass / r.max(mass * 2.0 + 1e-9);
    let sin_t = theta.sin();
    let g = [
        -f,
        1.0 / f,
        r * r,
        r * r * sin_t * sin_t,
    ];
    let u = state.u;
    (g[0] * u[0] * u[0]
        + g[1] * u[1] * u[1]
        + g[2] * u[2] * u[2]
        + g[3] * u[3] * u[3])
        .abs()
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

/// 盘面交点：使用 [`DiskFrame`] 法线判定穿越，与相机/发射约定一致。
fn disk_crossing(
    prev: &GeodesicState,
    next: &GeodesicState,
    frame: DiskFrame,
    inner: f64,
    outer: f64,
    rs: f64,
) -> Option<(f64, Vec3)> {
    let n = frame.normal.normalize();
    let p0 = prev.cartesian();
    let p1 = next.cartesian();
    let d0 = p0.dot(n);
    let d1 = p1.dot(n);
    if d0.signum() == d1.signum() {
        return None;
    }
    let t = d0.abs() / (d0.abs() + d1.abs());
    let hit = p0.add(p1.sub(p0).scale(t));
    let r = hit.length();
    if r > rs && r >= inner && r <= outer {
        Some((r, hit))
    } else {
        None
    }
}

struct IntegrateOutcome {
    diagnostic: TraceDiagnostic,
    hit: Option<RayHit>,
    escape_direction: Option<Vec3>,
}

fn integrate_geodesic<F>(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
    disk_enabled: bool,
    single_disk_hit: bool,
    mut on_disk: F,
) -> IntegrateOutcome
where
    F: FnMut(f64, Vec3) -> bool,
{
    let mass = bh.mass;
    if !mass.is_finite() || mass <= 0.0 {
        return IntegrateOutcome {
            diagnostic: TraceDiagnostic {
                termination: RayTermination::DomainError,
                steps: 0,
                null_residual: 0.0,
                disk_crossings: 0,
            },
            hit: None,
            escape_direction: None,
        };
    }
    let rs = bh.event_horizon_radius();
    if origin.length() <= rs {
        return IntegrateOutcome {
            diagnostic: TraceDiagnostic {
                termination: RayTermination::OriginInsideHorizon,
                steps: 0,
                null_residual: 0.0,
                disk_crossings: 0,
            },
            hit: None,
            escape_direction: None,
        };
    }
    let u0 = match schwarzschild_null_four_velocity(mass, origin, direction) {
        Some(u) => u,
        None => {
            return IntegrateOutcome {
                diagnostic: TraceDiagnostic {
                    termination: RayTermination::InitFailed,
                    steps: 0,
                    null_residual: 0.0,
                    disk_crossings: 0,
                },
                hit: None,
                escape_direction: None,
            };
        }
    };
    let sph = SphericalCoords::from_cartesian(origin);
    let mut state = GeodesicState {
        t: 0.0,
        r: sph.r,
        theta: sph.theta,
        phi: sph.phi,
        u: u0,
    };
    let dlambda = config.step_size;
    let mut disk_crossings = 0u32;
    let mut steps = 0u32;
    let mut residual = null_residual(mass, &state);

    for step in 0..config.max_steps {
        steps = step + 1;
        if state.r <= rs {
            return IntegrateOutcome {
                diagnostic: TraceDiagnostic {
                    termination: RayTermination::Captured,
                    steps,
                    null_residual: residual,
                    disk_crossings,
                },
                hit: Some(RayHit::Captured {
                    position: state.cartesian(),
                }),
                escape_direction: None,
            };
        }
        if state.r >= config.escape_radius {
            let dir = state.tangent_cartesian().normalize();
            return IntegrateOutcome {
                diagnostic: TraceDiagnostic {
                    termination: RayTermination::Escaped,
                    steps,
                    null_residual: residual,
                    disk_crossings,
                },
                hit: Some(RayHit::Escaped { direction: dir }),
                escape_direction: Some(dir),
            };
        }
        if !state.r.is_finite() || state.u.iter().any(|v| !v.is_finite()) {
            return IntegrateOutcome {
                diagnostic: TraceDiagnostic {
                    termination: RayTermination::NonFiniteState,
                    steps,
                    null_residual: residual,
                    disk_crossings,
                },
                hit: None,
                escape_direction: None,
            };
        }
        residual = null_residual(mass, &state);
        let prev = state;
        state = rk4_geodesic_step(mass, prev, dlambda);
        if step > 0 && disk_enabled {
            if let (Some(inner), Some(outer)) = (config.disk_inner, config.disk_outer) {
                if let Some((radius, hit_pos)) =
                    disk_crossing(&prev, &state, config.disk_frame, inner, outer, rs)
                {
                    disk_crossings += 1;
                    let stop = on_disk(radius, hit_pos);
                    if single_disk_hit || stop {
                        return IntegrateOutcome {
                            diagnostic: TraceDiagnostic {
                                termination: RayTermination::DiskCrossing,
                                steps,
                                null_residual: residual,
                                disk_crossings,
                            },
                            hit: Some(RayHit::Disk {
                                position: hit_pos,
                                radius,
                            }),
                            escape_direction: None,
                        };
                    }
                }
            }
        }
    }
    IntegrateOutcome {
        diagnostic: TraceDiagnostic {
            termination: RayTermination::StepBudgetExceeded,
            steps,
            null_residual: residual,
            disk_crossings,
        },
        hit: None,
        escape_direction: None,
    }
}

/// 诊断追迹：返回终止分类、步数、零测地线残差与盘面穿越次数。
pub fn trace_geodesic_3d_diagnostic(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
) -> TraceDiagnostic {
    let disk_on = config.disk_inner.is_some() && config.disk_outer.is_some();
    integrate_geodesic(bh, origin, direction, config, disk_on, true, |_r, _p| true).diagnostic
}

/// 完整 3D 零测地线：Christoffel RK4，与 `SchwarzschildOrbit` 独立 golden。
pub fn trace_geodesic_3d(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
) -> TonResult<RayHit> {
    let disk_on = config.disk_inner.is_some() && config.disk_outer.is_some();
    let out = integrate_geodesic(bh, origin, direction, config, disk_on, true, |_r, _p| true);
    match out.hit {
        Some(hit) => Ok(hit),
        None => match out.diagnostic.termination {
            RayTermination::StepBudgetExceeded => {
                Err(TonError::Domain("max steps exceeded".into()))
            }
            RayTermination::NonFiniteState => {
                Err(TonError::Domain("non-finite geodesic state".into()))
            }
            RayTermination::InitFailed => {
                Err(TonError::Domain("failed to initialize null four-velocity".into()))
            }
            RayTermination::OriginInsideHorizon => Err(TonError::Captured),
            RayTermination::DomainError => {
                Err(TonError::Domain("mass must be finite and positive".into()))
            }
            _ => Err(TonError::Domain("geodesic integration failed".into())),
        },
    }
}

/// 3D 测地线 + 薄盘多次穿越 HDR 累积（单次积分，附带诊断）。
pub fn trace_geodesic_3d_radiance_outcome<F, S>(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
    mut emit_disk: F,
    sky: S,
) -> RadianceTraceOutcome
where
    F: FnMut(f64, Vec3) -> DiskEmission,
    S: Fn(Vec3) -> Rgb,
{
    const MAX_DISK_HITS: u32 = 8;
    let disk_on = config.disk_inner.is_some() && config.disk_outer.is_some();
    let mut radiance = Rgb::BLACK;
    let mut transmittance = 1.0;
    let mut disk_hits = 0u32;

    let out = integrate_geodesic(
        bh,
        origin,
        direction,
        config,
        disk_on,
        false,
        |radius, hit_pos| {
            if disk_hits >= MAX_DISK_HITS {
                return false;
            }
            let sample = emit_disk(radius, hit_pos);
            radiance = radiance.add(sample.radiance.scale(transmittance));
            transmittance *= (1.0 - sample.absorption).clamp(0.0, 1.0);
            disk_hits += 1;
            sample.absorption >= 0.999
        },
    );

    let diagnostic = out.diagnostic;
    let ray = match diagnostic.termination {
        RayTermination::Captured | RayTermination::OriginInsideHorizon => {
            RayRadiance::new(radiance)
        }
        RayTermination::Escaped => {
            let sky_dir = out.escape_direction.unwrap_or(direction);
            RayRadiance::new(radiance.add(sky(sky_dir).scale(transmittance)))
        }
        RayTermination::DiskCrossing => RayRadiance::new(radiance),
        RayTermination::StepBudgetExceeded
        | RayTermination::NonFiniteState
        | RayTermination::InitFailed
        | RayTermination::DomainError => RayRadiance::BLACK,
    };
    RadianceTraceOutcome {
        radiance: ray,
        diagnostic,
    }
}

/// 3D 测地线 + 薄盘多次穿越 HDR 累积。
pub fn trace_geodesic_3d_radiance<F, S>(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
    emit_disk: F,
    sky: S,
) -> Result<RayRadiance, TraceDiagnostic>
where
    F: FnMut(f64, Vec3) -> DiskEmission,
    S: Fn(Vec3) -> Rgb,
{
    let out = trace_geodesic_3d_radiance_outcome(
        bh,
        origin,
        direction,
        config,
        emit_disk,
        sky,
    );
    if out.diagnostic.is_trace_failure() {
        Err(out.diagnostic)
    } else {
        Ok(out.radiance)
    }
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
    fn step_budget_reports_exceeded_not_escape() {
        let bh = BlackHole::schwarzschild(1.0);
        let cfg = TraceConfig {
            max_steps: 4,
            step_size: 0.5,
            escape_radius: 1.0e6,
            ..TraceConfig::default()
        };
        let diag = trace_geodesic_3d_diagnostic(
            bh,
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(-0.2, 0.01, 0.0),
            cfg,
        );
        assert_eq!(diag.termination, RayTermination::StepBudgetExceeded);
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

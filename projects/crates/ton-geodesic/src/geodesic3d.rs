//! 史瓦西 3D 零测地线（Christoffel + RK4，独立于赤道轨道方程）。

use ton_metric::{
    schwarzschild_geodesic_accel, schwarzschild_geodesic_impact_parameter,
    schwarzschild_null_four_velocity, spherical_velocity_to_cartesian, SphericalCoords,
};
use ton_types::{BlackHole, DiskFrame, Rgb, TonError, TonResult, Vec3};

use crate::radiance::{DiskEmission, DiskHit, RadianceTraceOutcome, RayRadiance};
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

/// Schwarzschild 球坐标零测地线的八标量状态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalState {
    /// 坐标时间。
    pub t: f64,
    /// Schwarzschild 半径。
    pub r: f64,
    /// 极角。
    pub theta: f64,
    /// 方位角。
    pub phi: f64,
    /// 时间分量。
    pub ut: f64,
    /// 径向分量。
    pub ur: f64,
    /// 极角分量。
    pub utheta: f64,
    /// 方位角分量。
    pub uphi: f64,
}

/// 使用独立 Rust 参考实现推进一个完整球坐标 RK4 步。
pub fn schwarzschild_geodesic_rk4_step(mass: f64, state: SphericalState, dlambda: f64) -> SphericalState {
    let state = GeodesicState {
        t: state.t,
        r: state.r,
        theta: state.theta,
        phi: state.phi,
        u: [state.ut, state.ur, state.utheta, state.uphi],
    };
    let state = rk4_geodesic_step(mass, state, dlambda);
    SphericalState {
        t: state.t,
        r: state.r,
        theta: state.theta,
        phi: state.phi,
        ut: state.u[0],
        ur: state.u[1],
        utheta: state.u[2],
        uphi: state.u[3],
    }
}

/// 径向 Schwarzschild 零测地线的四标量状态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialState {
    /// 坐标时间。
    pub t: f64,
    /// Schwarzschild 半径。
    pub r: f64,
    /// `u^t` 分量。
    pub ut: f64,
    /// `u^r` 分量。
    pub ur: f64,
}

/// 使用 TON 的独立 Rust 参考实现推进一个径向 RK4 步。
pub fn schwarzschild_radial_rk4_step(mass: f64, state: RadialState, dlambda: f64) -> RadialState {
    let to_internal = |state: RadialState| GeodesicState {
        t: state.t,
        r: state.r,
        theta: std::f64::consts::FRAC_PI_2,
        phi: 0.0,
        u: [state.ut, state.ur, 0.0, 0.0],
    };
    let from_internal = |state: GeodesicState| RadialState {
        t: state.t,
        r: state.r,
        ut: state.u[0],
        ur: state.u[1],
    };
    from_internal(rk4_geodesic_step(mass, to_internal(state), dlambda))
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

fn incoming_to_observer(prev: &GeodesicState) -> Vec3 {
    let tangent = prev.tangent_cartesian();
    if tangent.length() <= 1e-12 {
        return Vec3::new(0.0, 0.0, 1.0);
    }
    tangent.scale(-1.0).normalize()
}

/// 在 `prev.r > r_s` 与 `next.r <= r_s` 之间对仿射步长二分，精确定位视界穿越。
fn bisect_horizon_crossing(
    mass: f64,
    prev: GeodesicState,
    dlambda: f64,
    rs: f64,
) -> GeodesicState {
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..36 {
        let mid = (lo + hi) * 0.5;
        let trial = rk4_geodesic_step(mass, prev, dlambda * mid);
        if trial.r > rs {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    rk4_geodesic_step(mass, prev, dlambda * hi)
}

enum GeodesicAdvance {
    Normal(GeodesicState),
    CrossedHorizon(GeodesicState),
    NonFinite,
}

/// 单步推进：近视界带内自动缩小仿射步长，穿越时二分定位视界。
fn advance_geodesic_step(
    mass: f64,
    prev: GeodesicState,
    dlambda: f64,
    rs: f64,
) -> GeodesicAdvance {
    let approach_band = (rs * 0.12).max(0.35);
    let mut h = dlambda;
    if prev.r > rs {
        let margin = prev.r - rs;
        if margin < approach_band {
            h = h.min(margin * 0.42);
        }
    }
    for _ in 0..8 {
        let trial = rk4_geodesic_step(mass, prev, h);
        if !trial.r.is_finite() || trial.u.iter().any(|v| !v.is_finite()) {
            h *= 0.5;
            continue;
        }
        if prev.r > rs && trial.r <= rs {
            return GeodesicAdvance::CrossedHorizon(bisect_horizon_crossing(mass, prev, h, rs));
        }
        if prev.r <= rs {
            return GeodesicAdvance::CrossedHorizon(trial);
        }
        return GeodesicAdvance::Normal(trial);
    }
    let trial = rk4_geodesic_step(mass, prev, h);
    if !trial.r.is_finite() || trial.u.iter().any(|v| !v.is_finite()) {
        return GeodesicAdvance::NonFinite;
    }
    if prev.r > rs && trial.r <= rs {
        return GeodesicAdvance::CrossedHorizon(bisect_horizon_crossing(mass, prev, h, rs));
    }
    GeodesicAdvance::Normal(trial)
}

fn captured_outcome(
    state: GeodesicState,
    steps: u32,
    residual: f64,
    disk_crossings: u32,
) -> IntegrateOutcome {
    IntegrateOutcome {
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

/// 在光子球 / 视界附近缩小仿射步长；`r >= 5 r_p` 时恢复 `base`。
pub fn adaptive_geodesic_step_size(base: f64, r: f64, rs: f64, rp: f64) -> f64 {
    let base = base.max(1.0e-8);
    let refine_outer = rp * 8.0;
    if r >= refine_outer {
        return base;
    }
    let span = (refine_outer - rs).max(rp * 0.5);
    let t = ((r - rs) / span).clamp(0.0, 1.0);
    let scale = 0.01 + 0.99 * t * t.sqrt();
    base * scale
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
    F: FnMut(DiskHit) -> bool,
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
    let rp = bh.photon_sphere_radius();
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
    let mut disk_crossings = 0u32;
    let mut steps = 0u32;
    let mut residual = null_residual(mass, &state);
    let bc = bh.critical_impact_parameter();
    // `b < b_c` 的束缚零测地线可能先向外到达 `escape_radius` 再折返落入视界，不得仅凭半径判逃逸。
    let bound_infall = schwarzschild_geodesic_impact_parameter(mass, origin, direction)
        .map(|b| b < bc)
        .unwrap_or(false);

    for step in 0..config.max_steps {
        steps = step + 1;
        if state.r <= rs {
            return captured_outcome(state, steps, residual, disk_crossings);
        }
        if !bound_infall && state.r >= config.escape_radius {
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
        residual = null_residual(mass, &state);
        let prev = state;
        let dlambda = if config.adaptive_near_horizon {
            adaptive_geodesic_step_size(config.step_size, prev.r, rs, rp)
        } else {
            config.step_size
        };
        match advance_geodesic_step(mass, prev, dlambda, rs) {
            GeodesicAdvance::CrossedHorizon(crossed) => {
                return captured_outcome(crossed, steps, residual, disk_crossings);
            }
            GeodesicAdvance::NonFinite => {
                // 视界/光子球附近 RK4 可能发散；按守恒冲击参数回落到解析命运。
                if bound_infall {
                    return captured_outcome(prev, steps, residual, disk_crossings);
                }
                let dir = prev.tangent_cartesian().normalize();
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
            GeodesicAdvance::Normal(next) => state = next,
        }
        if step > 0 && disk_enabled {
            if let (Some(inner), Some(outer)) = (config.disk_inner, config.disk_outer) {
                if let Some((radius, hit_pos)) =
                    disk_crossing(&prev, &state, config.disk_frame, inner, outer, rs)
                {
                    disk_crossings += 1;
                    let hit = DiskHit {
                        radius,
                        position: hit_pos,
                        incoming_to_observer: incoming_to_observer(&prev),
                    };
                    let stop = on_disk(hit);
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
    if bound_infall {
        return captured_outcome(state, steps, residual, disk_crossings);
    }
    let dir = state.tangent_cartesian().normalize();
    IntegrateOutcome {
        diagnostic: TraceDiagnostic {
            termination: RayTermination::Escaped,
            steps,
            null_residual: residual,
            disk_crossings,
        },
        hit: Some(RayHit::Escaped { direction: dir }),
        escape_direction: Some(dir),
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
    integrate_geodesic(bh, origin, direction, config, disk_on, true, |_| true).diagnostic
}

/// 完整 3D 零测地线：Christoffel RK4，与 `SchwarzschildOrbit` 独立 golden。
pub fn trace_geodesic_3d(
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
) -> TonResult<RayHit> {
    let disk_on = config.disk_inner.is_some() && config.disk_outer.is_some();
    let out = integrate_geodesic(bh, origin, direction, config, disk_on, true, |_| true);
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
    F: FnMut(DiskHit) -> DiskEmission,
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
        |hit| {
            if disk_hits >= MAX_DISK_HITS {
                return false;
            }
            let sample = emit_disk(hit);
            radiance = radiance.add(sample.radiance.scale(transmittance));
            transmittance *= (1.0 - sample.absorption).clamp(0.0, 1.0);
            disk_hits += 1;
            sample.stops_integration()
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
    F: FnMut(DiskHit) -> DiskEmission,
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
    use ton_metric::schwarzschild_geodesic_analytic_captured;
    use ton_types::{Camera, TonError};

    #[test]
    fn cinematic_grazing_boundary_rays_classify_cleanly() {
        let bh = BlackHole::schwarzschild(1.0);
        let cam = Camera::cinematic(bh);
        let trace = TraceConfig::with_step_size(0.08);
        let w = 48u32;
        let h = 27u32;
        let aspect = w as f64 / h as f64;
        let mut failures = 0u32;
        let mut mismatches = 0u32;
        for y in 0..h {
            for x in 0..w {
                let u = (x as f64 / w as f64) * 2.0 - 1.0;
                let v = 1.0 - (y as f64 / h as f64) * 2.0;
                let dir = cam.ray_direction(u * aspect, v);
                let diag = trace_geodesic_3d_diagnostic(bh, cam.position, dir, trace);
                if diag.is_trace_failure() {
                    failures += 1;
                    continue;
                }
                let analytic = schwarzschild_geodesic_analytic_captured(bh, cam.position, dir);
                let captured = matches!(diag.termination, RayTermination::Captured);
                if analytic != captured {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(failures, 0, "trace failures on cinematic grazing grid");
        let total = w * h;
        let mismatch_frac = mismatches as f64 / total as f64;
        assert!(
            mismatch_frac < 0.06,
            "classification mismatch {:.1}% exceeds 6%",
            mismatch_frac * 100.0
        );
    }

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
    fn adaptive_step_shrinks_near_horizon() {
        let rs = 2.0;
        let rp = 3.0;
        let base = 0.08;
        let far = adaptive_geodesic_step_size(base, 30.0, rs, rp);
        let mid = adaptive_geodesic_step_size(base, 6.0, rs, rp);
        let near = adaptive_geodesic_step_size(base, 2.5, rs, rp);
        assert!((far - base).abs() < 1e-12);
        assert!(mid < base * 0.8);
        assert!(near < base * 0.2);
    }

    #[test]
    fn bound_ray_does_not_escape_at_outer_radius() {
        let bh = BlackHole::schwarzschild(1.0);
        let origin = Vec3::new(10.0, 0.0, 0.0);
        let dir = Vec3::new(-1.0, 0.0, 0.12).normalize();
        let b = schwarzschild_geodesic_impact_parameter(bh.mass, origin, dir).expect("b");
        assert!(b < bh.critical_impact_parameter());
        let diag = trace_geodesic_3d_diagnostic(
            bh,
            origin,
            dir,
            TraceConfig::with_step_size(0.04),
        );
        assert_ne!(diag.termination, RayTermination::Escaped);
        assert_eq!(diag.termination, RayTermination::Captured);
    }

    #[test]
    fn unbound_ray_at_step_budget_classified_as_escaped() {
        let bh = BlackHole::schwarzschild(1.0);
        let origin = Vec3::new(20.0, 0.0, 0.0);
        let dir = Vec3::new(-0.2, 0.99, 0.0).normalize();
        let b = schwarzschild_geodesic_impact_parameter(bh.mass, origin, dir).expect("b");
        assert!(b >= bh.critical_impact_parameter());
        let cfg = TraceConfig {
            max_steps: 4,
            step_size: 0.5,
            escape_radius: 1.0e6,
            ..TraceConfig::default()
        };
        let diag = trace_geodesic_3d_diagnostic(bh, origin, dir, cfg);
        assert_eq!(diag.termination, RayTermination::Escaped);
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

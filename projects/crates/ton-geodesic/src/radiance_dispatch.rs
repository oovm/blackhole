//! 按积分器分派 HDR 辐射追迹。

use ton_types::{BlackHole, IntegratorKind, TonError, TonResult, Rgb, Vec3};

use crate::geodesic3d::trace_geodesic_3d_radiance_outcome;
use crate::radiance::RadianceTraceOutcome;
use crate::kerr_equatorial::trace_kerr_equatorial;
use crate::radiance::{DiskEmission, RayRadiance};
use crate::ray::RayHit;
use crate::sem_classical::trace_sem_classical;
use crate::termination::TraceDiagnostic;
use crate::trace::{trace_schwarzschild, TraceConfig};

fn hit_to_radiance(
    hit: RayHit,
    emit_disk: &mut impl FnMut(f64, Vec3) -> DiskEmission,
    sky: &impl Fn(Vec3) -> Rgb,
) -> RayRadiance {
    match hit {
        RayHit::Captured { .. } => RayRadiance::BLACK,
        RayHit::Escaped { direction } => RayRadiance::new(sky(direction)),
        RayHit::Disk { position, radius } => {
            RayRadiance::new(emit_disk(radius, position).radiance)
        }
    }
}

fn legacy_radiance_outcome<F, S>(
    hit: TonResult<RayHit>,
    emit_disk: &mut F,
    sky: &S,
) -> RadianceTraceOutcome
where
    F: FnMut(f64, Vec3) -> DiskEmission,
    S: Fn(Vec3) -> Rgb,
{
    match hit {
        Ok(h) => RadianceTraceOutcome {
            radiance: hit_to_radiance(h, emit_disk, sky),
            diagnostic: TraceDiagnostic {
                termination: match h {
                    RayHit::Captured { .. } => crate::termination::RayTermination::Captured,
                    RayHit::Escaped { .. } => crate::termination::RayTermination::Escaped,
                    RayHit::Disk { .. } => crate::termination::RayTermination::DiskCrossing,
                },
                steps: 0,
                null_residual: 0.0,
                disk_crossings: if matches!(h, RayHit::Disk { .. }) { 1 } else { 0 },
            },
        },
        Err(TonError::Captured) => RadianceTraceOutcome {
            radiance: RayRadiance::BLACK,
            diagnostic: TraceDiagnostic {
                termination: crate::termination::RayTermination::Captured,
                steps: 0,
                null_residual: 0.0,
                disk_crossings: 0,
            },
        },
        Err(TonError::Escaped) => RadianceTraceOutcome {
            radiance: RayRadiance::BLACK,
            diagnostic: TraceDiagnostic {
                termination: crate::termination::RayTermination::Escaped,
                steps: 0,
                null_residual: 0.0,
                disk_crossings: 0,
            },
        },
        Err(TonError::Domain(msg)) => RadianceTraceOutcome {
            radiance: RayRadiance::BLACK,
            diagnostic: TraceDiagnostic {
                termination: if msg.contains("max steps") {
                    crate::termination::RayTermination::StepBudgetExceeded
                } else if msg.contains("non-finite") {
                    crate::termination::RayTermination::NonFiniteState
                } else {
                    crate::termination::RayTermination::DomainError
                },
                steps: 0,
                null_residual: 0.0,
                disk_crossings: 0,
            },
        },
    }
}

/// HDR 辐射追迹：单次积分返回辐射与终止诊断。
pub fn trace_ray_radiance_outcome<F, S>(
    kind: IntegratorKind,
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
    match kind {
        IntegratorKind::Geodesic3d => trace_geodesic_3d_radiance_outcome(
            bh,
            origin,
            direction,
            config,
            emit_disk,
            sky,
        ),
        IntegratorKind::SchwarzschildOrbit => legacy_radiance_outcome(
            trace_schwarzschild(bh, origin, direction, config),
            &mut emit_disk,
            &sky,
        ),
        IntegratorKind::SemiClassicalRayMarch => legacy_radiance_outcome(
            trace_sem_classical(bh, origin, direction, config),
            &mut emit_disk,
            &sky,
        ),
        IntegratorKind::KerrEquatorial => legacy_radiance_outcome(
            trace_kerr_equatorial(bh, origin, direction, config),
            &mut emit_disk,
            &sky,
        ),
    }
}

/// HDR 辐射追迹：失败时返回 [`TraceDiagnostic`]，禁止静默回退星空。
pub fn trace_ray_radiance_result<F, S>(
    kind: IntegratorKind,
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
    mut emit_disk: F,
    sky: S,
) -> Result<RayRadiance, TraceDiagnostic>
where
    F: FnMut(f64, Vec3) -> DiskEmission,
    S: Fn(Vec3) -> Rgb,
{
    let out = trace_ray_radiance_outcome(
        kind,
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

/// HDR 辐射追迹（兼容 [`TonResult`]）。
pub fn trace_ray_radiance<F, S>(
    kind: IntegratorKind,
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
    emit_disk: F,
    sky: S,
) -> TonResult<RayRadiance>
where
    F: FnMut(f64, Vec3) -> DiskEmission,
    S: Fn(Vec3) -> Rgb,
{
    trace_ray_radiance_result(kind, bh, origin, direction, config, emit_disk, sky).map_err(|d| {
        TonError::Domain(format!("trace failed: {:?}", d.termination))
    })
}

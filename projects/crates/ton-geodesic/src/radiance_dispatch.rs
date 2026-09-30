//! 按积分器分派 HDR 辐射追迹。

use ton_types::{BlackHole, IntegratorKind, TonError, TonResult, Rgb, Vec3};

use crate::geodesic3d::trace_geodesic_3d_radiance;
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

fn legacy_radiance<F, S>(
    hit: TonResult<RayHit>,
    emit_disk: &mut F,
    sky: &S,
) -> Result<RayRadiance, TraceDiagnostic>
where
    F: FnMut(f64, Vec3) -> DiskEmission,
    S: Fn(Vec3) -> Rgb,
{
    match hit {
        Ok(h) => Ok(hit_to_radiance(h, emit_disk, sky)),
        Err(TonError::Captured) => Ok(RayRadiance::BLACK),
        Err(TonError::Escaped) => Ok(RayRadiance::BLACK),
        Err(TonError::Domain(msg)) => Err(TraceDiagnostic {
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
        }),
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
    match kind {
        IntegratorKind::Geodesic3d => {
            trace_geodesic_3d_radiance(bh, origin, direction, config, emit_disk, sky)
        }
        IntegratorKind::SchwarzschildOrbit => legacy_radiance(
            trace_schwarzschild(bh, origin, direction, config),
            &mut emit_disk,
            &sky,
        ),
        IntegratorKind::SemiClassicalRayMarch => legacy_radiance(
            trace_sem_classical(bh, origin, direction, config),
            &mut emit_disk,
            &sky,
        ),
        IntegratorKind::KerrEquatorial => legacy_radiance(
            trace_kerr_equatorial(bh, origin, direction, config),
            &mut emit_disk,
            &sky,
        ),
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

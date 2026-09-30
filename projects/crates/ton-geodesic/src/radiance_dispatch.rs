//! 按积分器分派 HDR 辐射追迹。

use ton_types::{BlackHole, IntegratorKind, TonResult, Rgb, Vec3};

use crate::geodesic3d::trace_geodesic_3d_radiance;
use crate::kerr_equatorial::trace_kerr_equatorial;
use crate::radiance::{DiskEmission, RayRadiance};
use crate::ray::RayHit;
use crate::sem_classical::trace_sem_classical;
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

/// HDR 辐射追迹：Geodesic3d 走多次盘面累积，其余积分器暂用单次命中映射。
pub fn trace_ray_radiance<F, S>(
    kind: IntegratorKind,
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
    mut emit_disk: F,
    sky: S,
) -> TonResult<RayRadiance>
where
    F: FnMut(f64, Vec3) -> DiskEmission,
    S: Fn(Vec3) -> Rgb,
{
    match kind {
        IntegratorKind::Geodesic3d => {
            trace_geodesic_3d_radiance(bh, origin, direction, config, emit_disk, sky)
        }
        IntegratorKind::SchwarzschildOrbit => {
            let hit = trace_schwarzschild(bh, origin, direction, config)?;
            Ok(hit_to_radiance(hit, &mut emit_disk, &sky))
        }
        IntegratorKind::SemiClassicalRayMarch => {
            let hit = trace_sem_classical(bh, origin, direction, config)?;
            Ok(hit_to_radiance(hit, &mut emit_disk, &sky))
        }
        IntegratorKind::KerrEquatorial => {
            let hit = trace_kerr_equatorial(bh, origin, direction, config)?;
            Ok(hit_to_radiance(hit, &mut emit_disk, &sky))
        }
    }
}

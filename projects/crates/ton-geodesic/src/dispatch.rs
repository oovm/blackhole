//! 按 `IntegratorKind` 分派光线追迹。

use ton_types::{BlackHole, IntegratorKind, TonResult, Vec3};

use crate::geodesic3d::trace_geodesic_3d;
use crate::kerr_equatorial::trace_kerr_equatorial;
use crate::ray::RayHit;
use crate::sem_classical::trace_sem_classical;
use crate::trace::{trace_schwarzschild, TraceConfig};

/// 统一入口：每种积分器独立 golden，不互相替代。
pub fn trace_ray(
    kind: IntegratorKind,
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
) -> TonResult<RayHit> {
    match kind {
        IntegratorKind::SchwarzschildOrbit => trace_schwarzschild(bh, origin, direction, config),
        IntegratorKind::SemiClassicalRayMarch => {
            trace_sem_classical(bh, origin, direction, config)
        }
        IntegratorKind::Geodesic3d => trace_geodesic_3d(bh, origin, direction, config),
        IntegratorKind::KerrEquatorial => trace_kerr_equatorial(bh, origin, direction, config),
    }
}

pub use crate::radiance_dispatch::trace_ray_radiance;

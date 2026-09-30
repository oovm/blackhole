//! 按积分器分派诊断追迹（终止分类图 / 捕获轮廓）。

use ton_types::{BlackHole, IntegratorKind, TonError, Vec3};

use crate::geodesic3d::trace_geodesic_3d_diagnostic;
use crate::ray::RayHit;
use crate::termination::{RayTermination, TraceDiagnostic};
use crate::trace::TraceConfig;

fn hit_to_termination(hit: RayHit) -> RayTermination {
    match hit {
        RayHit::Captured { .. } => RayTermination::Captured,
        RayHit::Escaped { .. } => RayTermination::Escaped,
        RayHit::Disk { .. } => RayTermination::DiskCrossing,
    }
}

fn ton_err_to_termination(err: TonError) -> RayTermination {
    match err {
        TonError::Captured => RayTermination::Captured,
        TonError::Escaped => RayTermination::Escaped,
        TonError::Domain(msg) => {
            if msg.contains("max steps") {
                RayTermination::StepBudgetExceeded
            } else if msg.contains("non-finite") {
                RayTermination::NonFiniteState
            } else if msg.contains("initialize") {
                RayTermination::InitFailed
            } else {
                RayTermination::DomainError
            }
        }
    }
}

/// 单条光线诊断：不着色盘面或星空，仅报告积分如何结束。
pub fn trace_ray_diagnostic(
    kind: IntegratorKind,
    bh: BlackHole,
    origin: Vec3,
    direction: Vec3,
    config: TraceConfig,
) -> TraceDiagnostic {
    match kind {
        IntegratorKind::Geodesic3d => trace_geodesic_3d_diagnostic(bh, origin, direction, config),
        IntegratorKind::SchwarzschildOrbit
        | IntegratorKind::SemiClassicalRayMarch
        | IntegratorKind::KerrEquatorial => {
            use crate::dispatch::trace_ray;
            match trace_ray(kind, bh, origin, direction, config) {
                Ok(hit) => TraceDiagnostic {
                    termination: hit_to_termination(hit),
                    steps: 0,
                    null_residual: 0.0,
                    disk_crossings: if matches!(hit, RayHit::Disk { .. }) {
                        1
                    } else {
                        0
                    },
                },
                Err(err) => TraceDiagnostic {
                    termination: ton_err_to_termination(err),
                    steps: 0,
                    null_residual: 0.0,
                    disk_crossings: 0,
                },
            }
        }
    }
}

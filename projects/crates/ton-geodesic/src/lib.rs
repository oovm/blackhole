//! 零测地线积分。
#![warn(missing_docs)]

mod dispatch;
mod diagnostic_dispatch;
mod geodesic3d;
mod kerr_equatorial;
mod orbit;
mod radiance;
mod radiance_dispatch;
mod ray;
mod sem_classical;
mod termination;
mod trace;

pub use dispatch::trace_ray;
pub use radiance_dispatch::{trace_ray_radiance, trace_ray_radiance_result};
pub use diagnostic_dispatch::trace_ray_diagnostic;
pub use geodesic3d::{
    adaptive_geodesic_step_size, trace_geodesic_3d, trace_geodesic_3d_diagnostic,
    trace_geodesic_3d_radiance, trace_geodesic_3d_radiance_outcome,
};
pub use radiance::RadianceTraceOutcome;
pub use radiance_dispatch::trace_ray_radiance_outcome;
pub use termination::{
    DiagnosticColorMode, RayTermination, TerminationHistogram, TraceDiagnostic,
};
pub use kerr_equatorial::trace_kerr_equatorial;
pub use orbit::OrbitState;
pub use radiance::{DiskEmission, RayRadiance};
pub use ray::RayHit;
pub use sem_classical::trace_sem_classical;
pub use ton_metric::{
    schwarzschild_geodesic_analytic_captured, schwarzschild_geodesic_impact_parameter,
};
pub use trace::{trace_schwarzschild, TraceConfig};

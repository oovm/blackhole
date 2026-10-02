//! 零测地线积分。
#![warn(missing_docs)]

mod dispatch;
mod adaptive_rk4;
mod adaptive_events;
mod adaptive_trace;
mod null_energy;
mod cartesian_null;
mod cartesian_adaptive;
mod cartesian_events;
mod cartesian_trace;
mod diagnostic_dispatch;
mod geodesic3d;
mod kerr_equatorial;
mod orbit;
mod radiance;
mod disk_frequency;
pub use disk_frequency::schwarzschild_disk_frequency_shift;
mod radiance_dispatch;
mod ray;
mod sem_classical;
mod termination;
mod trace;

pub use dispatch::trace_ray;
pub use adaptive_rk4::{schwarzschild_adaptive_rk4_step, AdaptiveRk4Config, AdaptiveRk4ConfigBuilder, AdaptiveRk4Step};
pub use adaptive_events::{schwarzschild_accepted_step_events, SchwarzschildEventConfig, SchwarzschildEventConfigBuilder, SchwarzschildStepEvent, SchwarzschildStepEventKind};
pub use adaptive_trace::{schwarzschild_adaptive_trace, AdaptiveTraceDiskEvent, AdaptiveTraceTermination, SchwarzschildAdaptiveTraceConfig, SchwarzschildAdaptiveTraceConfigBuilder, SchwarzschildAdaptiveTraceResult};
pub use null_energy::{schwarzschild_null_energy_rk4_step, SchwarzschildNullEnergyState};
pub use cartesian_null::{schwarzschild_cartesian_null_rk4_step, schwarzschild_cartesian_conserved_null_rk4_step, SchwarzschildCartesianNullState};
pub use cartesian_adaptive::{schwarzschild_cartesian_adaptive_rk4_step, CartesianAdaptiveRk4Step};
pub use cartesian_events::{schwarzschild_cartesian_accepted_step_events, CartesianStepEvent};
pub use cartesian_trace::{schwarzschild_cartesian_adaptive_trace, CartesianAdaptiveTraceResult, CartesianTraceDiskEvent};
pub use radiance_dispatch::{trace_ray_radiance, trace_ray_radiance_result};
pub use diagnostic_dispatch::trace_ray_diagnostic;
pub use geodesic3d::{
    adaptive_geodesic_step_size, trace_geodesic_3d, trace_geodesic_3d_diagnostic,
    trace_geodesic_3d_radiance, trace_geodesic_3d_radiance_outcome,
    schwarzschild_geodesic_initial_state, schwarzschild_geodesic_rk4_step, schwarzschild_radial_rk4_step, RadialState, SphericalState,
};
pub use radiance::RadianceTraceOutcome;
pub use radiance_dispatch::trace_ray_radiance_outcome;
pub use termination::{
    DiagnosticColorMode, RayTermination, TerminationHistogram, TraceDiagnostic,
};
pub use kerr_equatorial::trace_kerr_equatorial;
pub use orbit::OrbitState;
pub use radiance::{DiskEmission, DiskHit, RayRadiance};
pub use ray::RayHit;
pub use sem_classical::trace_sem_classical;
pub use ton_metric::{
    schwarzschild_geodesic_analytic_captured, schwarzschild_geodesic_impact_parameter,
};
pub use trace::{trace_schwarzschild, TraceConfig};

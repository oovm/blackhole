//! 零测地线积分。
#![warn(missing_docs)]

mod dispatch;
mod geodesic3d;
mod kerr_equatorial;
mod orbit;
mod radiance;
mod radiance_dispatch;
mod ray;
mod sem_classical;
mod trace;

pub use dispatch::{trace_ray, trace_ray_radiance};
pub use geodesic3d::{trace_geodesic_3d, trace_geodesic_3d_radiance};
pub use kerr_equatorial::trace_kerr_equatorial;
pub use orbit::OrbitState;
pub use radiance::{DiskEmission, RayRadiance};
pub use ray::RayHit;
pub use sem_classical::trace_sem_classical;
pub use trace::{trace_schwarzschild, TraceConfig};

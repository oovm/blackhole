//! 零测地线积分。
#![warn(missing_docs)]

mod dispatch;
mod geodesic3d;
mod orbit;
mod ray;
mod sem_classical;
mod trace;

pub use dispatch::trace_ray;
pub use geodesic3d::trace_geodesic_3d;
pub use orbit::OrbitState;
pub use ray::RayHit;
pub use sem_classical::trace_sem_classical;
pub use trace::{trace_schwarzschild, TraceConfig};

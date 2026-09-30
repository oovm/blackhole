//! 时空度规。
#![warn(missing_docs)]

mod geodesic;
mod kerr;
mod metric;
mod schwarzschild;
mod sem_classical;
mod spherical;

pub use geodesic::{
    schwarzschild_geodesic_accel, schwarzschild_geodesic_analytic_captured,
    schwarzschild_geodesic_impact_parameter, schwarzschild_null_four_velocity,
    spherical_velocity_to_cartesian,
};
pub use kerr::Kerr;
pub use metric::Metric;
pub use schwarzschild::Schwarzschild;
pub use sem_classical::semi_classical_acceleration;
pub use spherical::SphericalCoords;

//! 时空度规。
#![warn(missing_docs)]

mod kerr;
mod metric;
mod schwarzschild;
mod sem_classical;
mod spherical;

pub use kerr::Kerr;
pub use metric::Metric;
pub use schwarzschild::Schwarzschild;
pub use sem_classical::semi_classical_acceleration;
pub use spherical::SphericalCoords;

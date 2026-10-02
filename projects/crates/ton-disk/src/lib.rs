//! 吸积盘辐射模型。
#![warn(missing_docs)]

mod disk;
mod fbm;
mod sample;
mod spectrum;
mod graybody;
mod transfer;

pub use disk::{AccretionDisk, DiskConfig};
pub use fbm::procedural_fbm_color;
pub use sample::{sample_disk_radiance, DiskSample};
pub use spectrum::temperature_to_rgb;
pub use graybody::{GraybodyDiskPattern, GraybodyDiskPatternBuilder};
pub use transfer::{accumulate_radiative_events, RadiativeEvent, RadiativeTransferResult};

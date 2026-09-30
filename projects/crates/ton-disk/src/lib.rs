//! 吸积盘辐射模型。
#![warn(missing_docs)]

mod disk;
mod fbm;
mod spectrum;

pub use disk::{AccretionDisk, DiskConfig};
pub use fbm::procedural_fbm_color;
pub use spectrum::temperature_to_rgb;

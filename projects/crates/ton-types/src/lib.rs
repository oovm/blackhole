//! 黑洞模拟共享类型（栈底，无物理算法）。
#![warn(missing_docs)]

mod black_hole;
mod camera;
mod disk_frame;
mod color;
mod error;
mod kind;
mod post_flags;
mod units;
mod vec3;

pub use black_hole::BlackHole;
pub use camera::Camera;
pub use disk_frame::DiskFrame;
pub use color::Rgb;
pub use error::{TonError, TonResult};
pub use kind::{DiskModelKind, IntegratorKind};
pub use post_flags::{
    post_bloom_from_flags, use_ucf_post, POST_RAW, POST_UCF_BLOOM, POST_UCF_TONE_MAP,
};
pub use units::{geom_to_solar_radii, solar_mass_to_geom};
pub use vec3::Vec3;

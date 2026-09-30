//! 沿测地线累积的 HDR 辐射（非单次命中着色）。

use ton_types::Rgb;

/// 单条光线的 HDR 辐射结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayRadiance {
    pub radiance: Rgb,
}

impl RayRadiance {
    pub const BLACK: Self = Self {
        radiance: Rgb::BLACK,
    };

    pub fn new(radiance: Rgb) -> Self {
        Self { radiance }
    }
}

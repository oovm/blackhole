//! 沿测地线累积的 HDR 辐射（非单次命中着色）。

use ton_types::Rgb;

/// 薄盘局部发射与吸收（供测地线透射率累积）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskEmission {
    pub radiance: Rgb,
    /// 局部吸收系数 0..1。
    pub absorption: f64,
}

impl DiskEmission {
    pub fn opaque(radiance: Rgb) -> Self {
        Self {
            radiance,
            absorption: 0.0,
        }
    }

    pub fn with_absorption(radiance: Rgb, absorption: f64) -> Self {
        Self {
            radiance,
            absorption: absorption.clamp(0.0, 0.95),
        }
    }
}

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

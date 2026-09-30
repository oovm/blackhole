//! 沿测地线累积的 HDR 辐射（非单次命中着色）。

use ton_types::{Rgb, Vec3};

/// 薄盘命中上下文：含沿测地线指向观测者的入射方向。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskHit {
    pub radius: f64,
    pub position: Vec3,
    /// 沿测地线指向观测者的单位方向（逆仿射参数）。
    pub incoming_to_observer: Vec3,
}

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

    /// 完全不透明薄盘：命中后终止积分，不再透射背景。
    pub fn opaque_sheet(radiance: Rgb) -> Self {
        Self {
            radiance,
            absorption: 1.0,
        }
    }

    pub fn stops_integration(self) -> bool {
        self.absorption >= 1.0 - 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_sheet_stops_integration() {
        assert!(DiskEmission::opaque_sheet(Rgb::WHITE).stops_integration());
        assert!(!DiskEmission::with_absorption(Rgb::WHITE, 0.95).stops_integration());
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

/// 辐射追迹结果：始终附带终止诊断（单次积分，无二次追迹）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadianceTraceOutcome {
    pub radiance: RayRadiance,
    pub diagnostic: crate::termination::TraceDiagnostic,
}

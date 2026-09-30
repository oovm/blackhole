//! 盘面辐射采样（HDR 路径累积用，非一次命中着色）。

use ton_types::Rgb;

use crate::{AccretionDisk, temperature_to_rgb};

/// 沿光线局部采样得到的盘面辐射贡献。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskSample {
    /// 发射辐射（线性 HDR RGB）。
    pub emissive_radiance: Rgb,
    /// 局部吸收系数 0..1。
    pub absorption: f64,
    /// 有效温度（K）。
    pub temperature: f64,
    /// 简化相对论频移因子（Doppler）。
    pub redshift_factor: f64,
}

/// 按半径与 Doppler 因子采样盘面辐射（工程近似，非辐射转移求解）。
pub fn sample_disk_radiance(disk: AccretionDisk, radius: f64, doppler: f64) -> DiskSample {
    let temperature = disk.effective_temperature(radius);
    let base = temperature_to_rgb(temperature);
    let brightness = doppler.clamp(0.0, 8.0).powi(3);
    DiskSample {
        emissive_radiance: base.scale(brightness),
        absorption: 0.0,
        temperature,
        redshift_factor: doppler,
    }
}

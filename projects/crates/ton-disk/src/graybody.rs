//! 有界、随圆轨道平流的灰体发射纹理，不是流体或相对论盘通量模型。

use ton_types::{TonError, TonResult};

/// 频率无关的发射系数，仅描述黑色背衬上的灰体发射，不包含透射。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraybodyDiskPattern {
    contrast: f64,
    arms: u32,
    pitch: f64,
    radial_frequency: f64,
}

/// 使用有界参数配置盘面发射纹理。
#[derive(Debug, Clone, Copy)]
pub struct GraybodyDiskPatternBuilder {
    pattern: GraybodyDiskPattern,
}

impl GraybodyDiskPattern {
    /// 默认无调制，启用结构必须显式设置 contrast。
    pub fn builder() -> GraybodyDiskPatternBuilder {
        GraybodyDiskPatternBuilder { pattern: Self { contrast: 0.0, arms: 3, pitch: 14.0, radial_frequency: 48.0 } }
    }

    /// 灰体对比度，零值完全保留原始黑体。
    pub fn contrast(self) -> f64 { self.contrast }
    /// 周向整数谐波数，保证角度接缝连续。
    pub fn arms(self) -> u32 { self.arms }
    /// 对数螺旋径向相位系数。
    pub fn pitch(self) -> f64 { self.pitch }
    /// 对数径向条纹频率。
    pub fn radial_frequency(self) -> f64 { self.radial_frequency }

    /// 独立 F64 参考，单位 G=c=1，时间为实际发射的坐标时间。
    /// Schwarzschild 稳定赤道圆轨道 Ω=sqrt(M/r³)，禁止传入逆向光线的正向累计时间。
    pub fn emissivity(self, mass: f64, inner: f64, x: f64, y: f64, emission_time: f64) -> TonResult<f64> {
        if [mass, inner, x, y, emission_time].iter().any(|value| !value.is_finite()) || mass <= 0.0 {
            return Err(TonError::Domain("invalid graybody coordinates or mass".into()));
        }
        let radius = (x / mass).hypot(y / mass);
        let inner = inner / mass;
        let time = emission_time / mass;
        if !radius.is_finite() || !inner.is_finite() || !time.is_finite()
            || inner < 6.0 || radius < inner || radius > 10000.0 || time.abs() > 4096.0 {
            return Err(TonError::Domain("graybody pattern exceeds stable orbit or phase domain".into()));
        }
        let angle = y.atan2(x) - time / (radius * radius * radius).sqrt();
        let log_radius = (radius / inner).ln();
        let phase = self.arms as f64 * angle + self.pitch * log_radius;
        let spiral = 0.5 + 0.5 * phase.cos();
        let rings = 0.5 + 0.5 * (self.radial_frequency * log_radius + 2.0 * phase.sin()).cos();
        let fine = 0.5 + 0.5 * (2.73 * self.radial_frequency * log_radius - 3.0 * phase.cos()).cos();
        let signal = 0.5 * spiral + 0.3 * rings + 0.2 * fine;
        Ok(1.0 - self.contrast + self.contrast * signal * signal)
    }
}

impl GraybodyDiskPatternBuilder {
    /// contrast 必须在 0..1。
    pub fn contrast(mut self, value: f64) -> Self { self.pattern.contrast = value; self }
    /// 周向整数谐波数，支持 1..8。
    pub fn arms(mut self, value: u32) -> Self { self.pattern.arms = value; self }
    /// 对数螺旋相位，绝对值最多 32。
    pub fn pitch(mut self, value: f64) -> Self { self.pattern.pitch = value; self }
    /// 径向条纹频率，支持 0..128。
    pub fn radial_frequency(mut self, value: f64) -> Self { self.pattern.radial_frequency = value; self }
    /// 冻结可验证的发射参数，不构建 UCF 程序。
    pub fn build(self) -> TonResult<GraybodyDiskPattern> {
        let pattern = self.pattern;
        if !pattern.contrast.is_finite() || !(0.0..=1.0).contains(&pattern.contrast)
            || !(1..=8).contains(&pattern.arms) || !pattern.pitch.is_finite() || pattern.pitch.abs() > 32.0
            || !pattern.radial_frequency.is_finite() || !(0.0..=128.0).contains(&pattern.radial_frequency) {
            return Err(TonError::Domain("invalid bounded graybody pattern parameters".into()));
        }
        Ok(pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graybody_emissivity_is_bounded_and_zero_contrast_is_identity() {
        let identity = GraybodyDiskPattern::builder().build().unwrap();
        for contrast in [0.0, 0.5, 1.0] {
            let pattern = GraybodyDiskPattern::builder().contrast(contrast).build().unwrap();
            let mut minimum = 1.0f64;
            let mut maximum = 0.0f64;
            for radial in 0..64 {
                let radius = 6.1 + radial as f64 * 13.9 / 63.0;
                for sample in 0..128 {
                    let angle = sample as f64 * std::f64::consts::TAU / 128.0;
                    let (x, y) = (radius * angle.cos(), radius * angle.sin());
                    let value = pattern.emissivity(1.0, 6.0, x, y, -100.0).unwrap();
                    assert!(value >= 1.0 - contrast && value <= 1.0);
                    assert_eq!(identity.emissivity(1.0, 6.0, x, y, -100.0).unwrap(), 1.0);
                    minimum = minimum.min(value); maximum = maximum.max(value);
                }
            }
            if contrast > 0.0 { assert!(maximum - minimum > 0.5 * contrast); }
        }
    }

    #[test]
    fn graybody_is_continuous_at_azimuth_seam_and_advects_with_orbit() {
        let pattern = GraybodyDiskPattern::builder().contrast(0.8).build().unwrap();
        let radius = 12.0f64;
        let time = -150.0;
        let delta = 20.0;
        let angle = 0.7;
        let advanced = angle + delta / radius.powi(3).sqrt();
        let before = pattern.emissivity(1.0, 6.0, radius * angle.cos(), radius * angle.sin(), time).unwrap();
        let after = pattern.emissivity(1.0, 6.0, radius * advanced.cos(), radius * advanced.sin(), time + delta).unwrap();
        assert!((before - after).abs() < 1.0e-13);
        let above = pattern.emissivity(1.0, 6.0, -radius, 1.0e-9, time).unwrap();
        let below = pattern.emissivity(1.0, 6.0, -radius, -1.0e-9, time).unwrap();
        assert!((above - below).abs() < 1.0e-9);
        let scaled = pattern.emissivity(7.0, 42.0, 7.0 * radius * angle.cos(), 7.0 * radius * angle.sin(), 7.0 * time).unwrap();
        assert!((before - scaled).abs() < 1.0e-13);
    }

    #[test]
    fn graybody_rejects_invalid_parameters_or_phase_domains() {
        for builder in [GraybodyDiskPattern::builder().contrast(-0.1), GraybodyDiskPattern::builder().contrast(f64::NAN),
            GraybodyDiskPattern::builder().arms(0), GraybodyDiskPattern::builder().arms(9),
            GraybodyDiskPattern::builder().pitch(33.0), GraybodyDiskPattern::builder().radial_frequency(129.0)] {
            assert!(builder.build().is_err());
        }
        let pattern = GraybodyDiskPattern::builder().build().unwrap();
        for (mass, inner, x, time) in [(0.0, 6.0, 8.0, 0.0), (1.0, 5.0, 8.0, 0.0),
            (1.0, 6.0, 5.9, 0.0), (1.0, 6.0, 10001.0, 0.0), (1.0, 6.0, 8.0, 4097.0), (1.0, 6.0, f64::NAN, 0.0)] {
            assert!(pattern.emissivity(mass, inner, x, 0.0, time).is_err());
        }
    }
}

//! 独立可见光谱积分与线性 sRGB 转换，不使用温度到颜色的经验分段调色。

use ton_types::{TonError, TonResult};
use crate::observed_blackbody_spectral_radiance;

/// CIE 1931 2 度观察者响应的非对称高斯近似。不是官方逐纳米数据表。
/// 波长单位 nm，只用于 380..780nm 可见波段。
pub fn cie1931_fit(wavelength_nm: f64) -> TonResult<[f64; 3]> {
    if !wavelength_nm.is_finite() || !(380.0..=780.0).contains(&wavelength_nm) {
        return Err(TonError::Domain("CIE fit wavelength outside visible range".into()));
    }
    let gaussian = |center: f64, left: f64, right: f64| {
        let distance = (wavelength_nm - center) * if wavelength_nm < center { left } else { right };
        (-0.5 * distance * distance).exp()
    };
    Ok([
        1.056 * gaussian(599.8, 0.0264, 0.0323) + 0.362 * gaussian(442.0, 0.0624, 0.0374) - 0.065 * gaussian(501.1, 0.0490, 0.0382),
        0.821 * gaussian(568.8, 0.0213, 0.0247) + 0.286 * gaussian(530.9, 0.0613, 0.0322),
        1.217 * gaussian(437.0, 0.0845, 0.0278) + 0.681 * gaussian(459.0, 0.0385, 0.0725),
    ])
}

/// 以 1nm 梯形积分获得 XYZ，Y 的单位 cd/m²。使用近似观察者响应，未作色适应。
/// 使用 B_lambda=B_nu*c/lambda²，不能直接对 B_nu 作等波长积分。
pub fn observed_blackbody_xyz(temperature_kelvin: f64, frequency_shift: f64) -> TonResult<[f64; 3]> {
    if !temperature_kelvin.is_finite() || temperature_kelvin < 0.0 || !frequency_shift.is_finite() || frequency_shift <= 0.0 {
        return Err(TonError::Domain("invalid spectral color temperature or shift".into()));
    }
    if temperature_kelvin == 0.0 { return Ok([0.0; 3]); }
    let mut xyz = [0.0; 3];
    for wavelength in 380..=780 {
        let wavelength_m = wavelength as f64 * 1.0e-9;
        let radiance_nu = observed_blackbody_spectral_radiance(temperature_kelvin, 299792458.0 / wavelength_m, frequency_shift)?;
        let weight = if wavelength == 380 || wavelength == 780 { 0.5 } else { 1.0 };
        let scale = radiance_nu * 299792458.0 / wavelength_m.powi(2) * 1.0e-9 * 683.0 * weight;
        for (target, response) in xyz.iter_mut().zip(cie1931_fit(wavelength as f64)?) { *target += scale * response; }
    }
    if xyz.iter().any(|value| !value.is_finite() || *value < 0.0) { return Err(TonError::Domain("spectral XYZ exceeds numerical range".into())); }
    Ok(xyz)
}

/// XYZ 转线性 sRGB。保留负分量和 HDR 值，色域与曝光处理属于显示阶段。
pub fn xyz_to_linear_srgb(xyz: [f64; 3]) -> TonResult<[f64; 3]> {
    if xyz.iter().any(|value| !value.is_finite()) { return Err(TonError::Domain("nonfinite spectral XYZ".into())); }
    let [x, y, z] = xyz;
    Ok([3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z])
}

/// 无内缘力矩牛顿薄盘的受控温度基线，不声称是 Novikov-Thorne 相对论盘。
/// 形状为 r^-3(1-sqrt(r_inner/r))，峰值在 49/36 倍内缘。
pub fn zero_torque_disk_temperature(radius: f64, inner_radius: f64, peak_kelvin: f64) -> TonResult<f64> {
    if [radius, inner_radius, peak_kelvin].iter().any(|value| !value.is_finite())
        || inner_radius <= 0.0 || radius < inner_radius || peak_kelvin <= 0.0 {
        return Err(TonError::Domain("invalid zero torque disk temperature input".into()));
    }
    let ratio = inner_radius / radius;
    let shape = ratio.powi(3) * (1.0 - ratio.sqrt());
    let peak_shape = (36.0f64 / 49.0).powi(3) / 7.0;
    Ok(peak_kelvin * (shape / peak_shape).sqrt().sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_color_preserves_transport_luminance_and_hdr() {
        let xyz = observed_blackbody_xyz(6000.0, 1.0).unwrap();
        assert!(xyz.iter().all(|value| *value > 1.0e7));
        assert_eq!(observed_blackbody_xyz(6000.0, 0.5).unwrap(), observed_blackbody_xyz(3000.0, 1.0).unwrap());
        let cold = observed_blackbody_xyz(3000.0, 1.0).unwrap();
        assert!(cold[1] < xyz[1]);
        let rgb = xyz_to_linear_srgb(xyz).unwrap();
        assert!(rgb.iter().all(|value| *value > 1.0));
        assert_eq!(observed_blackbody_xyz(0.0, 1.0).unwrap(), [0.0; 3]);
        assert!(observed_blackbody_xyz(6000.0, -1.0).is_err());
        assert!(cie1931_fit(800.0).is_err());
    }

    #[test]
    fn radial_temperature_has_zero_inner_edge_and_normalized_peak() {
        assert_eq!(zero_torque_disk_temperature(6.0, 6.0, 6500.0).unwrap(), 0.0);
        assert!((zero_torque_disk_temperature(6.0 * 49.0 / 36.0, 6.0, 6500.0).unwrap() - 6500.0).abs() < 1.0e-9);
        assert!(zero_torque_disk_temperature(20.0, 6.0, 6500.0).unwrap() < 6500.0);
        assert!(zero_torque_disk_temperature(5.0, 6.0, 6500.0).is_err());
    }
}

//! 独立 Planck 谱参考。温度与频率使用 SI 单位，不混用几何质量。

use ton_types::{TonError, TonResult};

/// 返回黑体比强度 B_nu，单位 W m^-2 sr^-1 Hz^-1。
pub fn blackbody_spectral_radiance(temperature_kelvin: f64, frequency_hz: f64) -> TonResult<f64> {
    if !temperature_kelvin.is_finite() || temperature_kelvin <= 0.0
        || !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(TonError::Domain("invalid blackbody temperature or frequency".into()));
    }
    let planck = 6.62607015e-34;
    let boltzmann = 1.380649e-23;
    let light_speed = 299792458.0;
    let exponent = (planck / boltzmann) * (frequency_hz / temperature_kelvin);
    let intensity = if exponent > 700.0 { 0.0 } else {
        let frequency_thz = frequency_hz * 1.0e-12;
        (2.0 * planck / (light_speed * light_speed) * 1.0e36)
            * frequency_thz.powi(3) / exponent.exp_m1()
    };
    if !intensity.is_finite() || intensity < 0.0 {
        return Err(TonError::Domain("blackbody spectral radiance exceeds numerical range".into()));
    }
    Ok(intensity)
}

/// 利用 I_nu/nu^3 不变量计算观测谱。完整黑体等价于温度变为 g*T。
/// 不裁剪 g，不将 g^4 的总频率积分因子误用到单个频率样本。
pub fn observed_blackbody_spectral_radiance(
    temperature_kelvin: f64, observer_frequency_hz: f64, frequency_shift: f64,
) -> TonResult<f64> {
    if !frequency_shift.is_finite() || frequency_shift <= 0.0 {
        return Err(TonError::Domain("invalid blackbody frequency shift".into()));
    }
    blackbody_spectral_radiance(temperature_kelvin * frequency_shift, observer_frequency_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planck_spectrum_matches_transport_invariant_and_rayleigh_jeans_limit() {
        for shift in [0.2f64, 0.7, 1.0, 2.0] {
            let actual = observed_blackbody_spectral_radiance(6000.0, 5.0e14, shift).unwrap();
            let transported = shift.powi(3) * blackbody_spectral_radiance(6000.0, 5.0e14 / shift).unwrap();
            assert!((actual / transported - 1.0).abs() < 1.0e-13);
        }
        let actual = blackbody_spectral_radiance(6000.0, 1.0e6).unwrap();
        let expected = 2.0 * 1.380649e-23 * 6000.0 * 1.0e12 / 299792458.0f64.powi(2);
        assert!((actual / expected - 1.0).abs() < 1.0e-8);
        assert!(blackbody_spectral_radiance(3000.0, 5.0e14).unwrap()
            < blackbody_spectral_radiance(6000.0, 5.0e14).unwrap());
        assert_eq!(blackbody_spectral_radiance(1.0, 1.0e18).unwrap(), 0.0);
        assert!(blackbody_spectral_radiance(0.0, 5.0e14).is_err());
        assert!(observed_blackbody_spectral_radiance(6000.0, 5.0e14, -1.0).is_err());
    }
}

//! 独立参考路径的 RK4 步长加倍误差控制。

use ton_types::{TonError, TonResult};

use crate::{schwarzschild_geodesic_rk4_step, SphericalState};

/// 已校验的自适应参考积分配置。
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveRk4Config {
    absolute_tolerance: f64,
    relative_tolerance: f64,
    minimum_step: f64,
    maximum_step: f64,
    maximum_retries: u32,
}

/// 参考积分配置构造器。
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveRk4ConfigBuilder {
    config: AdaptiveRk4Config,
}

impl AdaptiveRk4Config {
    /// 构造容差与步长范围配置。
    pub fn builder() -> AdaptiveRk4ConfigBuilder {
        AdaptiveRk4ConfigBuilder {
            config: Self {
                absolute_tolerance: 1.0e-9,
                relative_tolerance: 1.0e-7,
                minimum_step: 1.0e-8,
                maximum_step: 0.08,
                maximum_retries: 16,
            },
        }
    }
}

impl AdaptiveRk4ConfigBuilder {
    /// 设置逐分量绝对和相对误差容差。
    pub fn tolerances(mut self, absolute: f64, relative: f64) -> Self {
        self.config.absolute_tolerance = absolute;
        self.config.relative_tolerance = relative;
        self
    }

    /// 设置可接受步长范围。
    pub fn step_range(mut self, minimum: f64, maximum: f64) -> Self {
        self.config.minimum_step = minimum;
        self.config.maximum_step = maximum;
        self
    }

    /// 设置单步最多重试次数。
    pub fn maximum_retries(mut self, retries: u32) -> Self {
        self.config.maximum_retries = retries;
        self
    }

    /// 拒绝非法容差、步长范围与无界重试。
    pub fn build(self) -> TonResult<AdaptiveRk4Config> {
        let config = self.config;
        if !config.absolute_tolerance.is_finite() || config.absolute_tolerance <= 0.0
            || !config.relative_tolerance.is_finite() || config.relative_tolerance < 0.0
            || !config.minimum_step.is_finite() || config.minimum_step <= 0.0
            || !config.maximum_step.is_finite() || config.maximum_step < config.minimum_step
            || config.maximum_retries > 64
        {
            return Err(TonError::Domain("invalid adaptive RK4 configuration".into()));
        }
        Ok(config)
    }
}

/// 通过局部误差门禁的独立参考步。
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveRk4Step {
    /// 两个半步得到的参考状态，不包含失败候选。
    pub state: SphericalState,
    /// 本次实际接受的仿射步长。
    pub step_used: f64,
    /// 下一步建议步长。
    pub next_step: f64,
    /// 最大逐分量加权误差，接受时不大于一。
    pub error_ratio: f64,
    /// 本次被拒绝的候选步数。
    pub rejections: u32,
}

fn components(state: SphericalState) -> [f64; 8] {
    [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi]
}

/// 独立 Rust RK4 的有界误差控制，不调用 UCF 或 GPU。
///
/// 单步失败不修改输入。误差通过一整步和两个半步之差除以十五估计，
/// 每个分量按绝对容差加相对容差归一化。该契约不替代视界事件定位。
pub fn schwarzschild_adaptive_rk4_step(
    mass: f64,
    state: SphericalState,
    requested_step: f64,
    config: AdaptiveRk4Config,
) -> TonResult<AdaptiveRk4Step> {
    if !mass.is_finite() || mass <= 0.0 || !requested_step.is_finite() || requested_step <= 0.0
        || components(state).iter().any(|value| !value.is_finite())
        || state.r <= 2.0 * mass
    {
        return Err(TonError::Domain("invalid adaptive RK4 initial state or step".into()));
    }
    let mut step = requested_step.clamp(config.minimum_step, config.maximum_step);
    for rejections in 0..=config.maximum_retries {
        let full = schwarzschild_geodesic_rk4_step(mass, state, step);
        let half = schwarzschild_geodesic_rk4_step(mass, state, step * 0.5);
        let refined = schwarzschild_geodesic_rk4_step(mass, half, step * 0.5);
        let mut error_ratio = 0.0f64;
        for ((initial, full), refined) in components(state).into_iter().zip(components(full)).zip(components(refined)) {
            let scale = config.absolute_tolerance + config.relative_tolerance * initial.abs().max(refined.abs());
            let ratio = ((refined - full) / 15.0).abs() / scale;
            if !full.is_finite() || !refined.is_finite() || !ratio.is_finite() {
                error_ratio = f64::INFINITY;
                break;
            }
            error_ratio = error_ratio.max(ratio);
        }
        if error_ratio <= 1.0 {
            let factor = if error_ratio == 0.0 { 2.0 } else { (0.9 * error_ratio.powf(-0.2)).clamp(0.5, 2.0) };
            return Ok(AdaptiveRk4Step {
                state: refined,
                step_used: step,
                next_step: (step * factor).clamp(config.minimum_step, config.maximum_step),
                error_ratio,
                rejections,
            });
        }
        if step <= config.minimum_step {
            return Err(TonError::Domain(format!("adaptive RK4 minimum step exhausted: error_ratio={error_ratio}")));
        }
        let factor = (0.9 * error_ratio.powf(-0.2)).clamp(0.1, 0.5);
        step = (step * factor).max(config.minimum_step);
    }
    Err(TonError::Domain("adaptive RK4 retry budget exhausted".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initial() -> SphericalState {
        SphericalState { t: 0.0, r: 12.0, theta: 1.0, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.22, uphi: 0.08 }
    }

    #[test]
    fn builder_rejects_invalid_configuration() {
        assert!(AdaptiveRk4Config::builder().tolerances(0.0, 1.0e-7).build().is_err());
        assert!(AdaptiveRk4Config::builder().step_range(0.1, 0.01).build().is_err());
        assert!(AdaptiveRk4Config::builder().maximum_retries(65).build().is_err());
    }

    #[test]
    fn accepted_step_is_two_independent_half_steps() {
        let config = AdaptiveRk4Config::builder().build().unwrap();
        let report = schwarzschild_adaptive_rk4_step(1.0, initial(), 0.01, config).unwrap();
        let half = schwarzschild_geodesic_rk4_step(1.0, initial(), report.step_used * 0.5);
        assert_eq!(report.state, schwarzschild_geodesic_rk4_step(1.0, half, report.step_used * 0.5));
        assert!(report.error_ratio <= 1.0);
        assert!(report.next_step > 0.0 && report.next_step <= 0.08);
    }

    #[test]
    fn strict_tolerance_rejects_without_unbounded_retry() {
        let config = AdaptiveRk4Config::builder().tolerances(1.0e-30, 0.0).maximum_retries(0).build().unwrap();
        assert!(schwarzschild_adaptive_rk4_step(1.0, initial(), 0.08, config).unwrap_err().to_string().contains("retry budget"));
        assert!(schwarzschild_adaptive_rk4_step(1.0, initial(), f64::NAN, config).is_err());
    }
}

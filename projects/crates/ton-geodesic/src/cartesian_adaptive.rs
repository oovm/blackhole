//! 笛卡尔状态上的独立步长加倍误差控制。

use ton_types::{TonError, TonResult};
use crate::{schwarzschild_cartesian_null_rk4_step, schwarzschild_cartesian_conserved_null_rk4_step, AdaptiveRk4Config, SchwarzschildCartesianNullState};

/// 通过局部误差门禁的参考状态，不含失败候选。
#[derive(Debug, Clone, Copy)]
pub struct CartesianAdaptiveRk4Step {
    /// 两个半步的状态，不做 Richardson 投影。
    pub state: SchwarzschildCartesianNullState,
    /// 实际接受的仿射步长。
    pub step_used: f64,
    /// 下一步建议步长。
    pub next_step: f64,
    /// 最大逐分量加权误差。
    pub error_ratio: f64,
    /// 被拒绝的候选数。
    pub rejections: u32,
    /// 可选全程初始 L²。事件重积分必须使用同一个力方程参数。
    pub conserved_angular_squared: Option<f64>,
}

/// 独立 Rust 积分器。域外中间状态与非有限候选都必须拒绝并缩短步长。
pub fn schwarzschild_cartesian_adaptive_rk4_step(
    mass: f64, state: SchwarzschildCartesianNullState, requested_step: f64, config: AdaptiveRk4Config,
) -> TonResult<CartesianAdaptiveRk4Step> {
    adaptive_step(mass, state, requested_step, config, None)
}

pub(crate) fn adaptive_step(mass: f64, state: SchwarzschildCartesianNullState, requested_step: f64,
    config: AdaptiveRk4Config, conserved_angular: Option<f64>,
) -> TonResult<CartesianAdaptiveRk4Step> {
    if !mass.is_finite() || mass <= 0.0 || !requested_step.is_finite() || requested_step <= 0.0
        || state.components().iter().any(|value| !value.is_finite()) || state.energy <= 0.0
        || !state.position.length().is_finite() || state.position.length() <= 2.0 * mass
        || conserved_angular.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(TonError::Domain("invalid Cartesian adaptive initial state or step".into()));
    }
    let (absolute, relative) = config.tolerances();
    let (minimum, maximum) = config.step_range();
    let mut step = requested_step.clamp(minimum, maximum);
    for rejections in 0..=config.maximum_retries() {
        let advance = |state, step| match conserved_angular {
            Some(angular) => schwarzschild_cartesian_conserved_null_rk4_step(mass, state, angular, step),
            None => schwarzschild_cartesian_null_rk4_step(mass, state, step),
        };
        let candidates = (|| {
            let full = advance(state, step)?;
            let half = advance(state, step * 0.5)?;
            let refined = advance(half, step * 0.5)?;
            Ok::<_, TonError>((full, refined))
        })();
        let mut error_ratio = f64::INFINITY;
        if let Ok((full, refined)) = candidates {
            error_ratio = 0.0;
            for ((initial, full), refined) in state.components().into_iter().zip(full.components()).zip(refined.components()) {
                let scale = absolute + relative * initial.abs().max(refined.abs());
                let ratio = ((refined - full) / 15.0).abs() / scale;
                if !ratio.is_finite() { error_ratio = f64::INFINITY; break; }
                error_ratio = error_ratio.max(ratio);
            }
            if error_ratio <= 1.0 {
                let factor = if error_ratio == 0.0 { 2.0 } else { (0.9 * error_ratio.powf(-0.2)).clamp(0.5, 2.0) };
                return Ok(CartesianAdaptiveRk4Step { state: refined, step_used: step,
                    next_step: (step * factor).clamp(minimum, maximum), error_ratio, rejections,
                    conserved_angular_squared: conserved_angular });
            }
        }
        if step <= minimum { return Err(TonError::Domain(format!("Cartesian adaptive minimum step exhausted: ratio={error_ratio}"))); }
        step = (step * (0.9 * error_ratio.powf(-0.2)).clamp(0.1, 0.5)).max(minimum);
    }
    Err(TonError::Domain("Cartesian adaptive retry budget exhausted".into()))
}

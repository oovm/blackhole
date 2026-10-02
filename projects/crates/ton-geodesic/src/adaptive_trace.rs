//! 独立自适应追踪参考：接受步、物理事件和守恒诊断分层。

use ton_types::{TonError, TonResult};

use crate::{
    schwarzschild_accepted_step_events, schwarzschild_adaptive_rk4_step,
    AdaptiveRk4Config, SchwarzschildEventConfig, SchwarzschildStepEvent,
    SchwarzschildStepEventKind, SphericalState,
};

/// 独立参考追踪配置。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildAdaptiveTraceConfig {
    mass: f64,
    integration: AdaptiveRk4Config,
    events: SchwarzschildEventConfig,
    initial_step: f64,
    maximum_steps: u32,
}

/// 完整参考追踪构造器。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildAdaptiveTraceConfigBuilder {
    mass: f64,
    integration: AdaptiveRk4Config,
    events: Option<SchwarzschildEventConfig>,
    initial_step: f64,
    maximum_steps: u32,
}

impl SchwarzschildAdaptiveTraceConfig {
    /// 默认接受步预算为一万步，默认初始步长为 `0.08M`。
    pub fn builder(mass: f64) -> SchwarzschildAdaptiveTraceConfigBuilder {
        SchwarzschildAdaptiveTraceConfigBuilder {
            mass, integration: AdaptiveRk4Config::builder().build().expect("valid reference defaults"),
            events: None, initial_step: 0.08 * mass, maximum_steps: 10_000,
        }
    }
}

impl SchwarzschildAdaptiveTraceConfigBuilder {
    /// 设置独立积分误差控制，不传递可执行回调。
    pub fn integration(mut self, config: AdaptiveRk4Config) -> Self { self.integration = config; self }

    /// 设置捕获、逃逸与薄盘事件配置。
    pub fn events(mut self, config: SchwarzschildEventConfig) -> Self { self.events = Some(config); self }

    /// 设置初始请求仿射步长。
    pub fn initial_step(mut self, step: f64) -> Self { self.initial_step = step; self }

    /// 设置接受步上限。
    pub fn maximum_steps(mut self, steps: u32) -> Self { self.maximum_steps = steps; self }

    /// 校验质量一致性、有限正步长与有界预算。
    pub fn build(self) -> TonResult<SchwarzschildAdaptiveTraceConfig> {
        if !self.mass.is_finite() || self.mass <= 0.0 || !self.initial_step.is_finite() || self.initial_step <= 0.0
            || self.maximum_steps == 0 || self.maximum_steps > 1_000_000
        {
            return Err(TonError::Domain("invalid Schwarzschild adaptive trace configuration".into()));
        }
        let events = match self.events {
            Some(events) if events.mass() == self.mass => events,
            Some(_) => return Err(TonError::Domain("event and trace masses differ".into())),
            None => SchwarzschildEventConfig::builder(self.mass).build()?,
        };
        Ok(SchwarzschildAdaptiveTraceConfig {
            mass: self.mass, integration: self.integration, events,
            initial_step: self.initial_step, maximum_steps: self.maximum_steps,
        })
    }
}

/// 完整追踪的终止分类，不复用单步积分控制状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveTraceTermination {
    /// 穿过配置的捕获面。
    Captured,
    /// 向外穿过配置的逃逸面。
    Escaped,
    /// 有效接受步预算耗尽，不声称捕获或逃逸。
    StepBudgetExhausted,
    /// 积分误差控制未能接受下一步。
    IntegrationFailure,
    /// 已括住事件未通过定位残差门禁。
    EventLocalizationFailure,
}

/// 盘交点的轨迹时间与定位状态。
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveTraceDiskEvent {
    /// 自初始状态累计的仿射参数。
    pub affine_parameter: f64,
    /// 包含此事件的接受步序号，从零开始。
    pub step_index: u32,
    /// 未经几何投影的参考交点。
    pub event: SchwarzschildStepEvent,
}

/// 独立追踪结果，失败保留最后一个有效状态。
#[derive(Debug, Clone)]
pub struct SchwarzschildAdaptiveTraceResult {
    /// 光线终止原因。
    pub termination: AdaptiveTraceTermination,
    /// 最后接受状态或定位后的终止事件状态。
    pub state: SphericalState,
    /// 已接受步数。
    pub accepted_steps: u32,
    /// 已接受步的累计拒绝试验次数，不包括失败步骤内部的试验。
    pub rejected_trials: u64,
    /// 到最终状态的累计仿射参数。
    pub affine_parameter: f64,
    /// 全部有效环带交点，不因首次命中而忽略透镜后的再次穿越。
    pub disk_events: Vec<AdaptiveTraceDiskEvent>,
    /// 每个接受步端点、盘交点和最终终止状态中的最大归一化零约束残差。
    pub maximum_null_residual: f64,
    /// 相同采样范围的最大能量相对漂移。
    pub maximum_energy_drift: f64,
    /// 相同采样范围的最大角动量平方相对漂移。
    pub maximum_angular_squared_drift: f64,
    /// 失败详情。成功与纯预算终止没有错误字符串。
    pub failure: Option<String>,
}

fn invariants(mass: f64, state: SphericalState) -> (f64, f64, f64) {
    let f = 1.0 - 2.0 * mass / state.r;
    let angular = state.utheta.powi(2) + state.theta.sin().powi(2) * state.uphi.powi(2);
    let temporal = f * state.ut.powi(2);
    let spatial = state.ur.powi(2) / f + state.r.powi(2) * angular;
    let null = (spatial - temporal).abs() / (spatial.abs() + temporal.abs());
    (f * state.ut, state.r.powi(4) * angular, null)
}

fn observe(result: &mut SchwarzschildAdaptiveTraceResult, mass: f64, state: SphericalState, initial: (f64, f64, f64)) {
    let (energy, angular, null) = invariants(mass, state);
    result.maximum_null_residual = result.maximum_null_residual.max(null);
    result.maximum_energy_drift = result.maximum_energy_drift.max(((energy - initial.0) / initial.0).abs());
    let angular_error = if initial.1 == 0.0 { angular.abs() } else { ((angular - initial.1) / initial.1).abs() };
    result.maximum_angular_squared_drift = result.maximum_angular_squared_drift.max(angular_error);
}

/// 独立 Rust 参考追踪。UCF 前端与 backend 不得调用本函数生成输出。
///
/// 事件完整性仍需对最大接受步长做收敛对照。端点同侧的切触或多次穿越
/// 不能仅凭有限输出证明不存在。
pub fn schwarzschild_adaptive_trace(
    initial: SphericalState, config: SchwarzschildAdaptiveTraceConfig,
) -> TonResult<SchwarzschildAdaptiveTraceResult> {
    let components = [initial.t, initial.r, initial.theta, initial.phi, initial.ut, initial.ur, initial.utheta, initial.uphi];
    let constants = invariants(config.mass, initial);
    if components.iter().any(|value| !value.is_finite()) || initial.r <= 2.0 * config.mass
        || !constants.0.is_finite() || constants.0 <= 0.0 || !constants.1.is_finite() || !constants.2.is_finite()
    {
        return Err(TonError::Domain("invalid Schwarzschild adaptive trace initial state".into()));
    }
    let mut result = SchwarzschildAdaptiveTraceResult {
        termination: AdaptiveTraceTermination::StepBudgetExhausted, state: initial,
        accepted_steps: 0, rejected_trials: 0, affine_parameter: 0.0, disk_events: Vec::new(),
        maximum_null_residual: constants.2, maximum_energy_drift: 0.0,
        maximum_angular_squared_drift: 0.0, failure: None,
    };
    if initial.r <= config.events.capture_radius() {
        result.termination = AdaptiveTraceTermination::Captured;
        return Ok(result);
    }
    if initial.r >= config.events.escape_radius() && initial.ur > 0.0 {
        result.termination = AdaptiveTraceTermination::Escaped;
        return Ok(result);
    }
    let mut requested = config.initial_step;
    for step_index in 0..config.maximum_steps {
        let accepted = match schwarzschild_adaptive_rk4_step(config.mass, result.state, requested, config.integration) {
            Ok(accepted) => accepted,
            Err(error) => {
                result.termination = AdaptiveTraceTermination::IntegrationFailure;
                result.failure = Some(error.to_string());
                return Ok(result);
            }
        };
        let events = match schwarzschild_accepted_step_events(result.state, accepted, config.events) {
            Ok(events) => events,
            Err(error) => {
                result.termination = AdaptiveTraceTermination::EventLocalizationFailure;
                result.failure = Some(error.to_string());
                return Ok(result);
            }
        };
        result.accepted_steps += 1;
        result.rejected_trials += accepted.rejections as u64;
        for event in events {
            observe(&mut result, config.mass, event.state, constants);
            let affine = result.affine_parameter + accepted.step_used * event.fraction;
            match event.kind {
                SchwarzschildStepEventKind::DiskCrossing => {
                    result.disk_events.push(AdaptiveTraceDiskEvent { affine_parameter: affine, step_index, event });
                }
                SchwarzschildStepEventKind::Capture | SchwarzschildStepEventKind::Escape => {
                    result.state = event.state;
                    result.affine_parameter = affine;
                    result.termination = if event.kind == SchwarzschildStepEventKind::Capture {
                        AdaptiveTraceTermination::Captured
                    } else { AdaptiveTraceTermination::Escaped };
                    return Ok(result);
                }
            }
        }
        observe(&mut result, config.mass, accepted.state, constants);
        result.state = accepted.state;
        result.affine_parameter += accepted.step_used;
        requested = accepted.next_step;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schwarzschild_geodesic_initial_state;
    use ton_types::{BlackHole, Camera, Vec3};

    #[test]
    fn trace_builder_rejects_invalid_budgets_and_mass_mismatch() {
        assert!(SchwarzschildAdaptiveTraceConfig::builder(0.0).build().is_err());
        assert!(SchwarzschildAdaptiveTraceConfig::builder(1.0).maximum_steps(0).build().is_err());
        assert!(SchwarzschildAdaptiveTraceConfig::builder(1.0).initial_step(f64::NAN).build().is_err());
        let wrong_mass = SchwarzschildEventConfig::builder(2.0).build().unwrap();
        assert!(SchwarzschildAdaptiveTraceConfig::builder(1.0).events(wrong_mass).build().is_err());
    }

    #[test]
    fn adaptive_trace_capture_matches_independent_analytic_boundary() {
        let bh = BlackHole::schwarzschild(1.0);
        let position = Vec3::new(24.0, 0.0, 0.0);
        for angle in [0.0f64, 0.1, 0.19, 0.21, 0.23, 0.25, 0.4] {
            let direction = Vec3::new(-angle.cos(), angle.sin(), 0.0);
            let initial = schwarzschild_geodesic_initial_state(1.0, position, direction).unwrap();
            let expected = crate::schwarzschild_geodesic_analytic_captured(bh, position, direction);
            let result = schwarzschild_adaptive_trace(initial, SchwarzschildAdaptiveTraceConfig::builder(1.0).build().unwrap()).unwrap();
            assert_eq!(result.termination, if expected { AdaptiveTraceTermination::Captured } else { AdaptiveTraceTermination::Escaped }, "angle={angle}, result={result:?}");
            assert!(result.maximum_null_residual < 1.0e-5, "angle={angle}, residual={}", result.maximum_null_residual);
            assert!(result.maximum_energy_drift < 1.0e-5);
            assert!(result.maximum_angular_squared_drift < 1.0e-5);
        }
    }

    #[test]
    fn adaptive_trace_polar_ray_escapes_with_per_step_invariants() {
        let camera = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let u = ((39.5 / 128.0) * 2.0 - 1.0) * (128.0 / 72.0);
        let v = 1.0 - (0.5 / 72.0) * 2.0;
        let initial = schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
        let result = schwarzschild_adaptive_trace(initial, SchwarzschildAdaptiveTraceConfig::builder(1.0).build().unwrap()).unwrap();
        assert_eq!(result.termination, AdaptiveTraceTermination::Escaped, "{result:?}");
        assert!((result.state.r - 80.0).abs() <= 1.0e-8);
        assert!(result.maximum_null_residual < 1.0e-5);
        assert!(result.maximum_energy_drift < 1.0e-5);
        assert!(result.maximum_angular_squared_drift < 1.0e-5);
    }

    #[test]
    fn trace_failure_and_budget_preserve_last_valid_state() {
        let initial = schwarzschild_geodesic_initial_state(1.0, Vec3::new(24.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)).unwrap();
        let budget = SchwarzschildAdaptiveTraceConfig::builder(1.0).maximum_steps(1).build().unwrap();
        let result = schwarzschild_adaptive_trace(initial, budget).unwrap();
        assert_eq!(result.termination, AdaptiveTraceTermination::StepBudgetExhausted);
        assert_eq!(result.accepted_steps, 1);
        assert!(result.failure.is_none());
        let strict = AdaptiveRk4Config::builder().tolerances(1.0e-30, 0.0).maximum_retries(0).build().unwrap();
        let config = SchwarzschildAdaptiveTraceConfig::builder(1.0).integration(strict).build().unwrap();
        let result = schwarzschild_adaptive_trace(initial, config).unwrap();
        assert_eq!(result.termination, AdaptiveTraceTermination::IntegrationFailure);
        assert_eq!(result.state, initial);
        assert_eq!(result.accepted_steps, 0);
        assert!(result.failure.is_some());
    }

    #[test]
    fn disk_events_converge_when_maximum_step_is_halved() {
        let camera = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let events = SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0)
            .localization(1.0e-9, 48).build().unwrap();
        let configs = [0.08, 0.04].map(|maximum| {
            let integration = AdaptiveRk4Config::builder().tolerances(1.0e-10, 1.0e-8)
                .step_range(1.0e-9, maximum).build().unwrap();
            SchwarzschildAdaptiveTraceConfig::builder(1.0).integration(integration).events(events).build().unwrap()
        });
        let mut crossings = 0;
        let mut multiple_crossings = 0;
        for u in [-0.9, -0.7, -0.5, 0.0, 0.5, 0.7, 0.9] {
            for v in [-0.4, -0.2, 0.0, 0.2, 0.4] {
                let initial = schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
                let coarse = schwarzschild_adaptive_trace(initial, configs[0]).unwrap();
                let fine = schwarzschild_adaptive_trace(initial, configs[1]).unwrap();
                assert!(matches!(coarse.termination, AdaptiveTraceTermination::Captured | AdaptiveTraceTermination::Escaped), "u={u}, v={v}, result={coarse:?}");
                assert_eq!(coarse.termination, fine.termination);
                assert_eq!(coarse.disk_events.len(), fine.disk_events.len(), "u={u}, v={v}");
                crossings += coarse.disk_events.len();
                multiple_crossings += usize::from(coarse.disk_events.len() > 1);
                for (coarse, fine) in coarse.disk_events.iter().zip(&fine.disk_events) {
                    assert!((coarse.affine_parameter - fine.affine_parameter).abs() < 1.0e-4, "u={u}, v={v}");
                    assert!((coarse.event.state.r - fine.event.state.r).abs() < 1.0e-4);
                    assert!(coarse.event.surface_residual <= 1.0e-9);
                }
            }
        }
        eprintln!("converged disk events={crossings}, rays with multiple crossings={multiple_crossings}");
        assert!(crossings > 0);
        assert!(multiple_crossings > 0);
    }
}

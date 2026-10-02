//! 独立笛卡尔自适应追踪：有界积分、事件定位和不变量门禁。

use ton_types::{TonError, TonResult};
use crate::{schwarzschild_cartesian_accepted_step_events,
    AdaptiveTraceTermination, CartesianStepEvent, SchwarzschildAdaptiveTraceConfig,
    SchwarzschildCartesianNullState, SchwarzschildStepEventKind};

/// 全程累计仿射参数下的盘面事件。
#[derive(Debug, Clone, Copy)]
pub struct CartesianTraceDiskEvent {
    /// 从初态开始累计的仿射参数。
    pub affine_parameter: f64,
    /// 从零开始的接受步编号。
    pub step_index: u32,
    /// 原始定位事件。
    pub event: CartesianStepEvent,
}

/// 独立参考结果，失败时保留最后有效状态与已完成事件。
#[derive(Debug, Clone)]
pub struct CartesianAdaptiveTraceResult {
    /// 终止原因，与积分器控制状态分离。
    pub termination: AdaptiveTraceTermination,
    /// 最后接受状态或终止交点。
    pub state: SchwarzschildCartesianNullState,
    /// 接受步数。
    pub accepted_steps: u32,
    /// 已接受步累计拒绝次数。
    pub rejected_trials: u64,
    /// 最终状态累计仿射参数。
    pub affine_parameter: f64,
    /// 终止前全部盘面交点。
    pub disk_events: Vec<CartesianTraceDiskEvent>,
    /// 接受端点与有效交点的最大零约束残差。
    pub maximum_null_residual: f64,
    /// 角动量平方漂移除以 `max(L0²,(M E0)²)`，径向零基线采用物理尺度绝对误差。
    pub maximum_angular_squared_drift: f64,
    /// 同样采样范围的能量相对漂移。
    pub maximum_energy_drift: f64,
    /// 数值失败的详情。
    pub failure: Option<String>,
}

fn observe(result: &mut CartesianAdaptiveTraceResult, mass: f64, state: SchwarzschildCartesianNullState,
    initial_energy: f64, initial_angular: f64,
) {
    let angular = state.angular_momentum();
    let angular = angular.dot(angular);
    result.maximum_null_residual = result.maximum_null_residual.max(state.null_residual(mass));
    let angular_scale = initial_angular.max((mass * initial_energy).powi(2));
    result.maximum_angular_squared_drift = result.maximum_angular_squared_drift.max((angular - initial_angular).abs() / angular_scale);
    result.maximum_energy_drift = result.maximum_energy_drift.max((state.energy / initial_energy - 1.0).abs());
}

/// 独立物理参考，不允许 UCF 后端调用此函数填充执行结果。
pub fn schwarzschild_cartesian_adaptive_trace(initial: SchwarzschildCartesianNullState,
    config: SchwarzschildAdaptiveTraceConfig,
) -> TonResult<CartesianAdaptiveTraceResult> {
    let mass = config.mass();
    let angular = initial.angular_momentum();
    let angular_squared = angular.dot(angular);
    let null = initial.null_residual(mass);
    if initial.components().iter().any(|value| !value.is_finite()) || initial.position.length() <= 2.0 * mass
        || !angular_squared.is_finite() || !null.is_finite() || null > 1.0e-6 || initial.energy <= 0.0 {
        return Err(TonError::Domain("invalid Cartesian adaptive trace initial state".into()));
    }
    let mut result = CartesianAdaptiveTraceResult { termination: AdaptiveTraceTermination::StepBudgetExhausted,
        state: initial, accepted_steps: 0, rejected_trials: 0, affine_parameter: 0.0, disk_events: Vec::new(),
        maximum_null_residual: null, maximum_angular_squared_drift: 0.0, maximum_energy_drift: 0.0, failure: None };
    if initial.position.length() <= config.events().capture_radius() { result.termination = AdaptiveTraceTermination::Captured; return Ok(result); }
    if initial.position.length() >= config.events().escape_radius() && initial.position.dot(initial.velocity) > 0.0 {
        result.termination = AdaptiveTraceTermination::Escaped; return Ok(result);
    }
    let mut requested = config.initial_step();
    for step_index in 0..config.maximum_steps() {
        let accepted = match crate::cartesian_adaptive::adaptive_step(mass, result.state, requested, config.integration(), Some(angular_squared)) {
            Ok(accepted) => accepted,
            Err(error) => { result.termination = AdaptiveTraceTermination::IntegrationFailure; result.failure = Some(error.to_string()); return Ok(result); }
        };
        let events = match schwarzschild_cartesian_accepted_step_events(result.state, accepted, config.events()) {
            Ok(events) => events,
            Err(error) => { result.termination = AdaptiveTraceTermination::EventLocalizationFailure; result.failure = Some(error.to_string()); return Ok(result); }
        };
        result.accepted_steps += 1;
        result.rejected_trials += u64::from(accepted.rejections);
        for event in events {
            observe(&mut result, mass, event.state, initial.energy, angular_squared);
            let affine_parameter = result.affine_parameter + accepted.step_used * event.fraction;
            match event.kind {
                SchwarzschildStepEventKind::DiskCrossing => result.disk_events.push(CartesianTraceDiskEvent { affine_parameter, step_index, event }),
                SchwarzschildStepEventKind::Capture | SchwarzschildStepEventKind::Escape => {
                    result.state = event.state;
                    result.affine_parameter = affine_parameter;
                    result.termination = if event.kind == SchwarzschildStepEventKind::Capture { AdaptiveTraceTermination::Captured } else { AdaptiveTraceTermination::Escaped };
                    return Ok(result);
                }
            }
        }
        observe(&mut result, mass, accepted.state, initial.energy, angular_squared);
        result.state = accepted.state;
        result.affine_parameter += accepted.step_used;
        requested = accepted.next_step;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schwarzschild_cartesian_adaptive_rk4_step, schwarzschild_cartesian_accepted_step_events,
        AdaptiveRk4Config, SchwarzschildEventConfig};
    use ton_types::{BlackHole, Camera, Vec3};

    #[test]
    fn cartesian_radial_events_match_analytic_affine_position() {
        for (velocity, capture, escape, kind) in [(-1.0, 9.98, 80.0, SchwarzschildStepEventKind::Capture),
            (1.0, 2.0002, 10.02, SchwarzschildStepEventKind::Escape)] {
            let initial = SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, velocity)).unwrap();
            let accepted = schwarzschild_cartesian_adaptive_rk4_step(1.0, initial, 0.04, AdaptiveRk4Config::builder().build().unwrap()).unwrap();
            let config = SchwarzschildEventConfig::builder(1.0).radial_boundaries(capture, escape).build().unwrap();
            let events = schwarzschild_cartesian_accepted_step_events(initial, accepted, config).unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].kind, kind);
            assert!((events[0].fraction * accepted.step_used - 0.02).abs() < 1.0e-8);
            assert!(events[0].surface_residual <= 1.0e-8);
        }
    }

    #[test]
    fn cartesian_disk_events_are_localized_filtered_and_terminal_ordered() {
        let initial = SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(10.0, 0.0, 0.001), Vec3::new(1.0, 0.0, -1.0)).unwrap();
        let accepted = schwarzschild_cartesian_adaptive_rk4_step(1.0, initial, 0.04, AdaptiveRk4Config::builder().build().unwrap()).unwrap();
        let config = SchwarzschildEventConfig::builder(1.0).radial_boundaries(2.0002, 10.02).equatorial_disk(6.0, 20.0).build().unwrap();
        let events = schwarzschild_cartesian_accepted_step_events(initial, accepted, config).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, SchwarzschildStepEventKind::DiskCrossing);
        assert_eq!(events[1].kind, SchwarzschildStepEventKind::Escape);
        assert!(events[0].fraction < events[1].fraction);
        assert!(events.iter().all(|event| event.surface_residual <= 1.0e-8));
        let early = SchwarzschildEventConfig::builder(1.0).radial_boundaries(2.0002, 10.0001).equatorial_disk(6.0, 20.0).build().unwrap();
        let early_events = schwarzschild_cartesian_accepted_step_events(initial, accepted, early).unwrap();
        assert_eq!(early_events.len(), 1);
        assert_eq!(early_events[0].kind, SchwarzschildStepEventKind::Escape);
        let narrow = SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 9.0).build().unwrap();
        assert!(schwarzschild_cartesian_accepted_step_events(initial, accepted, narrow).unwrap().is_empty());
        let insufficient = SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).localization(1.0e-16, 1).build().unwrap();
        assert!(schwarzschild_cartesian_accepted_step_events(initial, accepted, insufficient).is_err());
    }

    #[test]
    fn cartesian_trace_distinguishes_budget_failure_and_invalid_initial_state() {
        let initial = SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)).unwrap();
        let budget = SchwarzschildAdaptiveTraceConfig::builder(1.0).maximum_steps(1).build().unwrap();
        let result = schwarzschild_cartesian_adaptive_trace(initial, budget).unwrap();
        assert_eq!(result.termination, AdaptiveTraceTermination::StepBudgetExhausted);
        assert_eq!(result.accepted_steps, 1);
        let impossible = AdaptiveRk4Config::builder().tolerances(1.0e-30, 0.0).step_range(0.08, 0.08).build().unwrap();
        let result = schwarzschild_cartesian_adaptive_trace(initial,
            SchwarzschildAdaptiveTraceConfig::builder(1.0).integration(impossible).build().unwrap()).unwrap();
        assert_eq!(result.termination, AdaptiveTraceTermination::IntegrationFailure);
        assert_eq!(result.state, initial);
        assert_eq!(result.accepted_steps, 0);
        assert!(result.failure.is_some());
        let invalid = SchwarzschildCartesianNullState { energy: 2.0, ..initial };
        assert!(schwarzschild_cartesian_adaptive_trace(invalid, budget).is_err());
    }

    #[test]
    fn cartesian_capture_classification_and_near_horizon_localization_match_reference() {
        for angle in [0.0f64, 0.1, 0.19, 0.21, 0.23, 0.25, 0.4] {
            let position = Vec3::new(24.0, 0.0, 0.0);
            let direction = Vec3::new(-angle.cos(), angle.sin(), 0.0);
            let initial = SchwarzschildCartesianNullState::from_ray(1.0, position, direction).unwrap();
            let result = schwarzschild_cartesian_adaptive_trace(initial, SchwarzschildAdaptiveTraceConfig::builder(1.0).build().unwrap()).unwrap();
            let captured = crate::schwarzschild_geodesic_analytic_captured(BlackHole::schwarzschild(1.0), position, direction);
            assert_eq!(result.termination, if captured { AdaptiveTraceTermination::Captured } else { AdaptiveTraceTermination::Escaped }, "angle={angle}, failure={:?}", result.failure);
            assert!((result.state.position.length() - if captured { 2.0002 } else { 80.0 }).abs() <= 1.0e-8);
            assert!(result.maximum_null_residual < 1.0e-5);
            assert!(result.maximum_angular_squared_drift < 1.0e-5);
        }
    }

    #[test]
    fn cartesian_full_trace_disk_events_converge_and_match_spherical_reference() {
        let camera = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let mut crossings = 0;
        let mut multiple = 0;
        for y in 0..5 {
            for x in 0..7 {
                let direction = camera.ray_direction(((x as f64 + 0.5) / 7.0 * 2.0 - 1.0) * 7.0 / 5.0,
                    1.0 - (y as f64 + 0.5) / 5.0 * 2.0);
                let initial = SchwarzschildCartesianNullState::from_ray(1.0, camera.position, direction).unwrap();
                let events = SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).build().unwrap();
                let config = |maximum| SchwarzschildAdaptiveTraceConfig::builder(1.0).events(events)
                    .integration(AdaptiveRk4Config::builder().step_range(1.0e-8, maximum).build().unwrap()).build().unwrap();
                let coarse = schwarzschild_cartesian_adaptive_trace(initial, config(0.08)).unwrap();
                let fine = schwarzschild_cartesian_adaptive_trace(initial, config(0.04)).unwrap();
                let spherical = crate::schwarzschild_adaptive_trace(crate::schwarzschild_geodesic_initial_state(1.0, camera.position, direction).unwrap(), config(0.04)).unwrap();
                assert!(matches!(fine.termination, AdaptiveTraceTermination::Captured | AdaptiveTraceTermination::Escaped), "pixel={x},{y}, failure={:?}", fine.failure);
                assert_eq!(coarse.termination, fine.termination);
                assert_eq!(fine.termination, spherical.termination);
                assert_eq!(coarse.disk_events.len(), fine.disk_events.len());
                assert_eq!(fine.disk_events.len(), spherical.disk_events.len());
                for ((coarse, fine), spherical) in coarse.disk_events.iter().zip(&fine.disk_events).zip(&spherical.disk_events) {
                    assert!((coarse.event.state.position.length() - fine.event.state.position.length()).abs() < 1.0e-4);
                    assert!((fine.event.state.position.length() - spherical.event.state.r).abs() < 1.0e-4);
                    assert!((fine.affine_parameter - spherical.affine_parameter).abs() < 1.0e-4);
                }
                assert!(fine.maximum_null_residual < 1.0e-5 && fine.maximum_angular_squared_drift < 1.0e-5,
                    "pixel={x},{y}, null={}, angular={}, initial_angular={:?}", fine.maximum_null_residual,
                    fine.maximum_angular_squared_drift, initial.angular_momentum());
                crossings += fine.disk_events.len();
                multiple += usize::from(fine.disk_events.len() > 1);
            }
        }
        eprintln!("Cartesian reference disk crossings={crossings}, multiple={multiple}");
        assert!(crossings > 0 && multiple > 0);
    }
}

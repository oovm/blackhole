//! 笛卡尔接受步的球面与赤道平面交点参考。

use ton_types::{TonError, TonResult};
use crate::{schwarzschild_cartesian_null_rk4_step, schwarzschild_cartesian_conserved_null_rk4_step, CartesianAdaptiveRk4Step,
    SchwarzschildCartesianNullState, SchwarzschildEventConfig, SchwarzschildStepEventKind};

/// 未经投影的笛卡尔事件状态。
#[derive(Debug, Clone, Copy)]
pub struct CartesianStepEvent {
    /// TON 物理事件类型。
    pub kind: SchwarzschildStepEventKind,
    /// 接受步中的仿射比例。
    pub fraction: f64,
    /// 重新积分到交点的状态。
    pub state: SchwarzschildCartesianNullState,
    /// 事件面实际残差。
    pub surface_residual: f64,
}

fn surface(state: SchwarzschildCartesianNullState, kind: SchwarzschildStepEventKind, config: SchwarzschildEventConfig) -> f64 {
    match kind {
        SchwarzschildStepEventKind::Capture => state.position.length() - config.capture_radius(),
        SchwarzschildStepEventKind::Escape => state.position.length() - config.escape_radius(),
        SchwarzschildStepEventKind::DiskCrossing => state.position.z,
    }
}

fn locate(start: SchwarzschildCartesianNullState, accepted: CartesianAdaptiveRk4Step,
    kind: SchwarzschildStepEventKind, config: SchwarzschildEventConfig,
) -> TonResult<CartesianStepEvent> {
    let endpoint = surface(accepted.state, kind, config);
    if endpoint.abs() <= config.surface_tolerance() {
        return Ok(CartesianStepEvent { kind, fraction: 1.0, state: accepted.state, surface_residual: endpoint.abs() });
    }
    let mut lower = 0.0;
    let mut upper = 1.0;
    let mut lower_value = surface(start, kind, config);
    for _ in 0..config.maximum_bisections() {
        let fraction = (lower + upper) * 0.5;
        let half_step = accepted.step_used * fraction * 0.5;
        let advance = |state| match accepted.conserved_angular_squared {
            Some(angular) => schwarzschild_cartesian_conserved_null_rk4_step(config.mass(), state, angular, half_step),
            None => schwarzschild_cartesian_null_rk4_step(config.mass(), state, half_step),
        };
        let half = advance(start)?;
        let state = advance(half)?;
        let value = surface(state, kind, config);
        if !value.is_finite() { return Err(TonError::Domain("nonfinite Cartesian event surface".into())); }
        if value.abs() <= config.surface_tolerance() {
            return Ok(CartesianStepEvent { kind, fraction, state, surface_residual: value.abs() });
        }
        if value.is_sign_positive() == lower_value.is_sign_positive() { lower = fraction; lower_value = value; }
        else { upper = fraction; }
    }
    Err(TonError::Domain("Cartesian event localization budget exhausted".into()))
}

/// 定位端点已括住的事件并截断终止后的盘命中，不保证同侧多穿越或切触。
pub fn schwarzschild_cartesian_accepted_step_events(
    start: SchwarzschildCartesianNullState, accepted: CartesianAdaptiveRk4Step, config: SchwarzschildEventConfig,
) -> TonResult<Vec<CartesianStepEvent>> {
    if start.components().iter().chain(accepted.state.components().iter()).any(|value| !value.is_finite())
        || start.position.length() <= 2.0 * config.mass() || accepted.state.position.length() <= 2.0 * config.mass()
        || !accepted.step_used.is_finite() || accepted.step_used <= 0.0
        || !accepted.error_ratio.is_finite() || !(0.0..=1.0).contains(&accepted.error_ratio) {
        return Err(TonError::Domain("invalid Cartesian accepted event step".into()));
    }
    let mut events = Vec::new();
    let before = start.position.length();
    let after = accepted.state.position.length();
    if before > config.capture_radius() && after <= config.capture_radius() {
        events.push(locate(start, accepted, SchwarzschildStepEventKind::Capture, config)?);
    }
    if before < config.escape_radius() && after >= config.escape_radius()
        && accepted.state.position.dot(accepted.state.velocity) > 0.0 {
        events.push(locate(start, accepted, SchwarzschildStepEventKind::Escape, config)?);
    }
    if let Some((inner, outer)) = config.disk_radii() {
        let before = start.position.z;
        let after = accepted.state.position.z;
        if before.abs() > config.surface_tolerance() && (before.is_sign_positive() != after.is_sign_positive() || after == 0.0) {
            let event = locate(start, accepted, SchwarzschildStepEventKind::DiskCrossing, config)?;
            if (inner..=outer).contains(&event.state.position.length()) { events.push(event); }
        }
    }
    events.sort_by(|left, right| left.fraction.total_cmp(&right.fraction));
    if let Some(terminal) = events.iter().position(|event| event.kind != SchwarzschildStepEventKind::DiskCrossing) { events.truncate(terminal + 1); }
    Ok(events)
}

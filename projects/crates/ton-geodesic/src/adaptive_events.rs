//! 自适应接受步的独立事件定位，不依赖 UCF 或渲染后处理。

use ton_types::{TonError, TonResult};

use crate::{schwarzschild_geodesic_rk4_step, AdaptiveRk4Step, SphericalState};

/// Schwarzschild 参考事件配置。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildEventConfig {
    mass: f64,
    capture_radius: f64,
    escape_radius: f64,
    disk_radii: Option<(f64, f64)>,
    surface_tolerance: f64,
    maximum_bisections: u32,
}

/// 事件配置构造器，所有长度以参考模型的几何单位表示。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildEventConfigBuilder {
    config: SchwarzschildEventConfig,
}

impl SchwarzschildEventConfig {
    pub(crate) fn mass(self) -> f64 { self.mass }
    pub(crate) fn capture_radius(self) -> f64 { self.capture_radius }
    pub(crate) fn escape_radius(self) -> f64 { self.escape_radius }

    /// 捕获面默认位于坐标视界之外，不声称定位到奇异坐标上的事件视界。
    pub fn builder(mass: f64) -> SchwarzschildEventConfigBuilder {
        SchwarzschildEventConfigBuilder { config: Self {
            mass, capture_radius: 2.0002 * mass, escape_radius: 80.0 * mass,
            disk_radii: None, surface_tolerance: 1.0e-8 * mass, maximum_bisections: 48,
        } }
    }
}

impl SchwarzschildEventConfigBuilder {
    /// 设置捕获面与远场逃逸面的半径。
    pub fn radial_boundaries(mut self, capture: f64, escape: f64) -> Self {
        self.config.capture_radius = capture;
        self.config.escape_radius = escape;
        self
    }

    /// 启用赤道面无厚度盘的环带交点检测。
    pub fn equatorial_disk(mut self, inner: f64, outer: f64) -> Self {
        self.config.disk_radii = Some((inner, outer));
        self
    }

    /// 设置事件面残差容差与二分次数上限。
    pub fn localization(mut self, tolerance: f64, maximum_bisections: u32) -> Self {
        self.config.surface_tolerance = tolerance;
        self.config.maximum_bisections = maximum_bisections;
        self
    }

    /// 拒绝无界定位、非法长度与位于坐标视界内的捕获面。
    pub fn build(self) -> TonResult<SchwarzschildEventConfig> {
        let config = self.config;
        let lengths = [config.mass, config.capture_radius, config.escape_radius, config.surface_tolerance];
        if lengths.iter().any(|value| !value.is_finite() || *value <= 0.0)
            || config.capture_radius <= 2.0 * config.mass || config.escape_radius <= config.capture_radius
            || config.maximum_bisections == 0 || config.maximum_bisections > 64
            || config.disk_radii.is_some_and(|(inner, outer)|
                !inner.is_finite() || !outer.is_finite() || inner <= config.capture_radius || outer <= inner)
        {
            return Err(TonError::Domain("invalid Schwarzschild event configuration".into()));
        }
        Ok(config)
    }
}

/// 物理事件与积分控制状态是两种不同的状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchwarzschildStepEventKind {
    /// 穿过视界外的数值捕获面。
    Capture,
    /// 向外穿过远场边界。
    Escape,
    /// 穿过赤道薄盘的有效环带。
    DiskCrossing,
}

/// 接受步内的有界定位结果。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildStepEvent {
    /// 领域事件种类，不进入 UCF 核心类型系统。
    pub kind: SchwarzschildStepEventKind,
    /// 相对于接受步的仿射参数比例，范围为零到一。
    pub fraction: f64,
    /// 重新积分到事件位置的状态，不投影到事件面。
    pub state: SphericalState,
    /// 实际事件面残差，必须满足配置容差。
    pub surface_residual: f64,
}

fn finite(state: SphericalState) -> bool {
    [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi]
        .iter().all(|value| value.is_finite())
}

fn surface(state: SphericalState, kind: SchwarzschildStepEventKind, config: SchwarzschildEventConfig) -> f64 {
    match kind {
        SchwarzschildStepEventKind::Capture => state.r - config.capture_radius,
        SchwarzschildStepEventKind::Escape => state.r - config.escape_radius,
        SchwarzschildStepEventKind::DiskCrossing => state.r * state.theta.cos(),
    }
}

fn locate(
    start: SphericalState, accepted: AdaptiveRk4Step, kind: SchwarzschildStepEventKind,
    config: SchwarzschildEventConfig,
) -> TonResult<SchwarzschildStepEvent> {
    let mut lower = 0.0;
    let mut upper = 1.0;
    let mut lower_value = surface(start, kind, config);
    let endpoint_value = surface(accepted.state, kind, config);
    if endpoint_value.abs() <= config.surface_tolerance {
        return Ok(SchwarzschildStepEvent { kind, fraction: 1.0, state: accepted.state, surface_residual: endpoint_value.abs() });
    }
    for _ in 0..config.maximum_bisections {
        let fraction = (lower + upper) * 0.5;
        let half_step = accepted.step_used * fraction * 0.5;
        let half = schwarzschild_geodesic_rk4_step(config.mass, start, half_step);
        let state = schwarzschild_geodesic_rk4_step(config.mass, half, half_step);
        if !finite(state) {
            return Err(TonError::Domain("non-finite Schwarzschild event candidate".into()));
        }
        let value = surface(state, kind, config);
        if value.abs() <= config.surface_tolerance {
            return Ok(SchwarzschildStepEvent { kind, fraction, state, surface_residual: value.abs() });
        }
        if value.is_sign_positive() == lower_value.is_sign_positive() {
            lower = fraction;
            lower_value = value;
        } else {
            upper = fraction;
        }
    }
    Err(TonError::Domain("Schwarzschild event localization budget exhausted".into()))
}

/// 定位接受步端点之间已括住的事件，按仿射参数顺序返回。
///
/// 两个半步重新积分定义步内轨迹。端点同侧的多次穿越与切触不在本接口
/// 的保证范围内，调用者必须以步长收敛或进一步细分验证事件完整性。
/// 起点不重复计盘命中。捕获或逃逸之后的盘交点不会返回。
pub fn schwarzschild_accepted_step_events(
    start: SphericalState, accepted: AdaptiveRk4Step, config: SchwarzschildEventConfig,
) -> TonResult<Vec<SchwarzschildStepEvent>> {
    if !finite(start) || !finite(accepted.state) || start.r <= 2.0 * config.mass
        || !accepted.step_used.is_finite() || accepted.step_used <= 0.0
        || !accepted.error_ratio.is_finite() || !(0.0..=1.0).contains(&accepted.error_ratio)
    {
        return Err(TonError::Domain("invalid accepted Schwarzschild event step".into()));
    }
    let mut events = Vec::new();
    if start.r > config.capture_radius && accepted.state.r <= config.capture_radius {
        events.push(locate(start, accepted, SchwarzschildStepEventKind::Capture, config)?);
    }
    if start.r < config.escape_radius && accepted.state.r >= config.escape_radius && accepted.state.ur > 0.0 {
        events.push(locate(start, accepted, SchwarzschildStepEventKind::Escape, config)?);
    }
    if let Some((inner, outer)) = config.disk_radii {
        let before = surface(start, SchwarzschildStepEventKind::DiskCrossing, config);
        let after = surface(accepted.state, SchwarzschildStepEventKind::DiskCrossing, config);
        if before.abs() > config.surface_tolerance &&
            (before.is_sign_positive() != after.is_sign_positive() || after == 0.0)
        {
            let event = locate(start, accepted, SchwarzschildStepEventKind::DiskCrossing, config)?;
            if (inner..=outer).contains(&event.state.r) { events.push(event); }
        }
    }
    events.sort_by(|left, right| left.fraction.total_cmp(&right.fraction));
    if let Some(terminal) = events.iter().position(|event| event.kind != SchwarzschildStepEventKind::DiskCrossing) {
        events.truncate(terminal + 1);
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schwarzschild_adaptive_rk4_step, AdaptiveRk4Config};

    fn radial(velocity: f64) -> SphericalState {
        SphericalState { t: 0.0, r: 10.0, theta: 1.2, phi: 0.3, ut: 1.25,
            ur: velocity, utheta: 0.0, uphi: 0.0 }
    }

    #[test]
    fn rejects_invalid_event_configuration() {
        assert!(SchwarzschildEventConfig::builder(0.0).build().is_err());
        assert!(SchwarzschildEventConfig::builder(1.0).radial_boundaries(2.0, 80.0).build().is_err());
        assert!(SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 5.0).build().is_err());
        assert!(SchwarzschildEventConfig::builder(1.0).localization(1.0e-8, 0).build().is_err());
    }

    #[test]
    fn radial_events_match_analytic_affine_position() {
        for (velocity, capture, escape, kind) in [
            (-1.0, 9.98, 80.0, SchwarzschildStepEventKind::Capture),
            (1.0, 2.0002, 10.02, SchwarzschildStepEventKind::Escape),
        ] {
            let start = radial(velocity);
            let accepted = schwarzschild_adaptive_rk4_step(1.0, start, 0.04, AdaptiveRk4Config::builder().build().unwrap()).unwrap();
            let config = SchwarzschildEventConfig::builder(1.0).radial_boundaries(capture, escape).build().unwrap();
            let events = schwarzschild_accepted_step_events(start, accepted, config).unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].kind, kind);
            assert!((events[0].fraction * accepted.step_used - 0.02).abs() < 1.0e-8);
            assert!(events[0].surface_residual <= 1.0e-8);
        }
    }

    #[test]
    fn equatorial_crossing_is_localized_and_filtered_by_annulus() {
        let start = SphericalState { theta: std::f64::consts::FRAC_PI_2 - 0.001, utheta: 0.1, ..radial(0.0) };
        let accepted = schwarzschild_adaptive_rk4_step(1.0, start, 0.04, AdaptiveRk4Config::builder().build().unwrap()).unwrap();
        let config = SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).build().unwrap();
        let events = schwarzschild_accepted_step_events(start, accepted, config).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, SchwarzschildStepEventKind::DiskCrossing);
        assert!(events[0].surface_residual <= 1.0e-8);
        let narrow = SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 9.0).build().unwrap();
        assert!(schwarzschild_accepted_step_events(start, accepted, narrow).unwrap().is_empty());
        let bounded = SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).localization(1.0e-16, 1).build().unwrap();
        assert!(schwarzschild_accepted_step_events(start, accepted, bounded).unwrap_err().to_string().contains("budget exhausted"));
    }

    #[test]
    fn events_are_ordered_and_truncated_at_escape() {
        let start = SphericalState { theta: std::f64::consts::FRAC_PI_2 - 0.001, utheta: 0.1, ..radial(1.0) };
        let accepted = schwarzschild_adaptive_rk4_step(1.0, start, 0.04, AdaptiveRk4Config::builder().build().unwrap()).unwrap();
        let config = SchwarzschildEventConfig::builder(1.0).radial_boundaries(2.0002, 10.02).equatorial_disk(6.0, 20.0).build().unwrap();
        let events = schwarzschild_accepted_step_events(start, accepted, config).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, SchwarzschildStepEventKind::DiskCrossing);
        assert_eq!(events[1].kind, SchwarzschildStepEventKind::Escape);
        assert!(events[0].fraction < events[1].fraction);
        let early = SchwarzschildEventConfig::builder(1.0).radial_boundaries(2.0002, 10.0001).equatorial_disk(6.0, 20.0).build().unwrap();
        let events = schwarzschild_accepted_step_events(start, accepted, early).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, SchwarzschildStepEventKind::Escape);
    }

    #[test]
    fn radial_infall_reaches_default_capture_surface_before_coordinate_horizon() {
        let mut state = SphericalState { r: 2.01, ut: 1.0 / (1.0 - 2.0 / 2.01), ..radial(-1.0) };
        let integration = AdaptiveRk4Config::builder().step_range(1.0e-10, 0.008).build().unwrap();
        let events = SchwarzschildEventConfig::builder(1.0).localization(1.0e-10, 48).build().unwrap();
        let mut step = 0.008;
        for _ in 0..10_000 {
            let accepted = schwarzschild_adaptive_rk4_step(1.0, state, step, integration).unwrap();
            let hits = schwarzschild_accepted_step_events(state, accepted, events).unwrap();
            if let Some(event) = hits.first() {
                assert_eq!(event.kind, SchwarzschildStepEventKind::Capture);
                assert!((event.state.r - 2.0002).abs() <= 1.0e-10);
                assert!(event.state.r > 2.0);
                let f = 1.0 - 2.0 / event.state.r;
                let temporal = f * event.state.ut.powi(2);
                let spatial = event.state.ur.powi(2) / f;
                assert!(((spatial - temporal) / (spatial + temporal)).abs() < 1.0e-5);
                return;
            }
            state = accepted.state;
            step = accepted.next_step;
        }
        panic!("radial capture event budget exhausted");
    }
}

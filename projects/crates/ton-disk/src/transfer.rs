//! 有界离散盘面辐射传输参考。

use ton_types::{TonError, TonResult};

/// 一个沿光线排序的盘面发射事件。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiativeEvent {
    /// 已经包含温度、频移和灰体系数的源强度，必须为非负 HDR 值。
    pub source: f64,
    /// 当前事件吸收并发射的比例，0 为透明，1 为不透明。
    pub opacity: f64,
}

/// 离散传输结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiativeTransferResult {
    /// 事件处理后的出射强度。
    pub intensity: f64,
    /// 所有事件后的剩余透射率。
    pub transmittance: f64,
    /// 实际处理的事件数量。
    pub event_count: u32,
}

/// 从固定盘面事件构造标量辐射传输输入的 TON 参考模型。
///
/// 该模型只负责把已定位的事件转换为 source/opacity，不执行测地线积分，
/// 也不声明自己是完整流体盘模型。事件布局为
/// `[affine,t,x,y,z,E,vx,vy,vz]`。
#[derive(Debug, Clone, Copy)]
pub struct DiskRadiativeModel {
    mass: f64,
    inner_radius: f64,
    opacity_scale: f64,
}

impl DiskRadiativeModel {
    /// 创建有界标量盘面辐射模型。
    pub fn new(mass: f64, inner_radius: f64, opacity_scale: f64) -> TonResult<Self> {
        if !mass.is_finite() || mass <= 0.0 || !inner_radius.is_finite() || inner_radius < 6.0 * mass
            || !opacity_scale.is_finite() || opacity_scale < 0.0 {
            return Err(TonError::Domain("invalid disk radiative model parameters".into()));
        }
        Ok(Self { mass, inner_radius, opacity_scale })
    }

    /// 将一个固定布局事件转换为参考 source/opacity。
    pub fn event(&self, packed: [f64; 9]) -> TonResult<RadiativeEvent> {
        if packed.iter().any(|value| !value.is_finite()) || packed[5] <= 0.0 {
            return Err(TonError::Domain("invalid packed disk radiative event".into()));
        }
        let radius = packed[2].hypot(packed[3]).hypot(packed[4]);
        if !radius.is_finite() || radius < self.inner_radius || radius > 10_000.0 * self.mass {
            return Err(TonError::Domain("packed disk event lies outside radiative disk domain".into()));
        }
        let radius_ratio = self.inner_radius / radius;
        let source = packed[5].powi(4) * radius_ratio.powf(0.75);
        let opacity = 1.0 - (-self.opacity_scale * radius_ratio).exp();
        if !source.is_finite() || source < 0.0 || !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(TonError::Domain("non-finite packed disk radiative output".into()));
        }
        Ok(RadiativeEvent { source, opacity })
    }
}

/// 对有限事件序列执行前向离散传输。
///
/// 事件必须按光线传播方向排序。每一步采用 `I'=alpha*S+(1-alpha)*I`，
/// 该式是离散源函数传输，不把首个盘面命中误称为完整流体辐射模型。
pub fn accumulate_radiative_events(events: &[RadiativeEvent], background: f64) -> TonResult<RadiativeTransferResult> {
    if !background.is_finite() || background < 0.0 || events.len() > 64 {
        return Err(TonError::Domain("invalid radiative transfer background or event count".into()));
    }
    let mut intensity = background;
    let mut transmittance = 1.0;
    for event in events {
        if !event.source.is_finite() || event.source < 0.0 || !event.opacity.is_finite() || !(0.0..=1.0).contains(&event.opacity) {
            return Err(TonError::Domain("invalid radiative transfer event".into()));
        }
        intensity = event.opacity * event.source + (1.0 - event.opacity) * intensity;
        transmittance *= 1.0 - event.opacity;
    }
    Ok(RadiativeTransferResult { intensity, transmittance, event_count: events.len() as u32 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_is_transparent_and_opaque_at_the_expected_limits() {
        let events = [RadiativeEvent { source: 2.0, opacity: 0.25 }, RadiativeEvent { source: 8.0, opacity: 0.5 }];
        let result = accumulate_radiative_events(&events, 1.0).unwrap();
        assert!((result.intensity - 4.625).abs() < 1.0e-12);
        assert!((result.transmittance - 0.375).abs() < 1.0e-12);
        assert_eq!(accumulate_radiative_events(&[], 1.0).unwrap(), RadiativeTransferResult { intensity: 1.0, transmittance: 1.0, event_count: 0 });
        assert_eq!(accumulate_radiative_events(&[RadiativeEvent { source: 7.0, opacity: 1.0 }], 1.0).unwrap().intensity, 7.0);
    }

    #[test]
    fn transfer_is_order_sensitive_and_rejects_invalid_events() {
        let first = [RadiativeEvent { source: 2.0, opacity: 0.5 }, RadiativeEvent { source: 8.0, opacity: 0.5 }];
        let reverse = [first[1], first[0]];
        assert_ne!(accumulate_radiative_events(&first, 0.0).unwrap().intensity, accumulate_radiative_events(&reverse, 0.0).unwrap().intensity);
        assert!(accumulate_radiative_events(&[RadiativeEvent { source: -1.0, opacity: 0.5 }], 0.0).is_err());
        assert!(accumulate_radiative_events(&[RadiativeEvent { source: 1.0, opacity: 1.1 }], 0.0).is_err());
        assert!(accumulate_radiative_events(&[], -1.0).is_err());
    }

    #[test]
    fn disk_radiative_model_maps_packed_event_to_bounded_transport_input() {
        let model = DiskRadiativeModel::new(1.0, 6.0, 0.8).unwrap();
        let event = model.event([0.5, 1.0, 8.0, 0.0, 0.0, 1.2, 0.0, 0.0, 1.0]).unwrap();
        assert!(event.source > 0.0);
        assert!((0.0..=1.0).contains(&event.opacity));
        assert!(DiskRadiativeModel::new(1.0, 5.0, 0.8).is_err());
        assert!(model.event([0.5, 1.0, 5.0, 0.0, 0.0, 1.2, 0.0, 0.0, 1.0]).is_err());
    }
}

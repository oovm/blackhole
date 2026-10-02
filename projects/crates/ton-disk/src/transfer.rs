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
}

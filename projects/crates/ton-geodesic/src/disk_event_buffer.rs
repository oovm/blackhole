//! 固定容量盘面事件包，拒绝静默丢弃多次交点。

use ton_types::{TonError, TonResult};
use crate::CartesianTraceDiskEvent;

/// 一个盘面交点的 GPU 传输输入布局。
/// `[affine,t,x,y,z,E,vx,vy,vz]`。
pub type PackedDiskEvent = [f64; 9];

/// 有界盘面事件包。`count` 只统计实际写入的事件。
#[derive(Debug, Clone, PartialEq)]
pub struct DiskEventBuffer<const MAX_EVENTS: usize> {
    /// 固定容量事件数组，count 之后的内容必须忽略。
    pub events: [PackedDiskEvent; MAX_EVENTS],
    /// 已写入的有效事件数。
    pub count: usize,
}

impl<const MAX_EVENTS: usize> DiskEventBuffer<MAX_EVENTS> {
    /// 从独立笛卡尔追踪结果打包全部事件，超过容量时显式失败。
    pub fn from_events(events: &[CartesianTraceDiskEvent]) -> TonResult<Self> {
        if MAX_EVENTS == 0 || events.len() > MAX_EVENTS {
            return Err(TonError::Domain(format!("disk event buffer capacity exceeded: events={}, capacity={MAX_EVENTS}", events.len())));
        }
        let mut packed = [[0.0; 9]; MAX_EVENTS];
        for (slot, event) in events.iter().enumerate() {
            let state = event.event.state;
            packed[slot] = [event.affine_parameter, state.t, state.position.x, state.position.y, state.position.z,
                state.energy, state.velocity.x, state.velocity.y, state.velocity.z];
            if packed[slot].iter().any(|value| !value.is_finite()) {
                return Err(TonError::Domain(format!("non-finite disk event at index {slot}")));
            }
        }
        Ok(Self { events: packed, count: events.len() })
    }

    /// 展平为 GPU buffer 顺序，包含固定容量尾部零填充。
    pub fn flatten(&self) -> Vec<f64> { self.events.iter().flatten().copied().collect() }

    /// 展平为 UCF 事件资源顺序：`[count, overflow, event[0..N]]`。
    ///
    /// 该构造只接受未超容量的参考结果，因此 overflow 固定为零。超容量
    /// 结果必须由 `from_events` 拒绝，不能以截断数据进入后续辐射计算。
    pub fn flatten_with_header(&self) -> Vec<f64> {
        let mut output = Vec::with_capacity(2 + MAX_EVENTS * 9);
        output.push(self.count as f64);
        output.push(0.0);
        output.extend(self.flatten());
        output
    }

    /// UCF 事件资源的固定 F64 元素数量。
    pub const fn element_len() -> usize { 2 + MAX_EVENTS * 9 }

    /// UCF 事件资源的固定字节数。
    pub const fn byte_len() -> usize { Self::element_len() * std::mem::size_of::<f64>() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_has_fixed_layout() {
        let buffer = DiskEventBuffer::<4>::from_events(&[]).unwrap();
        assert_eq!(buffer.count, 0);
        assert_eq!(buffer.flatten().len(), 36);
        assert_eq!(buffer.flatten_with_header().len(), DiskEventBuffer::<4>::element_len());
        assert_eq!(DiskEventBuffer::<4>::byte_len(), 38 * 8);
        assert!(buffer.flatten().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn capacity_overflow_is_explicit() {
        let events = (0..3).map(|index| {
            let state = crate::SchwarzschildCartesianNullState::from_ray(1.0,
                ton_types::Vec3::new(0.0, 0.0, 10.0 + index as f64), ton_types::Vec3::new(0.0, 1.0, -1.0)).unwrap();
            CartesianTraceDiskEvent {
                affine_parameter: index as f64,
                step_index: index,
                event: crate::CartesianStepEvent { kind: crate::SchwarzschildStepEventKind::DiskCrossing, fraction: 0.5, state, surface_residual: 0.0 },
            }
        }).collect::<Vec<_>>();
        assert!(DiskEventBuffer::<2>::from_events(&events).is_err());
        let buffer = DiskEventBuffer::<4>::from_events(&events).unwrap();
        assert_eq!(buffer.count, 3);
        assert_eq!(buffer.events[2][0], 2.0);
        assert_eq!(buffer.events[3], [0.0; 9]);
        let flat = buffer.flatten_with_header();
        assert_eq!(flat[0], 3.0);
        assert_eq!(flat[1], 0.0);
        assert_eq!(flat[2], 0.0);
        assert_eq!(flat[2 + 2 * 9], 2.0);
    }
}

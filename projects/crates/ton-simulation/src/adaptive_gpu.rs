//! TON 自适应追踪的原生 GPU 编排，不调用物理参考实现。

use std::collections::BTreeMap;
use ton_geodesic::SphericalState;
use ton_types::{TonError, TonResult};
use ton_ucf_frontend::SchwarzschildAdaptiveTraceChunkBuilder;
use ucf_backend_vulkan::VulkanBackend;
use ucf_ir::{Graph, ResourceGraph, TaskGraph};
use ucf_scheduler::Backend;
use ucf_types::{Access, Domain, ParamValue, ResourceId, ResourceKind, ResourceNode};

/// GPU 程序终止与宿主总预算耗尽分别表示。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTraceTermination {
    /// 到达捕获面。
    Captured,
    /// 到达逃逸面。
    Escaped,
    /// 尚未终止，但接受步总预算已耗尽。
    StepBudgetExhausted,
    /// 自适应积分失败。
    IntegrationFailure,
    /// 事件交点定位失败。
    EventLocalizationFailure,
    /// 程序拒绝输入。
    InvalidInput,
}

/// 真实原生执行后的数据和实际执行程序哈希。
#[derive(Debug)]
pub struct GpuTraceBatch {
    /// 原生程序的二十九标量输出，不混入 CPU 物理计算。
    pub records: Vec<[f32; 29]>,
    /// 每实例终止原因。
    pub terminations: Vec<GpuTraceTermination>,
    /// 按调度顺序记录已经成功执行的程序哈希。
    pub executed_program_hashes: Vec<u64>,
}

/// 显式设置物理边界、F32 误差预算与有界执行预算。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildGpuTraceBuilder {
    mass: f32,
    escape: f32,
    inner: f32,
    outer: f32,
    initial_step: f32,
    maximum_step: f32,
    absolute: f32,
    relative: f32,
    maximum_steps: u32,
    chunk_steps: u32,
}

impl SchwarzschildGpuTraceBuilder {
    /// 以质量为长度尺度，默认最多 16384 个接受步。
    pub fn new(mass: f32) -> Self {
        Self { mass, escape: 80.0 * mass, inner: 6.0 * mass, outer: 20.0 * mass,
            initial_step: 0.08 * mass, maximum_step: 0.08 * mass,
            absolute: 1.0e-6, relative: 1.0e-5, maximum_steps: 16384, chunk_steps: 64 }
    }
    /// 指定逃逸球面。
    pub fn escape_radius(mut self, radius: f32) -> Self { self.escape = radius; self }
    /// 指定赤道薄盘环带。
    pub fn disk_radii(mut self, inner: f32, outer: f32) -> Self { self.inner = inner; self.outer = outer; self }
    /// 指定初始步长与允许的最大步长。
    pub fn step_range(mut self, initial: f32, maximum: f32) -> Self { self.initial_step = initial; self.maximum_step = maximum; self }
    /// 指定积分器的绝对和相对误差容差。
    pub fn tolerances(mut self, absolute: f32, relative: f32) -> Self { self.absolute = absolute; self.relative = relative; self }
    /// 指定每实例接受步总预算，最后一次 dispatch 不超出剩余预算。
    pub fn maximum_steps(mut self, steps: u32) -> Self { self.maximum_steps = steps; self }
    /// 指定单次 dispatch 的静态接受步上限。
    pub fn chunk_steps(mut self, steps: u32) -> Self { self.chunk_steps = steps; self }

    fn validate(self) -> TonResult<()> {
        let values = [self.mass, self.escape, self.inner, self.outer, self.initial_step,
            self.maximum_step, self.absolute, self.relative];
        if values.iter().any(|value| !value.is_finite()) || self.mass <= 0.0
            || self.inner <= 2.0002 * self.mass || self.outer <= self.inner
            || self.escape <= self.outer || self.initial_step < 1.0e-8 * self.mass
            || self.maximum_step < self.initial_step || self.absolute <= 0.0 || self.relative < 0.0
            || self.maximum_steps == 0 || self.maximum_steps > 16_777_216
            || self.chunk_steps == 0 || self.chunk_steps > 64
        {
            return Err(TonError::Domain("invalid adaptive GPU trace configuration".into()));
        }
        Ok(())
    }

    /// 创建原生后端。配置错误在创建 GPU 设备之前返回。
    pub fn execute(self, states: &[SphericalState]) -> TonResult<GpuTraceBatch> {
        self.validate()?;
        if states.is_empty() {
            return Ok(GpuTraceBatch { records: Vec::new(), terminations: Vec::new(), executed_program_hashes: Vec::new() });
        }
        let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create adaptive Vulkan backend: {error}")))?;
        self.execute_with_backend(&mut backend, states)
    }

    pub(crate) fn execute_with_backend(self, backend: &mut VulkanBackend, states: &[SphericalState]) -> TonResult<GpuTraceBatch> {
        self.validate()?;
        let count = u32::try_from(states.len()).map_err(|_| TonError::Domain("adaptive batch exceeds u32 capacity".into()))?;
        let mut records = states.iter().map(|state| {
            let mut record = [0.0; 29];
            record[..8].copy_from_slice(&[state.t, state.r, state.theta, state.phi, state.ut,
                state.ur, state.utheta, state.uphi].map(|value| value as f32));
            record[9] = self.initial_step;
            record
        }).collect::<Vec<_>>();
        let mut hashes = Vec::new();
        let mut scheduled = 0;
        let mut prepared_steps = 0;
        let mut graph = None;
        while scheduled < self.maximum_steps && records.iter().any(|record| record[8] == 0.0) {
            let steps = self.chunk_steps.min(self.maximum_steps - scheduled);
            if steps != prepared_steps {
                let program = SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(steps).conserved_null_energy()
                    .build().map_err(|error| TonError::Domain(format!("build adaptive program: {error}")))?;
                let uniforms = [self.mass, self.absolute, self.relative, 1.0e-8 * self.mass,
                    self.maximum_step, 2.0002 * self.mass, self.escape, self.inner, self.outer, 1.0e-5 * self.mass];
                let mut params = uniforms.into_iter().enumerate().map(|(slot, value)|
                    (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
                params.insert("buffer_2".into(), ParamValue::I64(2));
                let task = crate::ucf_physics::dispatch_task(56, &program, count, ResourceId(3), 0, 1, Some(ResourceId(1)), params)?;
                let next = Graph {
                    resources: ResourceGraph { nodes: [(1, 29, Access::Read), (2, 29, Access::Write), (3, 1, Access::Write)]
                        .into_iter().map(|(id, stride, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer,
                            domain: Domain::Vram, access, byte_size: Some(u64::from(count) * stride * 4) }).collect() },
                    tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
                };
                next.validate().map_err(|error| TonError::Domain(format!("validate adaptive graph: {error}")))?;
                backend.prepare(&next).map_err(|error| TonError::Domain(format!("prepare adaptive graph: {error}")))?;
                graph = Some((next, ton_ucf_frontend::program_hash(&program)));
                prepared_steps = steps;
            }
            let (current, hash) = graph.as_ref().unwrap();
            let input = records.iter().flatten().copied().collect::<Vec<_>>();
            backend.write_f32(ResourceId(1), &input).map_err(|error| TonError::Domain(format!("upload adaptive states: {error}")))?;
            backend.run_prepared(current).map_err(|error| TonError::Domain(format!("execute adaptive graph: {error}")))?;
            let output = backend.read_f32(ResourceId(2)).map_err(|error| TonError::Domain(format!("read adaptive states: {error}")))?;
            if output.len() != records.len() * 29 || output.iter().any(|value| !value.is_finite()) {
                return Err(TonError::Domain("adaptive GPU output has invalid size or nonfinite values".into()));
            }
            records = output.chunks_exact(29).map(|values| values.try_into().unwrap()).collect();
            hashes.push(*hash);
            scheduled += steps;
        }
        let terminations = records.iter().map(|record| match record[8] {
            0.0 => Ok(GpuTraceTermination::StepBudgetExhausted),
            1.0 => Ok(GpuTraceTermination::Captured),
            2.0 => Ok(GpuTraceTermination::Escaped),
            3.0 => Ok(GpuTraceTermination::IntegrationFailure),
            4.0 => Ok(GpuTraceTermination::EventLocalizationFailure),
            5.0 => Ok(GpuTraceTermination::InvalidInput),
            status => Err(TonError::Domain(format!("unknown adaptive GPU status: {status}"))),
        }).collect::<TonResult<Vec<_>>>()?;
        Ok(GpuTraceBatch { records, terminations, executed_program_hashes: hashes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::Vec3;

    fn initial(angle: f64) -> SphericalState {
        ton_geodesic::schwarzschild_geodesic_initial_state(1.0, Vec3::new(24.0, 0.0, 0.0),
            Vec3::new(-angle.cos(), angle.sin(), 0.0)).unwrap()
    }

    #[test]
    fn adaptive_gpu_builder_rejects_invalid_configuration_before_device_creation() {
        for builder in [SchwarzschildGpuTraceBuilder::new(f32::NAN),
            SchwarzschildGpuTraceBuilder::new(1.0).maximum_steps(0),
            SchwarzschildGpuTraceBuilder::new(1.0).chunk_steps(65),
            SchwarzschildGpuTraceBuilder::new(1.0).step_range(0.1, 0.08),
            SchwarzschildGpuTraceBuilder::new(1.0).disk_radii(6.0, 5.0)] {
            assert!(builder.execute(&[]).is_err());
        }
        let empty = SchwarzschildGpuTraceBuilder::new(1.0).execute(&[]).unwrap();
        assert!(empty.records.is_empty() && empty.executed_program_hashes.is_empty());
    }

    #[test]
    fn adaptive_gpu_facade_obeys_non_multiple_total_budget() {
        let builder = SchwarzschildGpuTraceBuilder::new(1.0).maximum_steps(5).chunk_steps(3);
        let result = builder.execute(&[initial(0.0)]).unwrap();
        assert_eq!(result.terminations, [GpuTraceTermination::StepBudgetExhausted]);
        assert_eq!(result.records[0][10], 5.0);
        assert_eq!(result.records[0][8], 0.0);
        let expected = [3, 2].map(|steps| ton_ucf_frontend::program_hash(
            &SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(steps).conserved_null_energy().build().unwrap()));
        assert_eq!(result.executed_program_hashes, expected);
    }

    #[test]
    fn adaptive_gpu_facade_matches_independent_capture_reference() {
        let states = [0.0, 0.19, 0.23, 0.4].map(initial);
        let result = SchwarzschildGpuTraceBuilder::new(1.0).execute(&states).unwrap();
        for ((state, record), termination) in states.iter().zip(&result.records).zip(&result.terminations) {
            let reference = ton_geodesic::schwarzschild_adaptive_trace(*state,
                ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).build().unwrap()).unwrap();
            let expected = match reference.termination {
                ton_geodesic::AdaptiveTraceTermination::Captured => GpuTraceTermination::Captured,
                ton_geodesic::AdaptiveTraceTermination::Escaped => GpuTraceTermination::Escaped,
                other => panic!("unexpected reference termination {other:?}"),
            };
            assert_eq!(*termination, expected);
            assert!(record[16] < 1.0e-3 && record[17] < 1.0e-3 && record[18] < 2.0e-3);
            assert!(record[10] <= 16384.0);
        }
        assert!(!result.executed_program_hashes.is_empty());
    }

    #[test]
    fn adaptive_gpu_small_frame_matches_independent_reference() {
        let camera = ton_types::Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let (width, height) = (16, 9);
        let uv = (0..height).flat_map(|y| (0..width).map(move |x|
            (((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
             1.0 - (y as f64 + 0.5) / height as f64 * 2.0))).collect::<Vec<_>>();
        let directions = crate::ucf_physics::camera_ray_ucf_vulkan_batch(camera, &uv).unwrap();
        let rays = directions.iter().map(|direction| (camera.position, *direction)).collect::<Vec<_>>();
        let states = crate::ucf_physics::schwarzschild_initial_state_ucf_vulkan_batch(1.0,
            3.0 * 3.0f64.sqrt(), &rays).unwrap();
        let result = SchwarzschildGpuTraceBuilder::new(1.0).execute(&states).unwrap();
        let mut captured = 0;
        let mut escaped = 0;
        for (index, ((u, v), termination)) in uv.iter().zip(&result.terminations).enumerate() {
            let reference_state = ton_geodesic::schwarzschild_geodesic_initial_state(1.0,
                camera.position, camera.ray_direction(*u, *v)).unwrap();
            let reference = ton_geodesic::schwarzschild_adaptive_trace(reference_state,
                ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).build().unwrap()).unwrap();
            let expected = match reference.termination {
                ton_geodesic::AdaptiveTraceTermination::Captured => { captured += 1; GpuTraceTermination::Captured },
                ton_geodesic::AdaptiveTraceTermination::Escaped => { escaped += 1; GpuTraceTermination::Escaped },
                other => panic!("reference pixel={index} termination={other:?}"),
            };
            let record = &result.records[index];
            assert_eq!(*termination, expected, "pixel={index}, record={record:?}");
            assert!(record[16] < 1.0e-3, "pixel={index}, null={}", record[16]);
            assert!(record[17] < 1.0e-3, "pixel={index}, energy={}", record[17]);
            assert!(record[18] < 2.0e-3, "pixel={index}, angular={}", record[18]);
        }
        eprintln!("adaptive GPU frame {width}x{height}: captured={captured}, escaped={escaped}, dispatches={}", result.executed_program_hashes.len());
        assert!(captured > 0 && escaped > 0);
    }
}

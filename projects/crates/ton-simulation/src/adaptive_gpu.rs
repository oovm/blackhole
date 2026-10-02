//! TON 自适应追踪的原生 GPU 编排，不调用物理参考实现。

use std::collections::BTreeMap;
use ton_geodesic::{SphericalState, SchwarzschildCartesianNullState};
use ton_types::{Camera, TonError, TonResult, Vec3};
use ton_ucf_frontend::SchwarzschildAdaptiveTraceChunkBuilder;
use ucf_backend_vulkan::VulkanBackend;
use ucf_ir::{Graph, ResourceGraph, TaskGraph};
use ucf_scheduler::Backend;
use ucf_types::{Access, Domain, ParamValue, ResourceId, ResourceKind, ResourceNode};

/// 在原生 Vulkan UCF backend 上执行笛卡尔追踪记录的圆轨道频移 Builder。
/// 输入和输出仍是 TON 负责解释的布局，UCF 只执行通用程序。
pub fn execute_cartesian_frequency(
    records: &[[f32; 29]], mass: f32, observer_radius: f32,
) -> TonResult<Vec<[f32; 4]>> {
    let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create frequency Vulkan backend: {error}")))?;
    execute_cartesian_frequency_with_backend(&mut backend, records, mass, observer_radius)
}

pub(crate) fn execute_cartesian_frequency_with_backend(
    backend: &mut VulkanBackend, records: &[[f32; 29]], mass: f32, observer_radius: f32,
) -> TonResult<Vec<[f32; 4]>> {
    if records.is_empty() || !mass.is_finite() || !observer_radius.is_finite()
        || mass <= 0.0 || observer_radius <= 2.0 * mass
        || records.iter().any(|record| record.iter().any(|value| !value.is_finite())) {
        return Err(TonError::Domain("invalid Cartesian frequency input".into()));
    }
    let count = u32::try_from(records.len()).map_err(|_| TonError::Domain("frequency batch exceeds u32 capacity".into()))?;
    let program = ton_ucf_frontend::SchwarzschildDiskFrequencyBuilder::new().build().map_err(TonError::Domain)?;
    let params = BTreeMap::from([
        ("uniform_0".into(), ParamValue::F64(mass as f64)),
        ("uniform_1".into(), ParamValue::F64(observer_radius as f64)),
        ("buffer_2".into(), ParamValue::I64(42)),
    ]);
    let task = crate::ucf_physics::dispatch_task(73, &program, count, ResourceId(43), 2, 1, Some(ResourceId(41)), params)?;
    let graph = Graph {
        resources: ResourceGraph { nodes: [(41, 29, Access::Read), (42, 4, Access::Write), (43, 1, Access::Write)].into_iter().map(|(id, stride, access)|
            ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access,
                byte_size: Some(u64::from(count) * stride * 4) }).collect() },
        tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
    };
    graph.validate().map_err(|error| TonError::Domain(format!("validate Cartesian frequency graph: {error}")))?;
    backend.prepare(&graph).map_err(|error| TonError::Domain(format!("prepare Cartesian frequency graph: {error}")))?;
    backend.write_f32(ResourceId(41), &records.iter().flatten().copied().collect::<Vec<_>>())
        .map_err(|error| TonError::Domain(format!("upload Cartesian frequency records: {error}")))?;
    backend.run_prepared(&graph).map_err(|error| TonError::Domain(format!("execute Cartesian frequency graph: {error}")))?;
    let output = backend.read_f32(ResourceId(42)).map_err(|error| TonError::Domain(format!("read Cartesian frequency output: {error}")))?;
    if output.len() != records.len() * 4 || output.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(TonError::Domain("Cartesian frequency output is invalid".into()));
    }
    Ok(output.chunks_exact(4).map(|value| value.try_into().unwrap()).collect())
}

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

/// TON 输出状态布局，不进入 UCF 核心。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTraceStateLayout {
    /// `[t,r,theta,phi,ut,ur,utheta,uphi]`。
    SphericalFourVelocity,
    /// `[t,x,y,z,E,vx,vy,vz]`。
    CartesianNullEnergy,
}

/// 真实原生执行后的数据和实际执行程序哈希。
#[derive(Debug)]
pub struct GpuTraceBatch {
    /// 主状态与首盘面状态的共同布局，禁止按另一坐标契约解码。
    pub state_layout: GpuTraceStateLayout,
    /// 原生程序的二十九标量输出，不混入 CPU 物理计算。
    pub records: Vec<[f32; 29]>,
    /// 每实例终止原因。
    pub terminations: Vec<GpuTraceTermination>,
    /// 按调度顺序记录已经成功执行的程序哈希。
    pub executed_program_hashes: Vec<u64>,
}

impl GpuTraceBatch {
    /// 渲染前同时检查终止原因与全轨迹不变量，不把正常终止视为物理正确。
    pub fn require_renderable(&self) -> TonResult<()> {
        if self.records.len() != self.terminations.len() {
            return Err(TonError::Domain("adaptive trace record/status count mismatch".into()));
        }
        for (index, (record, termination)) in self.records.iter().zip(&self.terminations).enumerate() {
            if !matches!(termination, GpuTraceTermination::Captured | GpuTraceTermination::Escaped)
                || record.iter().any(|value| !value.is_finite())
                || record[16] < 0.0 || record[16] >= 1.0e-3
                || record[17] < 0.0 || record[17] >= 1.0e-3
                || record[18] < 0.0 || record[18] >= 2.0e-3 {
                return Err(TonError::Domain(format!(
                    "adaptive trace instance {index} cannot enter Beauty: termination={termination:?}, null={}, energy_drift={}, angular_squared_drift={}",
                    record[16], record[17], record[18])));
            }
        }
        Ok(())
    }
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
            return Ok(GpuTraceBatch { state_layout: GpuTraceStateLayout::SphericalFourVelocity,
                records: Vec::new(), terminations: Vec::new(), executed_program_hashes: Vec::new() });
        }
        let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create adaptive Vulkan backend: {error}")))?;
        self.execute_with_backend(&mut backend, states)
    }

    pub(crate) fn execute_with_backend(self, backend: &mut VulkanBackend, states: &[SphericalState]) -> TonResult<GpuTraceBatch> {
        self.validate()?;
        let records = states.iter().map(|state| {
            let mut record = [0.0; 29];
            record[..8].copy_from_slice(&[state.t, state.r, state.theta, state.phi, state.ut,
                state.ur, state.utheta, state.uphi].map(|value| value as f32));
            record[9] = self.initial_step;
            record
        }).collect::<Vec<_>>();
        self.execute_packed(backend, records, GpuTraceStateLayout::SphericalFourVelocity)
    }

    /// 原生追踪笛卡尔零状态，坐标布局和能量定义与球坐标入口显式分离。
    pub fn execute_cartesian(self, states: &[SchwarzschildCartesianNullState]) -> TonResult<GpuTraceBatch> {
        self.validate()?;
        if states.is_empty() {
            return Ok(GpuTraceBatch { state_layout: GpuTraceStateLayout::CartesianNullEnergy,
                records: Vec::new(), terminations: Vec::new(), executed_program_hashes: Vec::new() });
        }
        let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create adaptive Vulkan backend: {error}")))?;
        let records = states.iter().map(|state| {
            let mut record = [0.0; 29];
            record[..8].copy_from_slice(&[state.t, state.position.x, state.position.y, state.position.z, state.energy,
                state.velocity.x, state.velocity.y, state.velocity.z].map(|value| value as f32));
            record[9] = self.initial_step;
            record
        }).collect();
        self.execute_packed(&mut backend, records, GpuTraceStateLayout::CartesianNullEnergy)
    }

    /// GPU 相机程序、GPU 笛卡尔初态程序与融合追踪，不调用任何 Rust 物理参考函数。
    /// 当前阶段之间显式 readback 并重新上传，不声称资源始终驻留 GPU。
    pub fn execute_camera(self, camera: Camera, width: u32, height: u32) -> TonResult<GpuTraceBatch> {
        self.validate()?;
        let count = width.checked_mul(height).filter(|count| *count > 0)
            .ok_or_else(|| TonError::Domain("adaptive camera resolution is zero or exceeds u32 capacity".into()))?;
        let uv = (0..height).flat_map(|y| (0..width).map(move |x|
            (((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
                1.0 - (y as f64 + 0.5) / height as f64 * 2.0))).collect::<Vec<_>>();
        let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create adaptive camera backend: {error}")))?;
        let directions = crate::ucf_physics::camera_ray_ucf_vulkan_batch_with_backend(&mut backend, camera, &uv)?;
        if directions.len() != count as usize { return Err(TonError::Domain("adaptive camera output length mismatch".into())); }
        let rays = directions.iter().map(|direction| (camera.position, *direction)).collect::<Vec<_>>();
        let (records, initial_hash) = self.initialize_rays(&mut backend, &rays)?;
        let mut result = self.execute_packed(&mut backend, records, GpuTraceStateLayout::CartesianNullEnergy)?;
        let camera_hash = ton_ucf_frontend::program_hash(&ton_ucf_frontend::camera_ray_batch());
        result.executed_program_hashes.splice(0..0, [camera_hash, initial_hash]);
        Ok(result)
    }

    fn initialize_rays(self, backend: &mut VulkanBackend, rays: &[(Vec3, Vec3)]) -> TonResult<(Vec<[f32; 29]>, u64)> {
        let count = u32::try_from(rays.len()).map_err(|_| TonError::Domain("adaptive initial batch exceeds u32 capacity".into()))?;
        let program = ton_ucf_frontend::SchwarzschildCartesianInitialStateBuilder::new().build()
            .map_err(|error| TonError::Domain(format!("build Cartesian initial program: {error}")))?;
        let params = BTreeMap::from([("uniform_0".into(), ParamValue::F64(self.mass as f64)),
            ("uniform_1".into(), ParamValue::F64(self.initial_step as f64)), ("buffer_2".into(), ParamValue::I64(32))]);
        let task = crate::ucf_physics::dispatch_task(72, &program, count, ResourceId(33), 2, 1, Some(ResourceId(31)), params)?;
        let graph = Graph { resources: ResourceGraph { nodes: [(31, 6, Access::Read), (32, 29, Access::Write), (33, 1, Access::Write)]
            .into_iter().map(|(id, stride, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer,
                domain: Domain::Vram, access, byte_size: Some(u64::from(count) * stride * 4) }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
        graph.validate().map_err(|error| TonError::Domain(format!("validate Cartesian initial graph: {error}")))?;
        backend.prepare(&graph).map_err(|error| TonError::Domain(format!("prepare Cartesian initial graph: {error}")))?;
        let input = rays.iter().flat_map(|(origin, direction)| [origin.x, origin.y, origin.z,
            direction.x, direction.y, direction.z].map(|value| value as f32)).collect::<Vec<_>>();
        backend.write_f32(ResourceId(31), &input).map_err(|error| TonError::Domain(format!("upload Cartesian rays: {error}")))?;
        backend.run_prepared(&graph).map_err(|error| TonError::Domain(format!("execute Cartesian initial graph: {error}")))?;
        let output = backend.read_f32(ResourceId(32)).map_err(|error| TonError::Domain(format!("read Cartesian initial states: {error}")))?;
        if output.len() != rays.len() * 29 || output.iter().any(|value| !value.is_finite()) {
            return Err(TonError::Domain("Cartesian initial output has invalid length or nonfinite values".into()));
        }
        Ok((output.chunks_exact(29).map(|values| values.try_into().unwrap()).collect(), ton_ucf_frontend::program_hash(&program)))
    }

    fn execute_packed(self, backend: &mut VulkanBackend, mut records: Vec<[f32; 29]>, state_layout: GpuTraceStateLayout) -> TonResult<GpuTraceBatch> {
        let count = u32::try_from(records.len()).map_err(|_| TonError::Domain("adaptive batch exceeds u32 capacity".into()))?;
        let mut hashes = Vec::new();
        let mut scheduled = 0;
        let mut prepared_steps = 0;
        let mut graph = None;
        while scheduled < self.maximum_steps && records.iter().any(|record| record[8] == 0.0) {
            let steps = self.chunk_steps.min(self.maximum_steps - scheduled);
            if steps != prepared_steps {
                let builder = SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(steps);
                let builder = match state_layout {
                    GpuTraceStateLayout::SphericalFourVelocity => builder.conserved_null_energy(),
                    GpuTraceStateLayout::CartesianNullEnergy => builder.cartesian_null(),
                };
                let program = builder.build().map_err(|error| TonError::Domain(format!("build adaptive program: {error}")))?;
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
        Ok(GpuTraceBatch { state_layout, records, terminations, executed_program_hashes: hashes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::Vec3;

    #[test]
    fn blackbody_spectrum_vulkan_matches_independent_si_reference() {
        let inputs = [[6000.0f32, 500.0, 1.0], [6000.0, 500.0, 0.7], [6000.0, 500.0, 2.0],
            [10000.0, 700.0, 1.0], [6000.0, 0.000001, 1.0], [1.0, 1000000.0, 1.0],
            [0.0, 500.0, 1.0], [6000.0, 500.0, -1.0]];
        let count = inputs.len() as u32;
        let program = ton_ucf_frontend::BlackbodySpectrumBuilder::new().build().unwrap();
        let task = crate::ucf_physics::dispatch_task(74, &program, count, ResourceId(53), 2, 1,
            Some(ResourceId(51)), BTreeMap::from([("buffer_2".into(), ParamValue::I64(52))])).unwrap();
        let graph = Graph { resources: ResourceGraph { nodes: [(51, 3, Access::Read), (52, 2, Access::Write), (53, 1, Access::Write)]
            .into_iter().map(|(id, stride, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer,
                domain: Domain::Vram, access, byte_size: Some(u64::from(count) * stride * 4) }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
        graph.validate().unwrap();
        let mut backend = VulkanBackend::new().unwrap();
        backend.prepare(&graph).unwrap();
        backend.write_f32(ResourceId(51), &inputs.iter().flatten().copied().collect::<Vec<_>>()).unwrap();
        backend.run_prepared(&graph).unwrap();
        let output = backend.read_f32(ResourceId(52)).unwrap();
        let returns = backend.read_f32(ResourceId(53)).unwrap();
        assert_eq!(output.len(), inputs.len() * 2);
        let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
        cpu.with_buffer(1, inputs.iter().flatten().copied().map(ton_ucf_frontend::Value::F32).collect());
        cpu.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); inputs.len() * 2]);
        cpu.run_parallel(&program, count).unwrap();
        let cpu = cpu.buffer(2).unwrap();
        for (index, (input, actual)) in inputs.iter().zip(output.chunks_exact(2)).enumerate() {
            assert_eq!(actual[1], returns[index]);
            assert_eq!(actual[1] as f64, cpu[index * 2 + 1].as_f64().unwrap());
            let expected = ton_geodesic::observed_blackbody_spectral_radiance(input[0] as f64,
                input[1] as f64 * 1.0e12, input[2] as f64);
            match expected {
                Err(_) => assert_eq!(actual, &[0.0, 2.0]),
                Ok(expected) => {
                    assert_eq!(actual[1], 1.0);
                    if expected == 0.0 { assert_eq!(actual[0], 0.0); }
                    else {
                        assert!((actual[0] as f64 / expected - 1.0).abs() < 2.0e-5, "input={input:?}, GPU={actual:?}, reference={expected}");
                        assert!((cpu[index * 2].as_f64().unwrap() / expected - 1.0).abs() < 2.0e-5);
                    }
                }
            }
        }
    }

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

    #[test]
    fn adaptive_gpu_frame_gate_matches_per_instance_diagnostics() {
        let camera = ton_types::Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let (width, height) = (128, 72);
        let uv = (0..height).flat_map(|y| (0..width).map(move |x|
            (((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
             1.0 - (y as f64 + 0.5) / height as f64 * 2.0))).collect::<Vec<_>>();
        let directions = crate::ucf_physics::camera_ray_ucf_vulkan_batch(camera, &uv).unwrap();
        let rays = directions.iter().map(|direction| (camera.position, *direction)).collect::<Vec<_>>();
        let states = crate::ucf_physics::schwarzschild_initial_state_ucf_vulkan_batch(1.0,
            3.0 * 3.0f64.sqrt(), &rays).unwrap();
        let result = SchwarzschildGpuTraceBuilder::new(1.0).execute(&states).unwrap();
        let mut failures = Vec::new();
        let mut maxima = [0.0f32; 3];
        for (index, (record, status)) in result.records.iter().zip(&result.terminations).enumerate() {
            for component in 0..3 { maxima[component] = maxima[component].max(record[16 + component]); }
            if !matches!(status, GpuTraceTermination::Captured | GpuTraceTermination::Escaped)
                || record[16] >= 1.0e-3 || record[17] >= 1.0e-3 || record[18] >= 2.0e-3 {
                failures.push((index, *status, record[1], record[10], record[16], record[17], record[18]));
            }
        }
        eprintln!("native frame {width}x{height}: invariant maxima={maxima:?}, failures={}, dispatches={}", failures.len(), result.executed_program_hashes.len());
        assert_eq!(result.require_renderable().is_ok(), failures.is_empty());
    }

    #[test]
    fn adaptive_gpu_near_axis_tolerance_sweep_enforces_accuracy_gate() {
        let camera = ton_types::Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let states = [1712usize, 7503].map(|index| {
            let u = (((index % 128) as f64 + 0.5) / 128.0 * 2.0 - 1.0) * 128.0 / 72.0;
            let v = 1.0 - ((index / 128) as f64 + 0.5) / 72.0 * 2.0;
            ton_geodesic::schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap()
        });
        for (index, state) in states.iter().enumerate() {
            let reference = ton_geodesic::schwarzschild_adaptive_trace(*state,
                ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).build().unwrap()).unwrap();
            eprintln!("axis reference ray={index}, termination={:?}, null={}, angular={}",
                reference.termination, reference.maximum_null_residual, reference.maximum_angular_squared_drift);
            assert_eq!(reference.termination, ton_geodesic::AdaptiveTraceTermination::Escaped);
            assert!(reference.maximum_null_residual < 1.0e-3 && reference.maximum_angular_squared_drift < 2.0e-3);
        }
        for (absolute, relative) in [(1.0e-6, 1.0e-5), (1.0e-7, 1.0e-6)] {
            let result = SchwarzschildGpuTraceBuilder::new(1.0).tolerances(absolute, relative).execute(&states).unwrap();
            for (index, record) in result.records.iter().enumerate() {
                eprintln!("axis ray={index} tolerances={absolute}/{relative}, status={}, steps={}, null={}, angular={}", record[8], record[10], record[16], record[18]);
                assert_eq!(record[8], 2.0);
            }
            let reliable = result.records.iter().all(|record| record[16] < 1.0e-3
                && record[17] < 1.0e-3 && record[18] < 2.0e-3);
            assert_eq!(result.require_renderable().is_ok(), reliable);
        }
    }

    #[test]
    fn render_gate_rejects_terminal_but_inaccurate_records() {
        let mut batch = GpuTraceBatch { state_layout: GpuTraceStateLayout::SphericalFourVelocity, records: vec![[0.0; 29]],
            terminations: vec![GpuTraceTermination::Escaped], executed_program_hashes: Vec::new() };
        assert!(batch.require_renderable().is_ok());
        for (component, value) in [(16, 0.035), (17, 0.01), (18, 0.075), (16, f32::NAN)] {
            batch.records[0] = [0.0; 29];
            batch.records[0][component] = value;
            assert!(batch.require_renderable().is_err());
        }
        batch.records[0] = [0.0; 29];
        batch.terminations[0] = GpuTraceTermination::StepBudgetExhausted;
        assert!(batch.require_renderable().is_err());
    }

    #[test]
    fn cartesian_frequency_builder_executes_on_vulkan_and_matches_cpu_and_reference() {
        let mut record = [0.0f32; 29];
        record[13] = 1.0;
        record[21] = 123.0;
        record[22] = 8.0;
        record[25] = 1.0;
        record[27] = 0.25;
        let gpu = execute_cartesian_frequency(&[record], 1.0, 24.0).unwrap();
        let program = ton_ucf_frontend::SchwarzschildDiskFrequencyBuilder::new().build().unwrap();
        let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
        for (slot, value) in [1.0, 24.0].into_iter().enumerate() {
            cpu.with_uniform(slot as u32, ton_ucf_frontend::Value::F32(value));
        }
        cpu.with_buffer(1, record.into_iter().map(ton_ucf_frontend::Value::F32).collect());
        cpu.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); 4]);
        cpu.run_parallel(&program, 1).unwrap();
        let expected = cpu.buffer(2).unwrap();
        for (actual, expected) in gpu[0].iter().zip(expected) {
            assert!((*actual - expected.as_f64().unwrap() as f32).abs() < 2.0e-4,
                "GPU={actual}, CPU={}", expected.as_f64().unwrap());
        }
        let reference = ton_geodesic::schwarzschild_disk_frequency_shift(1.0, 24.0,
            SchwarzschildCartesianNullState { t: 123.0, position: Vec3::new(8.0, 0.0, 0.0),
                energy: 1.0, velocity: Vec3::new(0.0, 0.25, 0.0) }).unwrap();
        assert_eq!(gpu[0][0], 8.0);
        assert_eq!(gpu[0][3], 1.0);
        assert!((gpu[0][1] as f64 - reference).abs() < 1.0e-6);
        assert!((gpu[0][2] as f64 - reference.powi(4)).abs() < 2.0e-6);
    }

    #[test]
    fn cartesian_camera_trace_frequency_chain_matches_independent_disk_reference() {
        let camera = Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let trace = SchwarzschildGpuTraceBuilder::new(1.0).execute_camera(camera, 16, 9).unwrap();
        trace.require_renderable().unwrap();
        let frequency = execute_cartesian_frequency(&trace.records, 1.0, camera.position.length() as f32).unwrap();
        let mut disk_pixels = 0;
        let mut maximum_error = 0.0f64;
        for (index, (record, output)) in trace.records.iter().zip(&frequency).enumerate() {
            if record[13] == 0.0 {
                assert_eq!(*output, [0.0; 4], "pixel={index}");
                continue;
            }
            disk_pixels += 1;
            assert_eq!(output[3], 1.0, "pixel={index}, frequency={output:?}");
            let state = SchwarzschildCartesianNullState { t: record[21] as f64,
                position: Vec3::new(record[22] as f64, record[23] as f64, record[24] as f64),
                energy: record[25] as f64,
                velocity: Vec3::new(record[26] as f64, record[27] as f64, record[28] as f64) };
            let expected = ton_geodesic::schwarzschild_disk_frequency_shift(1.0,
                (camera.position.length() as f32) as f64, state).unwrap();
            let error = (output[1] as f64 - expected).abs();
            maximum_error = maximum_error.max(error);
            assert!(error < 2.0e-6, "pixel={index}, actual={}, reference={expected}", output[1]);
            assert!((output[2] as f64 - expected.powi(4)).abs() < 1.0e-5, "pixel={index}");
            assert!((output[0] as f64 - state.position.length()).abs() < 3.0e-6, "pixel={index}");
        }
        assert!(disk_pixels > 0);
        eprintln!("GPU camera/trace/frequency 16x9: disk_pixels={disk_pixels}, max_same_state_shift_error={maximum_error}");
    }

    #[test]
    fn cartesian_adaptive_program_matches_reference_single_step() {
        let initial = ton_geodesic::SchwarzschildCartesianNullState::from_ray(1.0,
            Vec3::new(12.0, 3.0, 5.0), Vec3::new(-1.0, 0.2, -0.1)).unwrap();
        let input = [initial.t, initial.position.x, initial.position.y, initial.position.z, initial.energy,
            initial.velocity.x, initial.velocity.y, initial.velocity.z, 0.08].map(|value| value as f32);
        let program = ton_ucf_frontend::SchwarzschildCartesianAdaptiveChunkBuilder::new().build().unwrap();
        let mut interpreter = ton_ucf_frontend::CpuInterpreter::new();
        for (slot, value) in [1.0, 1.0e-6, 1.0e-5, 1.0e-8, 0.08].into_iter().enumerate() {
            interpreter.with_uniform(slot as u32, ton_ucf_frontend::Value::F32(value));
        }
        interpreter.with_buffer(1, input.into_iter().map(ton_ucf_frontend::Value::F32).collect());
        interpreter.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); 16]);
        interpreter.run_parallel(&program, 1).unwrap();
        let output = interpreter.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
        assert_eq!(output[8], 1.0);
        let rounded = ton_geodesic::SchwarzschildCartesianNullState { t: input[0] as f64,
            position: Vec3::new(input[1] as f64, input[2] as f64, input[3] as f64), energy: input[4] as f64,
            velocity: Vec3::new(input[5] as f64, input[6] as f64, input[7] as f64) };
        let half = ton_geodesic::schwarzschild_cartesian_null_rk4_step(1.0, rounded, output[9] * 0.5).unwrap();
        let expected = ton_geodesic::schwarzschild_cartesian_null_rk4_step(1.0, half, output[9] * 0.5).unwrap();
        for (actual, expected) in output[..8].iter().zip([expected.t, expected.position.x, expected.position.y,
            expected.position.z, expected.energy, expected.velocity.x, expected.velocity.y, expected.velocity.z]) {
            assert!((*actual - expected).abs() < 16.0 * f32::EPSILON as f64 * expected.abs().max(1.0));
        }
        assert!(output[13] < 1.0e-5 && output[14] < 1.0e-5);
    }

    #[test]
    fn cartesian_adaptive_cpu_and_vulkan_problem_rays_preserve_invariants() {
        let camera = ton_types::Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let initial = [1712usize, 7503].map(|index| {
            let u = (((index % 128) as f64 + 0.5) / 128.0 * 2.0 - 1.0) * 128.0 / 72.0;
            let v = 1.0 - ((index / 128) as f64 + 0.5) / 72.0 * 2.0;
            ton_geodesic::SchwarzschildCartesianNullState::from_ray(1.0, camera.position, camera.ray_direction(u, v)).unwrap()
        });
        let seed = initial.into_iter().flat_map(|state| [state.t, state.position.x, state.position.y, state.position.z,
            state.energy, state.velocity.x, state.velocity.y, state.velocity.z, 0.08].map(|value| value as f32)).collect::<Vec<_>>();
        let program = ton_ucf_frontend::SchwarzschildCartesianAdaptiveChunkBuilder::new().maximum_steps(64).build().unwrap();
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08];
        let mut params = uniforms.into_iter().enumerate().map(|(slot, value)|
            (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(2));
        let task = crate::ucf_physics::dispatch_task(70, &program, 2, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
        let graph = Graph { resources: ResourceGraph { nodes: [(1, 18, Access::Read), (2, 32, Access::Write), (3, 2, Access::Write)]
            .into_iter().map(|(id, elements, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer,
                domain: Domain::Vram, access, byte_size: Some(elements * 4) }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
        graph.validate().unwrap();
        let mut backend = VulkanBackend::new().unwrap();
        backend.prepare(&graph).unwrap();
        for native in [false, true] {
            let mut input = seed.clone();
            let mut maxima = [0.0f64; 2];
            let mut escaped = false;
            let mut chunks = 0;
            for _ in 0..256 {
                let output = if native {
                    backend.write_f32(ResourceId(1), &input).unwrap();
                    backend.run_prepared(&graph).unwrap();
                    backend.read_f32(ResourceId(2)).unwrap()
                } else {
                    let mut interpreter = ton_ucf_frontend::CpuInterpreter::new();
                    for (slot, value) in uniforms.into_iter().enumerate() { interpreter.with_uniform(slot as u32, ton_ucf_frontend::Value::F32(value)); }
                    interpreter.with_buffer(1, input.iter().copied().map(ton_ucf_frontend::Value::F32).collect());
                    interpreter.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); 32]);
                    interpreter.run_parallel(&program, 2).unwrap();
                    interpreter.buffer(2).unwrap().iter().map(|value| {
                        let ton_ucf_frontend::Value::F32(value) = value else { panic!("Cartesian program lost F32 precision"); };
                        *value
                    }).collect()
                };
                chunks += 1;
                assert!(output.iter().all(|value| value.is_finite()));
                let mut all_escaped = true;
                input.clear();
                for (record, original) in output.chunks_exact(16).zip(initial) {
                    assert_eq!(record[8], 1.0);
                    let state = ton_geodesic::SchwarzschildCartesianNullState { t: record[0] as f64,
                        position: Vec3::new(record[1] as f64, record[2] as f64, record[3] as f64), energy: record[4] as f64,
                        velocity: Vec3::new(record[5] as f64, record[6] as f64, record[7] as f64) };
                    let angular = state.angular_momentum();
                    let original_angular = original.angular_momentum();
                    let drift = (angular.dot(angular) / original_angular.dot(original_angular) - 1.0).abs();
                    maxima[0] = maxima[0].max(record[13] as f64).max(state.null_residual(1.0));
                    let initial_squared = original_angular.dot(original_angular);
                    let chunk_baseline_ratio = record[15] as f64 / initial_squared;
                    let global_drift_bound = (chunk_baseline_ratio - 1.0).abs()
                        + chunk_baseline_ratio.abs() * record[14] as f64;
                    maxima[1] = maxima[1].max(global_drift_bound).max(drift);
                    assert!(maxima[0] < 1.0e-3 && maxima[1] < 2.0e-3, "native={native}, maxima={maxima:?}");
                    all_escaped &= state.position.length() >= 80.0 && state.position.dot(state.velocity) > 0.0;
                    input.extend_from_slice(&record[..8]);
                    input.push(record[10]);
                }
                if all_escaped { escaped = true; break; }
            }
            eprintln!("Cartesian native={native}, chunks={chunks}, maxima={maxima:?}");
            assert!(escaped);
        }
    }

    #[test]
    fn cartesian_event_localization_cpu_and_vulkan_match_independent_reference() {
        use ton_geodesic::{SchwarzschildCartesianNullState, SchwarzschildStepEventKind};
        for (position, direction, surface_kind, capture, escape, expected_kind) in [
            (Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0), 0.0f32, 9.98, 80.0, SchwarzschildStepEventKind::Capture),
            (Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, 1.0), 0.0, 2.0002, 10.02, SchwarzschildStepEventKind::Escape),
            (Vec3::new(10.0, 0.0, 0.001), Vec3::new(1.0, 0.0, -1.0), 1.0, 2.0002, 80.0, SchwarzschildStepEventKind::DiskCrossing),
        ] {
            let initial = SchwarzschildCartesianNullState::from_ray(1.0, position, direction).unwrap();
            let accepted = ton_geodesic::schwarzschild_cartesian_adaptive_rk4_step(1.0, initial, 0.04,
                ton_geodesic::AdaptiveRk4Config::builder().build().unwrap()).unwrap();
            let config = ton_geodesic::SchwarzschildEventConfig::builder(1.0).radial_boundaries(capture, escape);
            let config = if surface_kind == 1.0 { config.equatorial_disk(6.0, 20.0) } else { config };
            let config = config.build().unwrap();
            let events = ton_geodesic::schwarzschild_cartesian_accepted_step_events(initial, accepted, config).unwrap();
            let expected = events.iter().find(|event| event.kind == expected_kind).unwrap();
            let pack = |state: SchwarzschildCartesianNullState| [state.t, state.position.x, state.position.y,
                state.position.z, state.energy, state.velocity.x, state.velocity.y, state.velocity.z].map(|value| value as f32);
            let mut input = pack(initial).to_vec();
            input.extend(pack(accepted.state));
            input.push(accepted.step_used as f32);
            let program = ton_ucf_frontend::SchwarzschildEventLocalizationBuilder::new().cartesian_null().build().unwrap();
            let radius = if expected_kind == SchwarzschildStepEventKind::Escape { escape } else { capture };
            let uniforms = [1.0f32, surface_kind, radius as f32, 1.0e-5];
            let mut interpreter = ton_ucf_frontend::CpuInterpreter::new();
            for (slot, value) in uniforms.into_iter().enumerate() { interpreter.with_uniform(slot as u32, ton_ucf_frontend::Value::F32(value)); }
            interpreter.with_buffer(1, input.iter().copied().map(ton_ucf_frontend::Value::F32).collect());
            interpreter.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); 11]);
            interpreter.run_parallel(&program, 1).unwrap();
            let cpu = interpreter.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap() as f32).collect::<Vec<_>>();
            let mut params = uniforms.into_iter().enumerate().map(|(slot, value)|
                (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
            params.insert("buffer_2".into(), ParamValue::I64(2));
            let task = crate::ucf_physics::dispatch_task(71, &program, 1, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
            let graph = Graph { resources: ResourceGraph { nodes: [(1, 17, Access::Read), (2, 11, Access::Write), (3, 1, Access::Write)]
                .into_iter().map(|(id, elements, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer,
                    domain: Domain::Vram, access, byte_size: Some(elements * 4) }).collect() },
                tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
            graph.validate().unwrap();
            let mut backend = VulkanBackend::new().unwrap();
            backend.prepare(&graph).unwrap();
            backend.write_f32(ResourceId(1), &input).unwrap();
            backend.run_prepared(&graph).unwrap();
            let gpu = backend.read_f32(ResourceId(2)).unwrap();
            for output in [&cpu, &gpu] {
                assert_eq!(output[10], 1.0);
                assert!(output[9] <= 1.0e-5);
                assert!((output[8] as f64 - expected.fraction).abs() < 1.0e-3);
                for (actual, expected) in output[..8].iter().zip(pack(expected.state)) {
                    assert!((*actual - expected).abs() < 2.0e-4, "kind={expected_kind:?}");
                }
            }
        }
    }

    #[test]
    fn cartesian_fused_gpu_trace_matches_independent_events_and_invariants() {
        let camera = ton_types::Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let mut states = (0..5).flat_map(|y| (0..7).map(move |x| {
            let direction = camera.ray_direction(((x as f64 + 0.5) / 7.0 * 2.0 - 1.0) * 7.0 / 5.0,
                1.0 - (y as f64 + 0.5) / 5.0 * 2.0);
            SchwarzschildCartesianNullState::from_ray(1.0, camera.position, direction).unwrap()
        })).collect::<Vec<_>>();
        for index in [1712usize, 7503] {
            let u = (((index % 128) as f64 + 0.5) / 128.0 * 2.0 - 1.0) * 128.0 / 72.0;
            let v = 1.0 - ((index / 128) as f64 + 0.5) / 72.0 * 2.0;
            states.push(SchwarzschildCartesianNullState::from_ray(1.0, camera.position, camera.ray_direction(u, v)).unwrap());
        }
        let result = SchwarzschildGpuTraceBuilder::new(1.0).execute_cartesian(&states).unwrap();
        assert_eq!(result.state_layout, GpuTraceStateLayout::CartesianNullEnergy);
        result.require_renderable().unwrap();
        let expected_hash = ton_ucf_frontend::program_hash(&SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(64).cartesian_null().build().unwrap());
        assert!(result.executed_program_hashes.iter().all(|hash| *hash == expected_hash));
        let events = ton_geodesic::SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).build().unwrap();
        let config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).events(events).build().unwrap();
        let mut disk_count = 0;
        let mut multiple = 0;
        for (index, ((record, termination), initial)) in result.records.iter().zip(&result.terminations).zip(states).enumerate() {
            let reference = ton_geodesic::schwarzschild_cartesian_adaptive_trace(initial, config).unwrap();
            let expected = match reference.termination {
                ton_geodesic::AdaptiveTraceTermination::Captured => GpuTraceTermination::Captured,
                ton_geodesic::AdaptiveTraceTermination::Escaped => GpuTraceTermination::Escaped,
                other => panic!("reference {index}: {other:?}, {:?}", reference.failure),
            };
            assert_eq!(*termination, expected, "instance={index}");
            assert_eq!(record[13], reference.disk_events.len() as f32, "instance={index}");
            let radius = Vec3::new(record[1] as f64, record[2] as f64, record[3] as f64).length();
            assert!((radius - reference.state.position.length()).abs() < 2.0e-5, "instance={index}, radius={radius}");
            if let Some(first) = reference.disk_events.first() {
                assert!((record[14] as f64 - first.event.state.position.length()).abs() < 2.0e-3, "instance={index}");
                assert!((record[15] as f64 - first.affine_parameter).abs() < 2.0e-3, "instance={index}");
                assert!(record[24].abs() <= 1.0e-5);
                assert!((record[25] as f64 - first.event.state.energy).abs() < 1.0e-5);
            }
            disk_count += reference.disk_events.len();
            multiple += usize::from(reference.disk_events.len() > 1);
        }
        eprintln!("Cartesian fused GPU instances={}, disk_count={disk_count}, multiple={multiple}, dispatches={}", result.records.len(), result.executed_program_hashes.len());
        assert!(disk_count > 0 && multiple > 0);
    }

    #[test]
    fn cartesian_fused_gpu_trace_obeys_exact_non_multiple_budget() {
        let state = SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)).unwrap();
        let result = SchwarzschildGpuTraceBuilder::new(1.0).maximum_steps(5).chunk_steps(3).execute_cartesian(&[state]).unwrap();
        assert_eq!(result.terminations, [GpuTraceTermination::StepBudgetExhausted]);
        assert_eq!(result.records[0][10], 5.0);
        assert!(result.require_renderable().is_err());
        assert_eq!(result.executed_program_hashes.len(), 2);
        assert_ne!(result.executed_program_hashes[0], result.executed_program_hashes[1]);
    }

    #[test]
    fn cartesian_fused_cpu_trace_executes_decoded_program_and_matches_reference() {
        let camera = ton_types::Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let mut states = [1712usize, 7503].map(|index| {
            let u = (((index % 128) as f64 + 0.5) / 128.0 * 2.0 - 1.0) * 128.0 / 72.0;
            let v = 1.0 - ((index / 128) as f64 + 0.5) / 72.0 * 2.0;
            SchwarzschildCartesianNullState::from_ray(1.0, camera.position, camera.ray_direction(u, v)).unwrap()
        }).to_vec();
        states.push(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)).unwrap());
        states.push(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(10.0, 0.0, 0.001), Vec3::new(1.0, 0.0, -1.0)).unwrap());
        let program = SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(64).cartesian_null().build().unwrap();
        let encoded = serde_json::to_string(&program).unwrap();
        let decoded: ton_ucf_frontend::Program = serde_json::from_str(&encoded).unwrap();
        ton_ucf_frontend::validate_program(&decoded).unwrap();
        assert_eq!(ton_ucf_frontend::program_hash(&decoded), ton_ucf_frontend::program_hash(&program));
        let mut input = states.iter().flat_map(|state| {
            let mut record = [0.0f32; 29];
            record[..8].copy_from_slice(&[state.t, state.position.x, state.position.y, state.position.z, state.energy,
                state.velocity.x, state.velocity.y, state.velocity.z].map(|value| value as f32));
            record[9] = 0.08;
            record
        }).collect::<Vec<_>>();
        let execute = |input: &[f32]| {
            let mut interpreter = ton_ucf_frontend::CpuInterpreter::new();
            for (slot, value) in [1.0, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 80.0, 6.0, 20.0, 1.0e-5].into_iter().enumerate() {
                interpreter.with_uniform(slot as u32, ton_ucf_frontend::Value::F32(value));
            }
            interpreter.with_buffer(1, input.iter().copied().map(ton_ucf_frontend::Value::F32).collect());
            interpreter.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); input.len()]);
            interpreter.run_parallel(&decoded, states.len() as u32).unwrap();
            interpreter.buffer(2).unwrap().iter().map(|value| {
                let ton_ucf_frontend::Value::F32(value) = value else { panic!("fused Cartesian trace lost F32 type"); };
                *value
            }).collect::<Vec<_>>()
        };
        for _ in 0..256 {
            input = execute(&input);
            if input.chunks_exact(29).all(|record| record[8] != 0.0) { break; }
        }
        let events = ton_geodesic::SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).build().unwrap();
        let config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).events(events).build().unwrap();
        for (record, state) in input.chunks_exact(29).zip(states.iter()) {
            let reference = ton_geodesic::schwarzschild_cartesian_adaptive_trace(*state, config).unwrap();
            assert_eq!(record[8], match reference.termination {
                ton_geodesic::AdaptiveTraceTermination::Captured => 1.0,
                ton_geodesic::AdaptiveTraceTermination::Escaped => 2.0,
                other => panic!("unexpected reference status: {other:?}"),
            });
            assert_eq!(record[13], reference.disk_events.len() as f32);
            assert!(record[16] < 1.0e-3 && record[17] < 1.0e-3 && record[18] < 2.0e-3);
            if let Some(first) = reference.disk_events.first() {
                assert!((record[14] as f64 - first.event.state.position.length()).abs() < 2.0e-3);
            }
        }
        assert_eq!(execute(&input), input);
    }

    #[test]
    fn cartesian_fused_gpu_frame_passes_invariants_and_analytic_capture() {
        let bh = ton_types::BlackHole::schwarzschild(1.0);
        let camera = ton_types::Camera::cinematic(bh);
        let (width, height) = (128, 72);
        let rays = (0..height).flat_map(|y| (0..width).map(move |x|
            camera.ray_direction(((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
                1.0 - (y as f64 + 0.5) / height as f64 * 2.0))).collect::<Vec<_>>();
        let states = rays.iter().map(|direction| SchwarzschildCartesianNullState::from_ray(1.0,
            camera.position, *direction).unwrap()).collect::<Vec<_>>();
        let result = SchwarzschildGpuTraceBuilder::new(1.0).execute_cartesian(&states).unwrap();
        result.require_renderable().unwrap();
        let mut captured = 0;
        let mut maxima = [0.0f32; 3];
        for (index, ((record, termination), direction)) in result.records.iter().zip(&result.terminations).zip(rays).enumerate() {
            let expected = if ton_metric::schwarzschild_geodesic_analytic_captured(bh, camera.position, direction) {
                captured += 1;
                GpuTraceTermination::Captured
            } else { GpuTraceTermination::Escaped };
            assert_eq!(*termination, expected, "pixel={index}");
            for component in 0..3 { maxima[component] = maxima[component].max(record[16 + component]); }
        }
        eprintln!("Cartesian fused GPU frame {width}x{height}: captured={captured}, escaped={}, maxima={maxima:?}, dispatches={}",
            states.len() - captured, result.executed_program_hashes.len());
        assert!(captured > 0 && captured < states.len());
    }

    #[test]
    fn cartesian_initial_cpu_and_gpu_match_independent_ray_reference() {
        let rays = [(Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -3.0)),
            (Vec3::new(12.0, 3.0, 5.0), Vec3::new(-1.0, 0.2, -0.1)),
            (Vec3::new(5.0, -2.0, -10.0), Vec3::new(1.0, 0.4, 0.1)),
            (Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO)];
        let program = ton_ucf_frontend::SchwarzschildCartesianInitialStateBuilder::new().build().unwrap();
        let encoded = serde_json::to_string(&program).unwrap();
        let decoded: ton_ucf_frontend::Program = serde_json::from_str(&encoded).unwrap();
        assert_eq!(ton_ucf_frontend::program_hash(&program), ton_ucf_frontend::program_hash(&decoded));
        let input = rays.iter().flat_map(|(origin, direction)| [origin.x, origin.y, origin.z,
            direction.x, direction.y, direction.z].map(|value| ton_ucf_frontend::Value::F32(value as f32))).collect::<Vec<_>>();
        let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
        cpu.with_uniform(0, ton_ucf_frontend::Value::F32(1.0));
        cpu.with_uniform(1, ton_ucf_frontend::Value::F32(0.08));
        cpu.with_buffer(1, input);
        cpu.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); rays.len() * 29]);
        cpu.run_parallel(&decoded, rays.len() as u32).unwrap();
        let cpu = cpu.buffer(2).unwrap().iter().map(|value| {
            let ton_ucf_frontend::Value::F32(value) = value else { panic!("initial program lost F32 type"); };
            *value
        }).collect::<Vec<_>>();
        let mut backend = VulkanBackend::new().unwrap();
        let (gpu, hash) = SchwarzschildGpuTraceBuilder::new(1.0).initialize_rays(&mut backend, &rays).unwrap();
        assert_eq!(hash, ton_ucf_frontend::program_hash(&decoded));
        for (index, ((origin, direction), native)) in rays.iter().zip(&gpu).enumerate() {
            for record in [&cpu[index * 29..(index + 1) * 29], native.as_slice()] {
                match SchwarzschildCartesianNullState::from_ray(1.0, *origin, *direction) {
                    Ok(reference) => {
                        assert_eq!(record[8], 0.0);
                        for (actual, expected) in record[..8].iter().zip([reference.t, reference.position.x, reference.position.y,
                            reference.position.z, reference.energy, reference.velocity.x, reference.velocity.y, reference.velocity.z]) {
                            assert!((*actual as f64 - expected).abs() < 16.0 * f32::EPSILON as f64 * expected.abs().max(1.0));
                        }
                        let state = SchwarzschildCartesianNullState { t: record[0] as f64,
                            position: Vec3::new(record[1] as f64, record[2] as f64, record[3] as f64), energy: record[4] as f64,
                            velocity: Vec3::new(record[5] as f64, record[6] as f64, record[7] as f64) };
                        assert!(state.null_residual(1.0) < 1.0e-6);
                    }
                    Err(_) => { assert_eq!(record[8], 5.0); assert!(record.iter().all(|value| value.is_finite())); }
                }
            }
        }
    }

    #[test]
    fn cartesian_gpu_camera_chain_small_frame_matches_independent_trace() {
        let camera = Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let (width, height) = (16, 9);
        let result = SchwarzschildGpuTraceBuilder::new(1.0).execute_camera(camera, width, height).unwrap();
        result.require_renderable().unwrap();
        assert_eq!(result.state_layout, GpuTraceStateLayout::CartesianNullEnergy);
        assert_eq!(result.executed_program_hashes[0], ton_ucf_frontend::program_hash(&ton_ucf_frontend::camera_ray_batch()));
        assert_eq!(result.executed_program_hashes[1], ton_ucf_frontend::program_hash(&ton_ucf_frontend::SchwarzschildCartesianInitialStateBuilder::new().build().unwrap()));
        let events = ton_geodesic::SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).build().unwrap();
        let config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).events(events).build().unwrap();
        let mut captured = 0;
        let mut crossings = 0;
        for (index, (record, termination)) in result.records.iter().zip(&result.terminations).enumerate() {
            let u = (((index % width as usize) as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64;
            let v = 1.0 - ((index / width as usize) as f64 + 0.5) / height as f64 * 2.0;
            let initial = SchwarzschildCartesianNullState::from_ray(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
            let reference = ton_geodesic::schwarzschild_cartesian_adaptive_trace(initial, config).unwrap();
            let expected = match reference.termination {
                ton_geodesic::AdaptiveTraceTermination::Captured => { captured += 1; GpuTraceTermination::Captured },
                ton_geodesic::AdaptiveTraceTermination::Escaped => GpuTraceTermination::Escaped,
                other => panic!("pixel={index}, reference={other:?}"),
            };
            assert_eq!(*termination, expected, "pixel={index}");
            assert_eq!(record[13], reference.disk_events.len() as f32, "pixel={index}, gpu_record={:?}, reference_events={}", record, reference.disk_events.len());
            if let Some(first) = reference.disk_events.first() {
                assert!((record[14] as f64 - first.event.state.position.length()).abs() < 5.0e-3, "pixel={index}, gpu_radius={}, ref_radius={}, record={record:?}", record[14], first.event.state.position.length());
                assert!((record[15] as f64 - first.affine_parameter).abs() < 5.0e-3, "pixel={index}");
            }
            crossings += reference.disk_events.len();
        }
        eprintln!("GPU camera chain {width}x{height}: captured={captured}, disk_crossings={crossings}");
        assert!(captured > 0 && captured < result.records.len() && crossings > 0);
    }

    #[test]
    fn cartesian_gpu_camera_chain_large_frame_passes_analytic_and_invariant_gates() {
        let bh = ton_types::BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let (width, height) = (128, 72);
        let result = SchwarzschildGpuTraceBuilder::new(1.0).execute_camera(camera, width, height).unwrap();
        result.require_renderable().unwrap();
        let mut captured = 0;
        let mut maxima = [0.0f32; 3];
        for (index, (record, termination)) in result.records.iter().zip(&result.terminations).enumerate() {
            let u = (((index % width as usize) as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64;
            let v = 1.0 - ((index / width as usize) as f64 + 0.5) / height as f64 * 2.0;
            let expected = if ton_metric::schwarzschild_geodesic_analytic_captured(bh, camera.position, camera.ray_direction(u, v)) {
                captured += 1;
                GpuTraceTermination::Captured
            } else { GpuTraceTermination::Escaped };
            assert_eq!(*termination, expected, "pixel={index}");
            for component in 0..3 { maxima[component] = maxima[component].max(record[16 + component]); }
        }
        eprintln!("GPU camera chain {width}x{height}: captured={captured}, escaped={}, maxima={maxima:?}, executed_programs={}",
            result.records.len() - captured, result.executed_program_hashes.len());
    }

    #[test]
    fn cartesian_near_critical_disk_error_sources_and_step_sweep() {
        let camera = Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let (u, v) = (((9.5 / 16.0) * 2.0 - 1.0) * 16.0 / 9.0, 1.0 - (1.5 / 9.0) * 2.0);
        let ideal = SchwarzschildCartesianNullState::from_ray(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
        let mut backend = VulkanBackend::new().unwrap();
        let directions = crate::ucf_physics::camera_ray_ucf_vulkan_batch_with_backend(&mut backend, camera, &[(u, v)]).unwrap();
        let (records, _) = SchwarzschildGpuTraceBuilder::new(1.0).initialize_rays(&mut backend, &[(camera.position, directions[0])]).unwrap();
        let seed = records[0];
        let rounded = SchwarzschildCartesianNullState { t: seed[0] as f64,
            position: Vec3::new(seed[1] as f64, seed[2] as f64, seed[3] as f64), energy: seed[4] as f64,
            velocity: Vec3::new(seed[5] as f64, seed[6] as f64, seed[7] as f64) };
        let events = ton_geodesic::SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).build().unwrap();
        let config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).events(events)
            .integration(ton_geodesic::AdaptiveRk4Config::builder().step_range(1.0e-8, 0.02).build().unwrap()).build().unwrap();
        let ideal = ton_geodesic::schwarzschild_cartesian_adaptive_trace(ideal, config).unwrap();
        let rounded = ton_geodesic::schwarzschild_cartesian_adaptive_trace(rounded, config).unwrap();
        let ideal_radius = ideal.disk_events[0].event.state.position.length();
        let rounded_radius = rounded.disk_events[0].event.state.position.length();
        eprintln!("pixel 25 initial-source radius error={}, ideal={ideal_radius}, rounded={rounded_radius}", (rounded_radius - ideal_radius).abs());
        for step in [0.08, 0.04, 0.02] {
            let mut input = records.clone();
            input[0][9] = step;
            let result = SchwarzschildGpuTraceBuilder::new(1.0).step_range(step, step)
                .execute_packed(&mut backend, input, GpuTraceStateLayout::CartesianNullEnergy).unwrap();
            result.require_renderable().unwrap();
            assert_eq!(result.terminations, [GpuTraceTermination::Escaped]);
            assert_eq!(result.records[0][13], rounded.disk_events.len() as f32);
            eprintln!("pixel 25 step={step}, disk_radius={}, same-input error={}, ideal error={}, angular_drift={}",
                result.records[0][14], (result.records[0][14] as f64 - rounded_radius).abs(),
                (result.records[0][14] as f64 - ideal_radius).abs(), result.records[0][18]);
        }
    }
}

//! F64 通用 UCF 程序的 GPU 相机与分块追踪编排。

use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use ton_types::{Camera, TonError, TonResult};
use ucf_backend_vulkan::{ProgramCacheStats, VulkanBackend};
use ucf_ir::{Graph, ResourceGraph, TaskGraph};
use ucf_scheduler::Backend;
use ucf_types::{Access, Domain, ParamValue, ResourceId, ResourceKind, ResourceNode};
use crate::GpuTraceTermination;

/// 主机观察到的 F64 追踪各阶段耗时，dispatch 含同步等待，不是 GPU timestamp。
#[derive(Debug, Clone, Copy, Default)]
pub struct F64TraceTiming {
    /// GPU 相机初态链的完整耗时，包含准备与传输。
    pub camera: Duration,
    /// 追踪程序构建与图资源准备耗时，不包含运行时 lowering。
    pub preparation: Duration,
    /// 逐 chunk 记录上传耗时。
    pub upload: Duration,
    /// 逐 chunk 原生执行、首次 lowering 与同步等待耗时。
    pub dispatch: Duration,
    /// 输出与状态读回耗时。
    pub readback: Duration,
    /// 完整调用耗时，包含设备创建和主机编排。
    pub total: Duration,
}

/// 实际 F64 GPU 输出，状态布局与 F32 版本的字节宽度明确分离。
#[derive(Debug)]
pub struct F64GpuTraceBatch {
    /// `[state(8),status,step,steps,rejections,affine,disk_count,radius,affine,max_null,max_E,max_L2,E0,L0²,first_disk_state(8)]`。
    pub records: Vec<[f64; 29]>,
    /// 物理终止或失败，不把 chunk 返回状态 0 当作成功。
    pub terminations: Vec<GpuTraceTermination>,
    /// 仅在实际执行与 readback 成功后记录的程序哈希。
    pub executed_program_hashes: Vec<u64>,
    /// 为保持末批资源容量一致而实际执行、但不进入图像的重复实例数。
    pub padded_instances: u32,
    /// 实测主机阶段耗时，不以调色掩盖性能问题。
    pub timing: F64TraceTiming,
    /// 追踪后端的编译缓存统计，不代表实际 dispatch 数量。
    pub program_cache: ProgramCacheStats,
}

/// F64 GPU 追踪与固定容量盘面事件输出。
///
/// 事件资源每实例固定为 `[count, overflow, event[0..4]]`，每个事件九个
/// F64。该结果只允许进入辐射阶段，事件 overflow 或非终止状态都必须被
/// 上层拒绝，不能静默生成 Beauty。
#[derive(Debug)]
pub struct F64GpuEventTraceBatch {
    /// 与普通 F64 追踪相同的 29 标量记录。
    pub records: Vec<[f64; 29]>,
    /// 每实例固定 38 元素事件包。
    pub events: Vec<[f64; 38]>,
    /// 物理终止或失败分类。
    pub terminations: Vec<GpuTraceTermination>,
    /// 实际执行的事件追踪程序哈希。
    pub executed_program_hashes: Vec<u64>,
}

impl F64GpuEventTraceBatch {
    /// 将同一份已执行记录投影为普通渲染轨迹，不重新执行 GPU 程序。
    pub fn render_trace(&self) -> F64GpuTraceBatch {
        F64GpuTraceBatch {
            records: self.records.clone(),
            terminations: self.terminations.clone(),
            executed_program_hashes: self.executed_program_hashes.clone(),
            padded_instances: 0,
            timing: F64TraceTiming::default(),
            program_cache: ProgramCacheStats::default(),
        }
    }

    /// 检查事件包是否允许进入辐射阶段。
    pub fn require_radiative(&self) -> TonResult<()> {
        if self.records.len() != self.events.len() || self.records.len() != self.terminations.len() {
            return Err(TonError::Domain("F64 event record and packet count mismatch".into()));
        }
        for (index, ((record, event), termination)) in self.records.iter().zip(&self.events).zip(&self.terminations).enumerate() {
            if !matches!(termination, GpuTraceTermination::Captured | GpuTraceTermination::Escaped)
                || record.iter().any(|value| !value.is_finite()) || event.iter().any(|value| !value.is_finite())
                || event[0] < 0.0 || event[0] > 4.0 || event[0].fract() != 0.0 || event[1] != 0.0 {
                return Err(TonError::Domain(format!("F64 event instance {index} cannot enter radiative stage: termination={termination:?}, count={}, overflow={}", event[0], event[1])));
            }
        }
        Ok(())
    }
}

impl F64GpuTraceBatch {
    /// 检查终止和不变量，不能替代独立轨迹及步长收敛验证。
    pub fn require_renderable(&self) -> TonResult<()> {
        if self.records.len() != self.terminations.len() { return Err(TonError::Domain("F64 trace record/status count mismatch".into())); }
        for (index, (record, status)) in self.records.iter().zip(&self.terminations).enumerate() {
            if !matches!(status, GpuTraceTermination::Captured | GpuTraceTermination::Escaped)
                || record.iter().any(|value| !value.is_finite()) || record[16] < 0.0 || record[16] >= 1.0e-6
                || record[17] < 0.0 || record[17] >= 1.0e-12 || record[18] < 0.0 || record[18] >= 1.0e-7 {
                return Err(TonError::Domain(format!("F64 instance {index} cannot enter Beauty: status={status:?}, null={}, energy={}, angular={}", record[16], record[17], record[18])));
            }
        }
        Ok(())
    }
}

/// F64 相机和完整捕获/逃逸追踪，所有轨迹计算均由通用 UCF 程序执行。
/// 追踪 chunk 之间使用 GPU 双缓冲，相机与辐射阶段之间仍有主机传输。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildF64GpuTraceBuilder {
    mass: f64,
    step: f64,
    capture: f64,
    escape: f64,
    inner: f64,
    outer: f64,
    maximum_steps: u32,
    chunk_steps: u32,
    batch_instances: u32,
}

impl SchwarzschildF64GpuTraceBuilder {
    /// 默认最大步长 0.02M，总预算 32768，捕获面位于坐标视界外。
    pub fn new(mass: f64) -> Self {
        Self { mass, step: 0.02 * mass, capture: 2.0002 * mass, escape: 80.0 * mass,
            inner: 6.0 * mass, outer: 20.0 * mass, maximum_steps: 32768, chunk_steps: 256, batch_instances: 8192 }
    }
    /// 设置最大仿射步长，近视界仍受位移上限约束。
    pub fn maximum_step(mut self, value: f64) -> Self { self.step = value; self }
    /// 设置捕获与逃逸边界。
    pub fn radial_boundaries(mut self, capture: f64, escape: f64) -> Self { self.capture = capture; self.escape = escape; self }
    /// 设置赤道薄盘环带。
    pub fn disk_radii(mut self, inner: f64, outer: f64) -> Self { self.inner = inner; self.outer = outer; self }
    /// 设置逐实例全局预算。
    pub fn maximum_steps(mut self, value: u32) -> Self { self.maximum_steps = value; self }
    /// 设置单次 dispatch 的有界积分预算。
    pub fn chunk_steps(mut self, value: u32) -> Self { self.chunk_steps = value; self }
    /// 限制单个 GPU dispatch 的实例规模，避免长循环大图触发设备超时。
    pub fn batch_instances(mut self, value: u32) -> Self { self.batch_instances = value; self }

    fn validate(self) -> TonResult<()> {
        if [self.mass, self.step, self.capture, self.escape, self.inner, self.outer].iter().any(|value| !value.is_finite())
            || self.mass <= 0.0 || self.step <= 0.0 || self.step > 0.1 * self.mass
            || self.capture <= 2.0 * self.mass || self.inner < 6.0 * self.mass || self.inner <= self.capture
            || self.outer <= self.inner || self.escape <= self.outer || self.maximum_steps == 0
            || self.chunk_steps == 0 || self.chunk_steps > 65536 || self.batch_instances == 0 || self.batch_instances > 8192 {
            return Err(TonError::Domain("invalid F64 GPU trace configuration".into()));
        }
        Ok(())
    }

    /// GPU 相机、GPU 初态与 GPU 追踪，不调用 TON 参考初态或参考渲染。
    pub fn execute_camera(self, camera: Camera, width: u32, height: u32) -> TonResult<F64GpuTraceBatch> {
        let started = Instant::now();
        self.validate()?;
        width.checked_mul(height).filter(|count| *count > 0)
            .ok_or_else(|| TonError::Domain("F64 camera resolution is zero or exceeds u32 capacity".into()))?;
        let coordinates = (0..height).flat_map(|y| (0..width).map(move |x|
            (((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
                1.0 - (y as f64 + 0.5) / height as f64 * 2.0))).collect::<Vec<_>>();
        let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create F64 trace backend: {error}")))?;
        let capacity = coordinates.len().min(self.batch_instances as usize);
        let mut result = F64GpuTraceBatch { records: Vec::with_capacity(coordinates.len()), terminations: Vec::with_capacity(coordinates.len()),
            executed_program_hashes: Vec::new(), padded_instances: 0, timing: F64TraceTiming::default(), program_cache: ProgramCacheStats::default() };
        for batch in coordinates.chunks(capacity) {
            let mut padded = batch.to_vec();
            padded.resize(capacity, batch[0]);
            let camera_started = Instant::now();
            let (initial, hash) = crate::adaptive_gpu::initialize_f64_camera(&mut backend, camera, &padded, self.mass)?;
            result.timing.camera += camera_started.elapsed();
            let output = self.execute_initial_with_backend(&mut backend, &initial)?;
            result.timing.preparation += output.timing.preparation;
            result.timing.upload += output.timing.upload;
            result.timing.dispatch += output.timing.dispatch;
            result.timing.readback += output.timing.readback;
            result.records.extend(output.records.into_iter().take(batch.len()));
            result.terminations.extend(output.terminations.into_iter().take(batch.len()));
            result.executed_program_hashes.push(hash);
            result.executed_program_hashes.extend(output.executed_program_hashes);
            result.padded_instances += (capacity - batch.len()) as u32;
        }
        result.timing.total = started.elapsed();
        result.program_cache = backend.program_cache_stats();
        Ok(result)
    }

    /// 执行相机 F64 追踪并保留每像素固定容量盘面事件。
    ///
    /// 该入口和 `execute_camera` 使用相同的 UCF 物理程序与相机初态，
    /// 但绑定额外的事件资源。事件资源在所有 chunk 间保持 resident，
    /// 不通过主机标签拼接结果。
    pub fn execute_event_camera(self, camera: Camera, width: u32, height: u32) -> TonResult<F64GpuEventTraceBatch> {
        let self_copy = self;
        self_copy.validate()?;
        let coordinates = (0..height).flat_map(|y| (0..width).map(move |x|
            (((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
                1.0 - (y as f64 + 0.5) / height as f64 * 2.0))).collect::<Vec<_>>();
        if coordinates.is_empty() { return Err(TonError::Domain("F64 event camera resolution is zero".into())); }
        let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create F64 event backend: {error}")))?;
        let capacity = coordinates.len().min(self_copy.batch_instances as usize);
        let mut records = Vec::with_capacity(coordinates.len());
        let mut events = Vec::with_capacity(coordinates.len());
        let mut terminations = Vec::with_capacity(coordinates.len());
        let mut hashes = Vec::new();
        for batch in coordinates.chunks(capacity) {
            let mut padded = batch.to_vec();
            padded.resize(capacity, batch[0]);
            let (initial, camera_hash) = crate::adaptive_gpu::initialize_f64_camera(&mut backend, camera, &padded, self_copy.mass)?;
            let output = self_copy.execute_initial_events_with_backend(&mut backend, &initial)?;
            records.extend(output.records.into_iter().take(batch.len()));
            events.extend(output.events.into_iter().take(batch.len()));
            terminations.extend(output.terminations.into_iter().take(batch.len()));
            hashes.push(camera_hash);
            hashes.extend(output.executed_program_hashes);
        }
        Ok(F64GpuEventTraceBatch { records, events, terminations, executed_program_hashes: hashes })
    }

    pub(crate) fn execute_initial_with_backend(self, backend: &mut VulkanBackend, initial: &[[f64; 9]]) -> TonResult<F64GpuTraceBatch> {
        let started = Instant::now();
        let mut timing = F64TraceTiming::default();
        self.validate()?;
        if initial.is_empty() { return Err(TonError::Domain("empty F64 trace batch".into())); }
        let count = u32::try_from(initial.len()).map_err(|_| TonError::Domain("F64 trace batch exceeds u32 capacity".into()))?;
        let mut records = initial.iter().map(|initial| {
            let mut record = [0.0; 29];
            record[..8].copy_from_slice(&initial[..8]); record[9] = self.step; record[19] = initial[4]; record[20] = initial[8]; record
        }).collect::<Vec<_>>();
        let mut hashes = Vec::new();
        let mut scheduled = 0;
        let mut prepared_steps = 0;
        let mut prepared = None;
        let mut statuses = vec![0.0; records.len()];
        let mut input_id = ResourceId(91);
        let mut output_id = ResourceId(92);
        let mut last_output = input_id;
        while scheduled < self.maximum_steps && statuses.iter().any(|status| *status == 0.0) {
            let steps = self.chunk_steps.min(self.maximum_steps - scheduled);
            if steps != prepared_steps {
                let prepare_started = Instant::now();
                let program = ton_ucf_frontend::SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(steps)
                    .surface_tolerance(1.0e-11 * self.mass).build().map_err(TonError::Domain)?;
                let mut params = [self.mass, self.step, self.capture, self.escape, self.inner, self.outer]
                    .into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value))).collect::<BTreeMap<_, _>>();
                params.insert("buffer_2".into(), ParamValue::I64(92));
                let task = crate::ucf_physics::dispatch_task(78, &program, count, ResourceId(93), 2, 1, Some(ResourceId(91)), params)?;
                let graph = Graph { resources: ResourceGraph { nodes: [(91, 29, Access::ReadWrite), (92, 29, Access::ReadWrite), (93, 1, Access::Write)]
                    .into_iter().map(|(id, stride, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer,
                        domain: Domain::Vram, access, byte_size: Some(u64::from(count) * stride * 8) }).collect() },
                    tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
                graph.validate().map_err(|error| TonError::Domain(format!("validate F64 trace graph: {error}")))?;
                backend.prepare(&graph).map_err(|error| TonError::Domain(format!("prepare F64 trace graph: {error}")))?;
                prepared = Some((graph, ton_ucf_frontend::program_hash(&program)));
                prepared_steps = steps;
                timing.preparation += prepare_started.elapsed();
            }
            let (graph, hash) = prepared.as_mut().unwrap();
            let params = &mut graph.tasks.nodes[0].params;
            params.insert("buffer_0".into(), ParamValue::I64(input_id.0 as i64));
            params.insert("buffer_1".into(), ParamValue::I64(input_id.0 as i64));
            params.insert("buffer_2".into(), ParamValue::I64(output_id.0 as i64));
            if scheduled == 0 {
                let upload_started = Instant::now();
                backend.write_f64(input_id, &records.iter().flatten().copied().collect::<Vec<_>>())
                    .map_err(|error| TonError::Domain(format!("upload F64 trace records: {error}")))?;
                timing.upload += upload_started.elapsed();
            }
            let dispatch_started = Instant::now();
            backend.run_prepared(graph).map_err(|error| TonError::Domain(format!("execute F64 trace graph: {error}")))?;
            timing.dispatch += dispatch_started.elapsed();
            let readback_started = Instant::now();
            statuses = backend.read_f64(ResourceId(93)).map_err(|error| TonError::Domain(format!("read F64 trace status: {error}")))?;
            timing.readback += readback_started.elapsed();
            if statuses.len() != records.len() || statuses.iter().any(|status|
                !matches!(*status, 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0)) {
                return Err(TonError::Domain("invalid F64 trace scalar status".into()));
            }
            hashes.push(*hash);
            scheduled += steps;
            last_output = output_id;
            std::mem::swap(&mut input_id, &mut output_id);
        }
        let readback_started = Instant::now();
        let output = backend.read_f64(last_output).map_err(|error| TonError::Domain(format!("read F64 final trace records: {error}")))?;
        timing.readback += readback_started.elapsed();
        if output.len() != records.len() * 29 || output.iter().any(|value| !value.is_finite())
            || output.chunks_exact(29).zip(&statuses).any(|(record, status)| record[8] != *status) {
            return Err(TonError::Domain("invalid F64 final trace output or scalar status mismatch".into()));
        }
        records = output.chunks_exact(29).map(|record| record.try_into().unwrap()).collect();
        let terminations = records.iter().map(|record| match record[8] {
            0.0 => Ok(GpuTraceTermination::StepBudgetExhausted), 1.0 => Ok(GpuTraceTermination::Captured),
            2.0 => Ok(GpuTraceTermination::Escaped), 3.0 => Ok(GpuTraceTermination::IntegrationFailure),
            4.0 => Ok(GpuTraceTermination::EventLocalizationFailure), 5.0 => Ok(GpuTraceTermination::InvalidInput),
            status => Err(TonError::Domain(format!("unknown F64 trace status {status}"))),
        }).collect::<TonResult<Vec<_>>>()?;
        timing.total = started.elapsed();
        Ok(F64GpuTraceBatch { records, terminations, executed_program_hashes: hashes, padded_instances: 0, timing, program_cache: backend.program_cache_stats() })
    }

    fn execute_initial_events_with_backend(self, backend: &mut VulkanBackend, initial: &[[f64; 9]]) -> TonResult<F64GpuEventTraceBatch> {
        self.validate()?;
        if initial.is_empty() { return Err(TonError::Domain("empty F64 event trace batch".into())); }
        let count = u32::try_from(initial.len()).map_err(|_| TonError::Domain("F64 event trace batch exceeds u32 capacity".into()))?;
        let mut records = initial.iter().map(|initial| {
            let mut record = [0.0; 29];
            record[..8].copy_from_slice(&initial[..8]);
            record[9] = self.step;
            record[19] = initial[4];
            record[20] = initial[8];
            record
        }).collect::<Vec<_>>();
        let mut statuses = vec![0.0; records.len()];
        let mut input_id = ResourceId(191);
        let mut output_id = ResourceId(192);
        let mut last_output = output_id;
        let mut scheduled = 0;
        let mut prepared_steps = 0;
        let mut prepared = None;
        let mut hashes = Vec::new();
        while scheduled < self.maximum_steps && statuses.iter().any(|status| *status == 0.0) {
            let steps = self.chunk_steps.min(self.maximum_steps - scheduled);
            if steps != prepared_steps {
                let program = ton_ucf_frontend::SchwarzschildCartesianF64EventTraceBuilder::<4>::new()
                    .maximum_steps(steps).surface_tolerance(1.0e-11 * self.mass).build().map_err(TonError::Domain)?;
                let mut params = [self.mass, self.step, self.capture, self.escape, self.inner, self.outer]
                    .into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value))).collect::<BTreeMap<_, _>>();
                params.insert("buffer_2".into(), ParamValue::I64(output_id.0 as i64));
                params.insert("buffer_3".into(), ParamValue::I64(194));
                let task = crate::ucf_physics::dispatch_task(178, &program, count, ResourceId(193), 2, 1, Some(input_id), params)?;
                let graph = Graph { resources: ResourceGraph { nodes: [(191u64, 29u64, Access::ReadWrite), (192, 29, Access::ReadWrite),
                    (193, 1, Access::Write), (194, 38, Access::ReadWrite)].into_iter().map(|(id, stride, access)| ResourceNode {
                        id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access,
                        byte_size: Some(u64::from(count) * u64::from(stride) * 8),
                    }).collect() }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
                graph.validate().map_err(|error| TonError::Domain(format!("validate F64 event graph: {error}")))?;
                backend.prepare(&graph).map_err(|error| TonError::Domain(format!("prepare F64 event graph: {error}")))?;
                prepared = Some((graph, ton_ucf_frontend::program_hash(&program)));
                prepared_steps = steps;
            }
            let (graph, hash) = prepared.as_mut().unwrap();
            let params = &mut graph.tasks.nodes[0].params;
            params.insert("buffer_0".into(), ParamValue::I64(input_id.0 as i64));
            params.insert("buffer_1".into(), ParamValue::I64(input_id.0 as i64));
            params.insert("buffer_2".into(), ParamValue::I64(output_id.0 as i64));
            params.insert("buffer_3".into(), ParamValue::I64(194));
            if scheduled == 0 {
                backend.write_f64(input_id, &records.iter().flatten().copied().collect::<Vec<_>>())
                    .map_err(|error| TonError::Domain(format!("upload F64 event records: {error}")))?;
                backend.write_f64(ResourceId(194), &vec![0.0; initial.len() * 38])
                    .map_err(|error| TonError::Domain(format!("clear F64 event resource: {error}")))?;
            }
            backend.run_prepared(graph).map_err(|error| TonError::Domain(format!("execute F64 event graph: {error}")))?;
            statuses = backend.read_f64(ResourceId(193)).map_err(|error| TonError::Domain(format!("read F64 event status: {error}")))?;
            if statuses.len() != records.len() || statuses.iter().any(|status| !matches!(*status, 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0)) {
                return Err(TonError::Domain("invalid F64 event scalar status".into()));
            }
            hashes.push(*hash);
            scheduled += steps;
            last_output = output_id;
            std::mem::swap(&mut input_id, &mut output_id);
        }
        let output = backend.read_f64(last_output)
            .map_err(|error| TonError::Domain(format!("read F64 event records: {error}")))?;
        let packet = backend.read_f64(ResourceId(194)).map_err(|error| TonError::Domain(format!("read F64 event packets: {error}")))?;
        if output.len() != records.len() * 29 || packet.len() != records.len() * 38
            || output.iter().any(|value| !value.is_finite()) || packet.iter().any(|value| !value.is_finite()) {
            return Err(TonError::Domain("invalid F64 event trace output".into()));
        }
        records = output.chunks_exact(29).map(|record| record.try_into().unwrap()).collect();
        let events = packet.chunks_exact(38).map(|event| event.try_into().unwrap()).collect();
        let terminations = records.iter().map(|record| match record[8] {
            0.0 => Ok(GpuTraceTermination::StepBudgetExhausted), 1.0 => Ok(GpuTraceTermination::Captured),
            2.0 => Ok(GpuTraceTermination::Escaped), 3.0 => Ok(GpuTraceTermination::IntegrationFailure),
            4.0 => Ok(GpuTraceTermination::EventLocalizationFailure), 5.0 => Ok(GpuTraceTermination::InvalidInput),
            status => Err(TonError::Domain(format!("unknown F64 event trace status {status}"))),
        }).collect::<TonResult<Vec<_>>>()?;
        Ok(F64GpuEventTraceBatch { records, events, terminations, executed_program_hashes: hashes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_geodesic::SchwarzschildCartesianNullState;
    use ton_types::{BlackHole, Vec3};

    fn pack(state: SchwarzschildCartesianNullState) -> [f64; 9] {
        let angular = state.angular_momentum();
        [state.t, state.position.x, state.position.y, state.position.z, state.energy,
            state.velocity.x, state.velocity.y, state.velocity.z, angular.dot(angular)]
    }

    #[test]
    fn f64_native_resident_chunks_preserve_odd_even_and_partial_budget_outputs() {
        let seeds = [pack(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 79.9), Vec3::new(0.0, 0.0, 1.0)).unwrap()),
            pack(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0)).unwrap())];
        for budget in [4, 5, 6] {
            let mut backend = VulkanBackend::new().unwrap();
            let fused = SchwarzschildF64GpuTraceBuilder::new(1.0).maximum_steps(budget).chunk_steps(budget)
                .execute_initial_with_backend(&mut backend, &seeds).unwrap();
            for chunk in [1, 2, 3] {
                let mut backend = VulkanBackend::new().unwrap();
                let split = SchwarzschildF64GpuTraceBuilder::new(1.0).maximum_steps(budget).chunk_steps(chunk)
                    .execute_initial_with_backend(&mut backend, &seeds).unwrap();
                assert_eq!(split.records, fused.records, "budget={budget}, chunk={chunk}");
                assert_eq!(split.terminations, fused.terminations);
                assert_eq!(split.records[1][10], budget as f64);
                assert_eq!(split.terminations[1], GpuTraceTermination::StepBudgetExhausted);
                assert_eq!(split.executed_program_hashes.len(), budget.div_ceil(chunk) as usize);
            }
        }
    }

    #[test]
    fn f64_native_camera_batches_preserve_pixel_order_and_report_padding() {
        let camera = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let plain = SchwarzschildF64GpuTraceBuilder::new(1.0).maximum_steps(1).execute_camera(camera, 5, 3).unwrap();
        let batched = SchwarzschildF64GpuTraceBuilder::new(1.0).maximum_steps(1).batch_instances(8).execute_camera(camera, 5, 3).unwrap();
        assert_eq!(plain.records, batched.records);
        assert_eq!(plain.terminations, batched.terminations);
        assert_eq!(plain.padded_instances, 0); assert_eq!(batched.padded_instances, 1);
        assert_eq!(batched.records.len(), 15); assert_eq!(batched.executed_program_hashes.len(), 4);
        assert_eq!(batched.program_cache.lowerings, 2);
        assert_eq!(batched.program_cache.hits, 2);
        assert!(batched.timing.total >= batched.timing.camera + batched.timing.preparation + batched.timing.upload + batched.timing.dispatch + batched.timing.readback);
        assert!(SchwarzschildF64GpuTraceBuilder::new(1.0).batch_instances(8193).execute_camera(camera, 5, 3).is_err());
    }

    #[test]
    fn f64_native_trace_radial_events_match_cpu_and_analytic_reference() {
        let seeds = [pack(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0)).unwrap()),
            pack(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 79.99), Vec3::new(0.0, 0.0, 1.0)).unwrap())];
        let mut backend = VulkanBackend::new().unwrap();
        let result = SchwarzschildF64GpuTraceBuilder::new(1.0).radial_boundaries(9.98, 80.0).disk_radii(10.1, 20.0)
            .maximum_step(0.04).maximum_steps(4).chunk_steps(4).execute_initial_with_backend(&mut backend, &seeds).unwrap();
        result.require_renderable().unwrap();
        assert_eq!(result.terminations, [GpuTraceTermination::Captured, GpuTraceTermination::Escaped]);
        let program = ton_ucf_frontend::SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(4).build().unwrap();
        let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
        for (slot, value) in [1.0, 0.04, 9.98, 80.0, 10.1, 20.0].into_iter().enumerate() { cpu.with_uniform(slot as u32, ton_ucf_frontend::Value::F64(value)); }
        let input = seeds.iter().flat_map(|seed| {
            let mut record = [0.0; 29]; record[..8].copy_from_slice(&seed[..8]); record[9] = 0.04; record[19] = seed[4]; record[20] = seed[8]; record
        }).map(ton_ucf_frontend::Value::F64).collect();
        cpu.with_buffer(1, input); cpu.with_buffer(2, vec![ton_ucf_frontend::Value::F64(0.0); 58]); cpu.run_parallel(&program, 2).unwrap();
        for (actual, expected) in result.records.iter().flatten().zip(cpu.buffer(2).unwrap()) { assert!((actual - expected.as_f64().unwrap()).abs() < 1.0e-10); }
        for (record, radius, affine) in [(&result.records[0], 9.98, 0.02), (&result.records[1], 80.0, 0.01)] {
            assert!((record[3] - radius).abs() < 1.0e-11); assert!((record[12] - affine).abs() < 1.0e-11);
        }
        let near = pack(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 2.001), Vec3::new(0.0, 0.0, -1.0)).unwrap());
        let mut backend = VulkanBackend::new().unwrap();
        let horizon = SchwarzschildF64GpuTraceBuilder::new(1.0).maximum_steps(256).chunk_steps(256)
            .execute_initial_with_backend(&mut backend, &[near]).unwrap();
        horizon.require_renderable().unwrap();
        assert_eq!(horizon.terminations, [GpuTraceTermination::Captured]);
        let record = horizon.records[0];
        let expected_time = 2.001 - record[3] + 2.0 * ((2.001 - 2.0) / (record[3] - 2.0)).ln();
        assert!((record[0] - expected_time).abs() < 1.0e-8);
        eprintln!("F64 native radial horizon: time_error={}, steps={}", (record[0] - expected_time).abs(), record[10]);
    }

    #[test]
    fn f64_native_trace_budget_and_invalid_input_are_explicit() {
        let mut backend = VulkanBackend::new().unwrap();
        let valid = pack(SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0)).unwrap());
        let mut invalid = valid; invalid[4] = 2.0;
        let result = SchwarzschildF64GpuTraceBuilder::new(1.0).maximum_steps(1).chunk_steps(64)
            .execute_initial_with_backend(&mut backend, &[valid, invalid]).unwrap();
        assert_eq!(result.terminations, [GpuTraceTermination::StepBudgetExhausted, GpuTraceTermination::InvalidInput]);
        assert_eq!(result.records[0][10], 1.0); assert_eq!(result.records[1][10], 0.0);
        assert_eq!(&result.records[1][..8], &invalid[..8]);
        assert_eq!(result.executed_program_hashes.len(), 1);
        assert!(result.require_renderable().is_err());
    }

    #[test]
    fn f64_event_camera_keeps_fixed_packets_for_each_pixel() {
        let camera = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let result = SchwarzschildF64GpuTraceBuilder::new(1.0)
            .maximum_steps(32768).chunk_steps(256).batch_instances(32)
            .execute_event_camera(camera, 8, 4).unwrap();
        assert_eq!(result.records.len(), 32);
        assert_eq!(result.events.len(), 32);
        assert_eq!(result.terminations.len(), 32);
        assert!(result.executed_program_hashes.len() >= 2);
        assert!(result.events.iter().all(|event| event[0] >= 0.0 && event[0] <= 4.0 && event[1] == 0.0));
        assert!(result.events.iter().flatten().all(|value| value.is_finite()));
        assert!(result.events.iter().any(|event| event[0] > 0.0));
        result.require_radiative().unwrap();
    }

    #[test]
    fn f64_event_trace_vulkan_matches_reference_packet() {
        let initial = SchwarzschildCartesianNullState::from_ray(1.0, Vec3::new(10.0, 0.0, 0.01), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        let reference_events = ton_geodesic::SchwarzschildEventConfig::builder(1.0)
            .radial_boundaries(2.0002, 80.0).equatorial_disk(6.0, 20.0)
            .localization(1.0e-11, 64).build().unwrap();
        let integration = ton_geodesic::AdaptiveRk4Config::builder()
            .tolerances(1.0e-12, 1.0e-12).step_range(1.0e-8, 0.04).build().unwrap();
        let reference_config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0)
            .events(reference_events).integration(integration).initial_step(0.04)
            .maximum_steps(512).build().unwrap();
        let reference = ton_geodesic::schwarzschild_cartesian_adaptive_trace(initial, reference_config).unwrap();
        assert_eq!(reference.disk_events.len(), 1);
        let expected = ton_geodesic::DiskEventBuffer::<4>::from_events(&reference.disk_events).unwrap();

        let program = ton_ucf_frontend::SchwarzschildCartesianF64EventTraceBuilder::<4>::new()
            .maximum_steps(512).build().unwrap();
        let mut params = [1.0, 0.04, 2.0002, 80.0, 6.0, 20.0].into_iter().enumerate()
            .map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(92));
        params.insert("buffer_3".into(), ParamValue::I64(94));
        let task = crate::ucf_physics::dispatch_task(780, &program, 1, ResourceId(93), 2, 1, Some(ResourceId(91)), params).unwrap();
        let graph = Graph {
            resources: ResourceGraph { nodes: [(91u64, 29u64, Access::Read), (92, 29, Access::Write), (93, 1, Access::Write),
                (94, 38, Access::ReadWrite)].into_iter().map(|(id, stride, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access,
                    byte_size: Some(u64::from(stride) * 8),
                }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
        };
        graph.validate().unwrap();
        let mut backend = VulkanBackend::new().unwrap();
        backend.prepare(&graph).unwrap();
        let mut record = [0.0; 29];
        record[..8].copy_from_slice(&[
            initial.t, initial.position.x, initial.position.y, initial.position.z,
            initial.energy, initial.velocity.x, initial.velocity.y, initial.velocity.z,
        ]);
        record[9] = 0.04;
        record[19] = initial.energy;
        record[20] = initial.angular_momentum().dot(initial.angular_momentum());
        backend.write_f64(ResourceId(91), &record).unwrap();
        backend.write_f64(ResourceId(94), &[0.0; 38]).unwrap();
        backend.run_prepared(&graph).unwrap();
        let actual_record = backend.read_f64(ResourceId(92)).unwrap();
        let actual_events = backend.read_f64(ResourceId(94)).unwrap();
        assert_eq!(actual_events[0], expected.count as f64);
        assert_eq!(actual_events[1], 0.0);
        for (actual, expected) in actual_events[2..].iter().zip(expected.flatten()) {
            assert!((actual - expected).abs() < 1.0e-8, "event actual={actual}, expected={expected}");
        }
        assert_eq!(actual_record[13], 1.0);
        assert!((actual_record[14] - reference.disk_events[0].event.state.position.length()).abs() < 1.0e-8);
    }

    #[test]
    fn f64_native_camera_trace_frame_matches_independent_physics() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let (width, height) = (16, 9);
        let result = SchwarzschildF64GpuTraceBuilder::new(1.0).execute_camera(camera, width, height).unwrap();
        result.require_renderable().unwrap();
        let events = ton_geodesic::SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).localization(1.0e-11, 64).build().unwrap();
        let integration = ton_geodesic::AdaptiveRk4Config::builder().tolerances(1.0e-12, 1.0e-12).step_range(1.0e-8, 0.01).build().unwrap();
        let config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).events(events).integration(integration).maximum_steps(32768).build().unwrap();
        let mut captured = 0;
        let mut crossings = 0;
        let mut maximum_disk_error = 0.0f64;
        let mut maximum_terminal_time_error = 0.0f64;
        let mut maximum_terminal_position_error = 0.0f64;
        let mut maxima = [0.0f64; 3];
        for (index, record) in result.records.iter().enumerate() {
            let (x, y) = (index as u32 % width, index as u32 / width);
            let (horizontal, vertical) = (((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
                1.0 - (y as f64 + 0.5) / height as f64 * 2.0);
            let direction = camera.ray_direction(horizontal, vertical);
            let analytic_capture = ton_metric::schwarzschild_geodesic_analytic_captured(bh, camera.position, direction);
            assert_eq!(result.terminations[index], if analytic_capture { captured += 1; GpuTraceTermination::Captured } else { GpuTraceTermination::Escaped });
            let seed = SchwarzschildCartesianNullState::from_ray(1.0, camera.position, direction).unwrap();
            let reference = ton_geodesic::schwarzschild_cartesian_adaptive_trace(seed, config).unwrap();
            assert!(matches!(reference.termination, ton_geodesic::AdaptiveTraceTermination::Captured | ton_geodesic::AdaptiveTraceTermination::Escaped),
                "reference pixel {index}: termination={:?}, failure={:?}, steps={}", reference.termination, reference.failure, reference.accepted_steps);
            assert_eq!(record[13], reference.disk_events.len() as f64, "disk count pixel {index}");
            maximum_terminal_time_error = maximum_terminal_time_error.max((record[0] - reference.state.t).abs());
            let position_error = Vec3::new(record[1], record[2], record[3]).sub(reference.state.position).length();
            maximum_terminal_position_error = maximum_terminal_position_error.max(position_error);
            assert!(position_error < 1.0e-6, "terminal spatial error pixel {index}: {position_error}");
            assert!((record[0] - reference.state.t).abs() < 1.0e-5, "terminal coordinate time error pixel {index}: {}", (record[0] - reference.state.t).abs());
            crossings += reference.disk_events.len();
            if let Some(first) = reference.disk_events.first() {
                let error = (record[14] - first.event.state.position.length()).abs();
                maximum_disk_error = maximum_disk_error.max(error);
                assert!(error < 1.0e-6, "disk radius pixel {index}: {error}");
                assert!((record[15] - first.affine_parameter).abs() < 1.0e-6, "disk affine pixel {index}");
                assert!(record[24].abs() <= 1.0e-11);
            }
            for axis in 0..3 { maxima[axis] = maxima[axis].max(record[16 + axis]); }
        }
        eprintln!("F64 native frame: captured={captured}, escaped={}, crossings={crossings}, disk_error={maximum_disk_error}, invariant_maxima={maxima:?}, terminal_position_error={maximum_terminal_position_error}, terminal_time_error={maximum_terminal_time_error}, dispatches={}", result.records.len() - captured, result.executed_program_hashes.len());
        assert!(captured > 0 && captured < result.records.len() && crossings > 0);
        let critical_coordinates = (((9.5 / 16.0) * 2.0 - 1.0) * 16.0 / 9.0, 1.0 - (1.5 / 9.0) * 2.0);
        let mut backend = VulkanBackend::new().unwrap();
        let (initial, _) = crate::adaptive_gpu::initialize_f64_camera(&mut backend, camera, &[critical_coordinates], 1.0).unwrap();
        let mut record = [0.0; 29]; record[..8].copy_from_slice(&initial[0][..8]); record[9] = 0.02;
        record[19] = initial[0][4]; record[20] = initial[0][8];
        let program = ton_ucf_frontend::SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(256).build().unwrap();
        for _ in 0..128 {
            let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
            for (slot, value) in [1.0, 0.02, 2.0002, 80.0, 6.0, 20.0].into_iter().enumerate() { cpu.with_uniform(slot as u32, ton_ucf_frontend::Value::F64(value)); }
            cpu.with_buffer(1, record.into_iter().map(ton_ucf_frontend::Value::F64).collect());
            cpu.with_buffer(2, vec![ton_ucf_frontend::Value::F64(0.0); 29]); cpu.run_parallel(&program, 1).unwrap();
            record = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>().try_into().unwrap();
            if record[8] != 0.0 { break; }
        }
        assert_eq!(record[8], 2.0);
        let mut maximum_parity_error = 0.0f64;
        for (actual, expected) in result.records[25].iter().zip(record) {
            maximum_parity_error = maximum_parity_error.max((actual - expected).abs());
            assert!((actual - expected).abs() < 1.0e-8);
        }
        eprintln!("F64 complete critical trace CPU/GPU parity: maximum_error={maximum_parity_error}");
    }

    #[test]
    fn f64_native_large_frame_preserves_invariants_and_analytic_shadow() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let (width, height) = (128, 72);
        let result = SchwarzschildF64GpuTraceBuilder::new(1.0).maximum_step(0.01).execute_camera(camera, width, height).unwrap();
        result.require_renderable().unwrap();
        let mut captured = 0;
        let mut maxima = [0.0f64; 3];
        for (index, record) in result.records.iter().enumerate() {
            let (x, y) = (index as u32 % width, index as u32 / width);
            let direction = camera.ray_direction(((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * width as f64 / height as f64,
                1.0 - (y as f64 + 0.5) / height as f64 * 2.0);
            let expected = if ton_metric::schwarzschild_geodesic_analytic_captured(bh, camera.position, direction) {
                captured += 1; GpuTraceTermination::Captured
            } else { GpuTraceTermination::Escaped };
            assert_eq!(result.terminations[index], expected, "classification pixel {index}");
            for axis in 0..3 { maxima[axis] = maxima[axis].max(record[16 + axis]); }
        }
        eprintln!("F64 native 128x72: captured={captured}, escaped={}, invariant_maxima={maxima:?}, dispatches={}", result.records.len() - captured, result.executed_program_hashes.len());
        assert_eq!(captured, 2244);
    }
}

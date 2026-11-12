//! TON 物理参考与通用 UCF 程序的执行对照。

use std::collections::BTreeMap;

use ton_geodesic::{schwarzschild_geodesic_initial_state, schwarzschild_geodesic_rk4_step as reference_geodesic_rk4_step, schwarzschild_radial_rk4_step as reference_radial_rk4_step, RadialState, SphericalState};
use ton_metric::{schwarzschild_geodesic_analytic_captured, schwarzschild_geodesic_impact_parameter};
use ton_renderer::Framebuffer;
use ton_types::{BlackHole, Camera, Rgb, TonError, TonResult, Vec3};
use ton_ucf_frontend::{camera_ray_batch, compose_sky_batch, disk_shade_novikov_from_geodesic_batch, geodesic_capture_classify, geodesic_capture_classify_batch, schwarzschild_geodesic_rk4_chunk, schwarzschild_geodesic_rk4_chunk_with_disk, schwarzschild_geodesic_rk4_step, schwarzschild_geodesic_rk4_integrate, schwarzschild_impact_batch, schwarzschild_initial_state_batch, schwarzschild_radial_rk4_step, tone_map_aces_batch};
use ucf_backend_cpu::CpuBackend;
use ucf_backend_vulkan::VulkanBackend;

const VULKAN_GEODESIC_CHUNK_STEPS: u32 = 64;
use ucf_ir::{Dispatch, Graph, ResourceGraph, TaskGraph};
use ucf_scheduler::Backend;
use ucf_types::{Access, Domain, Objective, ParamValue, Priority, ResourceId, ResourceKind, ResourceNode, ShaderId, TaskId, TaskKind, TaskNode};
use ucf_program::Value;

fn program_param(program: &ucf_program::Program) -> TonResult<ParamValue> {
    let body = serde_json::to_string(program).map_err(|e| TonError::Domain(format!("encode UCF program: {e}")))?;
    Ok(ParamValue::Program { version: program.version, hash: ton_ucf_frontend::program_hash(program), body })
}

pub(crate) fn dispatch_task(
    id: u64,
    program: &ucf_program::Program,
    count: u32,
    output: ResourceId,
    output_slot: u32,
    input_slot: u32,
    input: Option<ResourceId>,
    uniforms: BTreeMap<String, ParamValue>,
) -> TonResult<TaskNode> {
    let mut params = uniforms;
    let hash = ton_ucf_frontend::program_hash(program);
    params.insert("program".into(), program_param(program)?);
    params.insert("program_hash_hex".into(), ParamValue::Str(format!("{hash:016x}")));
    params.insert("output".into(), ParamValue::I64(output.0 as i64));
    params
        .entry(format!("buffer_{output_slot}"))
        .or_insert(ParamValue::I64(output.0 as i64));
    if let Some(input) = input {
        params.insert("buffer_0".into(), ParamValue::I64(input.0 as i64));
        params.insert(format!("buffer_{input_slot}"), ParamValue::I64(input.0 as i64));
    }
    Ok(TaskNode { id: TaskId(id), kind: TaskKind::Dispatch, shader: ShaderId(id), params, dispatch: Dispatch::Explicit { x: count, y: 1, z: 1 }, objective: Objective::MaxUtilization, priority: Priority::Batch })
}

pub fn schwarzschild_capture_classify_ucf_cpu(bh: BlackHole, origin: Vec3, direction: Vec3) -> TonResult<bool> {
    if origin.length() <= bh.event_horizon_radius() { return Ok(true); }
    let b = schwarzschild_geodesic_impact_parameter(bh.mass, origin, direction).ok_or_else(|| TonError::Domain("invalid Schwarzschild ray state".into()))?;
    let program = geodesic_capture_classify();
    let report = CpuBackend::new().execute_program(&program, 1, [(0, Value::F64(b)), (1, Value::F64(bh.critical_impact_parameter()))]).map_err(TonError::Domain)?;
    Ok(report.instances.first().and_then(|r| r.value.as_f64().ok()).unwrap_or(0.0) >= 0.5)
}

pub fn schwarzschild_capture_classify_ucf_cpu_batch(bh: BlackHole, impacts: &[f64]) -> TonResult<Vec<bool>> {
    let program = geodesic_capture_classify_batch();
    let report = CpuBackend::new().execute_program_with_buffers(&program, impacts.len() as u32, [(0, Value::F32(bh.critical_impact_parameter() as f32))], [(0, impacts.iter().map(|v| Value::F32(*v as f32)).collect())]).map_err(TonError::Domain)?;
    Ok(report.instances.into_iter().map(|r| r.value.as_f64().map(|v| v >= 0.5).map_err(|_| TonError::Domain("UCF capture output type".into()))).collect::<TonResult<_>>()?)
}

pub fn schwarzschild_capture_classify_ucf_vulkan_batch(bh: BlackHole, impacts: &[f64]) -> TonResult<Vec<bool>> {
    if impacts.is_empty() { return Ok(Vec::new()); }
    let count = u32::try_from(impacts.len()).map_err(|_| TonError::Domain("impact batch exceeds u32 capacity".into()))?;
    let program = geodesic_capture_classify_batch();
    let task = dispatch_task(20, &program, count, ResourceId(10), 0, 0, Some(ResourceId(11)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(bh.critical_impact_parameter())), ("output".into(), ParamValue::I64(10))]))?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(10), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 4) }, ResourceNode { id: ResourceId(11), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 4) }] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan capture graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan capture graph: {e}")))?;
    backend.write_f32(ResourceId(11), &impacts.iter().map(|v| *v as f32).collect::<Vec<_>>()).map_err(|e| TonError::Domain(format!("upload impacts: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan capture graph: {e}")))?;
    Ok(backend.read_f32(ResourceId(10)).map_err(|e| TonError::Domain(format!("read capture output: {e}")))?.into_iter().map(|v| v >= 0.5).collect())
}

fn impact_input(position: Vec3, direction: Vec3) -> [f32; 6] {
    [position.x as f32, position.y as f32, position.z as f32, direction.x as f32, direction.y as f32, direction.z as f32]
}

fn camera_uniforms(camera: Camera) -> BTreeMap<String, ParamValue> {
    let values = [
        camera.forward.x, camera.forward.y, camera.forward.z,
        camera.right.x, camera.right.y, camera.right.z,
        camera.up.x, camera.up.y, camera.up.z,
        (camera.fov_y * 0.5).tan(),
    ];
    values.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value))).collect()
}

/// 由 UCF CPU Interpreter 从 UV 和相机参数生成归一化相机射线。
pub fn camera_ray_ucf_cpu_batch(camera: Camera, uv: &[(f64, f64)]) -> TonResult<Vec<Vec3>> {
    if uv.is_empty() { return Ok(Vec::new()); }
    let input = uv.iter().flat_map(|(u, v)| [Value::F32(*u as f32), Value::F32(*v as f32)]).collect::<Vec<_>>();
    let uniforms = camera_uniforms(camera).into_iter().filter_map(|(key, value)| {
        let slot = key.strip_prefix("uniform_")?.parse::<u32>().ok()?;
        match value {
            ParamValue::F64(value) => Some((slot, Value::F32(value as f32))),
            _ => None,
        }
    }).collect::<Vec<_>>();
    let output_seed = vec![Value::F32(0.0); uv.len() * 3];
    let report = CpuBackend::new().execute_program_with_buffers(&camera_ray_batch(), uv.len() as u32, uniforms, [(1, input), (2, output_seed)]).map_err(TonError::Domain)?;
    let output = report.buffers.get(&2).ok_or_else(|| TonError::Domain("UCF camera output buffer missing".into()))?;
    if output.len() < uv.len() * 3 { return Err(TonError::Domain("UCF camera output is short".into())); }
    output.chunks_exact(3).take(uv.len()).map(|v| Ok(Vec3::new(v[0].as_f64().map_err(|e| TonError::Domain(e.to_string()))?, v[1].as_f64().map_err(|e| TonError::Domain(e.to_string()))?, v[2].as_f64().map_err(|e| TonError::Domain(e.to_string()))?))).collect()
}

/// 由 UCF Vulkan backend 从 UV 和相机参数生成归一化相机射线。
pub fn camera_ray_ucf_vulkan_batch(camera: Camera, uv: &[(f64, f64)]) -> TonResult<Vec<Vec3>> {
    if uv.is_empty() { return Ok(Vec::new()); }
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    camera_ray_ucf_vulkan_batch_with_backend(&mut backend, camera, uv)
}

pub(crate) fn camera_ray_ucf_vulkan_batch_with_backend(
    backend: &mut VulkanBackend,
    camera: Camera,
    uv: &[(f64, f64)],
) -> TonResult<Vec<Vec3>> {
    if uv.is_empty() { return Ok(Vec::new()); }
    let count = u32::try_from(uv.len()).map_err(|_| TonError::Domain("UV batch exceeds u32 capacity".into()))?;
    let program = ton_ucf_frontend::camera_ray_batch();
    let mut params = camera_uniforms(camera);
    params.insert("output".into(), ParamValue::I64(10));
    params.insert("buffer_1".into(), ParamValue::I64(11));
    params.insert("buffer_2".into(), ParamValue::I64(12));
    let task = dispatch_task(22, &program, count, ResourceId(10), 2, 1, Some(ResourceId(11)), params)?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(10), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 4) }, ResourceNode { id: ResourceId(11), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 2 * 4) }, ResourceNode { id: ResourceId(12), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 3 * 4) }] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan camera graph: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan camera graph: {e}")))?;
    let input = uv.iter().flat_map(|(u, v)| [*u as f32, *v as f32]).collect::<Vec<_>>();
    backend.write_f32(ResourceId(11), &input).map_err(|e| TonError::Domain(format!("upload UV: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan camera graph: {e}")))?;
    let output = backend.read_f32(ResourceId(12)).map_err(|e| TonError::Domain(format!("read camera output: {e}")))?;
    if output.len() < uv.len() * 3 { return Err(TonError::Domain("Vulkan camera output is short".into())); }
    Ok(output.chunks_exact(3).take(uv.len()).map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).collect())
}

/// 由 UCF CPU Interpreter 从笛卡尔射线批量构造 Schwarzschild 球坐标初态。
///
/// 该函数只负责绑定通用程序和解码通用 buffer。物理参考结果仍由
/// `ton_geodesic::schwarzschild_geodesic_initial_state` 独立计算。
pub fn schwarzschild_initial_state_ucf_cpu_batch(
    mass: f64,
    critical_impact: f64,
    rays: &[(Vec3, Vec3)],
) -> TonResult<Vec<SphericalState>> {
    if rays.is_empty() {
        return Ok(Vec::new());
    }
    let input = rays
        .iter()
        .flat_map(|(origin, direction)| {
            [origin.x, origin.y, origin.z, direction.x, direction.y, direction.z]
                .into_iter()
                .map(|value| Value::F32(value as f32))
        })
        .collect::<Vec<_>>();
    let output_seed = vec![Value::F32(0.0); rays.len() * 10];
    let report = CpuBackend::new()
        .execute_program_with_buffers(
            &schwarzschild_initial_state_batch(),
            u32::try_from(rays.len()).map_err(|_| TonError::Domain("ray batch exceeds u32 capacity".into()))?,
            [
                (0, Value::F32(mass as f32)),
                (1, Value::F32(critical_impact as f32)),
            ],
            [(1, input), (2, output_seed)],
        )
        .map_err(TonError::Domain)?;
    let output = report
        .buffers
        .get(&2)
        .ok_or_else(|| TonError::Domain("UCF initial state output buffer missing".into()))?;
    if output.len() != rays.len() * 10 {
        return Err(TonError::Domain("UCF initial state output has invalid length".into()));
    }
    output
        .chunks_exact(10)
        .map(|values| {
            let scalar = |index: usize| {
                values[index]
                    .as_f64()
                    .map_err(|error| TonError::Domain(format!("UCF initial state value {index}: {error}")))
            };
            Ok(SphericalState {
                t: scalar(0)?,
                r: scalar(1)?,
                theta: scalar(2)?,
                phi: scalar(3)?,
                ut: scalar(4)?,
                ur: scalar(5)?,
                utheta: scalar(6)?,
                uphi: scalar(7)?,
            })
        })
        .collect()
}

/// 由原生 Vulkan UCF backend 批量构造 Schwarzschild 球坐标初态。
///
/// Vulkan 不支持该 Program 或资源绑定失败时直接返回错误，不回退 CPU。
pub fn schwarzschild_initial_state_ucf_vulkan_batch(
    mass: f64,
    critical_impact: f64,
    rays: &[(Vec3, Vec3)],
) -> TonResult<Vec<SphericalState>> {
    Ok(schwarzschild_initial_state_ucf_vulkan_batch_with_flags(mass, critical_impact, rays)?
        .0)
}

/// 执行 Vulkan 初态 Program，并保留后续测地线 chunk 所需的束缚入射标记。
pub fn schwarzschild_initial_state_ucf_vulkan_batch_with_flags(
    mass: f64,
    critical_impact: f64,
    rays: &[(Vec3, Vec3)],
) -> TonResult<(Vec<SphericalState>, Vec<bool>)> {
    if rays.is_empty() { return Ok((Vec::new(), Vec::new())); }
    let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create Vulkan initial state backend: {error}")))?;
    schwarzschild_initial_state_ucf_vulkan_batch_with_flags_backend(&mut backend, mass, critical_impact, rays)
}

fn schwarzschild_initial_state_ucf_vulkan_batch_with_flags_backend(
    backend: &mut VulkanBackend,
    mass: f64,
    critical_impact: f64,
    rays: &[(Vec3, Vec3)],
) -> TonResult<(Vec<SphericalState>, Vec<bool>)> {
    if rays.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let count = u32::try_from(rays.len()).map_err(|_| TonError::Domain("ray batch exceeds u32 capacity".into()))?;
    let program = schwarzschild_initial_state_batch();
    let input = rays
        .iter()
        .flat_map(|(origin, direction)| {
            [origin.x, origin.y, origin.z, direction.x, direction.y, direction.z]
                .into_iter()
                .map(|value| value as f32)
        })
        .collect::<Vec<_>>();
    let params = BTreeMap::from([
        ("uniform_0".into(), ParamValue::F64(mass)),
        ("uniform_1".into(), ParamValue::F64(critical_impact)),
        ("output".into(), ParamValue::I64(3)),
        ("buffer_1".into(), ParamValue::I64(1)),
        ("buffer_2".into(), ParamValue::I64(2)),
    ]);
    let task = dispatch_task(23, &program, count, ResourceId(3), 2, 1, Some(ResourceId(1)), params)?;
    let graph = Graph {
        resources: ResourceGraph {
            nodes: vec![
                ResourceNode { id: ResourceId(1), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 6 * 4) },
                ResourceNode { id: ResourceId(2), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 10 * 4) },
                ResourceNode { id: ResourceId(3), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 4) },
            ],
        },
        tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
    };
    graph.validate().map_err(|error| TonError::Domain(format!("Vulkan initial state graph: {error}")))?;
    backend.prepare(&graph).map_err(|error| TonError::Domain(format!("prepare Vulkan initial state graph: {error}")))?;
    backend.write_f32(ResourceId(1), &input).map_err(|error| TonError::Domain(format!("upload initial state rays: {error}")))?;
    backend.run_prepared(&graph).map_err(|error| TonError::Domain(format!("execute Vulkan initial state graph: {error}")))?;
    let output = backend.read_f32(ResourceId(2)).map_err(|error| TonError::Domain(format!("read Vulkan initial state output: {error}")))?;
    if output.len() != rays.len() * 10 {
        return Err(TonError::Domain("Vulkan initial state output has invalid length".into()));
    }
    let mut states = Vec::with_capacity(rays.len());
    let mut bound_flags = Vec::with_capacity(rays.len());
    for values in output.chunks_exact(10) {
        states.push(SphericalState {
            t: values[0] as f64,
            r: values[1] as f64,
            theta: values[2] as f64,
            phi: values[3] as f64,
            ut: values[4] as f64,
            ur: values[5] as f64,
            utheta: values[6] as f64,
            uphi: values[7] as f64,
        });
        bound_flags.push(values[9] >= 0.5);
    }
    Ok((states, bound_flags))
}

/// 由 UCF CPU Interpreter 批量计算 Schwarzschild 守恒冲击参数。
pub fn schwarzschild_impact_ucf_cpu_batch(mass: f64, rays: &[(Vec3, Vec3)]) -> TonResult<Vec<f64>> {
    let input = rays.iter().flat_map(|(position, direction)| impact_input(*position, *direction).into_iter().map(Value::F32)).collect::<Vec<_>>();
    let report = CpuBackend::new().execute_program_with_buffers(&schwarzschild_impact_batch(), rays.len() as u32, [(0, Value::F32(mass as f32))], [(0, input)]).map_err(TonError::Domain)?;
    report.instances.into_iter().map(|instance| instance.value.as_f64().map_err(|_| TonError::Domain("UCF impact output type".into()))).collect()
}

/// 由原生 Vulkan UCF backend 批量计算 Schwarzschild 守恒冲击参数。
pub fn schwarzschild_impact_ucf_vulkan_batch(mass: f64, rays: &[(Vec3, Vec3)]) -> TonResult<Vec<f64>> {
    if rays.is_empty() { return Ok(Vec::new()); }
    let count = u32::try_from(rays.len()).map_err(|_| TonError::Domain("ray batch exceeds u32 capacity".into()))?;
    let program = schwarzschild_impact_batch();
    let task = dispatch_task(21, &program, count, ResourceId(10), 0, 0, Some(ResourceId(11)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(mass))]))?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(10), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 4) }, ResourceNode { id: ResourceId(11), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 6 * 4) }] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan impact graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan impact graph: {e}")))?;
    let input = rays.iter().flat_map(|(position, direction)| impact_input(*position, *direction)).collect::<Vec<_>>();
    backend.write_f32(ResourceId(11), &input).map_err(|e| TonError::Domain(format!("upload rays: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan impact graph: {e}")))?;
    Ok(backend.read_f32(ResourceId(10)).map_err(|e| TonError::Domain(format!("read impact output: {e}")))?.into_iter().map(|value| value as f64).collect())
}

fn pixel_ray(x: u32, y: u32, w: u32, h: u32, camera: Camera) -> Vec3 {
    let aspect = w as f64 / h as f64;
    let u = ((x as f64 + 0.5) / w as f64 * 2.0 - 1.0) * aspect;
    let v = 1.0 - (y as f64 + 0.5) / h as f64 * 2.0;
    camera.ray_direction(u, v)
}

/// Schwarzschild 光线分类的 UCF 执行报告。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SchwarzschildClassificationReport {
    /// 参与分类的像素数量。
    pub total: u32,
    /// 解析参考判定为捕获的像素数量。
    pub analytic_captured: u32,
    /// UCF 程序判定为捕获的像素数量。
    pub captured: u32,
    /// UCF 程序判定为逃逸的像素数量。
    pub escaped: u32,
    /// UCF 程序耗尽步数预算的像素数量。
    pub budget_exhausted: u32,
    /// UCF 程序检测到非有限状态的像素数量。
    pub non_finite: u32,
    /// UCF 程序和解析参考分类一致的像素数量。
    pub agreement: u32,
}

fn classification_report(
    expected: &[bool],
    results: &[(SphericalState, u32)],
) -> SchwarzschildClassificationReport {
    let mut report = SchwarzschildClassificationReport {
        total: results.len() as u32,
        ..SchwarzschildClassificationReport::default()
    };
    for (expected_captured, (_, status)) in expected.iter().zip(results) {
        if *expected_captured {
            report.analytic_captured += 1;
        }
        match status {
            1 => report.captured += 1,
            2 => report.escaped += 1,
            3 => report.budget_exhausted += 1,
            4 => report.non_finite += 1,
            _ => {}
        }
        let actual_captured = *status == 1;
        if actual_captured == *expected_captured {
            report.agreement += 1;
        }
    }
    report
}

fn classification_rays(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
) -> (Vec<(Vec3, Vec3)>, Vec<bool>) {
    let mut rays = Vec::with_capacity((width * height) as usize);
    let mut expected = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let direction = pixel_ray(x, y, width, height, camera);
            expected.push(schwarzschild_geodesic_analytic_captured(
                bh,
                camera.position,
                direction,
            ));
            rays.push((camera.position, direction));
        }
    }
    (rays, expected)
}

/// 对小图执行 UCF CPU 测地线，并返回解析参考对照报告。
pub fn schwarzschild_classification_ucf_cpu(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    step: f64,
    escape_radius: f64,
    max_steps: u32,
) -> TonResult<SchwarzschildClassificationReport> {
    let (rays, expected) = classification_rays(bh, camera, width, height);
    let results = schwarzschild_geodesic_integrate_ucf_cpu_rays(
        bh.mass,
        &rays,
        step,
        escape_radius,
        max_steps,
    )?;
    Ok(classification_report(&expected, &results))
}

/// 对小图执行 UCF Vulkan 测地线，并返回解析参考对照报告。
pub fn schwarzschild_classification_ucf_vulkan(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    step: f64,
    escape_radius: f64,
    max_steps: u32,
) -> TonResult<SchwarzschildClassificationReport> {
    let (rays, expected) = classification_rays(bh, camera, width, height);
    let results = schwarzschild_geodesic_integrate_ucf_vulkan_rays(
        bh.mass,
        &rays,
        step,
        escape_radius,
        max_steps,
    )?;
    Ok(classification_report(&expected, &results))
}

pub fn schwarzschild_capture_silhouette_ucf_cpu(bh: BlackHole, camera: Camera, width: u32, height: u32) -> TonResult<Framebuffer> {
    let mut frame = Framebuffer::new(width, height);
    frame.encoding = ton_renderer::FramebufferEncoding::Display;
    for y in 0..height { for x in 0..width { let captured = schwarzschild_capture_classify_ucf_cpu(bh, camera.position, pixel_ray(x, y, width, height, camera))?; frame.set(x, y, if captured { Rgb::BLACK } else { Rgb::WHITE }); } }
    Ok(frame)
}

pub fn schwarzschild_capture_silhouette_ucf_vulkan(bh: BlackHole, camera: Camera, width: u32, height: u32) -> TonResult<Framebuffer> {
    let uv = (0..height).flat_map(|y| (0..width).map(move |x| {
        let aspect = width as f64 / height as f64;
        let u = ((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * aspect;
        let v = 1.0 - (y as f64 + 0.5) / height as f64 * 2.0;
        (u, v)
    })).collect::<Vec<_>>();
    let directions = camera_ray_ucf_vulkan_batch(camera, &uv)?;
    let rays = directions.into_iter().map(|direction| (camera.position, direction)).collect::<Vec<_>>();
    let impacts = schwarzschild_impact_ucf_vulkan_batch(bh.mass, &rays)?;
    let captured = schwarzschild_capture_classify_ucf_vulkan_batch(bh, &impacts)?;
    let mut frame = Framebuffer::new(width, height); frame.encoding = ton_renderer::FramebufferEncoding::Display;
    for (i, value) in captured.into_iter().enumerate() { frame.pixels[i] = if value { Rgb::BLACK } else { Rgb::WHITE }; }
    Ok(frame)
}

pub fn schwarzschild_radial_rk4_step_ucf_cpu(mass: f64, state: RadialState, step: f64) -> TonResult<RadialState> {
    let program = schwarzschild_radial_rk4_step();
    let values = vec![Value::F32(state.t as f32), Value::F32(state.r as f32), Value::F32(state.ut as f32), Value::F32(state.ur as f32)];
    let report = CpuBackend::new().execute_program_with_buffers(&program, 1, [(0, Value::F32(mass as f32)), (1, Value::F32(step as f32))], [(0, values)]).map_err(TonError::Domain)?;
    let out = report.buffers.get(&0).ok_or_else(|| TonError::Domain("UCF radial buffer missing".into()))?;
    let f = |i: usize| out.get(i).and_then(|v| v.as_f64().ok()).map(|v| v as f64).ok_or_else(|| TonError::Domain("UCF radial output type".into()));
    Ok(RadialState { t: f(0)?, r: f(1)?, ut: f(2)?, ur: f(3)? })
}

pub fn schwarzschild_radial_rk4_steps_ucf_cpu(mass: f64, mut state: RadialState, step: f64, count: u32) -> TonResult<RadialState> { for _ in 0..count { state = schwarzschild_radial_rk4_step_ucf_cpu(mass, state, step)?; } Ok(state) }

pub fn schwarzschild_radial_rk4_step_ucf_vulkan(mass: f64, state: RadialState, step: f64) -> TonResult<RadialState> {
    let program = schwarzschild_radial_rk4_step();
    let task = dispatch_task(30, &program, 1, ResourceId(10), 0, 0, Some(ResourceId(10)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(mass)), ("uniform_1".into(), ParamValue::F64(step))]))?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(10), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::ReadWrite, byte_size: Some(16) }] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan radial graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan radial graph: {e}")))?;
    backend.write_f32(ResourceId(10), &[state.t as f32, state.r as f32, state.ut as f32, state.ur as f32]).map_err(|e| TonError::Domain(format!("upload radial state: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan radial graph: {e}")))?;
    let out = backend.read_f32(ResourceId(10)).map_err(|e| TonError::Domain(format!("read radial state: {e}")))?;
    if out.len() < 4 { return Err(TonError::Domain("Vulkan radial output is short".into())); }
    Ok(RadialState { t: out[0] as f64, r: out[1] as f64, ut: out[2] as f64, ur: out[3] as f64 })
}

pub fn schwarzschild_radial_rk4_steps_ucf_vulkan(mass: f64, mut state: RadialState, step: f64, count: u32) -> TonResult<RadialState> { for _ in 0..count { state = schwarzschild_radial_rk4_step_ucf_vulkan(mass, state, step)?; } Ok(state) }

/// 由 UCF CPU Interpreter 推进一个完整球坐标 Schwarzschild RK4 步。
pub fn schwarzschild_geodesic_rk4_step_ucf_cpu(mass: f64, state: SphericalState, step: f64) -> TonResult<SphericalState> {
    let values = [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi].into_iter().map(|value| Value::F32(value as f32)).collect::<Vec<_>>();
    let report = CpuBackend::new().execute_program_with_buffers(&schwarzschild_geodesic_rk4_step(), 1, [(0, Value::F32(mass as f32)), (1, Value::F32(step as f32))], [(0, values)]).map_err(TonError::Domain)?;
    let output = report.buffers.get(&0).ok_or_else(|| TonError::Domain("UCF geodesic state buffer missing".into()))?;
    if output.len() < 8 { return Err(TonError::Domain("UCF geodesic state output is short".into())); }
    let value = |index: usize| output[index].as_f64().map_err(|error| TonError::Domain(error.to_string()));
    Ok(SphericalState { t: value(0)?, r: value(1)?, theta: value(2)?, phi: value(3)?, ut: value(4)?, ur: value(5)?, utheta: value(6)?, uphi: value(7)? })
}

/// 由单个 UCF 程序在 CPU 解释器中完成有界多步积分。
pub fn schwarzschild_geodesic_integrate_ucf_cpu(mass: f64, state: SphericalState, step: f64, escape_radius: f64, max_steps: u32) -> TonResult<(SphericalState, u32)> {
    let program = schwarzschild_geodesic_rk4_integrate(max_steps);
    let values = [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi, 0.0, 0.0]
        .into_iter().map(|value| Value::F32(value as f32)).collect::<Vec<_>>();
    let report = CpuBackend::new()
        .execute_program_with_buffers(&program, 1, [(0, Value::F32(mass as f32)), (1, Value::F32(step as f32)), (2, Value::F32(escape_radius as f32))], [(0, values)])
        .map_err(TonError::Domain)?;
    let output = report.buffers.get(&0).ok_or_else(|| TonError::Domain("UCF integrated geodesic state buffer missing".into()))?;
    if output.len() < 10 { return Err(TonError::Domain("UCF integrated geodesic state output is short".into())); }
    let value = |index: usize| output[index].as_f64().map_err(|error| TonError::Domain(error.to_string()));
    let status = value(8)? as u32;
    Ok((SphericalState { t: value(0)?, r: value(1)?, theta: value(2)?, phi: value(3)?, ut: value(4)?, ur: value(5)?, utheta: value(6)?, uphi: value(7)? }, status))
}

/// 由一次 CPU UCF dispatch 并行推进多条相机射线。
pub fn schwarzschild_geodesic_integrate_ucf_cpu_rays(mass: f64, rays: &[(Vec3, Vec3)], step: f64, escape_radius: f64, max_steps: u32) -> TonResult<Vec<(SphericalState, u32)>> {
    if rays.is_empty() { return Ok(Vec::new()); }
    let bound_flags = rays.iter().map(|(origin, direction)| schwarzschild_geodesic_impact_parameter(mass, *origin, *direction)
        .map(|impact| impact < BlackHole::schwarzschild(mass).critical_impact_parameter()).unwrap_or(false)).collect::<Vec<_>>();
    let mut states = schwarzschild_initial_state_ucf_cpu_batch(
        mass,
        BlackHole::schwarzschild(mass).critical_impact_parameter(),
        rays,
    )?;
    let chunk = schwarzschild_geodesic_rk4_chunk(16);
    let backend = CpuBackend::new();
    let mut statuses = vec![0u32; states.len()];
    for _ in 0..max_steps.div_ceil(16) {
        let input = states.iter().zip(bound_flags.iter()).flat_map(|(state, bound)| {
            [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi, 0.0, *bound as u8 as f64]
                .into_iter().map(|value| Value::F32(value as f32))
        }).collect::<Vec<_>>();
        let report = backend.execute_program_with_buffers(&chunk, states.len() as u32, [(0, Value::F32(mass as f32)), (1, Value::F32(step as f32)), (2, Value::F32(escape_radius as f32))], [(0, input)]).map_err(TonError::Domain)?;
        let output = report.buffers.get(&0).ok_or_else(|| TonError::Domain("CPU integrated geodesic chunk output is missing".into()))?;
        if output.len() < states.len() * 10 { return Err(TonError::Domain("CPU integrated geodesic chunk output is short".into())); }
        let mut next_states = Vec::with_capacity(states.len());
        let mut next_statuses = Vec::with_capacity(states.len());
        for index in 0..states.len() {
            let base = index * 10;
            let value = |offset: usize| output[base + offset].as_f64().map_err(|error| TonError::Domain(error.to_string()));
            next_states.push(SphericalState { t: value(0)?, r: value(1)?, theta: value(2)?, phi: value(3)?, ut: value(4)?, ur: value(5)?, utheta: value(6)?, uphi: value(7)? });
            next_statuses.push(value(8)? as u32);
        }
        states = next_states;
        statuses = next_statuses;
        if statuses.iter().all(|status| *status != 0) { break; }
    }
    for status in &mut statuses {
        if *status == 0 { *status = 3; }
    }
    Ok(states.into_iter().zip(statuses).collect())
}

/// 由原生 Vulkan UCF backend 在单个 dispatch 中完成有界多步积分。
pub fn schwarzschild_geodesic_integrate_ucf_vulkan(mass: f64, state: SphericalState, step: f64, escape_radius: f64, max_steps: u32) -> TonResult<(SphericalState, u32)> {
    let program = schwarzschild_geodesic_rk4_integrate(max_steps);
    let task = dispatch_task(32, &program, 1, ResourceId(13), 0, 0, Some(ResourceId(14)), BTreeMap::from([
        ("uniform_0".into(), ParamValue::F64(mass)),
        ("uniform_1".into(), ParamValue::F64(step)),
        ("uniform_2".into(), ParamValue::F64(escape_radius)),
    ]))?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![
        ResourceNode { id: ResourceId(13), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(36) },
        ResourceNode { id: ResourceId(14), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::ReadWrite, byte_size: Some(40) },
    ] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan integrated geodesic graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan integrated geodesic graph: {e}")))?;
    backend.write_f32(ResourceId(14), &[state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi, 0.0, 0.0].into_iter().map(|value| value as f32).collect::<Vec<_>>()).map_err(|e| TonError::Domain(format!("upload integrated geodesic state: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan integrated geodesic graph: {e}")))?;
    let output = backend.read_f32(ResourceId(14)).map_err(|e| TonError::Domain(format!("read integrated geodesic state: {e}")))?;
    if output.len() < 10 { return Err(TonError::Domain("Vulkan integrated geodesic state output is short".into())); }
    Ok((SphericalState { t: output[0] as f64, r: output[1] as f64, theta: output[2] as f64, phi: output[3] as f64, ut: output[4] as f64, ur: output[5] as f64, utheta: output[6] as f64, uphi: output[7] as f64 }, output[8] as u32))
}

/// 由一次 Vulkan dispatch 并行推进多个实例，每个实例占九个 `f32`。
pub fn schwarzschild_geodesic_integrate_ucf_vulkan_batch(mass: f64, states: &[SphericalState], step: f64, escape_radius: f64, max_steps: u32) -> TonResult<Vec<(SphericalState, u32)>> {
    let program = schwarzschild_geodesic_rk4_integrate(max_steps);
    schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_program(mass, states, &vec![false; states.len()], step, escape_radius, &program)
}

fn schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_program(mass: f64, states: &[SphericalState], bound_flags: &[bool], step: f64, escape_radius: f64, program: &ucf_program::Program) -> TonResult<Vec<(SphericalState, u32)>> {
    if states.is_empty() { return Ok(Vec::new()); }
    let count = u32::try_from(states.len()).map_err(|_| TonError::Domain("geodesic batch exceeds u32 capacity".into()))?;
    let task = dispatch_task(33, &program, count, ResourceId(15), 0, 0, Some(ResourceId(16)), BTreeMap::from([
        ("uniform_0".into(), ParamValue::F64(mass)),
        ("uniform_1".into(), ParamValue::F64(step)),
        ("uniform_2".into(), ParamValue::F64(escape_radius)),
    ]))?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![
        ResourceNode { id: ResourceId(15), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 4) },
        ResourceNode { id: ResourceId(16), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::ReadWrite, byte_size: Some(u64::from(count) * 10 * 4) },
    ] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan integrated geodesic batch graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan integrated geodesic batch graph: {e}")))?;
    let input = states.iter().zip(bound_flags.iter()).flat_map(|(state, bound)| [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi, 0.0, *bound as u8 as f64]).map(|value| value as f32).collect::<Vec<_>>();
    backend.write_f32(ResourceId(16), &input).map_err(|e| TonError::Domain(format!("upload integrated geodesic batch: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan integrated geodesic batch: {e}")))?;
    let output = backend.read_f32(ResourceId(16)).map_err(|e| TonError::Domain(format!("read integrated geodesic batch: {e}")))?;
    if output.len() < states.len() * 10 { return Err(TonError::Domain("Vulkan integrated geodesic batch output is short".into())); }
    Ok(states.iter().enumerate().map(|(index, _)| {
        let base = index * 10;
        (SphericalState { t: output[base] as f64, r: output[base + 1] as f64, theta: output[base + 2] as f64, phi: output[base + 3] as f64, ut: output[base + 4] as f64, ur: output[base + 5] as f64, utheta: output[base + 6] as f64, uphi: output[base + 7] as f64 }, output[base + 8] as u32)
    }).collect())
}

/// 由 Vulkan UCF backend 执行带盘面交点元数据的批量 chunk。
///
/// 每个实例使用十四个 `f32`，尾部为 `[disk_hit, disk_radius, disk_doppler, reserved]`。
pub fn schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk(
    mass: f64,
    states: &[SphericalState],
    bound_flags: &[bool],
    disk_hits: &[bool],
    disk_radii: &[f64],
    disk_dopplers: &[f64],
    inner_radius: f64,
    outer_radius: f64,
    step: f64,
    escape_radius: f64,
    max_steps: u32,
) -> TonResult<Vec<(SphericalState, u32, bool, f64, f64)>> {
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan disk geodesic backend: {e}")))?;
    schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk_backend(
        &mut backend,
        mass,
        states,
        &vec![0; states.len()],
        bound_flags,
        disk_hits,
        disk_radii,
        disk_dopplers,
        inner_radius,
        outer_radius,
        step,
        escape_radius,
        max_steps,
    )
}

fn schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk_backend(
    backend: &mut VulkanBackend,
    mass: f64,
    states: &[SphericalState],
    statuses: &[u32],
    bound_flags: &[bool],
    disk_hits: &[bool],
    disk_radii: &[f64],
    disk_dopplers: &[f64],
    inner_radius: f64,
    outer_radius: f64,
    step: f64,
    escape_radius: f64,
    max_steps: u32,
) -> TonResult<Vec<(SphericalState, u32, bool, f64, f64)>> {
    if states.is_empty() {
        return Ok(Vec::new());
    }
    if states.len() != statuses.len()
        || states.len() != bound_flags.len()
        || states.len() != disk_hits.len()
        || states.len() != disk_radii.len()
        || states.len() != disk_dopplers.len()
    {
        return Err(TonError::Domain("Vulkan disk geodesic metadata differs in length".into()));
    }
    let count = u32::try_from(states.len()).map_err(|_| TonError::Domain("geodesic disk batch exceeds u32 capacity".into()))?;
    let program = schwarzschild_geodesic_rk4_chunk_with_disk(max_steps);
    let task = dispatch_task(34, &program, count, ResourceId(17), 0, 0, Some(ResourceId(18)), BTreeMap::from([
        ("uniform_0".into(), ParamValue::F64(mass)),
        ("uniform_1".into(), ParamValue::F64(step)),
        ("uniform_2".into(), ParamValue::F64(escape_radius)),
        ("uniform_3".into(), ParamValue::F64(inner_radius)),
        ("uniform_4".into(), ParamValue::F64(outer_radius)),
    ]))?;
    let graph = Graph {
        resources: ResourceGraph {
            nodes: vec![
                ResourceNode { id: ResourceId(17), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 4) },
                ResourceNode { id: ResourceId(18), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::ReadWrite, byte_size: Some(u64::from(count) * 14 * 4) },
            ],
        },
        tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
    };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan disk geodesic graph: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan disk geodesic graph: {e}")))?;
    let input = states.iter().enumerate().flat_map(|(index, state)| [
        state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi,
        statuses[index] as f64, bound_flags[index] as u8 as f64,
        disk_hits[index] as u8 as f64, disk_radii[index], disk_dopplers[index], 0.0,
    ]).map(|value| value as f32).collect::<Vec<_>>();
    backend.write_f32(ResourceId(18), &input).map_err(|e| TonError::Domain(format!("upload Vulkan disk geodesic state: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan disk geodesic graph: {e}")))?;
    let output = backend.read_f32(ResourceId(18)).map_err(|e| TonError::Domain(format!("read Vulkan disk geodesic state: {e}")))?;
    if output.len() < states.len() * 14 {
        return Err(TonError::Domain("Vulkan disk geodesic output is short".into()));
    }
    Ok(states.iter().enumerate().map(|(index, _)| {
        let base = index * 14;
        (
            SphericalState { t: output[base] as f64, r: output[base + 1] as f64, theta: output[base + 2] as f64, phi: output[base + 3] as f64, ut: output[base + 4] as f64, ur: output[base + 5] as f64, utheta: output[base + 6] as f64, uphi: output[base + 7] as f64 },
            output[base + 8] as u32,
            output[base + 10] >= 0.5,
            output[base + 11] as f64,
            output[base + 12] as f64,
        )
    }).collect())
}

/// 由 Vulkan UCF backend 从扩展测地线状态资源生成批量盘面 radiance。
///
/// `state_buffer` 每个实例必须包含十四个 `f32`，盘面 Program 只读取其中的
/// `disk_radius` 与 `disk_doppler` 字段。输入输出均通过 UCF 资源提交。
pub fn disk_radiance_ucf_vulkan_from_geodesic_state(
    state_buffer: &[f32],
    count: u32,
    inner_radius: f64,
    outer_radius: f64,
    mass: f64,
    accretion_rate: f64,
) -> TonResult<Vec<f32>> {
    let expected = usize::try_from(count)
        .map_err(|_| TonError::Domain("Vulkan disk count exceeds usize capacity".into()))?
        .saturating_mul(14);
    if state_buffer.len() != expected {
        return Err(TonError::Domain("Vulkan disk state buffer has invalid length".into()));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let program = disk_shade_novikov_from_geodesic_batch();
    let task = dispatch_task(35, &program, count, ResourceId(19), 2, 1, Some(ResourceId(18)), BTreeMap::from([
        ("uniform_0".into(), ParamValue::F64(inner_radius)),
        ("uniform_1".into(), ParamValue::F64(outer_radius)),
        ("uniform_2".into(), ParamValue::F64(mass)),
        ("uniform_3".into(), ParamValue::F64(accretion_rate)),
    ]))?;
    let graph = Graph {
        resources: ResourceGraph {
            nodes: vec![
                ResourceNode { id: ResourceId(18), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 14 * 4) },
                ResourceNode { id: ResourceId(19), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 3 * 4) },
            ],
        },
        tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
    };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan disk radiance graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan disk radiance backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan disk radiance graph: {e}")))?;
    backend.write_f32(ResourceId(18), state_buffer).map_err(|e| TonError::Domain(format!("upload Vulkan disk state: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan disk radiance graph: {e}")))?;
    let output = backend.read_f32(ResourceId(19)).map_err(|e| TonError::Domain(format!("read Vulkan disk radiance: {e}")))?;
    if output.len() != usize::try_from(count).unwrap_or(0).saturating_mul(3) {
        return Err(TonError::Domain("Vulkan disk radiance output has invalid length".into()));
    }
    Ok(output)
}

/// 执行完整的 Vulkan UCF Schwarzschild 盘面 radiance 链。
///
/// 相机、初态、扩展 geodesic chunk 和盘面着色均由 UCF Program 完成。该函数
/// 只返回线性 radiance，不负责显示空间 tone map，也不在失败时切换 CPU。
pub fn schwarzschild_disk_radiance_ucf_vulkan(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    inner_radius: f64,
    outer_radius: f64,
    accretion_rate: f64,
    step: f64,
    escape_radius: f64,
    max_steps: u32,
) -> TonResult<Vec<f32>> {
    let uv = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let aspect = width.max(1) as f64 / height.max(1) as f64;
                (
                    ((x as f64 + 0.5) / width.max(1) as f64 * 2.0 - 1.0) * aspect,
                    1.0 - (y as f64 + 0.5) / height.max(1) as f64 * 2.0,
                )
            })
        })
        .collect::<Vec<_>>();
    if uv.is_empty() {
        return Ok(Vec::new());
    }
    let directions = camera_ray_ucf_vulkan_batch(camera, &uv)?;
    let rays = directions
        .into_iter()
        .map(|direction| (camera.position, direction))
        .collect::<Vec<_>>();
    let (mut states, bound_flags) = schwarzschild_initial_state_ucf_vulkan_batch_with_flags(
        bh.mass,
        bh.critical_impact_parameter(),
        &rays,
    )?;
    let mut disk_hits = vec![false; states.len()];
    let mut disk_radii = vec![0.0; states.len()];
    let mut disk_dopplers = vec![1.0; states.len()];
    let mut statuses = vec![0u32; states.len()];
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan disk trace backend: {e}")))?;
    for _ in 0..max_steps.div_ceil(VULKAN_GEODESIC_CHUNK_STEPS).max(1) {
        let results = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk_backend(
            &mut backend,
            bh.mass,
            &states,
            &statuses,
            &bound_flags,
            &disk_hits,
            &disk_radii,
            &disk_dopplers,
            inner_radius,
            outer_radius,
            step,
            escape_radius,
            VULKAN_GEODESIC_CHUNK_STEPS,
        )?;
        for (index, (state, status, hit, radius, doppler)) in results.into_iter().enumerate() {
            states[index] = state;
            statuses[index] = status;
            disk_hits[index] = hit;
            disk_radii[index] = radius;
            disk_dopplers[index] = doppler;
        }
        if statuses.iter().all(|status| *status != 0) {
            break;
        }
    }
    for status in &mut statuses {
        if *status == 0 {
            *status = 3;
        }
    }
    let state_buffer = states
        .iter()
        .enumerate()
        .flat_map(|(index, state)| {
            [
                state.t as f32,
                state.r as f32,
                state.theta as f32,
                state.phi as f32,
                state.ut as f32,
                state.ur as f32,
                state.utheta as f32,
                state.uphi as f32,
                statuses[index] as f32,
                bound_flags[index] as u8 as f32,
                disk_hits[index] as u8 as f32,
                disk_radii[index] as f32,
                disk_dopplers[index] as f32,
                0.0,
            ]
        })
        .collect::<Vec<_>>();
    disk_radiance_ucf_vulkan_from_geodesic_state(
        &state_buffer,
        states.len() as u32,
        inner_radius,
        outer_radius,
        bh.mass,
        accretion_rate,
    )
}

/// 由 Vulkan UCF backend 执行批量 ACES tone map。
pub fn tone_map_ucf_vulkan_batch(hdr: &[f32], count: u32) -> TonResult<Vec<f32>> {
    let expected = usize::try_from(count)
        .map_err(|_| TonError::Domain("Vulkan tone-map count exceeds usize capacity".into()))?
        .saturating_mul(3);
    if hdr.len() != expected {
        return Err(TonError::Domain("Vulkan tone-map input has invalid length".into()));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let program = tone_map_aces_batch();
    let task = dispatch_task(36, &program, count, ResourceId(20), 2, 1, Some(ResourceId(21)), BTreeMap::new())?;
    let graph = Graph {
        resources: ResourceGraph {
            nodes: vec![
                ResourceNode { id: ResourceId(21), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 3 * 4) },
                ResourceNode { id: ResourceId(20), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 3 * 4) },
            ],
        },
        tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
    };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan tone-map graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan tone-map backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan tone-map graph: {e}")))?;
    backend.write_f32(ResourceId(21), hdr).map_err(|e| TonError::Domain(format!("upload Vulkan HDR buffer: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan tone-map graph: {e}")))?;
    let output = backend.read_f32(ResourceId(20)).map_err(|e| TonError::Domain(format!("read Vulkan display buffer: {e}")))?;
    if output.len() != expected {
        return Err(TonError::Domain("Vulkan tone-map output has invalid length".into()));
    }
    Ok(output)
}

/// 由 Vulkan UCF backend 合成逃逸天空、捕获黑洞和盘面 radiance。
pub fn compose_sky_ucf_vulkan_batch(
    directions: &[f32],
    states: &[f32],
    radiance: &[f32],
    count: u32,
) -> TonResult<Vec<f32>> {
    let count_usize = usize::try_from(count).map_err(|_| TonError::Domain("Vulkan sky count exceeds usize capacity".into()))?;
    if directions.len() != count_usize * 3 || states.len() != count_usize * 14 || radiance.len() != count_usize * 3 {
        return Err(TonError::Domain("Vulkan sky composite inputs have invalid lengths".into()));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let program = compose_sky_batch();
    let mut params = BTreeMap::new();
    params.insert("buffer_0".into(), ParamValue::I64(22));
    params.insert("buffer_1".into(), ParamValue::I64(22));
    params.insert("buffer_2".into(), ParamValue::I64(23));
    params.insert("buffer_3".into(), ParamValue::I64(24));
    params.insert("buffer_4".into(), ParamValue::I64(25));
    params.insert("output".into(), ParamValue::I64(25));
    params.insert("program".into(), program_param(&program)?);
    let task = TaskNode {
        id: TaskId(37),
        kind: TaskKind::Dispatch,
        shader: ShaderId(37),
        params,
        dispatch: Dispatch::Explicit { x: count, y: 1, z: 1 },
        objective: Objective::MaxUtilization,
        priority: Priority::Batch,
    };
    let graph = Graph {
        resources: ResourceGraph {
            nodes: vec![
                ResourceNode { id: ResourceId(22), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 3 * 4) },
                ResourceNode { id: ResourceId(23), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 14 * 4) },
                ResourceNode { id: ResourceId(24), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 3 * 4) },
                ResourceNode { id: ResourceId(25), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 3 * 4) },
            ],
        },
        tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
    };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan sky composite graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan sky composite backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan sky composite graph: {e}")))?;
    backend.write_f32(ResourceId(22), directions).map_err(|e| TonError::Domain(format!("upload Vulkan directions: {e}")))?;
    backend.write_f32(ResourceId(23), states).map_err(|e| TonError::Domain(format!("upload Vulkan states: {e}")))?;
    backend.write_f32(ResourceId(24), radiance).map_err(|e| TonError::Domain(format!("upload Vulkan radiance: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan sky composite graph: {e}")))?;
    let output = backend.read_f32(ResourceId(25)).map_err(|e| TonError::Domain(format!("read Vulkan composite output: {e}")))?;
    if output.len() != count_usize * 3 {
        return Err(TonError::Domain("Vulkan sky composite output has invalid length".into()));
    }
    Ok(output)
}

/// 执行 Schwarzschild UCF Vulkan Beauty 链并读回显示帧。
///
/// 相机射线、初态、测地线、盘面辐射、天空合成和色调映射均通过 UCF
/// Program 执行。该入口不接受预算耗尽或非有限实例，也不在 GPU 失败时
/// 切换到 CPU 或原生 TON 渲染器。
pub fn schwarzschild_beauty_ucf_vulkan(
    bh: BlackHole,
    camera: Camera,
    width: u32,
    height: u32,
    inner_radius: f64,
    outer_radius: f64,
    accretion_rate: f64,
    step: f64,
    escape_radius: f64,
    max_steps: u32,
) -> TonResult<Framebuffer> {
    if width == 0 || height == 0 {
        return Err(TonError::Domain("Vulkan Beauty resolution must be non-zero".into()));
    }
    if max_steps == 0 {
        return Err(TonError::Domain("Vulkan Beauty step budget must be non-zero".into()));
    }
    let uv = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let aspect = width as f64 / height as f64;
                (
                    ((x as f64 + 0.5) / width as f64 * 2.0 - 1.0) * aspect,
                    1.0 - (y as f64 + 0.5) / height as f64 * 2.0,
                )
            })
        })
        .collect::<Vec<_>>();
    let count = u32::try_from(uv.len())
        .map_err(|_| TonError::Domain("Vulkan Beauty pixel count exceeds u32 capacity".into()))?;
    let mut beauty_backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan Beauty backend: {e}")))?;
    let directions = camera_ray_ucf_vulkan_batch_with_backend(&mut beauty_backend, camera, &uv)?;
    let rays = directions
        .iter()
        .copied()
        .map(|direction| (camera.position, direction))
        .collect::<Vec<_>>();
    let (mut states, bound_flags) = schwarzschild_initial_state_ucf_vulkan_batch_with_flags_backend(
        &mut beauty_backend,
        bh.mass,
        bh.critical_impact_parameter(),
        &rays,
    )?;
    let mut disk_hits = vec![false; states.len()];
    let mut disk_radii = vec![0.0; states.len()];
    let mut disk_dopplers = vec![1.0; states.len()];
    let mut statuses = vec![0u32; states.len()];
    let mut completed_steps = 0u32;
    while completed_steps < max_steps && statuses.iter().any(|status| *status == 0) {
        let chunk_steps = (max_steps - completed_steps).min(VULKAN_GEODESIC_CHUNK_STEPS);
        let results = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk_backend(
            &mut beauty_backend,
            bh.mass,
            &states,
            &statuses,
            &bound_flags,
            &disk_hits,
            &disk_radii,
            &disk_dopplers,
            inner_radius,
            outer_radius,
            step,
            escape_radius,
            chunk_steps,
        )?;
        for (index, (state, status, hit, radius, doppler)) in results.into_iter().enumerate() {
            states[index] = state;
            statuses[index] = status;
            disk_hits[index] = hit;
            disk_radii[index] = radius;
            disk_dopplers[index] = doppler;
        }
        completed_steps += chunk_steps;
    }
    for status in &mut statuses {
        if *status == 0 {
            *status = 3;
        }
    }
    if let Some((index, status)) = statuses
        .iter()
        .enumerate()
        .find(|(_, status)| **status == 3 || **status == 4)
    {
        let mut histogram = BTreeMap::<u32, usize>::new();
        for status in &statuses {
            *histogram.entry(*status).or_default() += 1;
        }
        let state = states[index];
        return Err(TonError::Domain(format!(
            "Vulkan Beauty contains non-beauty geodesic instance {index} with status {status}. \
             histogram={histogram:?}, steps={completed_steps}/{max_steps}, step={step}, escape_radius={escape_radius}, \
             r={}, ur={}, theta={}, ut={}, bound_infall={}, disk_hit={}",
            state.r, state.ur, state.theta, state.ut, bound_flags[index], disk_hits[index],
        )));
    }
    let state_buffer = states
        .iter()
        .enumerate()
        .flat_map(|(index, state)| {
            [
                state.t as f32,
                state.r as f32,
                state.theta as f32,
                state.phi as f32,
                state.ut as f32,
                state.ur as f32,
                state.utheta as f32,
                state.uphi as f32,
                statuses[index] as f32,
                bound_flags[index] as u8 as f32,
                disk_hits[index] as u8 as f32,
                disk_radii[index] as f32,
                disk_dopplers[index] as f32,
                0.0,
            ]
        })
        .collect::<Vec<_>>();
    let direction_buffer = directions
        .iter()
        .flat_map(|direction| [direction.x as f32, direction.y as f32, direction.z as f32])
        .collect::<Vec<_>>();
    let post_graph = beauty_post_graph(count, inner_radius, outer_radius, bh.mass, accretion_rate)?;
    beauty_backend.prepare(&post_graph).map_err(|e| TonError::Domain(format!("prepare Vulkan Beauty post graph: {e}")))?;
    beauty_backend.write_f32(ResourceId(18), &state_buffer).map_err(|e| TonError::Domain(format!("upload Vulkan Beauty terminal states: {e}")))?;
    beauty_backend.write_f32(ResourceId(22), &direction_buffer).map_err(|e| TonError::Domain(format!("upload Vulkan Beauty directions: {e}")))?;
    beauty_backend.run_prepared(&post_graph).map_err(|e| TonError::Domain(format!("execute Vulkan Beauty post graph: {e}")))?;
    let display = beauty_backend.read_f32(ResourceId(20)).map_err(|e| TonError::Domain(format!("read Vulkan Beauty display: {e}")))?;
    let expected = uv.len().saturating_mul(3);
    if display.len() != expected {
        return Err(TonError::Domain("Vulkan Beauty display buffer has invalid length".into()));
    }
    if display.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
        return Err(TonError::Domain("Vulkan Beauty display buffer is not finite display data".into()));
    }
    let pixels = display
        .chunks_exact(3)
        .map(|rgb| Rgb::new(rgb[0] as f64, rgb[1] as f64, rgb[2] as f64))
        .collect::<Vec<_>>();
    Ok(Framebuffer {
        width,
        height,
        pixels,
        encoding: ton_renderer::FramebufferEncoding::Display,
    })
}

fn beauty_post_graph(count: u32, inner_radius: f64, outer_radius: f64, mass: f64, accretion_rate: f64) -> TonResult<Graph> {
    let emission_binding = ton_renderer::beauty_disk_shade_kernel();
    let emission = dispatch_task(
        35, &emission_binding.program, count, ResourceId(19), 2, 1, Some(ResourceId(18)),
        BTreeMap::from([
            ("uniform_0".into(), ParamValue::F64(inner_radius)),
            ("uniform_1".into(), ParamValue::F64(outer_radius)),
            ("uniform_2".into(), ParamValue::F64(mass)),
            ("uniform_3".into(), ParamValue::F64(accretion_rate)),
        ]),
    )?;
    let composite_binding = ton_renderer::beauty_sky_composite_kernel();
    let composite = dispatch_task(
        37, &composite_binding.program, count, ResourceId(25), 4, 1, Some(ResourceId(22)),
        BTreeMap::from([
            ("buffer_2".into(), ParamValue::I64(18)),
            ("buffer_3".into(), ParamValue::I64(19)),
        ]),
    )?;
    let tone_map_binding = ton_renderer::beauty_tone_map_kernel();
    let tone_map = dispatch_task(36, &tone_map_binding.program, count, ResourceId(20), 2, 1, Some(ResourceId(25)), BTreeMap::new())?;
    let mut graph = Graph {
        resources: ResourceGraph {
            nodes: [(18, 14, Access::Read), (22, 3, Access::Read), (19, 3, Access::ReadWrite), (25, 3, Access::ReadWrite), (20, 3, Access::Write)]
                .into_iter()
                .map(|(id, stride, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access,
                    byte_size: Some(u64::from(count) * stride * 4),
                })
                .collect(),
        },
        tasks: TaskGraph { nodes: vec![emission, composite, tone_map], edges: ucf_ir::chain_edges(&[35, 37, 36]) },
    };
    ton_renderer::attach_binding(&mut graph, 35, &emission_binding);
    ton_renderer::attach_binding(&mut graph, 37, &composite_binding);
    ton_renderer::attach_binding(&mut graph, 36, &tone_map_binding);
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan Beauty post graph: {e}")))?;
    ton_renderer::validate_program_bindings(&graph)
        .map_err(|error| TonError::Domain(format!("Vulkan Beauty program binding contract: {error}")))?;
    Ok(graph)
}

/// 从相机射线开始，依次执行 Vulkan 初态 Program 与 Vulkan 有界测地线 chunk。
///
/// 初态和终止状态都来自 UCF GPU buffer。该入口不调用 TON 原生积分器，
/// 也不把 CPU 结果作为 GPU 失败时的替代路径。
pub fn schwarzschild_geodesic_integrate_ucf_vulkan_from_rays(
    mass: f64,
    critical_impact: f64,
    rays: &[(Vec3, Vec3)],
    step: f64,
    escape_radius: f64,
    max_steps: u32,
) -> TonResult<Vec<(SphericalState, u32)>> {
    if rays.is_empty() {
        return Ok(Vec::new());
    }
    let (mut states, bound_flags) = schwarzschild_initial_state_ucf_vulkan_batch_with_flags(
        mass,
        critical_impact,
        rays,
    )?;
    let program = schwarzschild_geodesic_rk4_chunk(16);
    let mut statuses = vec![0u32; states.len()];
    for _ in 0..max_steps.div_ceil(16) {
        let results = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_program(
            mass,
            &states,
            &bound_flags,
            step,
            escape_radius,
            &program,
        )?;
        states = results.iter().map(|(state, _)| *state).collect();
        statuses = results.iter().map(|(_, status)| *status).collect();
        if statuses.iter().all(|status| *status != 0) {
            break;
        }
    }
    for status in &mut statuses {
        if *status == 0 {
            *status = 3;
        }
    }
    Ok(states.into_iter().zip(statuses).collect())
}

/// 由 Vulkan UCF batch dispatch 并行推进多条相机射线。
pub fn schwarzschild_geodesic_integrate_ucf_vulkan_rays(mass: f64, rays: &[(Vec3, Vec3)], step: f64, escape_radius: f64, max_steps: u32) -> TonResult<Vec<(SphericalState, u32)>> {
    schwarzschild_geodesic_integrate_ucf_vulkan_from_rays(
        mass,
        BlackHole::schwarzschild(mass).critical_impact_parameter(),
        rays,
        step,
        escape_radius,
        max_steps,
    )
}

/// 生成 UCF Vulkan 测地线终止诊断帧。
///
/// 颜色只用于验证执行分类：黑色为捕获，白色为逃逸，红色为预算耗尽。
/// 该输出不是 Beauty，也不允许被后处理路径复用。
pub fn schwarzschild_geodesic_diagnostic_ucf_vulkan(bh: BlackHole, camera: Camera, width: u32, height: u32, step: f64, escape_radius: f64, max_steps: u32) -> TonResult<Framebuffer> {
    let uv = (0..height).flat_map(|y| (0..width).map(move |x| {
        let aspect = width as f64 / height.max(1) as f64;
        (((x as f64 + 0.5) / width.max(1) as f64 * 2.0 - 1.0) * aspect, 1.0 - (y as f64 + 0.5) / height.max(1) as f64 * 2.0)
    })).collect::<Vec<_>>();
    let directions = camera_ray_ucf_vulkan_batch(camera, &uv)?;
    let rays = directions.into_iter().map(|direction| (camera.position, direction)).collect::<Vec<_>>();
    let states = schwarzschild_geodesic_integrate_ucf_vulkan_from_rays(
        bh.mass,
        bh.critical_impact_parameter(),
        &rays,
        step,
        escape_radius,
        max_steps,
    )?;
    let mut frame = Framebuffer::new(width, height);
    frame.encoding = ton_renderer::FramebufferEncoding::Display;
    for (index, (_, status)) in states.into_iter().enumerate() {
        frame.pixels[index] = match status {
            1 => Rgb::BLACK,
            2 => Rgb::WHITE,
            _ => Rgb::new(1.0, 0.0, 0.0),
        };
    }
    Ok(frame)
}

/// 由原生 Vulkan UCF backend 推进一个完整球坐标 Schwarzschild RK4 步。
pub fn schwarzschild_geodesic_rk4_step_ucf_vulkan(mass: f64, state: SphericalState, step: f64) -> TonResult<SphericalState> {
    let program = schwarzschild_geodesic_rk4_step();
    let task = dispatch_task(31, &program, 1, ResourceId(12), 0, 0, Some(ResourceId(10)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(mass)), ("uniform_1".into(), ParamValue::F64(step))]))?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(12), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(4) }, ResourceNode { id: ResourceId(10), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::ReadWrite, byte_size: Some(32) }] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan geodesic graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan geodesic graph: {e}")))?;
    backend.write_f32(ResourceId(10), &[state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi].into_iter().map(|value| value as f32).collect::<Vec<_>>()).map_err(|e| TonError::Domain(format!("upload geodesic state: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan geodesic graph: {e}")))?;
    let output = backend.read_f32(ResourceId(10)).map_err(|e| TonError::Domain(format!("read geodesic state: {e}")))?;
    if output.len() < 8 { return Err(TonError::Domain("Vulkan geodesic state output is short".into())); }
    Ok(SphericalState { t: output[0] as f64, r: output[1] as f64, theta: output[2] as f64, phi: output[3] as f64, ut: output[4] as f64, ur: output[5] as f64, utheta: output[6] as f64, uphi: output[7] as f64 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adaptive_cpu_output(input: &[f32; 9], uniforms: [f32; 5], retries: u32) -> Vec<f64> {
        let program = ton_ucf_frontend::SchwarzschildAdaptiveStepBuilder::new().maximum_retries(retries).build().unwrap();
        adaptive_cpu_program_output(&program, input, uniforms)
    }

    fn adaptive_cpu_program_output(program: &ucf_program::Program, input: &[f32; 9], uniforms: [f32; 5]) -> Vec<f64> {
        let mut interpreter = ton_ucf_frontend::CpuInterpreter::new();
        for (slot, value) in uniforms.into_iter().enumerate() { interpreter.with_uniform(slot as u32, Value::F32(value)); }
        interpreter.with_buffer(1, input.iter().copied().map(Value::F32).collect());
        interpreter.with_buffer(2, vec![Value::F32(0.0); 13]);
        interpreter.run_parallel(program, 1).unwrap();
        interpreter.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect()
    }

    #[test]
    fn adaptive_ucf_chunk_matches_independent_reference_sequence() {
        let program = ton_ucf_frontend::SchwarzschildAdaptiveChunkBuilder::new().maximum_steps(8).build().unwrap();
        let input = [0.0f32, 12.0, 1.0, 0.3, 1.2, -0.7, 0.22, 0.08, 0.08];
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08];
        let actual = adaptive_cpu_program_output(&program, &input, uniforms);
        let mut expected = SphericalState { t: input[0] as f64, r: input[1] as f64, theta: input[2] as f64, phi: input[3] as f64,
            ut: input[4] as f64, ur: input[5] as f64, utheta: input[6] as f64, uphi: input[7] as f64 };
        let config = ton_geodesic::AdaptiveRk4Config::builder().tolerances(uniforms[1] as f64, uniforms[2] as f64)
            .step_range(uniforms[3] as f64, uniforms[4] as f64).build().unwrap();
        let mut step = input[8] as f64;
        let mut rejections = 0;
        for _ in 0..8 {
            let report = ton_geodesic::schwarzschild_adaptive_rk4_step(1.0, expected, step, config).unwrap();
            expected = report.state;
            step = report.next_step;
            rejections += report.rejections;
        }
        assert_eq!(actual[8], 1.0);
        assert_eq!(actual[12], rejections as f64);
        for (actual, expected) in actual[..8].iter().copied().zip([expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi]) {
            assert!((actual - expected).abs() < 1.0e-5, "actual={actual}, expected={expected}");
        }
        assert!((actual[10] - step).abs() < 1.0e-5);
        let failure = adaptive_cpu_program_output(&program, &input, [1.0, 1.0e-30, 0.0, 0.08, 0.08]);
        assert_eq!(failure[8], 2.0);
        assert_eq!(failure[12], 1.0);
        assert_eq!(failure[9], 0.0);
        for (actual, initial) in failure[..8].iter().zip(&input[..8]) { assert_eq!(*actual, *initial as f64); }
    }

    #[test]
    fn adaptive_null_energy_trace_vulkan_preserves_multiple_disk_crossings() {
        let camera = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let events = ton_geodesic::SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0).build().unwrap();
        let config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).events(events).build().unwrap();
        let mut selected = None;
        'search: for u in [-0.9, -0.7, -0.5, 0.0, 0.5, 0.7, 0.9] {
            for v in [-0.4, -0.2, 0.0, 0.2, 0.4] {
                let initial = schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
                let reference = ton_geodesic::schwarzschild_adaptive_trace(initial, config).unwrap();
                if reference.disk_events.len() > 1 { selected = Some((initial, reference)); break 'search; }
            }
        }
        let (initial, reference) = selected.expect("independent reference must contain a multiple-crossing ray");
        let mut input = [0.0f32; 29];
        input[..8].copy_from_slice(&[initial.t, initial.r, initial.theta, initial.phi, initial.ut, initial.ur, initial.utheta, initial.uphi].map(|value| value as f32));
        input[9] = 0.08;
        let program = ton_ucf_frontend::SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(64).conserved_null_energy().build().unwrap();
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 80.0, 6.0, 20.0, 1.0e-5];
        let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(2));
        let task = dispatch_task(57, &program, 1, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
        let graph = Graph {
            resources: ResourceGraph { nodes: [(1, 29, Access::Read), (2, 29, Access::Write), (3, 1, Access::Write)]
                .into_iter().map(|(id, elements, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
        };
        let mut backend = VulkanBackend::new().expect("Vulkan multiple disk crossings");
        backend.prepare(&graph).unwrap();
        let mut chunks = 0;
        while input[8] == 0.0 && chunks < 256 {
            backend.write_f32(ResourceId(1), &input).unwrap();
            backend.run_prepared(&graph).unwrap();
            input.copy_from_slice(&backend.read_f32(ResourceId(2)).unwrap());
            chunks += 1;
        }
        let expected_status = match reference.termination {
            ton_geodesic::AdaptiveTraceTermination::Captured => 1.0,
            ton_geodesic::AdaptiveTraceTermination::Escaped => 2.0,
            other => panic!("reference termination={other:?}"),
        };
        eprintln!("multiple disk crossings GPU={}, reference={}, first_radius={}, reference_first_radius={}",
            input[13], reference.disk_events.len(), input[14], reference.disk_events[0].event.state.r);
        assert_eq!(input[8], expected_status);
        assert_eq!(input[13], reference.disk_events.len() as f32);
        assert!((input[14] as f64 - reference.disk_events[0].event.state.r).abs() < 3.0e-3);
        assert!((input[15] as f64 - reference.disk_events[0].affine_parameter).abs() < 3.0e-3);
        assert!((input[22] as f64 - reference.disk_events[0].event.state.r).abs() < 3.0e-3);
        assert!(input.iter().all(|value| value.is_finite()));
        assert!(input[16] < 1.0e-3 && input[17] < 1.0e-3 && input[18] < 2.0e-3);
    }

    #[test]
    fn adaptive_trace_vulkan_batch_matches_capture_reference_and_invariants() {
        let angles = [0.0f64, 0.1, 0.19, 0.21, 0.23, 0.25, 0.4];
        let origin = Vec3::new(24.0, 0.0, 0.0);
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for angle in angles {
            let initial = schwarzschild_geodesic_initial_state(1.0, origin, Vec3::new(-angle.cos(), angle.sin(), 0.0)).unwrap();
            let reference = ton_geodesic::schwarzschild_adaptive_trace(initial,
                ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).build().unwrap()).unwrap();
            expected.push(match reference.termination {
                ton_geodesic::AdaptiveTraceTermination::Captured => 1.0f32,
                ton_geodesic::AdaptiveTraceTermination::Escaped => 2.0,
                other => panic!("reference angle={angle}, termination={other:?}"),
            });
            let mut ray = [0.0f32; 29];
            ray[..8].copy_from_slice(&[initial.t, initial.r, initial.theta, initial.phi, initial.ut, initial.ur, initial.utheta, initial.uphi].map(|value| value as f32));
            ray[9] = 0.08;
            input.extend(ray);
        }
        let program = ton_ucf_frontend::SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(64).conserved_null_energy().build().unwrap();
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 80.0, 6.0, 20.0, 1.0e-5];
        let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(2));
        let count = angles.len() as u32;
        let task = dispatch_task(56, &program, count, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
        let graph = Graph {
            resources: ResourceGraph { nodes: [(1, count as u64 * 29, Access::Read), (2, count as u64 * 29, Access::Write), (3, count as u64, Access::Write)]
                .into_iter().map(|(id, elements, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
        };
        let mut backend = VulkanBackend::new().expect("Vulkan adaptive trace capture batch");
        backend.prepare(&graph).unwrap();
        let mut chunks = 0;
        while input.chunks_exact(29).any(|ray| ray[8] == 0.0) && chunks < 256 {
            backend.write_f32(ResourceId(1), &input).unwrap();
            backend.run_prepared(&graph).unwrap();
            input = backend.read_f32(ResourceId(2)).unwrap();
            chunks += 1;
        }
        for ((ray, expected), angle) in input.chunks_exact(29).zip(expected).zip(angles) {
            eprintln!("capture batch angle={angle}, status={}, r={}, steps={}, max_null={}, energy={}, angular={}", ray[8], ray[1], ray[10], ray[16], ray[17], ray[18]);
            assert_eq!(ray[8], expected, "angle={angle}, state={ray:?}");
            assert!(ray.iter().all(|value| value.is_finite()));
            assert!(ray[16] < 1.0e-3, "angle={angle}, null={}", ray[16]);
            assert!(ray[17] < 1.0e-3, "angle={angle}, energy={}", ray[17]);
            assert!(ray[18] < 2.0e-3, "angle={angle}, angular={}", ray[18]);
        }
        backend.write_f32(ResourceId(1), &input).unwrap();
        backend.run_prepared(&graph).unwrap();
        assert_eq!(backend.read_f32(ResourceId(2)).unwrap(), input);
    }

    #[test]
    fn adaptive_trace_vulkan_polar_ray_escapes_with_per_step_diagnostics() {
        let camera = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let u = ((39.5 / 128.0) * 2.0 - 1.0) * (128.0 / 72.0);
        let v = 1.0 - (0.5 / 72.0) * 2.0;
        let initial = schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
        let mut input = [0.0f32; 29];
        input[..8].copy_from_slice(&[initial.t, initial.r, initial.theta, initial.phi, initial.ut, initial.ur, initial.utheta, initial.uphi].map(|value| value as f32));
        input[9] = 0.08;
        let program = ton_ucf_frontend::SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(64).conserved_null_energy().build().unwrap();
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 80.0, 6.0, 20.0, 1.0e-5];
        let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(2));
        let task = dispatch_task(55, &program, 1, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
        let graph = Graph {
            resources: ResourceGraph { nodes: [(1, 29, Access::Read), (2, 29, Access::Write), (3, 1, Access::Write)]
                .into_iter().map(|(id, elements, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
        };
        let mut backend = VulkanBackend::new().expect("Vulkan adaptive trace polar ray");
        backend.prepare(&graph).unwrap();
        let mut chunks = 0;
        while input[8] == 0.0 && chunks < 256 {
            backend.write_f32(ResourceId(1), &input).unwrap();
            backend.run_prepared(&graph).unwrap();
            let output = backend.read_f32(ResourceId(2)).unwrap();
            assert!(output.iter().all(|value| value.is_finite()));
            input.copy_from_slice(&output);
            chunks += 1;
        }
        eprintln!("fused Vulkan polar chunks={chunks}, steps={}, status={}, r={}, max_null={}, energy_drift={}, angular_drift={}",
            input[10], input[8], input[1], input[16], input[17], input[18]);
        assert_eq!(input[8], 2.0);
        assert!((input[1] - 80.0).abs() <= uniforms[9]);
        assert!(input[16] < 1.0e-3);
        assert!(input[17] < 1.0e-3);
        assert!(input[18] < 2.0e-3);
        assert!(input[10] > 64.0 && chunks < 256);
    }

    #[test]
    fn adaptive_trace_chunk_matches_reference_events_on_cpu_and_vulkan() {
        let program = ton_ucf_frontend::SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(8).build().unwrap();
        let encoded = serde_json::to_string(&program).unwrap();
        let decoded: ucf_program::Program = serde_json::from_str(&encoded).unwrap();
        assert_eq!(ton_ucf_frontend::program_hash(&program), ton_ucf_frontend::program_hash(&decoded));
        let mut input = [0.0f32; 29];
        input[..8].copy_from_slice(&[0.0, 10.0, std::f32::consts::FRAC_PI_2 - 0.001, 0.3, 1.25, 1.0, 0.1, 0.0]);
        input[9] = 0.04;
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 10.02, 6.0, 20.0, 1.0e-5];
        let initial = SphericalState { t: input[0] as f64, r: input[1] as f64, theta: input[2] as f64,
            phi: input[3] as f64, ut: input[4] as f64, ur: input[5] as f64, utheta: input[6] as f64, uphi: input[7] as f64 };
        let integration = ton_geodesic::AdaptiveRk4Config::builder().tolerances(uniforms[1] as f64, uniforms[2] as f64)
            .step_range(uniforms[3] as f64, uniforms[4] as f64).build().unwrap();
        let events = ton_geodesic::SchwarzschildEventConfig::builder(1.0).radial_boundaries(uniforms[5] as f64, uniforms[6] as f64)
            .equatorial_disk(6.0, 20.0).localization(uniforms[9] as f64, 48).build().unwrap();
        let config = ton_geodesic::SchwarzschildAdaptiveTraceConfig::builder(1.0).integration(integration).events(events).initial_step(input[9] as f64).build().unwrap();
        let expected = ton_geodesic::schwarzschild_adaptive_trace(initial, config).unwrap();
        assert_eq!(expected.termination, ton_geodesic::AdaptiveTraceTermination::Escaped);
        assert_eq!(expected.disk_events.len(), 1);
        let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
        for (slot, value) in uniforms.into_iter().enumerate() { cpu.with_uniform(slot as u32, Value::F32(value)); }
        cpu.with_buffer(1, input.into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 29]);
        cpu.run_parallel(&program, 1).unwrap();
        let cpu_output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
        assert_eq!(cpu_output[8], 2.0);
        assert_eq!(cpu_output[10], expected.accepted_steps as f64);
        assert_eq!(cpu_output[13], 1.0);
        assert!((cpu_output[14] - expected.disk_events[0].event.state.r).abs() < 2.0e-4);
        assert!((cpu_output[15] - expected.disk_events[0].affine_parameter).abs() < 2.0e-4);
        assert!((cpu_output[12] - expected.affine_parameter).abs() < 2.0e-4);
        let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(2));
        let task = dispatch_task(54, &program, 1, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
        let graph = Graph {
            resources: ResourceGraph { nodes: [(1, 29, Access::Read), (2, 29, Access::Write), (3, 1, Access::Write)]
                .into_iter().map(|(id, elements, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
        };
        let mut backend = VulkanBackend::new().expect("Vulkan adaptive trace chunk");
        backend.prepare(&graph).unwrap();
        backend.write_f32(ResourceId(1), &input).unwrap();
        backend.run_prepared(&graph).unwrap();
        let actual = backend.read_f32(ResourceId(2)).unwrap();
        assert_eq!(actual[8], 2.0, "{actual:?}");
        assert_eq!(actual[10], expected.accepted_steps as f32);
        assert_eq!(actual[13], 1.0);
        for index in 0..29 {
            assert!((actual[index] as f64 - cpu_output[index]).abs() < 3.0e-4, "index={index}, actual={}, cpu={}", actual[index], cpu_output[index]);
        }
        backend.write_f32(ResourceId(1), &actual).unwrap();
        backend.run_prepared(&graph).unwrap();
        assert_eq!(backend.read_f32(ResourceId(2)).unwrap(), actual);
        let mut invalid = input;
        invalid[9] = -1.0;
        backend.write_f32(ResourceId(1), &invalid).unwrap();
        backend.run_prepared(&graph).unwrap();
        let output = backend.read_f32(ResourceId(2)).unwrap();
        assert_eq!(output[8], 5.0);
        assert_eq!(&output[..8], &input[..8]);
        assert_eq!(output[10], 0.0);
    }

    #[test]
    fn accepted_step_event_program_matches_independent_reference_and_vulkan() {
        let program = ton_ucf_frontend::SchwarzschildEventLocalizationBuilder::new().build().unwrap();
        let encoded = serde_json::to_string(&program).unwrap();
        let decoded: ucf_program::Program = serde_json::from_str(&encoded).unwrap();
        assert_eq!(ton_ucf_frontend::program_hash(&program), ton_ucf_frontend::program_hash(&decoded));
        for disk in [false, true] {
            let start_values = [0.0f32, 10.0, if disk { std::f32::consts::FRAC_PI_2 - 0.001 } else { 1.2 },
                0.3, 1.25, if disk { 0.0 } else { -1.0 }, if disk { 0.1 } else { 0.0 }, 0.0];
            let start = SphericalState { t: start_values[0] as f64, r: start_values[1] as f64,
                theta: start_values[2] as f64, phi: start_values[3] as f64, ut: start_values[4] as f64,
                ur: start_values[5] as f64, utheta: start_values[6] as f64, uphi: start_values[7] as f64 };
            let accepted = ton_geodesic::schwarzschild_adaptive_rk4_step(1.0, start, 0.04,
                ton_geodesic::AdaptiveRk4Config::builder().build().unwrap()).unwrap();
            let target = 9.98f32;
            let config = if disk {
                ton_geodesic::SchwarzschildEventConfig::builder(1.0).equatorial_disk(6.0, 20.0)
            } else {
                ton_geodesic::SchwarzschildEventConfig::builder(1.0).radial_boundaries(target as f64, 80.0)
            }.localization(1.0e-5f32 as f64, 48).build().unwrap();
            let expected = ton_geodesic::schwarzschild_accepted_step_events(start, accepted, config).unwrap()[0];
            let end = accepted.state;
            let mut input = start_values.to_vec();
            input.extend([end.t, end.r, end.theta, end.phi, end.ut, end.ur, end.utheta, end.uphi, accepted.step_used].map(|value| value as f32));
            let uniforms = [1.0f32, if disk { 1.0 } else { 0.0 }, target, 1.0e-5];
            let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
            cpu.with_buffer(1, input.iter().copied().map(Value::F32).collect());
            cpu.with_buffer(2, vec![Value::F32(0.0); 11]);
            for (slot, value) in uniforms.iter().enumerate() { cpu.with_uniform(slot as u32, Value::F32(*value)); }
            cpu.run_parallel(&program, 1).unwrap();
            let cpu_output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert_eq!(cpu_output[10], 1.0);
            assert!((cpu_output[8] - expected.fraction).abs() < 1.0e-5);
            let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
            params.insert("buffer_2".into(), ParamValue::I64(2));
            let task = dispatch_task(53, &program, 1, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
            let graph = Graph {
                resources: ResourceGraph { nodes: [(1, 17, Access::Read), (2, 11, Access::Write), (3, 1, Access::Write)]
                    .into_iter().map(|(id, elements, access)| ResourceNode {
                        id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                    }).collect() },
                tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
            };
            let mut backend = VulkanBackend::new().expect("Vulkan accepted step event localization");
            backend.prepare(&graph).unwrap();
            backend.write_f32(ResourceId(1), &input).unwrap();
            backend.run_prepared(&graph).unwrap();
            let actual = backend.read_f32(ResourceId(2)).unwrap();
            assert_eq!(actual[10], 1.0, "disk={disk}, output={actual:?}");
            assert!(actual[9] <= uniforms[3]);
            assert!((actual[8] as f64 - expected.fraction).abs() < 2.0e-4);
            for (actual, expected) in actual[..8].iter().zip(&cpu_output[..8]) {
                assert!((*actual as f64 - expected).abs() < 2.0e-4);
            }
            let mut no_crossing = input.clone();
            no_crossing[8..16].copy_from_slice(&start_values);
            backend.write_f32(ResourceId(1), &no_crossing).unwrap();
            backend.run_prepared(&graph).unwrap();
            let actual = backend.read_f32(ResourceId(2)).unwrap();
            assert_eq!(actual[10], 2.0);
            assert_eq!(&actual[..8], &start_values);
            assert_eq!(actual[8], 0.0);
            let mut invalid = input.clone();
            invalid[16] = -0.04;
            backend.write_f32(ResourceId(1), &invalid).unwrap();
            backend.run_prepared(&graph).unwrap();
            let actual = backend.read_f32(ResourceId(2)).unwrap();
            assert_eq!(actual[10], 4.0);
            assert_eq!(&actual[..8], &start_values);
        }
    }

    #[test]
    fn adaptive_ucf_vulkan_chunk_matches_cpu_sequence() {
        let program = ton_ucf_frontend::SchwarzschildAdaptiveChunkBuilder::new().maximum_steps(32).build().unwrap();
        let input = [0.0f32, 12.0, 1.0, 0.3, 1.2, -0.7, 0.22, 0.08, 0.08];
        let uniforms = [1.0f32, 1.0e-5, 1.0e-4, 1.0e-8, 0.08];
        let expected = adaptive_cpu_program_output(&program, &input, uniforms);
        let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(2));
        let task = dispatch_task(51, &program, 1, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
        let graph = Graph {
            resources: ResourceGraph { nodes: [(1, 9, Access::Read), (2, 13, Access::Write), (3, 1, Access::Write)]
                .into_iter().map(|(id, elements, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
        };
        let mut backend = VulkanBackend::new().expect("Vulkan adaptive chunk");
        backend.prepare(&graph).unwrap();
        backend.write_f32(ResourceId(1), &input).unwrap();
        backend.run_prepared(&graph).unwrap();
        let actual = backend.read_f32(ResourceId(2)).unwrap();
        assert_eq!(actual[8], 1.0);
        for (actual, expected) in actual[..8].iter().zip(&expected[..8]) {
            assert!((*actual as f64 - expected).abs() < 3.0e-4, "actual={actual}, expected={expected}");
        }
        assert!((actual[10] as f64 - expected[10]).abs() < 1.0e-6);
        assert!((0.0..=1.0).contains(&actual[11]));
    }

    #[test]
    fn adaptive_ucf_vulkan_polar_chunks_escape_with_bounded_null_residual() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let u = ((39.5 / 128.0) * 2.0 - 1.0) * (128.0 / 72.0);
        let v = 1.0 - (0.5 / 72.0) * 2.0;
        let state = schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
        let program = ton_ucf_frontend::SchwarzschildAdaptiveChunkBuilder::new().build().unwrap();
        let mut input = [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi, 0.08].map(|value| value as f32);
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08];
        let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
        params.insert("buffer_2".into(), ParamValue::I64(2));
        let task = dispatch_task(52, &program, 1, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
        let graph = Graph {
            resources: ResourceGraph { nodes: [(1, 9, Access::Read), (2, 13, Access::Write), (3, 1, Access::Write)]
                .into_iter().map(|(id, elements, access)| ResourceNode {
                    id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                }).collect() },
            tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
        };
        let mut backend = VulkanBackend::new().expect("Vulkan adaptive polar trajectory");
        backend.prepare(&graph).unwrap();
        let initial_energy = (1.0 - 2.0 / input[1] as f64) * input[4] as f64;
        let initial_angular_squared = (input[1] as f64).powi(4) *
            ((input[6] as f64).powi(2) + (input[2] as f64).sin().powi(2) * (input[7] as f64).powi(2));
        let mut residual = 0.0f64;
        let mut energy_drift = 0.0f64;
        let mut angular_drift = 0.0f64;
        let mut chunks = 0;
        while input[1] < 80.0 && chunks < 256 {
            backend.write_f32(ResourceId(1), &input).unwrap();
            backend.run_prepared(&graph).unwrap();
            let output = backend.read_f32(ResourceId(2)).unwrap();
            assert_eq!(output[8], 1.0, "adaptive GPU chunk failure: {output:?}");
            assert!(output.iter().all(|value| value.is_finite()));
            let radius = output[1] as f64;
            let f = 1.0 - 2.0 / radius;
            let temporal = f * (output[4] as f64).powi(2);
            let spatial = (output[5] as f64).powi(2) / f + radius * radius *
                ((output[6] as f64).powi(2) + (output[2] as f64).sin().powi(2) * (output[7] as f64).powi(2));
            let relative = ((spatial - temporal) / (spatial.abs() + temporal.abs())).abs();
            assert!(relative.is_finite());
            residual = residual.max(relative);
            let energy = f * output[4] as f64;
            let angular_squared = radius.powi(4) *
                ((output[6] as f64).powi(2) + (output[2] as f64).sin().powi(2) * (output[7] as f64).powi(2));
            energy_drift = energy_drift.max(((energy - initial_energy) / initial_energy).abs());
            angular_drift = angular_drift.max(((angular_squared - initial_angular_squared) / initial_angular_squared).abs());
            input[..8].copy_from_slice(&output[..8]);
            input[8] = output[10];
            chunks += 1;
        }
        eprintln!("UCF Vulkan polar chunks={chunks}, max_chunk_boundary_null_residual={residual}, r={}", input[1]);
        eprintln!("UCF Vulkan max_chunk_boundary_energy_drift={energy_drift}, angular_squared_drift={angular_drift}");
        assert!(input[1] >= 80.0 && chunks < 256);
        assert!(residual < 1.0e-3);
        assert!(energy_drift < 1.0e-3);
        assert!(angular_drift < 2.0e-3);
    }

    #[test]
    fn adaptive_ucf_cpu_polar_chunks_escape_with_bounded_null_residual() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let u = ((39.5 / 128.0) * 2.0 - 1.0) * (128.0 / 72.0);
        let v = 1.0 - (0.5 / 72.0) * 2.0;
        let state = schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
        let program = ton_ucf_frontend::SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(64).conserved_null_energy().build().unwrap();
        let mut input = [0.0f32; 29];
        input[..8].copy_from_slice(&[state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi].map(|value| value as f32));
        input[9] = 0.08;
        let mut chunks = 0;
        while input[8] == 0.0 && chunks < 256 {
            let mut interpreter = ton_ucf_frontend::CpuInterpreter::new();
            for (slot, value) in [1.0, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 80.0, 6.0, 20.0, 1.0e-5].into_iter().enumerate() {
                interpreter.with_uniform(slot as u32, Value::F32(value));
            }
            interpreter.with_buffer(1, input.into_iter().map(Value::F32).collect());
            interpreter.with_buffer(2, vec![Value::F32(0.0); 29]);
            interpreter.run_parallel(&program, 1).unwrap();
            for (target, value) in input.iter_mut().zip(interpreter.buffer(2).unwrap()) {
                let Value::F32(value) = value else { panic!("F32 trace lost its value type"); };
                *target = *value;
            }
            chunks += 1;
        }
        eprintln!("UCF F32 CPU polar chunks={chunks}, max_step_null_residual={}, r={}", input[16], input[1]);
        assert_eq!(input[8], 2.0);
        assert!((input[1] - 80.0).abs() <= 1.0e-5 && chunks < 256);
        assert!(input[16] < 1.0e-3);
        assert!(input[17] < 1.0e-3);
        assert!(input[18] < 2.0e-3);
    }

    #[test]
    fn adaptive_ucf_cpu_matches_independent_reference_and_failures() {
        let program = ton_ucf_frontend::SchwarzschildAdaptiveStepBuilder::new().build().unwrap();
        let ParamValue::Program { hash, body, .. } = program_param(&program).unwrap() else { panic!("missing program body"); };
        let decoded: ucf_program::Program = serde_json::from_str(&body).unwrap();
        assert_eq!(hash, ton_ucf_frontend::program_hash(&decoded));
        assert_eq!(program, decoded);
        let input = [0.0f32, 12.0, 1.0, 0.3, 1.2, -0.7, 0.22, 0.08, 4.0];
        let state = SphericalState { t: input[0] as f64, r: input[1] as f64, theta: input[2] as f64, phi: input[3] as f64,
            ut: input[4] as f64, ur: input[5] as f64, utheta: input[6] as f64, uphi: input[7] as f64 };
        let uniforms = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 4.0];
        let actual = adaptive_cpu_output(&input, uniforms, 16);
        assert_eq!(actual[8], 1.0);
        assert!(actual[12] > 0.0);
        let half = ton_geodesic::schwarzschild_geodesic_rk4_step(1.0, state, actual[9] * 0.5);
        let expected = ton_geodesic::schwarzschild_geodesic_rk4_step(1.0, half, actual[9] * 0.5);
        for (actual, expected) in actual[..8].iter().copied().zip([expected.t, expected.r, expected.theta, expected.phi,
            expected.ut, expected.ur, expected.utheta, expected.uphi]) {
            let rounding_budget = 16.0 * f32::EPSILON as f64 * expected.abs().max(1.0);
            assert!((actual - expected).abs() < rounding_budget, "actual={actual}, expected={expected}");
        }
        assert!((uniforms[3] as f64..=uniforms[4] as f64).contains(&actual[9]));
        assert!((uniforms[3] as f64..=uniforms[4] as f64).contains(&actual[10]));
        assert!(actual[11] <= 1.0);
        for (uniforms, retries, status) in [
            ([1.0, 1.0e-30, 0.0, 1.0e-8, 0.08], 0, 3.0),
            ([1.0, 1.0e-30, 0.0, 0.08, 0.08], 16, 2.0),
            ([0.0, 1.0e-9, 1.0e-7, 1.0e-8, 0.08], 16, 4.0),
        ] {
            let output = adaptive_cpu_output(&input, uniforms, retries);
            assert_eq!(output[8], status);
            assert_eq!(output[9], 0.0);
            for (actual, initial) in output[..8].iter().zip(&input[..8]) { assert_eq!(*actual, *initial as f64); }
        }
    }

    #[test]
    fn adaptive_ucf_vulkan_executes_mixed_batch_and_minimum_step_failure() {
        let program = ton_ucf_frontend::SchwarzschildAdaptiveStepBuilder::new().maximum_retries(2).build().unwrap();
        let valid = [0.0f32, 12.0, 1.0, 0.3, 1.2, -0.7, 0.22, 0.08, 0.08];
        let mut invalid_step = valid;
        invalid_step[8] = -1.0;
        let mut invalid_radius = valid;
        invalid_radius[1] = 1.0;
        let input = [valid, invalid_step, invalid_radius].into_iter().flatten().collect::<Vec<_>>();
        let make_graph = |uniforms: [f32; 5]| {
            let mut params = uniforms.into_iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(value as f64))).collect::<BTreeMap<_, _>>();
            params.insert("buffer_2".into(), ParamValue::I64(2));
            let task = dispatch_task(50, &program, 3, ResourceId(3), 0, 1, Some(ResourceId(1)), params).unwrap();
            Graph {
                resources: ResourceGraph { nodes: [(1, 27, Access::Read), (2, 39, Access::Write), (3, 3, Access::Write)]
                    .into_iter().map(|(id, elements, access)| ResourceNode {
                        id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram, access, byte_size: Some(elements * 4),
                    }).collect() },
                tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
            }
        };
        let uniforms = [1.0, 1.0e-5, 1.0e-4, 1.0e-8, 0.08];
        let graph = make_graph(uniforms);
        graph.validate().unwrap();
        let mut backend = VulkanBackend::new().expect("Vulkan adaptive program");
        backend.prepare(&graph).unwrap();
        backend.write_f32(ResourceId(1), &input).unwrap();
        backend.run_prepared(&graph).unwrap();
        let actual = backend.read_f32(ResourceId(2)).unwrap();
        let expected = adaptive_cpu_output(&valid, uniforms, 2);
        assert_eq!(actual.len(), 39);
        assert_eq!([actual[8], actual[21], actual[34]], [1.0, 4.0, 4.0]);
        for (actual, expected) in actual[..8].iter().zip(&expected[..8]) {
            assert!((*actual as f64 - expected).abs() < 2.0e-5);
        }
        assert!((0.0..=1.0).contains(&actual[11]));
        for (offset, initial) in [(13, invalid_step), (26, invalid_radius)] {
            assert_eq!(&actual[offset..offset + 8], &initial[..8]);
            assert_eq!(actual[offset + 9], 0.0);
        }
        let graph = make_graph([1.0, 1.0e-25, 0.0, 0.08, 0.08]);
        backend.run_prepared(&graph).unwrap();
        let failure = backend.read_f32(ResourceId(2)).unwrap();
        assert_eq!(failure[8], 2.0);
        assert_eq!(&failure[..8], &valid[..8]);
        assert_eq!(failure[9], 0.0);
    }

    #[test]
    fn adaptive_polar_reference_escapes_with_bounded_null_residual() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let u = ((39.5 / 128.0) * 2.0 - 1.0) * (128.0 / 72.0);
        let v = 1.0 - (0.5 / 72.0) * 2.0;
        let mut state = schwarzschild_geodesic_initial_state(1.0, camera.position, camera.ray_direction(u, v)).unwrap();
        let config = ton_geodesic::AdaptiveRk4Config::builder().build().unwrap();
        let mut step = 0.08;
        let mut accepted = 0;
        let mut rejected = 0;
        let mut residual = 0.0f64;
        let mut minimum_step = step;
        while state.r < 80.0 && accepted < 20_000 {
            let report = ton_geodesic::schwarzschild_adaptive_rk4_step(1.0, state, step, config).unwrap();
            assert!(report.error_ratio <= 1.0);
            state = report.state;
            step = report.next_step;
            minimum_step = minimum_step.min(report.step_used);
            accepted += 1;
            rejected += report.rejections;
            let f = 1.0 - 2.0 / state.r;
            let temporal = f * state.ut * state.ut;
            let spatial = state.ur * state.ur / f + state.r * state.r *
                (state.utheta * state.utheta + state.theta.sin().powi(2) * state.uphi * state.uphi);
            let relative = ((spatial - temporal) / (spatial.abs() + temporal.abs())).abs();
            assert!(relative.is_finite());
            residual = residual.max(relative);
        }
        eprintln!("adaptive reference accepted={accepted}, rejected={rejected}, minimum_step={minimum_step}, max_null_residual={residual}");
        assert!(state.r >= 80.0 && accepted < 20_000);
        assert!(rejected > 0 && minimum_step < 0.005);
        assert!(residual < 1.0e-3, "adaptive null residual={residual}");
    }

    #[test]
    fn polar_scattering_reference_converges_under_step_refinement() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let u = ((39.5 / 128.0) * 2.0 - 1.0) * (128.0 / 72.0);
        let v = 1.0 - (0.5 / 72.0) * 2.0;
        let direction = camera.ray_direction(u, v);
        let initial = schwarzschild_geodesic_initial_state(1.0, camera.position, direction).unwrap();
        let mut results = Vec::new();
        for step in [0.08, 0.02, 0.005, 0.00125] {
            let mut state = initial;
            let mut residual = 0.0f64;
            for _ in 0..(200.0 / step) as usize {
                if state.r >= 80.0 || state.r <= 2.0002 { break; }
                state = reference_geodesic_rk4_step(1.0, state, step);
                let f = 1.0 - 2.0 / state.r;
                let temporal = f * state.ut * state.ut;
                let spatial = state.ur * state.ur / f + state.r * state.r *
                    (state.utheta * state.utheta + state.theta.sin().powi(2) * state.uphi * state.uphi);
                let relative = ((spatial - temporal) / (spatial.abs() + temporal.abs())).abs();
                if !relative.is_finite() { residual = f64::INFINITY; break; }
                residual = residual.max(relative);
            }
            eprintln!("reference step={step}, r={}, theta={}, max_null_residual={residual}", state.r, state.theta);
            results.push((state, residual));
        }
        let (finest, residual) = results.last().unwrap();
        assert!(finest.r >= 80.0 && finest.r.is_finite());
        assert!(*residual < 1.0e-3, "finest null residual={residual}");
        assert!(*residual < results[0].1);
    }

    #[test]
    fn signed_spherical_chunk_matches_independent_reference() {
        let state = SphericalState { t: 0.0, r: 12.0, theta: -0.8, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.22, uphi: 0.08 };
        let expected = reference_geodesic_rk4_step(1.0, state, 0.01);
        let (actual, status) = schwarzschild_geodesic_integrate_ucf_cpu(1.0, state, 0.01, 100.0, 1).unwrap();
        assert_eq!(status, 3);
        for (actual, expected) in [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi]
            .into_iter().zip([expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi]) {
            assert!((actual - expected).abs() < 2.0e-5, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn vulkan_signed_spherical_chunks_match_independent_reference() {
        let states = [-0.8, std::f64::consts::PI + 0.8].map(|theta| SphericalState {
            t: 0.0, r: 12.0, theta, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.22, uphi: 0.08,
        });
        let output = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk(
            1.0, &states, &[false; 2], &[false; 2], &[0.0; 2], &[1.0; 2],
            6.0, 40.0, 0.01, 100.0, 4,
        ).expect("Vulkan signed-angle RK4");
        for (initial, (actual, status, _, _, _)) in states.into_iter().zip(output) {
            let mut expected = initial;
            for _ in 0..4 { expected = reference_geodesic_rk4_step(1.0, expected, 0.01); }
            assert_eq!(status, 0);
            for (actual, expected) in [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi]
                .into_iter().zip([expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi]) {
                assert!((actual - expected).abs() < 2.0e-5, "actual={actual}, expected={expected}");
            }
        }
    }

    #[test]
    fn vulkan_geodesic_failure_preserves_last_finite_state() {
        let state = SphericalState { t: 0.0, r: 4.0, theta: 1.0, phi: 0.0, ut: 1.0, ur: -0.5, utheta: 0.0, uphi: 0.0 };
        let output = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk(
            1.0, &[state], &[false], &[false], &[0.0], &[1.0],
            6.0, 40.0, f64::INFINITY, 100.0, 4,
        ).expect("Vulkan numerical failure report");
        assert_eq!(output[0].1, 4);
        let actual = output[0].0;
        assert_eq!(
            [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi],
            [state.t, state.r, state.theta, state.phi, state.ut, state.ur, state.utheta, state.uphi],
        );
    }

    #[test]
    fn vulkan_geodesic_chunk_stops_at_boundaries_when_available() {
        let states = [2.0001, 100.0].map(|r| SphericalState {
            t: 0.0, r, theta: 1.0, phi: 0.0, ut: 1.0,
            ur: if r < 3.0 { -1.0 } else { 1.0 }, utheta: 0.0, uphi: 0.0,
        });
        let output = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk(
            1.0, &states, &[false; 2], &[false; 2], &[0.0; 2], &[1.0; 2],
            6.0, 40.0, 0.1, 100.0, 64,
        ).expect("Vulkan boundary termination");
        for (index, (state, status, _, _, _)) in output.iter().enumerate() {
            assert_eq!(*status, index as u32 + 1);
            assert!((state.r - states[index].r).abs() < 1.0e-6);
            assert_eq!(state.t, 0.0);
            assert_eq!(state.ur, states[index].ur);
        }
    }

    #[test]
    fn vulkan_disk_crossing_matches_reference_step_when_available() {
        let state = SphericalState { t: 0.0, r: 12.0, theta: 1.56, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.22, uphi: 0.01 };
        let expected = reference_geodesic_rk4_step(1.0, state, 0.1);
        let actual = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk(
            1.0, &[state], &[false], &[false], &[0.0], &[1.0], 6.0, 40.0, 0.1, 100.0, 1,
        ).expect("Vulkan disk crossing");
        let (actual, status, hit, radius, doppler) = actual[0];
        assert_eq!(status, 0);
        assert!(hit);
        assert!(radius > 11.8 && radius < 12.0);
        assert!(doppler.is_finite() && doppler > 0.0);
        for (actual, expected) in [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi]
            .into_iter().zip([expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi]) {
            assert!((actual - expected).abs() < 2.0e-5, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn vulkan_disk_chunk_preserves_terminal_state_and_metadata_when_available() {
        let mut backend = match VulkanBackend::new() {
            Ok(backend) => backend,
            Err(error) => { eprintln!("Vulkan unavailable: {error}"); return; }
        };
        let state = SphericalState { t: 1.0, r: 20.0, theta: 1.0, phi: 0.2, ut: 1.0, ur: 0.5, utheta: 0.0, uphi: 0.0 };
        let states = vec![state; 5];
        let statuses = [1, 2, 3, 4, 0];
        let output = schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk_backend(
            &mut backend, 1.0, &states, &statuses, &[false; 5], &[true; 5], &[8.0; 5], &[1.1; 5],
            6.0, 30.0, 0.01, 100.0, 4,
        ).unwrap();
        for (index, (actual, status, hit, radius, doppler)) in output.iter().enumerate().take(4) {
            assert_eq!(*status, statuses[index]);
            assert_eq!(actual.r, state.r);
            assert_eq!(actual.t, state.t);
            assert!(*hit);
            assert_eq!(*radius, 8.0);
            assert!((*doppler - 1.1).abs() < 1.0e-6);
        }
        assert!(output[4].0.r > state.r);
        assert_eq!(output[4].1, 0);
        assert_eq!(output[4].3, 8.0);
    }

    #[test]
    fn vulkan_resident_beauty_post_matches_separate_stages_when_available() {
        let mut backend = match VulkanBackend::new() {
            Ok(backend) => backend,
            Err(error) => { eprintln!("Vulkan unavailable: {error}"); return; }
        };
        let count = 3;
        let mut states = vec![0.0f32; count as usize * 14];
        states[8] = 1.0;
        states[14 + 8] = 2.0;
        states[28 + 8] = 2.0;
        states[28 + 10] = 1.0;
        states[28 + 11] = 8.0;
        states[28 + 12] = 1.1;
        let directions = vec![0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let radiance = disk_radiance_ucf_vulkan_from_geodesic_state(&states, count, 6.0, 30.0, 1.0, 1.0).unwrap();
        let hdr = compose_sky_ucf_vulkan_batch(&directions, &states, &radiance, count).unwrap();
        let expected = tone_map_ucf_vulkan_batch(&hdr, count).unwrap();
        let graph = beauty_post_graph(count, 6.0, 30.0, 1.0, 1.0).unwrap();
        assert_eq!(graph.tasks.topological_order().unwrap(), vec![TaskId(35), TaskId(37), TaskId(36)]);
        backend.prepare(&graph).unwrap();
        backend.write_f32(ResourceId(18), &states).unwrap();
        backend.write_f32(ResourceId(22), &directions).unwrap();
        for _ in 0..2 {
            backend.run_prepared(&graph).unwrap();
            let actual = backend.read_f32(ResourceId(20)).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(actual.is_finite());
                assert!((actual - expected).abs() < 1.0e-5, "actual={actual}, expected={expected}");
            }
        }
    }
    use ton_metric::schwarzschild_geodesic_impact_parameter;
    #[test]
    fn reference_radial_step_is_available() { let state = RadialState { t: 0.0, r: 10.0, ut: 1.0, ur: -0.8 }; assert!((schwarzschild_radial_rk4_step_ucf_cpu(1.0, state, 0.01).unwrap().r - reference_radial_rk4_step(1.0, state, 0.01).r).abs() < 2e-6); }

    #[test]
    fn cpu_geodesic_program_matches_reference() {
        let state = SphericalState { t: 0.0, r: 10.0, theta: 1.1, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.02, uphi: 0.01 };
        let actual = schwarzschild_geodesic_rk4_step_ucf_cpu(1.0, state, 0.01).unwrap();
        let expected = reference_geodesic_rk4_step(1.0, state, 0.01);
        let actual = [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi];
        let expected = [expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi];
        for (actual, expected) in actual.into_iter().zip(expected) { assert!((actual - expected).abs() < 3.0e-4, "actual={actual}, expected={expected}"); }
    }

    #[test]
    fn cpu_integrated_program_matches_repeated_reference_steps() {
        let state = SphericalState { t: 0.0, r: 10.0, theta: 1.1, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.02, uphi: 0.01 };
        let (actual, status) = schwarzschild_geodesic_integrate_ucf_cpu(1.0, state, 0.01, 100.0, 4).unwrap();
        let mut expected = state;
        for _ in 0..4 { expected = reference_geodesic_rk4_step(1.0, expected, 0.01); }
        let actual = [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi];
        let expected = [expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi];
        for (actual, expected) in actual.into_iter().zip(expected) { assert!((actual - expected).abs() < 2.0e-3, "actual={actual}, expected={expected}"); }
        assert_eq!(status, 3);
    }

    #[test]
    fn vulkan_integrated_program_matches_cpu_when_available() {
        let state = SphericalState { t: 0.0, r: 10.0, theta: 1.1, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.02, uphi: 0.01 };
        let cpu = schwarzschild_geodesic_integrate_ucf_cpu(1.0, state, 0.01, 100.0, 4).unwrap();
        let gpu = match schwarzschild_geodesic_integrate_ucf_vulkan(1.0, state, 0.01, 100.0, 4) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan integrated geodesic program: {error}"),
        };
        let a = [cpu.0.t, cpu.0.r, cpu.0.theta, cpu.0.phi, cpu.0.ut, cpu.0.ur, cpu.0.utheta, cpu.0.uphi];
        let b = [gpu.0.t, gpu.0.r, gpu.0.theta, gpu.0.phi, gpu.0.ut, gpu.0.ur, gpu.0.utheta, gpu.0.uphi];
        for (actual, expected) in a.into_iter().zip(b) { assert!((actual - expected).abs() < 2.0e-3, "actual={actual}, expected={expected}"); }
        assert_eq!(gpu.1, cpu.1);
    }

    #[test]
    fn vulkan_integrated_batch_matches_cpu_when_available() {
        let states = vec![
            SphericalState { t: 0.0, r: 10.0, theta: 1.1, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.02, uphi: 0.01 },
            SphericalState { t: 0.0, r: 12.0, theta: 1.0, phi: 0.2, ut: 1.0, ur: -0.4, utheta: 0.01, uphi: 0.02 },
        ];
        let gpu = match schwarzschild_geodesic_integrate_ucf_vulkan_batch(1.0, &states, 0.01, 100.0, 4) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan integrated geodesic batch: {error}"),
        };
        assert_eq!(gpu.len(), states.len());
        for (state, (actual, status)) in states.iter().zip(gpu) {
            let (expected, expected_status) = schwarzschild_geodesic_integrate_ucf_cpu(1.0, *state, 0.01, 100.0, 4).unwrap();
            let actual_values = [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi];
            let expected_values = [expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi];
            for (actual, expected) in actual_values.into_iter().zip(expected_values) {
                assert!((actual - expected).abs() < 2.0e-3, "actual={actual}, expected={expected}");
            }
            assert_eq!(status, expected_status);
        }
    }

    #[test]
    fn ray_batch_initialization_and_cpu_execution_are_stable() {
        let rays = vec![
            (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.2, 0.0, -1.0).normalize()),
        ];
        let output = schwarzschild_geodesic_integrate_ucf_cpu_rays(1.0, &rays, 0.01, 100.0, 4).unwrap();
        assert_eq!(output.len(), rays.len());
        assert!(output.iter().all(|(state, status)| state.r.is_finite() && *status == 3));
    }

    #[test]
    fn orbit_loop_and_ray_batch_programs_match_cpu_and_vulkan() {
        fn vulkan_values(
            backend: &mut VulkanBackend,
            program: &ucf_program::Program,
            count: u32,
            task_id: u64,
            resource_id: u64,
            uniforms: BTreeMap<String, ParamValue>,
        ) -> TonResult<Vec<f64>> {
            let task = dispatch_task(task_id, program, count, ResourceId(resource_id), 0, 0, None, uniforms)?;
            let graph = Graph {
                resources: ResourceGraph {
                    nodes: vec![ResourceNode {
                        id: ResourceId(resource_id),
                        kind: ResourceKind::Buffer,
                        domain: Domain::Vram,
                        access: Access::Write,
                        byte_size: Some(u64::from(count) * 8),
                    }],
                },
                tasks: TaskGraph { nodes: vec![task], edges: Vec::new() },
            };
            graph.validate().map_err(|error| TonError::Domain(format!("validate loop parity graph: {error}")))?;
            backend.prepare(&graph).map_err(|error| TonError::Domain(format!("prepare loop parity graph: {error}")))?;
            backend.run_prepared(&graph).map_err(|error| TonError::Domain(format!("execute loop parity graph: {error}")))?;
            backend.read_f64(ResourceId(resource_id)).map_err(|error| TonError::Domain(format!("read loop parity output: {error}")))
        }

        let orbit_program = ton_ucf_frontend::orbit_loop_accumulate();
        let orbit_cpu = CpuBackend::new().execute_program(&orbit_program, 1, [(0, Value::F64(0.25))]).unwrap();
        let ray_program = ton_ucf_frontend::ray_batch_steps();
        let ray_cpu = CpuBackend::new().execute_program(&ray_program, 32, []).unwrap();
        let mut backend = match VulkanBackend::new() {
            Ok(backend) => backend,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("create Vulkan loop parity backend: {error}"),
        };
        let orbit_gpu = vulkan_values(&mut backend, &orbit_program, 1, 190, 30, BTreeMap::from([("uniform_0".into(), ParamValue::F64(0.25))])).unwrap();
        let ray_gpu = vulkan_values(&mut backend, &ray_program, 32, 191, 31, BTreeMap::new()).unwrap();
        assert_eq!(orbit_gpu.len(), 1);
        assert!((orbit_gpu[0] - orbit_cpu.instances[0].value.as_f64().unwrap()).abs() < 1.0e-12);
        assert_eq!(ray_gpu.len(), 32);
        for (actual, expected) in ray_gpu.into_iter().zip(ray_cpu.instances) {
            assert!((actual - expected.value.as_f64().unwrap()).abs() < 1.0e-12, "actual={actual}, expected={expected:?}");
        }
    }

    #[test]
    fn ucf_initial_state_batch_matches_reference() {
        let mass = 1.0;
        let bh = BlackHole::schwarzschild(mass);
        let rays = vec![
            (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(3.0, -2.0, 18.0), Vec3::new(0.2, 0.1, -1.0).normalize()),
            (Vec3::new(-4.0, 2.5, 16.0), Vec3::new(-0.25, -0.15, -1.0).normalize()),
        ];
        let actual = schwarzschild_initial_state_ucf_cpu_batch(
            mass,
            bh.critical_impact_parameter(),
            &rays,
        )
        .expect("UCF initial state batch");
        assert_eq!(actual.len(), rays.len());
        for ((origin, direction), actual) in rays.into_iter().zip(actual) {
            let expected = schwarzschild_geodesic_initial_state(mass, origin, direction)
                .expect("reference initial state");
            let a = [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi];
            let b = [expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi];
            for (index, (actual, expected)) in a.into_iter().zip(b).enumerate() {
                if index == 3 && expected.sin().abs() < 1.0e-8 {
                    continue;
                }
                assert!((actual - expected).abs() < 3.0e-4, "index={index}, actual={actual}, expected={expected}");
            }
        }
    }

    #[test]
    fn vulkan_ray_chunks_match_cpu_chunks_when_available() {
        let rays = vec![
            (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.2, 0.0, -1.0).normalize()),
        ];
        let cpu = schwarzschild_geodesic_integrate_ucf_cpu_rays(1.0, &rays, 0.01, 100.0, 32).unwrap();
        let gpu = match schwarzschild_geodesic_integrate_ucf_vulkan_rays(1.0, &rays, 0.01, 100.0, 32) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan chunked geodesic rays: {error}"),
        };
        for ((cpu_state, cpu_status), (gpu_state, gpu_status)) in cpu.into_iter().zip(gpu) {
            assert!((cpu_state.r - gpu_state.r).abs() < 3.0e-3);
            assert_eq!(cpu_status, gpu_status);
        }
    }

    #[test]
    fn vulkan_initial_state_program_matches_cpu_when_available() {
        let bh = BlackHole::schwarzschild(1.0);
        let rays = vec![
            (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(3.0, -2.0, 18.0), Vec3::new(0.2, 0.1, -1.0).normalize()),
        ];
        let cpu = schwarzschild_initial_state_ucf_cpu_batch(
            bh.mass,
            bh.critical_impact_parameter(),
            &rays,
        )
        .expect("CPU initial state program");
        let gpu = match schwarzschild_initial_state_ucf_vulkan_batch(
            bh.mass,
            bh.critical_impact_parameter(),
            &rays,
        ) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan initial state program: {error}"),
        };
        assert_eq!(gpu.len(), cpu.len());
        for (actual, expected) in gpu.into_iter().zip(cpu) {
            let actual = [actual.t, actual.r, actual.theta, actual.phi, actual.ut, actual.ur, actual.utheta, actual.uphi];
            let expected = [expected.t, expected.r, expected.theta, expected.phi, expected.ut, expected.ur, expected.utheta, expected.uphi];
            for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                if index == 3 && expected.sin().abs() < 1.0e-8 {
                    continue;
                }
                assert!((actual - expected).abs() < 4.0e-3, "index={index}, actual={actual}, expected={expected}");
            }
        }
    }

    #[test]
    fn vulkan_initial_state_to_chunk_chain_matches_cpu_when_available() {
        let bh = BlackHole::schwarzschild(1.0);
        let rays = vec![
            (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::new(3.0, -2.0, 18.0), Vec3::new(0.2, 0.1, -1.0).normalize()),
        ];
        let cpu = schwarzschild_geodesic_integrate_ucf_cpu_rays(
            bh.mass,
            &rays,
            0.01,
            100.0,
            32,
        )
        .expect("CPU initial state and chunk chain");
        let gpu = match schwarzschild_geodesic_integrate_ucf_vulkan_from_rays(
            bh.mass,
            bh.critical_impact_parameter(),
            &rays,
            0.01,
            100.0,
            32,
        ) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan initial state to chunk chain: {error}"),
        };
        assert_eq!(gpu.len(), cpu.len());
        for ((actual, actual_status), (expected, expected_status)) in gpu.into_iter().zip(cpu) {
            assert_eq!(actual_status, expected_status);
            assert!((actual.r - expected.r).abs() < 4.0e-3);
            assert!((actual.theta - expected.theta).abs() < 4.0e-3);
        }
    }

    #[test]
    fn vulkan_geodesic_diagnostic_frame_is_nonempty_when_available() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let frame = match schwarzschild_geodesic_diagnostic_ucf_vulkan(bh, camera, 8, 5, 0.04, 100.0, 64) {
            Ok(frame) => frame,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan geodesic diagnostic frame: {error}"),
        };
        assert_eq!((frame.width, frame.height), (8, 5));
        assert_eq!(frame.pixels.len(), 40);
        assert!(frame.pixels.iter().any(|pixel| *pixel != Rgb::BLACK));
    }

    #[test]
    fn vulkan_geodesic_program_matches_cpu_and_reference_when_available() {
        let state = SphericalState { t: 0.0, r: 10.0, theta: 1.1, phi: 0.3, ut: 1.2, ur: -0.7, utheta: 0.02, uphi: 0.01 };
        let cpu = schwarzschild_geodesic_rk4_step_ucf_cpu(1.0, state, 0.01).unwrap();
        let gpu = match schwarzschild_geodesic_rk4_step_ucf_vulkan(1.0, state, 0.01) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan geodesic program: {error}"),
        };
        let reference = reference_geodesic_rk4_step(1.0, state, 0.01);
        let cpu = [cpu.t, cpu.r, cpu.theta, cpu.phi, cpu.ut, cpu.ur, cpu.utheta, cpu.uphi];
        let gpu = [gpu.t, gpu.r, gpu.theta, gpu.phi, gpu.ut, gpu.ur, gpu.utheta, gpu.uphi];
        let reference = [reference.t, reference.r, reference.theta, reference.phi, reference.ut, reference.ur, reference.utheta, reference.uphi];
        for ((cpu, gpu), reference) in cpu.into_iter().zip(gpu).zip(reference) {
            assert!((gpu - cpu).abs() < 3.0e-4, "cpu={cpu}, gpu={gpu}");
            assert!((gpu - reference).abs() < 3.0e-4, "gpu={gpu}, reference={reference}");
        }
    }
    #[test]
    fn cpu_silhouette_matches_analytic() { let bh = BlackHole::schwarzschild(1.0); let camera = Camera::cinematic(bh); let frame = schwarzschild_capture_silhouette_ucf_cpu(bh, camera, 8, 5).unwrap(); for y in 0..5 { for x in 0..8 { let expected = schwarzschild_geodesic_analytic_captured(bh, camera.position, pixel_ray(x, y, 8, 5, camera)); assert_eq!(frame.pixels[y as usize * 8 + x as usize] == Rgb::BLACK, expected); } } }

    #[test]
    fn cpu_classification_report_exposes_terminal_histogram() {
        let bh = BlackHole::schwarzschild(1.0);
        let report = schwarzschild_classification_ucf_cpu(
            bh,
            Camera::cinematic(bh),
            2,
            2,
            0.04,
            100.0,
            16,
        )
        .expect("CPU UCF classification report");
        assert_eq!(report.total, 4);
        assert_eq!(
            report.captured + report.escaped + report.budget_exhausted + report.non_finite,
            report.total
        );
        assert!(report.agreement <= report.total);
    }

    #[test]
    fn vulkan_beauty_rejects_invalid_resolution_before_backend_creation() {
        let bh = BlackHole::schwarzschild(1.0);
        let error = schwarzschild_beauty_ucf_vulkan(
            bh,
            Camera::cinematic(bh),
            0,
            2,
            6.0,
            24.0,
            1.0,
            0.04,
            100.0,
            16,
        )
        .expect_err("zero-width Beauty frame must fail");
        assert!(error.to_string().contains("resolution must be non-zero"));
    }

    #[test]
    fn vulkan_classification_report_matches_cpu_when_available() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let cpu = schwarzschild_classification_ucf_cpu(bh, camera, 2, 2, 0.04, 100.0, 16)
            .expect("CPU UCF classification report");
        let gpu = match schwarzschild_classification_ucf_vulkan(bh, camera, 2, 2, 0.04, 100.0, 16) {
            Ok(report) => report,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan UCF classification report: {error}"),
        };
        assert_eq!(gpu.total, cpu.total);
        assert_eq!(gpu.analytic_captured, cpu.analytic_captured);
        assert_eq!(gpu.captured, cpu.captured);
        assert_eq!(gpu.escaped, cpu.escaped);
        assert_eq!(gpu.budget_exhausted, cpu.budget_exhausted);
        assert_eq!(gpu.non_finite, cpu.non_finite);
        assert_eq!(gpu.agreement, cpu.agreement);
    }

    #[test]
    fn cpu_impact_program_matches_reference() {
        let rays = [
            (Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.15, 0.05)),
            (Vec3::new(24.0, -3.0, 5.0), Vec3::new(-0.7, 0.4, -0.2)),
            (Vec3::new(80.0, 12.0, -9.0), Vec3::new(-0.2, -0.8, 0.35)),
        ];
        let actual = schwarzschild_impact_ucf_cpu_batch(1.0, &rays).unwrap();
        for ((position, direction), value) in rays.iter().zip(actual) {
            let reference = schwarzschild_geodesic_impact_parameter(1.0, *position, *direction).unwrap();
            assert!((value - reference).abs() < 2.0e-5, "value={value}, reference={reference}");
        }
    }

    #[test]
    fn vulkan_camera_program_matches_reference_when_available() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let uv = [(-0.7, 0.5), (0.0, 0.0), (0.8, -0.4)];
        let actual = match camera_ray_ucf_vulkan_batch(camera, &uv) {
            Ok(values) => values,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan camera program: {error}"),
        };
        for ((u, v), value) in uv.into_iter().zip(actual) {
            let reference = camera.ray_direction(u, v);
            assert!((value.x - reference.x).abs() < 2.0e-5);
            assert!((value.y - reference.y).abs() < 2.0e-5, "u={u} v={v} actual={value:?} reference={reference:?}");
            assert!((value.z - reference.z).abs() < 2.0e-5);
        }
    }

    #[test]
    fn cpu_camera_program_matches_reference() {
        let bh = BlackHole::schwarzschild(1.0);
        let camera = Camera::cinematic(bh);
        let uv = [(-0.7, 0.5), (0.0, 0.0), (0.8, -0.4)];
        let actual = camera_ray_ucf_cpu_batch(camera, &uv).unwrap();
        for ((u, v), value) in uv.into_iter().zip(actual) {
            let reference = camera.ray_direction(u, v);
            assert!((value.x - reference.x).abs() < 2.0e-5);
            assert!((value.y - reference.y).abs() < 2.0e-5);
            assert!((value.z - reference.z).abs() < 2.0e-5);
        }
    }

    #[test]
    fn vulkan_impact_program_matches_cpu_and_reference_when_available() {
        let rays = [
            (Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.15, 0.05)),
            (Vec3::new(24.0, -3.0, 5.0), Vec3::new(-0.7, 0.4, -0.2)),
            (Vec3::new(80.0, 12.0, -9.0), Vec3::new(-0.2, -0.8, 0.35)),
        ];
        let cpu = schwarzschild_impact_ucf_cpu_batch(1.0, &rays).unwrap();
        let gpu = match schwarzschild_impact_ucf_vulkan_batch(1.0, &rays) {
            Ok(values) => values,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan impact program: {error}"),
        };
        for ((cpu_value, (position, direction)), gpu_value) in cpu.iter().zip(rays).zip(gpu) {
            let reference = schwarzschild_geodesic_impact_parameter(1.0, position, direction).unwrap();
            assert!((f64::from(*cpu_value as f32) - gpu_value).abs() < 2.0e-5);
            assert!((gpu_value - reference).abs() < 2.0e-5);
        }
    }
}

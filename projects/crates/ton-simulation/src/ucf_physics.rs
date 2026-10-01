//! TON 物理参考与通用 UCF 程序的执行对照。

use std::collections::BTreeMap;

use ton_geodesic::{schwarzschild_geodesic_rk4_step as reference_geodesic_rk4_step, schwarzschild_radial_rk4_step as reference_radial_rk4_step, RadialState, SphericalState};
use ton_metric::{schwarzschild_geodesic_analytic_captured, schwarzschild_geodesic_impact_parameter};
use ton_renderer::Framebuffer;
use ton_types::{BlackHole, Camera, Rgb, TonError, TonResult, Vec3};
use ton_ucf_frontend::{camera_ray_batch, geodesic_capture_classify, geodesic_capture_classify_batch, schwarzschild_geodesic_rk4_step, schwarzschild_impact_batch, schwarzschild_radial_rk4_step};
use ucf_backend_cpu::CpuBackend;
use ucf_backend_vulkan::VulkanBackend;
use ucf_ir::{Dispatch, Graph, ResourceGraph, TaskGraph};
use ucf_scheduler::Backend;
use ucf_types::{Access, Domain, Objective, ParamValue, Priority, ResourceId, ResourceKind, ResourceNode, ShaderId, TaskId, TaskKind, TaskNode};
use ucf_program::Value;

fn program_param(program: &ucf_program::Program) -> TonResult<ParamValue> {
    let body = serde_json::to_string(program).map_err(|e| TonError::Domain(format!("encode UCF program: {e}")))?;
    Ok(ParamValue::Program { version: program.version, hash: ton_ucf_frontend::program_hash(program), body })
}

fn dispatch_task(id: u64, program: &ucf_program::Program, count: u32, output: ResourceId, input: Option<ResourceId>, uniforms: BTreeMap<String, ParamValue>) -> TonResult<TaskNode> {
    let mut params = uniforms;
    params.insert("program".into(), program_param(program)?);
    params.insert("output".into(), ParamValue::I64(output.0 as i64));
    if let Some(input) = input { params.insert("buffer_0".into(), ParamValue::I64(input.0 as i64)); }
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
    let task = dispatch_task(20, &program, count, ResourceId(10), Some(ResourceId(11)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(bh.critical_impact_parameter())), ("output".into(), ParamValue::I64(10))]))?;
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
    let count = u32::try_from(uv.len()).map_err(|_| TonError::Domain("UV batch exceeds u32 capacity".into()))?;
    let program = ton_ucf_frontend::camera_ray_batch();
    let mut params = camera_uniforms(camera);
    params.insert("output".into(), ParamValue::I64(10));
    params.insert("buffer_1".into(), ParamValue::I64(11));
    params.insert("buffer_2".into(), ParamValue::I64(12));
    let task = dispatch_task(22, &program, count, ResourceId(10), Some(ResourceId(11)), params)?;
    let graph = Graph { resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(10), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 4) }, ResourceNode { id: ResourceId(11), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Read, byte_size: Some(u64::from(count) * 2 * 4) }, ResourceNode { id: ResourceId(12), kind: ResourceKind::Buffer, domain: Domain::Vram, access: Access::Write, byte_size: Some(u64::from(count) * 3 * 4) }] }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|e| TonError::Domain(format!("Vulkan camera graph: {e}")))?;
    let mut backend = VulkanBackend::new().map_err(|e| TonError::Domain(format!("create Vulkan backend: {e}")))?;
    backend.prepare(&graph).map_err(|e| TonError::Domain(format!("prepare Vulkan camera graph: {e}")))?;
    let input = uv.iter().flat_map(|(u, v)| [*u as f32, *v as f32]).collect::<Vec<_>>();
    backend.write_f32(ResourceId(11), &input).map_err(|e| TonError::Domain(format!("upload UV: {e}")))?;
    backend.run_prepared(&graph).map_err(|e| TonError::Domain(format!("execute Vulkan camera graph: {e}")))?;
    let output = backend.read_f32(ResourceId(12)).map_err(|e| TonError::Domain(format!("read camera output: {e}")))?;
    if output.len() < uv.len() * 3 { return Err(TonError::Domain("Vulkan camera output is short".into())); }
    Ok(output.chunks_exact(3).take(uv.len()).map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).collect())
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
    let task = dispatch_task(21, &program, count, ResourceId(10), Some(ResourceId(11)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(mass))]))?;
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
    let task = dispatch_task(30, &program, 1, ResourceId(10), Some(ResourceId(10)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(mass)), ("uniform_1".into(), ParamValue::F64(step))]))?;
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

/// 由原生 Vulkan UCF backend 推进一个完整球坐标 Schwarzschild RK4 步。
pub fn schwarzschild_geodesic_rk4_step_ucf_vulkan(mass: f64, state: SphericalState, step: f64) -> TonResult<SphericalState> {
    let program = schwarzschild_geodesic_rk4_step();
    let task = dispatch_task(31, &program, 1, ResourceId(12), Some(ResourceId(10)), BTreeMap::from([("uniform_0".into(), ParamValue::F64(mass)), ("uniform_1".into(), ParamValue::F64(step))]))?;
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

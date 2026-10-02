//! UCF 帧图 CPU 参考执行：按任务链顺序校验 `op` 标签并驱动后处理。

use ton_renderer::{
    apply_tone_map_ucf, bloom_pass_ucf, Framebuffer, Graph, DEFAULT_BLOOM_INTENSITY,
    DEFAULT_BLOOM_THRESHOLD,
};
use ton_geodesic::SphericalState;
use ton_types::{Camera, DiskModelKind, IntegratorKind, TonError, TonResult, Vec3};
use ton_ucf_frontend::program_hash;
use ucf_backend_cpu::CpuBackend;
use ucf_program::Value;
use ucf_types::ParamValue;

use crate::Scene;

/// CPU 参考执行报告（diagnostics / 回归用）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuGraphExecution {
    /// 按拓扑顺序记录的任务 `op` 名。
    pub executed_ops: Vec<String>,
}

/// 一次真实 UCF 相机 Program 执行的结果。
#[derive(Debug, Clone, PartialEq)]
pub struct CameraProgramExecution {
    /// 实际执行的 Program hash。
    pub program_hash: u64,
    /// 输出 ray buffer 解码后的方向。
    pub rays: Vec<Vec3>,
}

/// 一次真实 UCF 初态 Program 执行的结果。
#[derive(Debug, Clone, PartialEq)]
pub struct InitialStateProgramExecution {
    /// 实际执行的 Program hash。
    pub program_hash: u64,
    /// 输出 buffer 解码后的球坐标状态。
    pub states: Vec<SphericalState>,
}

fn camera_uniforms(camera: Camera) -> Vec<(u32, Value)> {
    [
        camera.forward.x,
        camera.forward.y,
        camera.forward.z,
        camera.right.x,
        camera.right.y,
        camera.right.z,
        camera.up.x,
        camera.up.y,
        camera.up.z,
        (camera.fov_y * 0.5).tan(),
    ]
    .into_iter()
    .enumerate()
    .map(|(slot, value)| (slot as u32, Value::F32(value as f32)))
    .collect()
}

/// 从帧图节点取出并执行相机 Program，不调用 TON 原生相机实现替代执行。
pub fn execute_camera_program_cpu(
    scene: &Scene,
    graph: &Graph,
) -> TonResult<CameraProgramExecution> {
    graph
        .validate()
        .map_err(|error| TonError::Domain(format!("camera graph validate: {error}")))?;
    let task = graph
        .tasks
        .nodes
        .iter()
        .find(|node| node.id.0 == 11)
        .ok_or_else(|| TonError::Domain("camera task 11 is missing".into()))?;
    let ParamValue::Program { hash, body, .. } = task
        .params
        .get("program")
        .ok_or_else(|| TonError::Domain("camera task has no UCF program body".into()))?
    else {
        return Err(TonError::Domain("camera task program binding has invalid type".into()));
    };
    let program: ucf_program::Program = serde_json::from_str(body)
        .map_err(|error| TonError::Domain(format!("decode camera UCF program: {error}")))?;
    let actual_hash = program_hash(&program);
    if actual_hash != *hash {
        return Err(TonError::Domain(format!(
            "camera program hash mismatch: expected {hash}, actual {actual_hash}"
        )));
    }
    let input_resource = match task.params.get("input_resource") {
        Some(ParamValue::I64(id)) => *id as u32,
        _ => return Err(TonError::Domain("camera input resource binding is missing".into())),
    };
    let output_resource = match task.params.get("output_resource") {
        Some(ParamValue::I64(id)) => *id as u32,
        _ => return Err(TonError::Domain("camera output resource binding is missing".into())),
    };
    let count = scene.width.saturating_mul(scene.height);
    let uv = (0..scene.height)
        .flat_map(|y| {
            (0..scene.width).map(move |x| {
                let aspect = scene.width.max(1) as f32 / scene.height.max(1) as f32;
                let u = ((x as f32 / scene.width.max(1) as f32) * 2.0 - 1.0) * aspect;
                let v = 1.0 - (y as f32 / scene.height.max(1) as f32) * 2.0;
                [u, v]
            })
        })
        .flatten()
        .map(Value::F32)
        .collect::<Vec<_>>();
    let output_seed = vec![Value::F32(0.0); count as usize * 3];
    let report = CpuBackend::new()
        .execute_program_with_buffers(
            &program,
            count,
            camera_uniforms(scene.camera),
            [(input_resource, uv), (output_resource, output_seed)],
        )
        .map_err(TonError::Domain)?;
    let output = report
        .buffers
        .get(&output_resource)
        .ok_or_else(|| TonError::Domain("camera output resource was not written".into()))?;
    if output.len() != count as usize * 3 {
        return Err(TonError::Domain("camera output resource has invalid length".into()));
    }
    let rays = output
        .chunks_exact(3)
        .map(|value| {
            Ok(Vec3::new(
                value[0].as_f64().map_err(|error| TonError::Domain(error.to_string()))?,
                value[1].as_f64().map_err(|error| TonError::Domain(error.to_string()))?,
                value[2].as_f64().map_err(|error| TonError::Domain(error.to_string()))?,
            ))
        })
        .collect::<TonResult<Vec<_>>>()?;
    Ok(CameraProgramExecution { program_hash: actual_hash, rays })
}

/// 从三维 Schwarzschild 帧图执行初态 Program。
///
/// 相机 Program 的输出资源是图资源 `2`，初态 Program 的输入资源是图资源
/// `2`，但二者在各自 UCF 正文中使用固定的程序 buffer 槽 `1` 和 `2`。
/// 这里显式完成图资源到程序槽的绑定，不根据任务名称调用领域函数。
pub fn execute_initial_state_program_cpu(
    scene: &Scene,
    graph: &Graph,
) -> TonResult<InitialStateProgramExecution> {
    if scene.integrator != IntegratorKind::Geodesic3d {
        return Err(TonError::Domain("initial state program requires Geodesic3d graph".into()));
    }
    let camera = execute_camera_program_cpu(scene, graph)?;
    let task = graph
        .tasks
        .nodes
        .iter()
        .find(|node| node.id.0 == 12)
        .ok_or_else(|| TonError::Domain("initial state task 12 is missing".into()))?;
    let ParamValue::Program { hash, body, .. } = task
        .params
        .get("program")
        .ok_or_else(|| TonError::Domain("initial state task has no UCF program body".into()))?
    else {
        return Err(TonError::Domain("initial state task program binding has invalid type".into()));
    };
    let program: ucf_program::Program = serde_json::from_str(body)
        .map_err(|error| TonError::Domain(format!("decode initial state UCF program: {error}")))?;
    let actual_hash = program_hash(&program);
    if actual_hash != *hash {
        return Err(TonError::Domain(format!(
            "initial state program hash mismatch: expected {hash}, actual {actual_hash}"
        )));
    }
    let input = camera
        .rays
        .iter()
        .flat_map(|direction| {
            [
                scene.camera.position.x,
                scene.camera.position.y,
                scene.camera.position.z,
                direction.x,
                direction.y,
                direction.z,
            ]
            .into_iter()
            .map(|value| Value::F32(value as f32))
        })
        .collect::<Vec<_>>();
    let output_seed = vec![Value::F32(0.0); camera.rays.len() * 10];
    let report = CpuBackend::new()
        .execute_program_with_buffers(
            &program,
            camera.rays.len() as u32,
            [
                (0, Value::F32(scene.black_hole.mass as f32)),
                (1, Value::F32(scene.black_hole.critical_impact_parameter() as f32)),
            ],
            [(1, input), (2, output_seed)],
        )
        .map_err(TonError::Domain)?;
    let output = report
        .buffers
        .get(&2)
        .ok_or_else(|| TonError::Domain("initial state output buffer was not written".into()))?;
    if output.len() != camera.rays.len() * 10 {
        return Err(TonError::Domain("initial state output has invalid length".into()));
    }
    let states = output
        .chunks_exact(10)
        .map(|values| {
            let scalar = |index: usize| {
                values[index]
                    .as_f64()
                    .map_err(|error| TonError::Domain(format!("initial state value {index}: {error}")))
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
        .collect::<TonResult<Vec<_>>>()?;
    Ok(InitialStateProgramExecution { program_hash: actual_hash, states })
}

fn task_op(graph: &Graph, task_id: u64) -> TonResult<&str> {
    let node = graph
        .tasks
        .nodes
        .iter()
        .find(|n| n.id.0 == task_id)
        .ok_or_else(|| TonError::Domain(format!("missing task {task_id}")))?;
    match node.params.get("op") {
        Some(ParamValue::Str(op)) => Ok(op.as_str()),
        _ => Err(TonError::Domain(format!("task {task_id} missing op param"))),
    }
}

fn expected_disk_op(disk: DiskModelKind) -> &'static str {
    match disk {
        DiskModelKind::None => "ton_shade_sky_only",
        DiskModelKind::NovikovThorneThin => "ton_shade_novikov_thorne",
        DiskModelKind::ProceduralFbm => "ton_shade_procedural_fbm",
        DiskModelKind::SolidUniform => "ton_shade_solid_uniform",
    }
}

fn task_chain(bloom: bool) -> &'static [u64] {
    if bloom {
        &[10, 11, 12, 13, 14, 15, 16]
    } else {
        &[10, 11, 12, 13, 14, 15]
    }
}

fn apply_post_ops(frame: &mut Framebuffer, ops: &[String]) {
    for op in ops {
        match op.as_str() {
            "ton_bloom" => bloom_pass_ucf(frame, DEFAULT_BLOOM_THRESHOLD, DEFAULT_BLOOM_INTENSITY),
            "ton_tone_map" => apply_tone_map_ucf(frame),
            _ => {}
        }
    }
}

/// 线性帧图 CPU 参考执行：校验 trace / shade / 后处理 `op` 与场景一致。
pub fn execute_frame_graph_cpu(
    scene: &Scene,
    graph: &Graph,
) -> TonResult<(Framebuffer, CpuGraphExecution)> {
    graph
        .validate()
        .map_err(|e| TonError::Domain(format!("graph validate: {e}")))?;
    let _ = scene;
    Err(TonError::Domain(
        "UCF CPU frame execution is unavailable until the graph is bound to real UCF programs and resources".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::{BlackHole, IntegratorKind};

    #[test]
    fn cpu_executor_does_not_claim_label_only_execution() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_default_disk()
            .with_integrator(IntegratorKind::SemiClassicalRayMarch)
            .with_disk_model(DiskModelKind::ProceduralFbm)
            .with_resolution(32, 24)
            .build();
        let error = scene.render_frame_ucf_cpu().expect_err("must reject fake execution");
        assert!(error.to_string().contains("real UCF programs"));
    }

    #[test]
    fn cpu_executor_rejects_bloom_label_chain() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_default_disk()
            .with_integrator(IntegratorKind::SchwarzschildOrbit)
            .with_disk_model(DiskModelKind::NovikovThorneThin)
            .with_post_bloom(true)
            .with_resolution(32, 24)
            .build();
        let error = scene.render_frame_ucf_cpu().expect_err("must reject fake execution");
        assert!(error.to_string().contains("real UCF programs"));
    }

    #[test]
    fn ucf_shade_path_matches_native_rgba_for_small_frame() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_default_disk()
            .with_integrator(IntegratorKind::Geodesic3d)
            .with_disk_model(DiskModelKind::NovikovThorneThin)
            .with_resolution(24, 16)
            .build();
        let native = scene.render_frame_with_histogram_opts(false).framebuffer;
        let ucf = scene.render_frame_with_histogram_opts(true).framebuffer;
        assert_eq!(native.pixels.len(), ucf.pixels.len());
        for (i, (a, b)) in native.pixels.iter().zip(ucf.pixels.iter()).enumerate() {
            assert!(
                (a.r - b.r).abs() < 1e-5 && (a.g - b.g).abs() < 1e-5 && (a.b - b.b).abs() < 1e-5,
                "pixel {i} native={a:?} ucf={b:?}"
            );
        }
    }

    #[test]
    fn cpu_graph_executes_bound_camera_program() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_resolution(4, 3)
            .build();
        let graph = scene.build_ucf_graph();
        let execution = execute_camera_program_cpu(&scene, &graph)
            .expect("bound camera program must execute through UCF CPU backend");
        assert_eq!(execution.rays.len(), 12);
        let center = execution.rays[1 + 4];
        let expected = scene.camera.ray_direction(
            (1.0 / 4.0 * 2.0 - 1.0) * 4.0 / 3.0,
            1.0 - 1.0 / 3.0 * 2.0,
        );
        assert!((center.x - expected.x).abs() < 2.0e-5);
        assert!((center.y - expected.y).abs() < 2.0e-5);
        assert!((center.z - expected.z).abs() < 2.0e-5);
    }

    #[test]
    fn cpu_graph_executes_bound_initial_state_program() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_integrator(IntegratorKind::Geodesic3d)
            .with_resolution(4, 3)
            .build();
        let graph = scene.build_ucf_graph();
        let execution = execute_initial_state_program_cpu(&scene, &graph)
            .expect("bound initial state program must execute through UCF CPU backend");
        assert_eq!(execution.states.len(), 12);
        let center = execution.states[1 + 4];
        let expected = ton_geodesic::schwarzschild_geodesic_initial_state(
            scene.black_hole.mass,
            scene.camera.position,
            scene.camera.ray_direction(
                (1.0 / 4.0 * 2.0 - 1.0) * 4.0 / 3.0,
                1.0 - 1.0 / 3.0 * 2.0,
            ),
        )
        .expect("reference initial state");
        assert!((center.r - expected.r).abs() < 3.0e-4);
        assert!((center.ut - expected.ut).abs() < 3.0e-4);
    }
}

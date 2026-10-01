//! UCF 帧图 CPU 参考执行：按任务链顺序校验 `op` 标签并驱动后处理。

use ton_renderer::{
    apply_tone_map_ucf, bloom_pass_ucf, Framebuffer, Graph, DEFAULT_BLOOM_INTENSITY,
    DEFAULT_BLOOM_THRESHOLD,
};
use ton_types::{DiskModelKind, TonError, TonResult};
use ucf_types::ParamValue;

use crate::Scene;

/// CPU 参考执行报告（diagnostics / 回归用）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuGraphExecution {
    /// 按拓扑顺序记录的任务 `op` 名。
    pub executed_ops: Vec<String>,
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

    let chain = task_chain(scene.post_bloom);
    let mut executed_ops = Vec::with_capacity(chain.len());
    for id in chain {
        executed_ops.push(task_op(graph, *id)?.to_string());
    }

    let trace_op = scene.integrator.ucf_op_name();
    if executed_ops[2] != trace_op {
        return Err(TonError::Domain(format!(
            "trace op mismatch: graph `{}` vs scene `{trace_op}`",
            executed_ops[2]
        )));
    }
    let shade_op = expected_disk_op(scene.disk_model);
    if executed_ops[3] != shade_op {
        return Err(TonError::Domain(format!(
            "shade op mismatch: graph `{}` vs scene `{shade_op}`",
            executed_ops[3]
        )));
    }

    let trace_node = graph
        .tasks
        .nodes
        .iter()
        .find(|n| n.id.0 == 12)
        .ok_or_else(|| TonError::Domain("missing trace task 12".into()))?;
    if !trace_node.params.contains_key("program_hash") {
        return Err(TonError::Domain(
            "trace task missing program_hash (KernelBinding)".into(),
        ));
    }

    let shade_node = graph
        .tasks
        .nodes
        .iter()
        .find(|n| n.id.0 == 13)
        .ok_or_else(|| TonError::Domain("missing shade task 13".into()))?;
    if !shade_node.params.contains_key("program_hash") {
        return Err(TonError::Domain(
            "shade task missing program_hash (KernelBinding)".into(),
        ));
    }

    let tone_task_id = if scene.post_bloom { 15 } else { 14 };
    let tone_node = graph
        .tasks
        .nodes
        .iter()
        .find(|n| n.id.0 == tone_task_id)
        .ok_or_else(|| TonError::Domain(format!("missing tone_map task {tone_task_id}")))?;
    if !tone_node.params.contains_key("program_hash") {
        return Err(TonError::Domain(
            "tone_map task missing program_hash (KernelBinding)".into(),
        ));
    }
    if scene.post_bloom {
        let bloom_node = graph
            .tasks
            .nodes
            .iter()
            .find(|n| n.id.0 == 14)
            .ok_or_else(|| TonError::Domain("missing bloom task 14".into()))?;
        if !bloom_node.params.contains_key("program_hash") {
            return Err(TonError::Domain(
                "bloom task missing program_hash (KernelBinding)".into(),
            ));
        }
    }

    let mut frame = scene
        .render_frame_with_histogram_opts(true)
        .framebuffer;
    apply_post_ops(&mut frame, &executed_ops);
    Ok((frame, CpuGraphExecution { executed_ops }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::{BlackHole, IntegratorKind};

    #[test]
    fn cpu_executor_runs_full_chain_for_sem_classical() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_default_disk()
            .with_integrator(IntegratorKind::SemiClassicalRayMarch)
            .with_disk_model(DiskModelKind::ProceduralFbm)
            .with_resolution(32, 24)
            .build();
        let (frame, report) = scene.render_frame_ucf_cpu().expect("execute");
        assert_eq!(report.executed_ops.len(), 6);
        assert_eq!(report.executed_ops[2], "ton_trace_sem_classical");
        assert_eq!(report.executed_ops[3], "ton_shade_procedural_fbm");
        assert_eq!(report.executed_ops[4], "ton_tone_map");
        assert_eq!(frame.width, 32);
        assert!(frame.to_rgba8().iter().any(|&b| b > 0));
    }

    #[test]
    fn cpu_executor_applies_bloom_when_enabled() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_default_disk()
            .with_integrator(IntegratorKind::SchwarzschildOrbit)
            .with_disk_model(DiskModelKind::NovikovThorneThin)
            .with_post_bloom(true)
            .with_resolution(32, 24)
            .build();
        let (frame, report) = scene.render_frame_ucf_cpu().expect("execute");
        assert_eq!(report.executed_ops.len(), 7);
        assert_eq!(report.executed_ops[4], "ton_bloom");
        assert_eq!(report.executed_ops[5], "ton_tone_map");
        assert!(frame.to_rgba8().iter().any(|&b| b > 0));
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
}

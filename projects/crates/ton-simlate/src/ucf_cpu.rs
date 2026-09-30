//! UCF 帧图 CPU 参考执行：按任务链顺序校验 `op` 标签并驱动 `Scene::render_frame`。

use ton_render::Framebuffer;
use ton_types::{DiskModelKind, IntegratorKind, TonError, TonResult};
use ton_ucf::Graph;
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
    }
}

/// 线性帧图 CPU 参考执行：校验 trace / shade `op` 与场景一致后渲染一帧。
pub fn execute_frame_graph_cpu(
    scene: &Scene,
    graph: &Graph,
) -> TonResult<(Framebuffer, CpuGraphExecution)> {
    graph
        .validate()
        .map_err(|e| TonError::Domain(format!("graph validate: {e}")))?;

    let chain = [10u64, 11, 12, 13, 14, 15];
    let mut executed_ops = Vec::with_capacity(chain.len());
    for id in chain {
        executed_ops.push(task_op(graph, id)?.to_string());
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

    let frame = scene.render_frame();
    Ok((frame, CpuGraphExecution { executed_ops }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::BlackHole;
    use ton_ucf::build_frame_graph;

    #[test]
    fn cpu_executor_runs_full_chain_for_sem_classical() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_default_disk()
            .with_integrator(IntegratorKind::SemiClassicalRayMarch)
            .with_disk_model(DiskModelKind::ProceduralFbm)
            .with_resolution(32, 24)
            .build();
        let graph = build_frame_graph(scene.integrator, scene.disk_model);
        let (frame, report) = execute_frame_graph_cpu(&scene, &graph).expect("execute");
        assert_eq!(report.executed_ops.len(), 6);
        assert_eq!(report.executed_ops[2], "ton_trace_sem_classical");
        assert_eq!(report.executed_ops[3], "ton_shade_procedural_fbm");
        assert_eq!(frame.width, 32);
        assert!(frame.to_rgba8().iter().any(|&b| b > 0));
    }
}

//! UCF 内核绑定：任务图节点关联 `ucf-program` 程序指纹（非领域 `op` 字符串替代）。

use ton_types::IntegratorKind;
use ucf_program::{orbit_step_once, program_hash, ray_batch_steps, vector_math_normalize};
use ucf_types::ParamValue;

use ucf_ir::Graph;

/// 单个 UCF Program 绑定摘要（可序列化进任务 `params`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelBinding {
    pub program_name: String,
    pub program_hash: u64,
}

/// 任务图上一项可执行内核：`op` 为调度标签，`binding` 为程序正文指纹。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelInstance {
    pub op: String,
    pub binding: KernelBinding,
}

impl KernelBinding {
    pub fn from_program_name_hash(name: impl Into<String>, hash: u64) -> Self {
        Self {
            program_name: name.into(),
            program_hash: hash,
        }
    }
}

/// 测地线 trace 任务当前阶段的程序绑定（随积分器族区分，后续替换为真实测地线 IR）。
pub fn trace_kernel(integrator: IntegratorKind) -> KernelInstance {
    let (name, hash) = match integrator {
        IntegratorKind::SchwarzschildOrbit | IntegratorKind::KerrEquatorial => {
            let p = orbit_step_once();
            ("orbit_step_once", program_hash(&p))
        }
        IntegratorKind::Geodesic3d | IntegratorKind::SemiClassicalRayMarch => {
            let p = ray_batch_steps();
            ("ray_batch_steps", program_hash(&p))
        }
    };
    KernelInstance {
        op: integrator.ucf_op_name().to_string(),
        binding: KernelBinding::from_program_name_hash(name, hash),
    }
}

/// 盘面着色任务绑定（VectorMath 切片占位，后续换 DiskShade 程序）。
pub fn shade_kernel() -> KernelBinding {
    let p = vector_math_normalize();
    KernelBinding::from_program_name_hash("vector_math_normalize", program_hash(&p))
}

/// 将 `program_hash` / `program_name` 写入指定任务节点 `params`。
pub fn attach_binding(graph: &mut Graph, task_id: u64, binding: &KernelBinding) {
    if let Some(node) = graph.tasks.nodes.iter_mut().find(|n| n.id.0 == task_id) {
        node.params.insert(
            "program_hash".into(),
            ParamValue::I64(binding.program_hash as i64),
        );
        node.params.insert(
            "program_name".into(),
            ParamValue::Str(binding.program_name.clone()),
        );
    }
}

/// 帧图构建后标注 trace / shade 的程序绑定。
pub fn annotate_frame_programs(graph: &mut Graph, integrator: IntegratorKind) {
    let trace = trace_kernel(integrator);
    attach_binding(graph, 12, &trace.binding);
    let shade = shade_kernel();
    attach_binding(graph, 13, &shade);
}

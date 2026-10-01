//! UCF 内核绑定：任务图节点关联 `ucf-program` 程序指纹（非领域 `op` 字符串替代）。

use ton_types::{DiskModelKind, IntegratorKind};
use ucf_program::{
    disk_shade_novikov, generate_camera_ray, geodesic_radial_infall_steps, orbit_step_once,
    post_bloom_extract, program_hash, tone_map_aces, vector_math_normalize,
};
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

/// 测地线 trace 任务程序绑定（随积分器族区分）。
pub fn trace_kernel(integrator: IntegratorKind) -> KernelInstance {
    let (name, hash) = match integrator {
        IntegratorKind::SchwarzschildOrbit | IntegratorKind::KerrEquatorial => {
            let p = orbit_step_once();
            ("orbit_step_once", program_hash(&p))
        }
        IntegratorKind::Geodesic3d | IntegratorKind::SemiClassicalRayMarch => {
            let p = geodesic_radial_infall_steps();
            ("geodesic_radial_infall_steps", program_hash(&p))
        }
    };
    KernelInstance {
        op: integrator.ucf_op_name().to_string(),
        binding: KernelBinding::from_program_name_hash(name, hash),
    }
}

/// 盘面着色任务程序绑定（Novikov-Thorne 薄盘走 `disk_shade_novikov`，其余仍用占位切片）。
pub fn shade_kernel(disk: DiskModelKind) -> KernelBinding {
    match disk {
        DiskModelKind::NovikovThorneThin => {
            let p = disk_shade_novikov();
            KernelBinding::from_program_name_hash("disk_shade_novikov", program_hash(&p))
        }
        _ => {
            let p = vector_math_normalize();
            KernelBinding::from_program_name_hash("vector_math_normalize", program_hash(&p))
        }
    }
}

/// 相机射线生成任务程序绑定。
pub fn generate_rays_kernel() -> KernelBinding {
    let p = generate_camera_ray();
    KernelBinding::from_program_name_hash("generate_camera_ray", program_hash(&p))
}

/// 色调映射任务程序绑定（ACES 拟合）。
pub fn tone_map_kernel() -> KernelBinding {
    let p = tone_map_aces();
    KernelBinding::from_program_name_hash("tone_map_aces", program_hash(&p))
}

/// 泛光亮部提取任务程序绑定（多尺度模糊仍由宿主完成）。
pub fn bloom_kernel() -> KernelBinding {
    let p = post_bloom_extract();
    KernelBinding::from_program_name_hash("post_bloom_extract", program_hash(&p))
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

/// 帧图构建后标注 generate_rays / trace / shade / 后处理的程序绑定。
pub fn annotate_frame_programs(
    graph: &mut Graph,
    integrator: IntegratorKind,
    disk: DiskModelKind,
    bloom: bool,
) {
    attach_binding(graph, 11, &generate_rays_kernel());
    let trace = trace_kernel(integrator);
    attach_binding(graph, 12, &trace.binding);
    let shade = shade_kernel(disk);
    attach_binding(graph, 13, &shade);
    if bloom {
        attach_binding(graph, 14, &bloom_kernel());
        attach_binding(graph, 15, &tone_map_kernel());
    } else {
        attach_binding(graph, 14, &tone_map_kernel());
    }
}

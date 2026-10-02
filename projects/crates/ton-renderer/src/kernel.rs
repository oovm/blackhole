//! UCF 内核绑定：任务图节点关联 `ucf-program` 程序指纹（非领域 `op` 字符串替代）。

use ton_types::{DiskModelKind, IntegratorKind};
use ton_ucf_frontend::{
    camera_ray_batch, disk_shade_novikov, geodesic_capture_classify, geodesic_radial_infall_steps,
    equatorial_orbit_rk4_step, post_bloom_extract, program_hash, schwarzschild_geodesic_rk4_chunk,
    tone_map_aces, vector_math_normalize,
    Program,
};
use ucf_types::ParamValue;

use ucf_ir::Graph;

/// 单个 UCF Program 绑定摘要（可序列化进任务 `params`）。
#[derive(Debug, Clone, PartialEq)]
pub struct KernelBinding {
    pub program_name: String,
    pub program_hash: u64,
    pub program: Program,
}

/// 任务图上一项可执行内核：`op` 为调度标签，`binding` 为程序正文指纹。
#[derive(Debug, Clone, PartialEq)]
pub struct KernelInstance {
    pub op: String,
    pub binding: KernelBinding,
}

impl KernelBinding {
    pub fn from_program(name: impl Into<String>, program: Program) -> Self {
        Self {
            program_name: name.into(),
            program_hash: program_hash(&program),
            program,
        }
    }
}

/// 测地线 trace 任务程序绑定（随积分器族区分）。
pub fn trace_kernel(integrator: IntegratorKind) -> KernelInstance {
    match integrator {
        IntegratorKind::SchwarzschildOrbit | IntegratorKind::KerrEquatorial => {
            let p = equatorial_orbit_rk4_step();
            return KernelInstance {
                op: integrator.ucf_op_name().to_string(),
                binding: KernelBinding::from_program("equatorial_orbit_rk4_step", p),
            };
        }
        IntegratorKind::Geodesic3d | IntegratorKind::SemiClassicalRayMarch => {
            let (name, p) = if matches!(integrator, IntegratorKind::Geodesic3d) {
                // 3D Schwarzschild 前端使用有界 chunk 程序。Kerr 仍需独立前端，不能复用该绑定。
                ("schwarzschild_geodesic_rk4_chunk", schwarzschild_geodesic_rk4_chunk(16))
            } else {
                ("geodesic_radial_infall_steps", geodesic_radial_infall_steps())
            };
            return KernelInstance {
                op: integrator.ucf_op_name().to_string(),
                binding: KernelBinding::from_program(name, p),
            };
        }
}
}

/// 解析捕获分类程序绑定（史瓦西 `b < b_c`）。
pub fn capture_classify_kernel() -> KernelBinding {
    let p = geodesic_capture_classify();
    KernelBinding::from_program("geodesic_capture_classify", p)
}

/// 盘面着色任务程序绑定（Novikov-Thorne 薄盘走 `disk_shade_novikov`，其余仍用占位切片）。
pub fn shade_kernel(disk: DiskModelKind) -> KernelBinding {
    match disk {
        DiskModelKind::NovikovThorneThin => {
            let p = disk_shade_novikov();
            KernelBinding::from_program("disk_shade_novikov", p)
        }
        _ => {
            let p = vector_math_normalize();
            KernelBinding::from_program("vector_math_normalize", p)
        }
    }
}

/// 相机射线生成任务程序绑定。
pub fn generate_rays_kernel() -> KernelBinding {
    let p = camera_ray_batch();
    KernelBinding::from_program("camera_ray_batch", p)
}

/// 色调映射任务程序绑定（ACES 拟合）。
pub fn tone_map_kernel() -> KernelBinding {
    let p = tone_map_aces();
    KernelBinding::from_program("tone_map_aces", p)
}

/// 泛光亮部提取任务程序绑定（多尺度模糊仍由宿主完成）。
pub fn bloom_kernel() -> KernelBinding {
    let p = post_bloom_extract();
    KernelBinding::from_program("post_bloom_extract", p)
}

/// 将 `program_hash` / `program_name` 写入指定任务节点 `params`。
pub fn attach_binding(graph: &mut Graph, task_id: u64, binding: &KernelBinding) {
    if let Some(node) = graph.tasks.nodes.iter_mut().find(|n| n.id.0 == task_id) {
        node.params.insert(
            "program_hash".into(),
            ParamValue::I64(binding.program_hash as i64),
        );
        let body = serde_json::to_string(&binding.program).expect("UCF program serializes");
        node.params.insert(
            "program".into(),
            ParamValue::Program {
                version: binding.program.version,
                hash: binding.program_hash,
                body,
            },
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

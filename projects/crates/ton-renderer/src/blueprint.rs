//! 单帧渲染 UCF 图：与 `规划设计/blackhole-基于UCF的黑洞渲染架构与实施计划.md` 对齐。

use ton_types::{DiskModelKind, IntegratorKind};
use ucf_ir::{Domain, Graph, GraphBuilder};

use crate::kernel::{annotate_frame_programs, trace_kernel, KernelInstance};

fn disk_shade_op(disk: DiskModelKind) -> &'static str {
    match disk {
        DiskModelKind::None => "ton_shade_sky_only",
        DiskModelKind::NovikovThorneThin => "ton_shade_novikov_thorne",
        DiskModelKind::ProceduralFbm => "ton_shade_procedural_fbm",
        DiskModelKind::SolidUniform => "ton_shade_solid_uniform",
    }
}

/// 构建一帧黑洞渲染的 UCF 任务链（默认无泛光）。
pub fn build_frame_graph(integrator: IntegratorKind, disk: DiskModelKind) -> Graph {
    build_frame_graph_with_post(integrator, disk, false)
}

/// 构建一帧黑洞渲染的 UCF 任务链；`bloom` 为真时插入 `ton_bloom` 任务。
pub fn build_frame_graph_with_post(
    integrator: IntegratorKind,
    disk: DiskModelKind,
    bloom: bool,
) -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Host);
    b.buffer(2, Domain::Host);
    b.buffer(3, Domain::Host);
    b.buffer(4, Domain::Host);
    b.buffer(5, Domain::Host);
    b.buffer(6, Domain::Host);
    b.gpu(10, "ton_upload_params");
    b.gpu(11, "ton_generate_rays");
    b.gpu(12, integrator.ucf_op_name());
    b.gpu(13, disk_shade_op(disk));
    if bloom {
        b.gpu(14, "ton_bloom");
        b.gpu(15, "ton_tone_map");
        b.gpu(16, "ton_present_or_readback");
        b.chain(&[10, 11, 12, 13, 14, 15, 16]);
    } else {
        b.gpu(14, "ton_tone_map");
        b.gpu(15, "ton_present_or_readback");
        b.chain(&[10, 11, 12, 13, 14, 15]);
    }
    let mut graph = b.build();
    annotate_frame_programs(&mut graph, integrator);
    graph
}

/// 当前帧图 trace 任务的 [`KernelInstance`]（含 `ucf-program` 指纹）。
pub fn trace_kernel_for(integrator: IntegratorKind) -> KernelInstance {
    trace_kernel(integrator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::{DiskModelKind, IntegratorKind};
    use ucf_types::ParamValue;

    #[test]
    fn trace_task_carries_program_hash() {
        let g = build_frame_graph(
            IntegratorKind::SchwarzschildOrbit,
            DiskModelKind::NovikovThorneThin,
        );
        let node = g.tasks.nodes.iter().find(|n| n.id.0 == 12).expect("trace");
        assert!(matches!(
            node.params.get("program_hash"),
            Some(ParamValue::I64(h)) if *h != 0
        ));
        assert!(matches!(
            node.params.get("program_name"),
            Some(ParamValue::Str(name)) if name == "orbit_step_once"
        ));
    }

    #[test]
    fn geodesic_graph_binds_geodesic_infall_program() {
        let g = build_frame_graph(IntegratorKind::Geodesic3d, DiskModelKind::NovikovThorneThin);
        let node = g.tasks.nodes.iter().find(|n| n.id.0 == 12).expect("trace");
        assert!(matches!(
            node.params.get("program_name"),
            Some(ParamValue::Str(name)) if name == "geodesic_radial_infall_steps"
        ));
    }

    #[test]
    fn schwarzschild_orbit_graph_validates() {
        let g = build_frame_graph(
            IntegratorKind::SchwarzschildOrbit,
            DiskModelKind::NovikovThorneThin,
        );
        g.validate().expect("graph must validate");
        assert_eq!(g.tasks.nodes.len(), 6);
        assert!(matches!(
            g.tasks.nodes[2].params.get("op"),
            Some(ParamValue::Str(op)) if op == "ton_trace_schwarzschild_orbit"
        ));
    }

    #[test]
    fn sem_classical_graph_uses_distinct_trace_op() {
        let g = build_frame_graph(
            IntegratorKind::SemiClassicalRayMarch,
            DiskModelKind::None,
        );
        g.validate().expect("graph must validate");
        assert!(matches!(
            g.tasks.nodes[2].params.get("op"),
            Some(ParamValue::Str(op)) if op == "ton_trace_sem_classical"
        ));
    }

    #[test]
    fn procedural_fbm_graph_uses_fbm_shade_op() {
        let g = build_frame_graph(
            IntegratorKind::SchwarzschildOrbit,
            DiskModelKind::ProceduralFbm,
        );
        g.validate().expect("graph must validate");
        assert!(matches!(
            g.tasks.nodes[3].params.get("op"),
            Some(ParamValue::Str(op)) if op == "ton_shade_procedural_fbm"
        ));
    }

    #[test]
    fn geodesic_3d_graph_uses_distinct_trace_op() {
        let g = build_frame_graph(
            IntegratorKind::Geodesic3d,
            DiskModelKind::NovikovThorneThin,
        );
        g.validate().expect("graph must validate");
        assert!(matches!(
            g.tasks.nodes[2].params.get("op"),
            Some(ParamValue::Str(op)) if op == "ton_trace_geodesic_3d"
        ));
    }

    #[test]
    fn kerr_equatorial_graph_uses_distinct_trace_op() {
        let g = build_frame_graph(
            IntegratorKind::KerrEquatorial,
            DiskModelKind::NovikovThorneThin,
        );
        g.validate().expect("graph must validate");
        assert!(matches!(
            g.tasks.nodes[2].params.get("op"),
            Some(ParamValue::Str(op)) if op == "ton_trace_kerr_equatorial"
        ));
    }

    #[test]
    fn bloom_graph_inserts_ton_bloom_task() {
        let g = build_frame_graph_with_post(
            IntegratorKind::SchwarzschildOrbit,
            DiskModelKind::NovikovThorneThin,
            true,
        );
        g.validate().expect("graph must validate");
        assert_eq!(g.tasks.nodes.len(), 7);
        assert!(matches!(
            g.tasks.nodes[4].params.get("op"),
            Some(ParamValue::Str(op)) if op == "ton_bloom"
        ));
        assert!(matches!(
            g.tasks.nodes[5].params.get("op"),
            Some(ParamValue::Str(op)) if op == "ton_tone_map"
        ));
    }
}

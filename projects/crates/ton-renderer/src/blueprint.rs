//! 单帧渲染 UCF 图。

use ton_types::{DiskModelKind, IntegratorKind};
use ucf_ir::{Dispatch, Domain, Graph, GraphBuilder};
use ucf_types::ParamValue;

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
    build_frame_graph_for_resolution(integrator, disk, false, 1, 1)
}

/// 构建一帧黑洞渲染的 UCF 任务链；`bloom` 为真时插入 `ton_bloom` 任务。
pub fn build_frame_graph_with_post(
    integrator: IntegratorKind,
    disk: DiskModelKind,
    bloom: bool,
) -> Graph {
    build_frame_graph_for_resolution(integrator, disk, bloom, 1, 1)
}

/// 按实际帧尺寸构建带资源布局和 dispatch 契约的 UCF 图。
pub fn build_frame_graph_for_resolution(
    integrator: IntegratorKind,
    disk: DiskModelKind,
    bloom: bool,
    width: u32,
    height: u32,
) -> Graph {
    let pixel_count = u64::from(width).saturating_mul(u64::from(height));
    let pixel_count_u32 = width.saturating_mul(height);
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
    let resource_sizes = [
        (1, pixel_count.saturating_mul(2 * 4)),
        (2, pixel_count.saturating_mul(3 * 4)),
        (3, pixel_count.saturating_mul(10 * 4)),
        (4, pixel_count.saturating_mul(3 * 4)),
        (5, pixel_count.saturating_mul(4)),
        (6, 256),
    ];
    for node in &mut graph.resources.nodes {
        if let Some((_, byte_size)) = resource_sizes.iter().find(|(id, _)| *id == node.id.0) {
            node.byte_size = Some(*byte_size);
        }
    }
    let task_resources = [
        (10, None, Some(6)),
        (11, Some(1), Some(2)),
        (12, Some(2), Some(3)),
        (13, Some(3), Some(4)),
        (14, Some(4), Some(if bloom { 4 } else { 5 })),
        (15, Some(if bloom { 4 } else { 5 }), Some(5)),
        (16, Some(5), Some(5)),
    ];
    for task in &mut graph.tasks.nodes {
        task.dispatch = Dispatch::Explicit {
            x: pixel_count_u32.max(1),
            y: 1,
            z: 1,
        };
        if let Some((_, input, output)) = task_resources.iter().find(|(id, _, _)| *id == task.id.0) {
            if let Some(input) = input {
                task.params.insert("input_resource".into(), ParamValue::I64(*input));
            }
            if let Some(output) = output {
                task.params.insert("output_resource".into(), ParamValue::I64(*output));
            }
        }
        task.params.insert("dispatch_x".into(), ParamValue::I64(i64::from(pixel_count_u32)));
    }
    annotate_frame_programs(&mut graph, integrator, disk, bloom);
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
            Some(ParamValue::Str(name)) if name == "equatorial_orbit_rk4_step"
        ));
        let ParamValue::Program { hash, body, .. } = node.params.get("program").expect("program body") else {
            panic!("trace task must carry a structured UCF program");
        };
        let program: ton_ucf_frontend::Program = serde_json::from_str(body).expect("program json");
        assert_eq!(*hash, ton_ucf_frontend::program_hash(&program));
    }

    #[test]
    fn geodesic_graph_binds_geodesic_infall_program() {
        let g = build_frame_graph(IntegratorKind::Geodesic3d, DiskModelKind::NovikovThorneThin);
        let node = g.tasks.nodes.iter().find(|n| n.id.0 == 12).expect("trace");
        assert!(matches!(
            node.params.get("program_name"),
            Some(ParamValue::Str(name)) if name == "schwarzschild_geodesic_rk4_chunk"
        ));
    }

    #[test]
    fn novikov_graph_binds_disk_shade_program() {
        let g = build_frame_graph(IntegratorKind::Geodesic3d, DiskModelKind::NovikovThorneThin);
        let node = g.tasks.nodes.iter().find(|n| n.id.0 == 13).expect("shade");
        assert!(matches!(
            node.params.get("program_name"),
            Some(ParamValue::Str(name)) if name == "disk_shade_novikov"
        ));
        assert!(matches!(
            node.params.get("program_hash"),
            Some(ParamValue::I64(h)) if *h != 0
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
    fn generate_rays_task_carries_program_hash() {
        let g = build_frame_graph(
            IntegratorKind::Geodesic3d,
            DiskModelKind::NovikovThorneThin,
        );
        let node = g.tasks.nodes.iter().find(|n| n.id.0 == 11).expect("rays");
        let expected = crate::kernel::generate_rays_kernel();
        assert!(matches!(
            node.params.get("program_name"),
            Some(ParamValue::Str(name)) if name.as_str() == expected.program_name
        ));
        assert!(matches!(
            node.params.get("program_hash"),
            Some(ParamValue::I64(h)) if *h == expected.program_hash as i64
        ));
    }

    #[test]
    fn tone_map_and_shade_hashes_match_prelude_programs() {
        use crate::kernel::{shade_kernel, tone_map_kernel};
        let g = build_frame_graph(
            IntegratorKind::Geodesic3d,
            DiskModelKind::NovikovThorneThin,
        );
        let shade = shade_kernel(DiskModelKind::NovikovThorneThin);
        let tone = tone_map_kernel();
        let shade_node = g.tasks.nodes.iter().find(|n| n.id.0 == 13).expect("shade");
        let tone_node = g.tasks.nodes.iter().find(|n| n.id.0 == 14).expect("tone");
        assert!(matches!(
            shade_node.params.get("program_hash"),
            Some(ParamValue::I64(h)) if *h == shade.program_hash as i64
        ));
        assert!(matches!(
            tone_node.params.get("program_hash"),
            Some(ParamValue::I64(h)) if *h == tone.program_hash as i64
        ));
        let _ = crate::kernel::capture_classify_kernel();
    }

    #[test]
    fn tone_map_task_carries_program_hash() {
        let g = build_frame_graph(
            IntegratorKind::Geodesic3d,
            DiskModelKind::NovikovThorneThin,
        );
        let node = g.tasks.nodes.iter().find(|n| n.id.0 == 14).expect("tone_map");
        assert!(matches!(
            node.params.get("program_name"),
            Some(ParamValue::Str(name)) if name == "tone_map_aces"
        ));
        assert!(matches!(
            node.params.get("program_hash"),
            Some(ParamValue::I64(h)) if *h != 0
        ));
    }

    #[test]
    fn bloom_graph_binds_post_bloom_extract() {
        let g = build_frame_graph_with_post(
            IntegratorKind::SchwarzschildOrbit,
            DiskModelKind::NovikovThorneThin,
            true,
        );
        let bloom = g.tasks.nodes.iter().find(|n| n.id.0 == 14).expect("bloom");
        assert!(matches!(
            bloom.params.get("program_name"),
            Some(ParamValue::Str(name)) if name == "post_bloom_extract"
        ));
        let tone = g.tasks.nodes.iter().find(|n| n.id.0 == 15).expect("tone_map");
        assert!(matches!(
            tone.params.get("program_name"),
            Some(ParamValue::Str(name)) if name == "tone_map_aces"
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

    #[test]
    fn sized_graph_declares_pixel_dispatch_and_resource_buffers() {
        let graph = build_frame_graph_for_resolution(
            IntegratorKind::Geodesic3d,
            DiskModelKind::NovikovThorneThin,
            false,
            16,
            9,
        );
        let rays = graph.tasks.nodes.iter().find(|node| node.id.0 == 11).expect("ray task");
        assert_eq!(rays.dispatch, Dispatch::Explicit { x: 144, y: 1, z: 1 });
        assert_eq!(rays.params.get("input_resource"), Some(&ParamValue::I64(1)));
        assert_eq!(rays.params.get("output_resource"), Some(&ParamValue::I64(2)));
        assert_eq!(
            graph.resources.nodes.iter().find(|node| node.id.0 == 2).expect("ray buffer").byte_size,
            Some(144 * 3 * 4)
        );
    }
}

//! 黑洞模拟场景编排。
#![warn(missing_docs)]

mod convergence;
mod adaptive_gpu;
mod params;
mod scene;
mod ucf_cpu;
mod ucf_physics;

pub use convergence::{capture_convergence_samples, CaptureStepSample};
pub use adaptive_gpu::{SchwarzschildGpuTraceBuilder, GpuTraceBatch, GpuTraceTermination};
pub use params::black_hole_for_render;
pub use scene::{Scene, SceneBuilder};
pub use ton_renderer::RenderMode;
pub use ucf_cpu::{execute_camera_program_cpu, execute_disk_shade_program_cpu, execute_frame_graph_cpu, execute_geodesic_beauty_frame_cpu, execute_geodesic_chunk_program_cpu, execute_geodesic_diagnostic_frame_cpu, execute_geodesic_disk_radiance_cpu, execute_initial_state_program_cpu, CameraProgramExecution, CpuGraphExecution, GeodesicChunkProgramExecution, InitialStateProgramExecution};
pub use ucf_physics::{
    schwarzschild_classification_ucf_cpu, schwarzschild_classification_ucf_vulkan,
    SchwarzschildClassificationReport,
    schwarzschild_capture_classify_ucf_cpu, schwarzschild_capture_classify_ucf_cpu_batch,
    schwarzschild_capture_classify_ucf_vulkan_batch,
    schwarzschild_impact_ucf_cpu_batch, schwarzschild_impact_ucf_vulkan_batch,
    schwarzschild_capture_silhouette_ucf_cpu,
    schwarzschild_capture_silhouette_ucf_vulkan,
    schwarzschild_radial_rk4_step_ucf_cpu, schwarzschild_radial_rk4_step_ucf_vulkan,
    schwarzschild_radial_rk4_steps_ucf_cpu, schwarzschild_radial_rk4_steps_ucf_vulkan,
    schwarzschild_geodesic_integrate_ucf_cpu,
    schwarzschild_geodesic_integrate_ucf_vulkan,
    schwarzschild_geodesic_integrate_ucf_vulkan_batch,
    schwarzschild_geodesic_integrate_ucf_vulkan_batch_with_disk,
    disk_radiance_ucf_vulkan_from_geodesic_state,
    schwarzschild_disk_radiance_ucf_vulkan,
    tone_map_ucf_vulkan_batch,
    compose_sky_ucf_vulkan_batch,
    schwarzschild_beauty_ucf_vulkan,
    schwarzschild_geodesic_integrate_ucf_cpu_rays, schwarzschild_geodesic_integrate_ucf_vulkan_rays,
    schwarzschild_geodesic_diagnostic_ucf_vulkan,
    schwarzschild_initial_state_ucf_cpu_batch, schwarzschild_initial_state_ucf_vulkan_batch,
    schwarzschild_initial_state_ucf_vulkan_batch_with_flags,
    schwarzschild_geodesic_integrate_ucf_vulkan_from_rays,
};

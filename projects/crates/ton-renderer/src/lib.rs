//! TON 渲染器：UCF 任务图蓝图 + CPU 光线追踪帧缓冲与后处理。

#![warn(missing_docs)]

mod blueprint;
mod capture_ucf;
mod framebuffer;
mod kernel;
mod post;
mod post_ucf;
mod rays_ucf;
mod renderer;
mod shade_ucf;
mod shadow_overlay;
mod sky;

pub use capture_ucf::{analytic_captured_ucf, CaptureClassifyUcf};
pub use blueprint::{build_frame_graph, build_frame_graph_for_resolution, build_frame_graph_with_post, trace_kernel_for};
pub use kernel::{attach_binding, KernelBinding, KernelInstance};
pub use framebuffer::{Framebuffer, FramebufferEncoding};
pub use post::{
    aces_filmic, apply_post_process, apply_tone_map, bloom_pass, composite_bloom_from_bright,
    extract_bright_pass, tone_map_rgb, DEFAULT_EXPOSURE,
};
pub use post_ucf::{
    apply_tone_map_ucf, bloom_pass_ucf, extract_bright_pass_ucf, DEFAULT_BLOOM_INTENSITY,
    DEFAULT_BLOOM_THRESHOLD,
};
pub use shade_ucf::{sample_disk_radiance_ucf, DiskShadeUcf};
pub use rays_ucf::{camera_ray_direction_ucf, CameraRayUcf};
pub use renderer::{
    default_camera, default_integrator, RenderConfig, RenderMode, RenderOutput, Renderer,
};
pub use shadow_overlay::{
    capture_classification_agreement, draw_analytic_shadow_ring, on_axis_shadow_radius_ndc,
    pixel_ray_dir, schwarzschild_boundary_radius_ndc,
};
pub use ton_geodesic::TerminationHistogram;
pub use sky::sky_color;
pub use ucf_ir::Graph;

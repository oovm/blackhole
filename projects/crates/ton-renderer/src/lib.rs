//! TON 渲染器：UCF 任务图蓝图 + CPU 光线追踪帧缓冲与后处理。

#![warn(missing_docs)]

mod blueprint;
mod framebuffer;
mod kernel;
mod post;
mod renderer;
mod shadow_overlay;
mod sky;

pub use blueprint::{build_frame_graph, build_frame_graph_with_post, trace_kernel_for};
pub use kernel::{attach_binding, KernelBinding, KernelInstance};
pub use framebuffer::{Framebuffer, FramebufferEncoding};
pub use post::{
    aces_filmic, apply_post_process, apply_tone_map, bloom_pass, tone_map_rgb, DEFAULT_EXPOSURE,
};
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

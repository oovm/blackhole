//! TON 渲染器：UCF 任务图蓝图 + CPU 光线追踪帧缓冲与后处理。

#![warn(missing_docs)]

mod blueprint;
mod framebuffer;
mod kernel;
mod post;
mod renderer;
mod sky;

pub use blueprint::{build_frame_graph, build_frame_graph_with_post, trace_kernel_for};
pub use kernel::{attach_binding, KernelBinding, KernelInstance};
pub use framebuffer::Framebuffer;
pub use post::{apply_post_process, apply_tone_map, bloom_pass, tone_map_rgb};
pub use renderer::{default_camera, default_integrator, RenderConfig, Renderer};
pub use sky::sky_color;
pub use ucf_ir::Graph;

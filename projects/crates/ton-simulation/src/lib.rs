//! 黑洞模拟场景编排。
#![warn(missing_docs)]

mod params;
mod scene;
mod ucf_cpu;

pub use params::black_hole_for_render;
pub use scene::{Scene, SceneBuilder};
pub use ton_renderer::RenderMode;
pub use ucf_cpu::{execute_frame_graph_cpu, CpuGraphExecution};

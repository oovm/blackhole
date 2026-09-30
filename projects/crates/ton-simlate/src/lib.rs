//! 黑洞模拟场景编排。
#![warn(missing_docs)]

mod scene;
mod ucf_cpu;

pub use scene::{Scene, SceneBuilder};
pub use ucf_cpu::{execute_frame_graph_cpu, CpuGraphExecution};

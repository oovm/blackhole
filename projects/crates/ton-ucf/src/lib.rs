//! TON × UCF 任务图蓝图。

#![warn(missing_docs)]

mod blueprint;

pub use blueprint::{build_frame_graph, build_frame_graph_with_post};
pub use ucf_ir::Graph;

# ton-renderer

黑洞渲染器：UCF 单帧任务图蓝图 + CPU 光线追踪帧缓冲与后处理。

```rust
use ton_types::{DiskModelKind, IntegratorKind};
use ton_renderer::{build_frame_graph, Renderer, RenderConfig};

let graph = build_frame_graph(
    IntegratorKind::SemiClassicalRayMarch,
    DiskModelKind::NovikovThorneThin,
);
graph.validate()?;
```

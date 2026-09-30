# `ton-ucf`

黑洞单帧 UCF IR 蓝图：`IntegratorKind` × `DiskModelKind` → 可 `validate()` 的任务链。

```rust
use ton_types::{DiskModelKind, IntegratorKind};
use ton_ucf::build_frame_graph;

let graph = build_frame_graph(
    IntegratorKind::SemiClassicalRayMarch,
    DiskModelKind::NovikovThorneThin,
);
graph.validate()?;
```

CPU 执行器与 GPU 后端在后续工作包接入同一 IR。

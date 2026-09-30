# packages

npm 面前端与 Web 演示（Rust `ton-*` 为权威算法层）。

| 目录        | 包名              | 说明                                      |
|-------------|-------------------|-------------------------------------------|
| `blackhole` | `@ton/blackhole`  | `ton-wasm` + UCF L2/L0 宿主呈现 → `lib/` |
| `homepage`  | `@ton/homepage`   | VMZ 黑洞可视化演示页（`@vmz/vmz`）      |

UCF 不单独发 npm；当前上屏为 L0 readback，GPU 热路径后置 UCF compute。

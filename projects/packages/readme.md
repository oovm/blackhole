# packages

npm 面前端与 Web 演示（Rust `ton-*` 为权威算法层）。

| 目录        | 包名              | 说明                                      |
|-------------|-------------------|-------------------------------------------|
| `blackhole` | `@ton/blackhole`  | `ton-wasm` 加载器 + WebGPU 呈现 → `lib/`  |
| `homepage`  | `@ton/homepage`   | Vite 黑洞可视化演示页                     |

UCF 不单独发 npm；GPU 呈现经 `@ton/blackhole` 的 `ton_host` WebGPU 胶水封装。

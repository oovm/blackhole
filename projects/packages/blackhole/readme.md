# `@doki-land/blackhole`

黑洞宿主 TS 库：`src/wasm` 与 `src/node` 同构（加载绑定 → 渲染 RGBA8）。

| 子路径 | 说明 |
|--------|------|
| `src/wasm` | `ton-wasm` 加载 + UCF L2/L0 Web 呈现 |
| `src/node` | `ton-napi` 加载 + CPU RGBA8 |

平台 npm 袋 `@doki-land/blackhole-<platform>` 只携带产物并 re-export 入口。

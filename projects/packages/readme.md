# packages

按平台拆分的 npm 袋与 VMZ 演示（对齐 Spark `sparkle-engine-unknown-wasm32` / VMZ `@vmz/vmz-<triple>`）。

| 目录 | 包名 | 说明 |
|------|------|------|
| [`blackhole-unknown-wasm32`](blackhole-unknown-wasm32/readme.md) | `@doki-land/blackhole-unknown-wasm32` | `ton-wasm` cdylib + UCF L2/L0 Web 呈现 |
| [`blackhole-win32-x64`](blackhole-win32-x64/readme.md) | `@doki-land/blackhole-win32-x64` | `ton-napi` N-API `.node` 袋 |
| [`homepage`](homepage/readme.md) | `@doki-land/blackhole-homepage` | VMZ 黑洞可视化演示（`@vmz/vmz`） |

Rust 绑定 crate 在 [`../crates/`](../crates/readme.md)（`ton-wasm` / `ton-napi`）。

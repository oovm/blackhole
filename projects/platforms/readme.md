# platforms

按目标平台拆分的原生 npm 袋（对齐 Spark `projects/platforms/native` 与 VMZ `@vmz/vmz-<triple>`）。

| 目录 | 包名 | 说明 |
|------|------|------|
| [`native/ton-win32-x64`](native/ton-win32-x64/readme.md) | `@doki-land/blackhole-win32-x64` | `ton-napi` N-API `.node` 袋 |

Wasm 平台袋在 [`../packages/blackhole-unknown-wasm32`](../packages/blackhole-unknown-wasm32/readme.md)（`@doki-land/blackhole-unknown-wasm32`）。

Rust 绑定 crate 在 [`../crates/`](../crates/readme.md)（`ton-wasm` / `ton-napi`）。

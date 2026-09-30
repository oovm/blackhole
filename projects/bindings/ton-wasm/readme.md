# ton-wasm

`wasm32-unknown-unknown` cdylib：导出稳定 C ABI，供 `@ton/blackhole` 浏览器宿主加载。

- **物理渲染**：`ton-sim` CPU 史瓦西光线追踪 → RGBA8
- **UCF**：同模块链 `ucf` CPU 后端（应用契约诊断探针）
- **GPU 呈现**：经 `ton_host` import 由 TS 侧 WebGPU 上传帧缓冲（UCF Web 宿主模式）

## 构建

```bash
rustup target add wasm32-unknown-unknown
cargo build -p ton-wasm --target wasm32-unknown-unknown --release
```

产物拷贝见 `projects/packages/blackhole/scripts/copy-from-cargo.mjs`。

## 导出（C ABI）

| 符号 | 说明 |
|------|------|
| `ton_version_code` | `major * 1_000_000 + minor * 1_000 + patch` |
| `ton_target_triple` | 只读 C 字符串 |
| `ton_render_schwarzschild_rgba8` | 渲染一帧 RGBA8 到线性内存 |
| `ton_ucf_application_contract_diag_mask` | UCF 诊断 kind 位掩码 |
| `ton_l2_present_rgba8_webgpu` | 经 `ton_host` 提交 WebGPU 纹理上屏 |

## 验收

```bash
cargo test -p ton-wasm
```

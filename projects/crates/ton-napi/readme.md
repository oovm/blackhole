# ton-napi

Node **N-API** cdylib：native 宿主直接调用 `ton-sim`（与 `ton-wasm` 并列，目标平台为 `projects/packages/blackhole-*` npm 袋）。

## 构建

```bash
pnpm --filter @doki-land/blackhole-win32-x64 run build:napi
```

或：

```bash
cargo build -p ton-napi --release
```

## 导出（JS）

| 符号 | 说明 |
|------|------|
| `tonVersionCode` | 版本编码 |
| `renderSchwarzschildRgba8` | 一帧 RGBA8 `Buffer` |

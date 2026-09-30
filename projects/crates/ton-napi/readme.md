# ton-napi

Node **N-API** cdylib：由 `@doki-land/blackhole/node` 加载，平台袋为 `projects/packages/blackhole-<triple>/`。

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

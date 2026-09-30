# TON 黑洞可视化首页（VMZ）

基于 `@ton/blackhole` 的 **VMZ** 演示页：史瓦西黑洞 CPU 光线追踪、UCF 运行时探针、L0 readback 上屏。

依赖 npm 上的 `@vmz/vmz` / `@vmz/core`（无需本地 `vmz-framework` 路径）。

## 开发

```bash
pnpm --filter @ton/blackhole run build:wasm
pnpm --filter @ton/blackhole run build
pnpm --filter @ton/homepage run dev
```

浏览器打开后点击「渲染一帧」。WebGPU 可用时走 **UCF L2**（`ton_l2_present_rgba8_webgpu`），否则 L0 readback。

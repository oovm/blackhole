# TON 黑洞可视化首页

基于 `@ton/blackhole` 的 Vite 演示：史瓦西黑洞 + 吸积盘 CPU 光线追踪，UCF 运行时探针，WebGPU / 2D 降级上屏。

## 开发

```bash
# 先构建 wasm 与 TS 库
pnpm --filter @ton/blackhole run build:wasm
pnpm --filter @ton/blackhole run build

pnpm --filter @ton/homepage run dev
```

浏览器打开后点击「渲染一帧」。默认 640×360；质量 M 可调。

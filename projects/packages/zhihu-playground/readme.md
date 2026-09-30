# TON 黑洞可视化（知乎 AI Works）

基于 `@doki-land/blackhole-unknown-wasm32` 的 **Vite** 静态演示页：史瓦西黑洞 CPU 光线追踪、UCF 运行时探针、L2/L0 上屏。

独立 npm 包（`vendor/` 内置 wasm 加载器），远端构建仅依赖 `vite build`，无需 VMZ 原生工具链。

## 开发

```bash
cd projects/packages/zhihu-playground
npm install
npm run dev
```

浏览器打开后点击「渲染一帧」。WebGPU 可用时走 **UCF L2**，否则 L0 readback。

## Vendor 同步

仓内更新 wasm 加载器后，在仓库根执行 `pnpm build:ts`，再在本目录运行：

```bash
node ./scripts/sync-vendor-runtime.mjs
```

产物写入 `vendor/blackhole-unknown-wasm32/runtime/`（避免 CloudBase ZIP 排除名为 `dist` 的目录）并刷新 `public/ton_wasm_bg.wasm`。

## CloudBase 交付

运行 `zhihu-ai-works-deploy-helper` 生成 `_tmp/frontend.deploy.json` 与根目录 `zhihu-playground.zip`，上传 ZIP 至知乎 AI Works 静态托管流水线。

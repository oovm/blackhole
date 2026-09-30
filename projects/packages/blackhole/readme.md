# `@ton/blackhole`

TON 黑洞模拟 **下游 Web 封装**：加载 `ton-wasm`（`ton-sim` + UCF CPU 探针），经 WebGPU 宿主上屏。

UCF 不单独发 npm；本包自行封装 wasm 与 `ton_host` WebGPU import。

## 构建

```bash
pnpm --filter @ton/blackhole run build:wasm
pnpm --filter @ton/blackhole run build
```

## API

```js
import { renderBlackholeFrame } from "@ton/blackhole";

const canvas = document.querySelector("canvas");
canvas.width = 640;
canvas.height = 360;
const { ton, tier } = await renderBlackholeFrame(canvas, { mass: 1.0 });
console.log(ton.info(), tier);
```

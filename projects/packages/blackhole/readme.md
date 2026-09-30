# `@ton/blackhole`

TON 黑洞模拟 **下游 Web 封装**：加载 `ton-wasm`（`ton-sim` + UCF CPU 探针），L0 readback 上屏。

UCF 不单独发 npm；本包封装 wasm、`ucf_host` / `ton_host` WebGPU 宿主胶水。CPU 测地线在 `ton-sim`，上屏优先 UCF L2，降级 L0 readback。

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
console.log(ton.info(), tier); // tier: "L2" | "L0"
```

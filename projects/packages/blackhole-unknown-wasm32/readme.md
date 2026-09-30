# `@doki-land/blackhole-unknown-wasm32`

`wasm32-unknown-unknown` npm 袋（`projects/packages/blackhole-unknown-wasm32`）：加载 `ton-wasm` cdylib，UCF L2/L0 Web 呈现。

UCF 不单独发 npm；本包封装 wasm、`ucf_host` / `ton_host` WebGPU 宿主胶水。CPU 测地线在 `ton-sim`，上屏优先 UCF L2，降级 L0 readback。

## 构建

```bash
pnpm --filter @doki-land/blackhole-unknown-wasm32 run build:wasm
pnpm --filter @doki-land/blackhole-unknown-wasm32 run build
```

## API

```js
import { renderBlackholeFrame } from "@doki-land/blackhole-unknown-wasm32";

const canvas = document.querySelector("canvas");
canvas.width = 640;
canvas.height = 360;
const { ton, tier } = await renderBlackholeFrame(canvas, { mass: 1.0 });
console.log(ton.info(), tier); // tier: "L2" | "L0"
```

/**
 * @ton/blackhole — TON 黑洞 Wasm + UCF L0 呈现（无手写 WGSL / `wgpu` crate）。
 */

export { tonWasmRustTarget } from "./load.js";
export {
    loadTonWasm,
    renderBlackholeFrame,
    presentRgba8ToCanvas2d,
} from "./present.js";

export type {
    LoadTonWasmOptions,
    TonWasmBindings,
    TonWasmInfo,
    RenderBlackholeOptions,
    RenderBlackholeResult,
    PresentTier,
} from "./present.js";

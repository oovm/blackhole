/**
 * `@doki-land/blackhole/wasm` — `ton-wasm` + UCF L2/L0 呈现（宿主 WebGPU 胶水，无 Rust `wgpu` crate）。
 */

export { tonWasmRustTarget } from "./load.js";
export {
    loadTonWasm,
    renderBlackholeFrame,
    presentRgba8ToCanvas2d,
    initWebGpuHost,
    UcfHostHandleRegistry,
} from "./present.js";

export type {
    LoadTonWasmOptions,
    TonWasmBindings,
    TonWasmInfo,
    RenderBlackholeOptions,
    RenderBlackholeResult,
    PresentTier,
} from "./present.js";

export {
    DiskModelKind,
    IntegratorKind,
    DISK_MODEL_OPTIONS,
    INTEGRATOR_OPTIONS,
} from "../kinds.js";

export { PostFlags, buildPostFlags, type PostFlagsValue } from "../post-flags.js";

import {
    DiskModelKind,
    IntegratorKind,
    DISK_MODEL_OPTIONS,
    INTEGRATOR_OPTIONS,
    initWebGpuHost,
    presentRgba8ToCanvas2d,
    tonWasmRustTarget,
    UcfHostHandleRegistry,
    loadTonWasm as loadTonWasmHost,
    renderBlackholeFrame as renderBlackholeFrameHost,
    type LoadTonWasmOptions,
    type PresentTier,
    type RenderBlackholeOptions,
    type RenderBlackholeResult,
    type TonWasmBindings,
    type TonWasmInfo,
} from "@doki-land/blackhole/wasm";

export const rustTarget = "wasm32-unknown-unknown" as const;
export const platformPackage = "blackhole-unknown-wasm32" as const;

const defaultWasmUrl = new URL("../lib/ton_wasm_bg.wasm", import.meta.url);

export async function loadTonWasm(options: LoadTonWasmOptions = {}) {
    return loadTonWasmHost({
        ...options,
        wasmUrl: options.wasmUrl ?? defaultWasmUrl,
        module: options.module,
    });
}

export async function renderBlackholeFrame(
    canvas: HTMLCanvasElement,
    options: RenderBlackholeOptions = {},
): Promise<RenderBlackholeResult> {
    return renderBlackholeFrameHost(canvas, {
        ...options,
        wasmUrl: options.wasmUrl ?? defaultWasmUrl,
    });
}

export {
    tonWasmRustTarget,
    presentRgba8ToCanvas2d,
    initWebGpuHost,
    UcfHostHandleRegistry,
    DiskModelKind,
    IntegratorKind,
    DISK_MODEL_OPTIONS,
    INTEGRATOR_OPTIONS,
    type LoadTonWasmOptions,
    type PresentTier,
    type RenderBlackholeOptions,
    type RenderBlackholeResult,
    type TonWasmBindings,
    type TonWasmInfo,
};

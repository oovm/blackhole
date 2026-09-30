import { loadTonWasm, type LoadTonWasmOptions, type TonWasmBindings } from "./load.js";

/** UCF Web 对齐：L0 = CPU readback → canvas 2D（不经手写 WGSL / `wgpu` crate）。 */
export type PresentTier = "L0";

export interface RenderBlackholeOptions extends LoadTonWasmOptions {
    mass?: number;
    /** RGBA8 写入 wasm 线性内存偏移；默认 65536。 */
    outPtr?: number;
}

export interface RenderBlackholeResult {
    ton: TonWasmBindings;
    tier: PresentTier;
}

const DEFAULT_OUT_PTR = 65536;

export function presentRgba8ToCanvas2d(
    ctx: CanvasRenderingContext2D,
    memory: WebAssembly.Memory,
    outPtr: number,
    width: number,
    height: number,
): void {
    const byteLen = width * height * 4;
    // 必须从 wasm 线性内存拷贝：memory.grow 会使旧 buffer 失效，且部分引擎不接受外部视图。
    const bytes = new Uint8ClampedArray(byteLen);
    bytes.set(new Uint8Array(memory.buffer, outPtr, byteLen));
    ctx.putImageData(new ImageData(bytes, width, height), 0, 0);
}

/**
 * 渲染史瓦西黑洞一帧并上屏（UCF L0：wasm RGBA8 readback → `ImageData`）。
 * GPU 测地线积分后置为 UCF compute 图，不在此手写 WebGPU pipeline。
 */
export async function renderBlackholeFrame(
    canvas: HTMLCanvasElement,
    options: RenderBlackholeOptions = {},
): Promise<RenderBlackholeResult> {
    const width = canvas.width;
    const height = canvas.height;
    const mass = options.mass ?? 1.0;
    const outPtr = options.outPtr ?? DEFAULT_OUT_PTR;
    const byteLen = width * height * 4;

    const ton = await loadTonWasm(options);
    const status = ton.renderSchwarzschildRgba8(width, height, mass, outPtr);
    if (status !== 0) {
        throw new Error(`ton_render_schwarzschild_rgba8 failed: ${status}`);
    }
    if (outPtr + byteLen > ton.memory.buffer.byteLength) {
        throw new Error("ton wasm memory too small for frame readback");
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("canvas 2d context unavailable (do not bind WebGPU on same canvas)");
    }
    presentRgba8ToCanvas2d(ctx, ton.memory, outPtr, width, height);
    return { ton, tier: "L0" };
}

export { loadTonWasm, type LoadTonWasmOptions, type TonWasmBindings, type TonWasmInfo } from "./load.js";

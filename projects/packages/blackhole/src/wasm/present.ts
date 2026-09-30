import { DiskModelKind, IntegratorKind } from "../kinds.js";
import { buildPostFlags, type PostFlagsValue } from "../post-flags.js";
import { initWebGpuHost } from "./ucf/index.js";
import { loadTonWasm, type LoadTonWasmOptions, type TonWasmBindings } from "./load.js";

/** UCF Web 呈现档位：L2 = WebGPU 宿主提交，L0 = CPU readback。 */
export type PresentTier = "L2" | "L0";

export interface RenderBlackholeOptions extends LoadTonWasmOptions {
    mass?: number;
    /** 克尔自旋参数 `a`（几何单位）；仅 `KerrEquatorial` 生效。 */
    spin?: number;
    integrator?: IntegratorKind;
    diskModel?: DiskModelKind;
    /** UCF 色调映射（`POST_UCF_TONE_MAP`）。 */
    ucfToneMap?: boolean;
    /** UCF 泛光（须同时启用 `ucfToneMap`）。 */
    bloom?: boolean;
    /** 直接传入 `post_flags`；覆盖 `ucfToneMap` / `bloom`。 */
    postFlags?: PostFlagsValue;
    /** RGBA8 写入 wasm 线性内存偏移；默认 65536。 */
    outPtr?: number;
    /** 强制 L0（调试）；默认 WebGPU 可用时走 L2。 */
    forceL0?: boolean;
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
    const bytes = new Uint8ClampedArray(byteLen);
    bytes.set(new Uint8Array(memory.buffer, outPtr, byteLen));
    ctx.putImageData(new ImageData(bytes, width, height), 0, 0);
}

function probeWebGpu(): boolean {
    return typeof navigator !== "undefined" && !!navigator.gpu;
}

async function presentL0Readback(
    canvas: HTMLCanvasElement,
    memory: WebAssembly.Memory,
    outPtr: number,
    width: number,
    height: number,
): Promise<void> {
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("canvas 2d context unavailable");
    }
    presentRgba8ToCanvas2d(ctx, memory, outPtr, width, height);
}

async function presentL0Blit(
    target: HTMLCanvasElement,
    memory: WebAssembly.Memory,
    outPtr: number,
    width: number,
    height: number,
): Promise<void> {
    const scratch = document.createElement("canvas");
    scratch.width = width;
    scratch.height = height;
    await presentL0Readback(scratch, memory, outPtr, width, height);
    const ctx = target.getContext("2d");
    if (!ctx) {
        throw new Error("target canvas 2d unavailable for L0 blit");
    }
    ctx.drawImage(scratch, 0, 0);
}

/**
 * 渲染史瓦西黑洞一帧并上屏。
 * CPU 测地线在 `ton-sim`；上屏优先 UCF L2 WebGPU（`ton_l2_present_rgba8_webgpu` + `ucf_host`），降级 L0 readback。
 */
export async function renderBlackholeFrame(
    canvas: HTMLCanvasElement,
    options: RenderBlackholeOptions = {},
): Promise<RenderBlackholeResult> {
    const width = canvas.width;
    const height = canvas.height;
    const mass = options.mass ?? 1.0;
    const spin = options.spin ?? 0.0;
    const integrator = options.integrator ?? IntegratorKind.SchwarzschildOrbit;
    const diskModel = options.diskModel ?? DiskModelKind.NovikovThorneThin;
    const postFlags =
        options.postFlags ??
        buildPostFlags(options.ucfToneMap ?? false, options.bloom ?? false);
    const outPtr = options.outPtr ?? DEFAULT_OUT_PTR;
    const byteLen = width * height * 4;

    const ton = await loadTonWasm(options);
    const status = ton.renderFrameRgba8(
        width,
        height,
        mass,
        spin,
        integrator,
        diskModel,
        postFlags,
        outPtr,
    );
    if (status !== 0) {
        throw new Error(`ton_render_frame_rgba8 failed: ${status}`);
    }
    if (outPtr + byteLen > ton.memory.buffer.byteLength) {
        throw new Error("ton wasm memory too small for frame readback");
    }

    if (!options.forceL0 && probeWebGpu()) {
        try {
            await initWebGpuHost(canvas);
            const l2 = ton.l2PresentRgba8Webgpu(width, height, outPtr, byteLen);
            if (l2 === 0) {
                return { ton, tier: "L2" };
            }
            throw new Error(`ton_l2_present_rgba8_webgpu failed: ${l2}`);
        } catch {
            // WebGPU canvas 可能已占用；降级 L0 blit
        }
    }

    try {
        await presentL0Readback(canvas, ton.memory, outPtr, width, height);
    } catch {
        await presentL0Blit(canvas, ton.memory, outPtr, width, height);
    }
    return { ton, tier: "L0" };
}

export {
    DiskModelKind,
    IntegratorKind,
    DISK_MODEL_OPTIONS,
    INTEGRATOR_OPTIONS,
} from "../kinds.js";
export { PostFlags, buildPostFlags, type PostFlagsValue } from "../post-flags.js";
export { loadTonWasm, type LoadTonWasmOptions, type TonWasmBindings, type TonWasmInfo } from "./load.js";
export { initWebGpuHost, UcfHostHandleRegistry } from "./ucf/index.js";

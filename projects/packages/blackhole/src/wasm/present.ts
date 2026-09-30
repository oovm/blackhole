import { DiskModelKind, IntegratorKind } from "../kinds.js";
import { PostFlags, buildPostFlags, type PostFlagsValue } from "../post-flags.js";
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

export interface ProgressivePassInfo {
    scale: number;
    width: number;
    height: number;
    pass: number;
    passes: number;
    final: boolean;
    tier: PresentTier;
}

export interface RenderBlackholeProgressiveOptions extends RenderBlackholeOptions {
    /** 相对目标分辨率的缩放阶梯，默认 `[0.125, 0.25, 0.5, 1]`。 */
    scales?: number[];
    onPass?: (info: ProgressivePassInfo) => void | Promise<void>;
    /** 返回 `false` 时中止后续 pass（参数已变更等）。 */
    shouldContinue?: () => boolean;
}

const DEFAULT_OUT_PTR = 65536;
const DEFAULT_PROGRESSIVE_SCALES = [0.125, 0.25, 0.5, 1];

function frameByteLen(width: number, height: number): number {
    return width * height * 4;
}

function snapRenderSize(targetW: number, targetH: number, scale: number): { width: number; height: number } {
    const width = Math.max(32, Math.round(targetW * scale));
    const height = Math.max(18, Math.round(targetH * scale));
    return { width, height };
}

export function presentRgba8ToCanvas2d(
    ctx: CanvasRenderingContext2D,
    memory: WebAssembly.Memory,
    outPtr: number,
    width: number,
    height: number,
): void {
    const byteLen = frameByteLen(width, height);
    const bytes = new Uint8ClampedArray(byteLen);
    bytes.set(new Uint8Array(memory.buffer, outPtr, byteLen));
    ctx.putImageData(new ImageData(bytes, width, height), 0, 0);
}

function probeWebGpu(): boolean {
    return typeof navigator !== "undefined" && !!navigator.gpu;
}

function yieldToBrowser(): Promise<void> {
    return new Promise((resolve) => {
        requestAnimationFrame(() => resolve());
    });
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
    ctx.imageSmoothingEnabled = width < target.width || height < target.height;
    ctx.drawImage(scratch, 0, 0, width, height, 0, 0, target.width, target.height);
}

interface FrameParams {
    mass: number;
    spin: number;
    integrator: IntegratorKind;
    diskModel: DiskModelKind;
    postFlags: PostFlagsValue;
    outPtr: number;
}

function resolveFrameParams(
    canvas: HTMLCanvasElement,
    options: RenderBlackholeOptions,
): FrameParams {
    return {
        mass: options.mass ?? 1.0,
        spin: options.spin ?? 0.0,
        integrator: options.integrator ?? IntegratorKind.SchwarzschildOrbit,
        diskModel: options.diskModel ?? DiskModelKind.NovikovThorneThin,
        postFlags:
            options.postFlags ??
            buildPostFlags(options.ucfToneMap ?? false, options.bloom ?? false),
        outPtr: options.outPtr ?? DEFAULT_OUT_PTR,
    };
}

async function presentFrameBuffer(
    canvas: HTMLCanvasElement,
    ton: TonWasmBindings,
    width: number,
    height: number,
    outPtr: number,
    forceL0: boolean,
    webGpuReady: boolean,
): Promise<PresentTier> {
    const byteLen = frameByteLen(width, height);
    if (outPtr + byteLen > ton.memory.buffer.byteLength) {
        throw new Error("ton wasm memory too small for frame readback");
    }

    if (!forceL0 && webGpuReady) {
        const l2 = ton.l2PresentRgba8Webgpu(width, height, outPtr, byteLen);
        if (l2 === 0) {
            return "L2";
        }
    }

    if (width === canvas.width && height === canvas.height) {
        try {
            await presentL0Readback(canvas, ton.memory, outPtr, width, height);
            return "L0";
        } catch {
            await presentL0Blit(canvas, ton.memory, outPtr, width, height);
            return "L0";
        }
    }

    await presentL0Blit(canvas, ton.memory, outPtr, width, height);
    return "L0";
}

async function simulateAndPresentPass(
    canvas: HTMLCanvasElement,
    ton: TonWasmBindings,
    width: number,
    height: number,
    params: FrameParams,
    forceL0: boolean,
    webGpuReady: boolean,
): Promise<PresentTier> {
    const byteLen = frameByteLen(width, height);
    const status = ton.renderFrameRgba8(
        width,
        height,
        params.mass,
        params.spin,
        params.integrator,
        params.diskModel,
        params.postFlags,
        params.outPtr,
    );
    if (status !== 0) {
        throw new Error(`ton_render_frame_rgba8 failed: ${status}`);
    }
    return presentFrameBuffer(canvas, ton, width, height, params.outPtr, forceL0, webGpuReady);
}

/**
 * 渐进多分辨率渲染：先低分辨率快速上屏，再逐级细化到目标分辨率。
 * 测地线仍在 wasm CPU；上屏优先 WebGPU L2（纹理放大到 canvas）。
 */
export async function renderBlackholeFrameProgressive(
    canvas: HTMLCanvasElement,
    options: RenderBlackholeProgressiveOptions = {},
): Promise<RenderBlackholeResult> {
    const targetW = canvas.width;
    const targetH = canvas.height;
    const scales = options.scales ?? DEFAULT_PROGRESSIVE_SCALES;
    const baseParams = resolveFrameParams(canvas, options);
    const finalPostFlags =
        options.postFlags ??
        buildPostFlags(options.ucfToneMap ?? false, options.bloom ?? false);

    const ton = await loadTonWasm(options);
    let webGpuReady = false;
    if (!options.forceL0 && probeWebGpu()) {
        try {
            await initWebGpuHost(canvas);
            webGpuReady = true;
        } catch {
            webGpuReady = false;
        }
    }

    let lastTier: PresentTier = "L0";
    for (let pass = 0; pass < scales.length; pass += 1) {
        if (options.shouldContinue && !options.shouldContinue()) {
            break;
        }
        const scale = scales[pass];
        const final = pass === scales.length - 1;
        const { width, height } = snapRenderSize(targetW, targetH, scale);
        const params: FrameParams = {
            ...baseParams,
            postFlags: final ? finalPostFlags : PostFlags.Raw,
        };
        lastTier = await simulateAndPresentPass(
            canvas,
            ton,
            width,
            height,
            params,
            options.forceL0 ?? false,
            webGpuReady,
        );
        await options.onPass?.({
            scale,
            width,
            height,
            pass,
            passes: scales.length,
            final,
            tier: lastTier,
        });
        if (!final) {
            await yieldToBrowser();
        }
    }

    return { ton, tier: lastTier };
}

/**
 * 渲染史瓦西黑洞一帧并上屏。
 * CPU 测地线在 `ton-simulation` / `ton-renderer`；上屏优先 UCF L2 WebGPU（`ton_l2_present_rgba8_webgpu` + `ucf_host`），降级 L0 readback。
 */
export async function renderBlackholeFrame(
    canvas: HTMLCanvasElement,
    options: RenderBlackholeOptions = {},
): Promise<RenderBlackholeResult> {
    const width = canvas.width;
    const height = canvas.height;
    const params = resolveFrameParams(canvas, options);

    const ton = await loadTonWasm(options);
    let webGpuReady = false;
    if (!options.forceL0 && probeWebGpu()) {
        try {
            await initWebGpuHost(canvas);
            webGpuReady = true;
        } catch {
            webGpuReady = false;
        }
    }

    const tier = await simulateAndPresentPass(
        canvas,
        ton,
        width,
        height,
        params,
        options.forceL0 ?? false,
        webGpuReady,
    );
    return { ton, tier };
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

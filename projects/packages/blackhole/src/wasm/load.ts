import type { DiskModelKind, IntegratorKind } from "../kinds.js";
import {
    buildTonHostImports,
    buildUcfHostImports,
    UcfHostHandleRegistry,
} from "./ucf/index.js";

export const tonWasmRustTarget = "wasm32-unknown-unknown" as const;

export interface TonWasmInfo {
    name: string;
    version: string;
    versionCode: number;
    targetTriple: string;
    ucfDiagMask: number;
}

export interface TonWasmBindings {
    info(): TonWasmInfo;
    renderFrameRgba8(
        width: number,
        height: number,
        mass: number,
        spin: number,
        integrator: IntegratorKind,
        diskModel: DiskModelKind,
        outPtr: number,
    ): number;
    /** @deprecated 使用 `renderFrameRgba8`（史瓦西轨道 + Novikov 薄盘）。 */
    renderSchwarzschildRgba8(
        width: number,
        height: number,
        mass: number,
        outPtr: number,
    ): number;
    l2PresentRgba8Webgpu(width: number, height: number, ptr: number, byteLen: number): number;
    ucfL2RasterTriWebgpu(width: number, height: number): number;
    memory: WebAssembly.Memory;
}

export interface LoadTonWasmOptions {
    wasmUrl?: string | URL;
    module?: WebAssembly.Module;
    registry?: UcfHostHandleRegistry;
}

type WasmExports = {
    memory: WebAssembly.Memory;
    ton_version_code: () => number;
    ton_target_triple: () => number;
    ton_ucf_application_contract_diag_mask: (outPtr: number) => number;
    ton_render_frame_rgba8: (
        width: number,
        height: number,
        mass: number,
        spin: number,
        integrator: number,
        diskModel: number,
        outPtr: number,
    ) => number;
    ton_render_schwarzschild_rgba8: (
        width: number,
        height: number,
        mass: number,
        outPtr: number,
    ) => number;
    ton_l2_present_rgba8_webgpu: (
        width: number,
        height: number,
        ptr: number,
        byteLen: number,
    ) => number;
    ton_ucf_l2_raster_tri_webgpu: (width: number, height: number) => number;
};

function decodeVersion(code: number): string {
    const major = Math.floor(code / 1_000_000);
    const minor = Math.floor((code % 1_000_000) / 1_000);
    const patch = code % 1_000;
    return `${major}.${minor}.${patch}`;
}

function readCString(memory: WebAssembly.Memory, ptr: number): string {
    const view = new Uint8Array(memory.buffer);
    let end = ptr;
    while (view[end] !== 0) {
        end += 1;
    }
    return new TextDecoder().decode(view.subarray(ptr, end));
}

export async function loadTonWasm(options: LoadTonWasmOptions = {}): Promise<TonWasmBindings> {
    if (!options.wasmUrl && !options.module) {
        throw new Error("loadTonWasm requires wasmUrl or module (use @doki-land/blackhole-unknown-wasm32)");
    }
    const url = options.wasmUrl!;
    const memoryRef: { current?: WebAssembly.Memory } = {};
    const registry = options.registry ?? new UcfHostHandleRegistry();
    const imports: WebAssembly.Imports = {
        ton_host: buildTonHostImports(memoryRef),
        ucf_host: buildUcfHostImports(registry),
    };

    let exports: WasmExports;
    if (options.module) {
        const instance = await WebAssembly.instantiate(options.module, imports);
        exports = instance.exports as unknown as WasmExports;
    } else if (typeof fetch !== "function") {
        throw new Error("fetch unavailable: provide wasm module or run in browser");
    } else {
        const result = await WebAssembly.instantiateStreaming(fetch(url.toString()), imports);
        exports = result.instance.exports as unknown as WasmExports;
    }

    if (
        !exports.memory ||
        (!exports.ton_render_frame_rgba8 && !exports.ton_render_schwarzschild_rgba8)
    ) {
        throw new Error("ton_wasm_bg.wasm unavailable: run build:wasm first");
    }

    memoryRef.current = exports.memory;
    const memory = exports.memory;

    return {
        memory,
        info: () => {
            const diagPtr = 512;
            let ucfDiagMask = 0;
            const diagStatus = exports.ton_ucf_application_contract_diag_mask(diagPtr);
            if (diagStatus === 0) {
                ucfDiagMask = new DataView(memory.buffer).getUint32(diagPtr, true);
            }
            return {
                name: "TON Blackhole Simulator",
                version: decodeVersion(exports.ton_version_code()),
                versionCode: exports.ton_version_code(),
                targetTriple: readCString(memory, exports.ton_target_triple()),
                ucfDiagMask,
            };
        },
        renderFrameRgba8: (width, height, mass, spin, integrator, diskModel, outPtr) => {
            if (exports.ton_render_frame_rgba8) {
                return exports.ton_render_frame_rgba8(
                    width,
                    height,
                    mass,
                    spin,
                    integrator,
                    diskModel,
                    outPtr,
                );
            }
            return exports.ton_render_schwarzschild_rgba8(width, height, mass, outPtr);
        },
        renderSchwarzschildRgba8: (width, height, mass, outPtr) =>
            exports.ton_render_schwarzschild_rgba8(width, height, mass, outPtr),
        l2PresentRgba8Webgpu: (width, height, ptr, byteLen) => {
            if (!exports.ton_l2_present_rgba8_webgpu) {
                return -50;
            }
            return exports.ton_l2_present_rgba8_webgpu(width, height, ptr, byteLen);
        },
        ucfL2RasterTriWebgpu: (width, height) => {
            if (!exports.ton_ucf_l2_raster_tri_webgpu) {
                return -50;
            }
            return exports.ton_ucf_l2_raster_tri_webgpu(width, height);
        },
    };
}

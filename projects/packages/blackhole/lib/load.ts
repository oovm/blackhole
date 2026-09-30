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
    renderSchwarzschildRgba8(
        width: number,
        height: number,
        mass: number,
        outPtr: number,
    ): number;
    memory: WebAssembly.Memory;
}

export interface LoadTonWasmOptions {
    wasmUrl?: string | URL;
    module?: WebAssembly.Module;
}

type WasmExports = {
    memory: WebAssembly.Memory;
    ton_version_code: () => number;
    ton_target_triple: () => number;
    ton_ucf_application_contract_diag_mask: (outPtr: number) => number;
    ton_render_schwarzschild_rgba8: (
        width: number,
        height: number,
        mass: number,
        outPtr: number,
    ) => number;
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

/** wasm 仍链入 `ucf-wasm` / `ton_host` import；宿主侧仅 no-op stub，呈现走 L0 readback。 */
function buildImports(): WebAssembly.Imports {
    return {
        ton_host: {
            ton_host_present_rgba8_webgpu: (): number => -30,
        },
        ucf_host: {
            ucf_host_bind_external_buffer: (): number => 0,
            ucf_host_l2_color_fill_webgpu: (): number => 0,
            ucf_host_l2_raster_clear_webgpu: (): number => 0,
            ucf_host_l2_raster_tri_webgpu: (): number => 0,
        },
    };
}

export async function loadTonWasm(options: LoadTonWasmOptions = {}): Promise<TonWasmBindings> {
    const url = options.wasmUrl ?? new URL("../lib/ton_wasm_bg.wasm", import.meta.url);
    const imports = buildImports();

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

    if (!exports.memory || !exports.ton_render_schwarzschild_rgba8) {
        throw new Error("ton_wasm_bg.wasm unavailable: run build:wasm first");
    }

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
        renderSchwarzschildRgba8: (width, height, mass, outPtr) =>
            exports.ton_render_schwarzschild_rgba8(width, height, mass, outPtr),
    };
}

import {
    hostL2ColorFillWebGpu,
    hostL2RasterClearWebGpu,
    hostL2RasterTriWebGpu,
    hostPresentRgba8WebGpu,
} from "./webgpu.js";

export {
    getWebGpuHostState,
    hostL2ColorFillWebGpu,
    hostL2RasterClearWebGpu,
    hostL2RasterTriWebGpu,
    hostPresentRgba8WebGpu,
    initWebGpuHost,
} from "./webgpu.js";
export type { WebGpuHostState } from "./webgpu.js";

/** 逻辑 `resource_id` → 宿主句柄表（UCF `ucf_host_bind_external_buffer`）。 */
export class UcfHostHandleRegistry {
    readonly #handles = new Map<number, { kind: string; native?: unknown }>();
    readonly #bindings = new Map<number, number>();

    register(handleId: number, handle: { kind: string; native?: unknown }): void {
        this.#handles.set(handleId, handle);
    }

    bindResource(resourceId: number, handleId: number): number {
        if (!this.#handles.has(handleId)) {
            return -1;
        }
        this.#bindings.set(resourceId, handleId);
        return 0;
    }
}

export function buildUcfHostImports(registry: UcfHostHandleRegistry): WebAssembly.ModuleImports {
    return {
        ucf_host_bind_external_buffer: (resourceId: number, handleId: number): number =>
            registry.bindResource(resourceId, handleId),
        ucf_host_l2_color_fill_webgpu: (
            width: number,
            height: number,
            r: number,
            g: number,
            b: number,
            a: number,
        ): number => hostL2ColorFillWebGpu(width, height, r, g, b, a),
        ucf_host_l2_raster_clear_webgpu: (
            width: number,
            height: number,
            r: number,
            g: number,
            b: number,
            a: number,
        ): number => hostL2RasterClearWebGpu(width, height, r, g, b, a),
        ucf_host_l2_raster_tri_webgpu: (width: number, height: number): number =>
            hostL2RasterTriWebGpu(width, height),
    } as WebAssembly.ModuleImports;
}

export function buildTonHostImports(memoryRef: {
    current?: WebAssembly.Memory;
}): WebAssembly.ModuleImports {
    return {
        ton_host_present_rgba8_webgpu: (
            width: number,
            height: number,
            ptr: number,
            byteLen: number,
        ): number => {
            const memory = memoryRef.current;
            if (!memory) {
                return -30;
            }
            return hostPresentRgba8WebGpu(width, height, memory, ptr, byteLen);
        },
    } as WebAssembly.ModuleImports;
}

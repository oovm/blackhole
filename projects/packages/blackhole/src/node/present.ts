import { loadTonNapi, type LoadTonNapiOptions, type TonNapiBindings } from "./load.js";

export interface RenderBlackholeRgba8Options extends LoadTonNapiOptions {
    width: number;
    height: number;
    mass?: number;
}

export interface RenderBlackholeRgba8Result {
    ton: TonNapiBindings;
    rgba: Uint8Array;
}

export async function renderBlackholeRgba8(
    options: RenderBlackholeRgba8Options,
): Promise<RenderBlackholeRgba8Result> {
    const { width, height } = options;
    const mass = options.mass ?? 1.0;
    const ton = await loadTonNapi(options);
    const rgba = ton.renderSchwarzschildRgba8(width, height, mass);
    return { ton, rgba };
}

export { loadTonNapi, type LoadTonNapiOptions, type TonNapiBindings } from "./load.js";

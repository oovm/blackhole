import { DiskModelKind, IntegratorKind } from "../kinds.js";
import { loadTonNapi, type LoadTonNapiOptions, type TonNapiBindings } from "./load.js";

export interface RenderBlackholeRgba8Options extends LoadTonNapiOptions {
    width: number;
    height: number;
    mass?: number;
    spin?: number;
    integrator?: IntegratorKind;
    diskModel?: DiskModelKind;
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
    const spin = options.spin ?? 0.0;
    const integrator = options.integrator ?? IntegratorKind.SchwarzschildOrbit;
    const diskModel = options.diskModel ?? DiskModelKind.NovikovThorneThin;
    const ton = await loadTonNapi(options);
    const rgba = ton.renderFrameRgba8(width, height, mass, spin, integrator, diskModel);
    return { ton, rgba };
}

export {
    DiskModelKind,
    IntegratorKind,
    DISK_MODEL_OPTIONS,
    INTEGRATOR_OPTIONS,
} from "../kinds.js";
export { loadTonNapi, type LoadTonNapiOptions, type TonNapiBindings } from "./load.js";

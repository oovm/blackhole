/// <reference types="node" />
import { createRequire } from "node:module";
import type { DiskModelKind, IntegratorKind } from "../kinds.js";
import type { PostFlagsValue } from "../post-flags.js";

export const tonNodeRustTarget = "node-napi" as const;

export interface TonNapiBindings {
    tonVersionCode(): number;
    renderFrameRgba8(
        width: number,
        height: number,
        mass: number,
        spin: number,
        integrator: IntegratorKind,
        diskModel: DiskModelKind,
        postFlags: PostFlagsValue,
    ): Uint8Array;
    renderSchwarzschildRgba8(width: number, height: number, mass: number): Uint8Array;
}

export interface LoadTonNapiOptions {
    addon?: TonNapiBindings;
}

const PLATFORM_PACKAGES: Record<string, string> = {
    "win32-x64": "@doki-land/blackhole-win32-x64",
    "win32-arm64": "@doki-land/blackhole-win32-arm64",
    "darwin-x64": "@doki-land/blackhole-darwin-x64",
    "darwin-arm64": "@doki-land/blackhole-darwin-arm64",
    "linux-x64": "@doki-land/blackhole-linux-x64",
    "linux-arm64": "@doki-land/blackhole-linux-arm64",
};

function resolvePlatformPackage(): string {
    const key = `${process.platform}-${process.arch}`;
    const pkg = PLATFORM_PACKAGES[key];
    if (!pkg) {
        throw new Error(`unsupported node platform: ${key}`);
    }
    return pkg;
}

export async function loadTonNapi(options: LoadTonNapiOptions = {}): Promise<TonNapiBindings> {
    if (options.addon) {
        return options.addon;
    }
    const pkg = resolvePlatformPackage();
    const require = createRequire(import.meta.url);
    const native = require(pkg) as {
        tonVersionCode: () => number;
        renderFrameRgba8?: (
            width: number,
            height: number,
            mass: number,
            spin: number,
            integrator: number,
            diskModel: number,
            postFlags: number,
        ) => Uint8Array;
        renderSchwarzschildRgba8: (width: number, height: number, mass: number) => Uint8Array;
    };
    return {
        tonVersionCode: () => native.tonVersionCode(),
        renderFrameRgba8: (width, height, mass, spin, integrator, diskModel, postFlags) => {
            if (native.renderFrameRgba8) {
                return native.renderFrameRgba8(
                    width,
                    height,
                    mass,
                    spin,
                    integrator,
                    diskModel,
                    postFlags,
                );
            }
            return native.renderSchwarzschildRgba8(width, height, mass);
        },
        renderSchwarzschildRgba8: (width, height, mass) =>
            native.renderSchwarzschildRgba8(width, height, mass),
    };
}

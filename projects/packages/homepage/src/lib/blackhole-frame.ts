import {
    IntegratorKind,
    type DiskModelKind,
    type IntegratorKind as IntegratorKindType,
    renderBlackholeFrame,
} from "@doki-land/blackhole/wasm";

const WASM_URL = "/ton_wasm_bg.wasm";

export interface RenderTonFrameOptions {
    mass: number;
    /** 克尔自旋 `a`（几何单位）；仅 `KerrEquatorial` 生效。 */
    spin: number;
    integrator: IntegratorKindType;
    diskModel: DiskModelKind;
}

export async function renderTonFrame(
    canvas: HTMLCanvasElement,
    options: RenderTonFrameOptions,
): Promise<{ tier: string; statusLine: string }> {
    const spin =
        options.integrator === IntegratorKind.KerrEquatorial ? options.spin : 0;
    const { ton, tier } = await renderBlackholeFrame(canvas, {
        ...options,
        spin,
        wasmUrl: WASM_URL,
    });
    const info = ton.info();
    const spinNote =
        options.integrator === IntegratorKind.KerrEquatorial
            ? ` · a=${spin.toFixed(2)}`
            : "";
    return {
        tier,
        statusLine:
            `${info.name} v${info.version} · UCF diag=0x${info.ucfDiagMask.toString(16)} · ` +
            `present ${tier} · ${canvas.width}×${canvas.height}${spinNote}`,
    };
}

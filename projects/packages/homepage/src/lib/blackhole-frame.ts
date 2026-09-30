import {
    type DiskModelKind,
    type IntegratorKind,
    renderBlackholeFrame,
} from "@doki-land/blackhole/wasm";

const WASM_URL = "/ton_wasm_bg.wasm";

export interface RenderTonFrameOptions {
    mass: number;
    integrator: IntegratorKind;
    diskModel: DiskModelKind;
}

export async function renderTonFrame(
    canvas: HTMLCanvasElement,
    options: RenderTonFrameOptions,
): Promise<{ tier: string; statusLine: string }> {
    const { ton, tier } = await renderBlackholeFrame(canvas, {
        ...options,
        wasmUrl: WASM_URL,
    });
    const info = ton.info();
    return {
        tier,
        statusLine:
            `${info.name} v${info.version} · UCF diag=0x${info.ucfDiagMask.toString(16)} · ` +
            `present ${tier} · ${canvas.width}×${canvas.height}`,
    };
}

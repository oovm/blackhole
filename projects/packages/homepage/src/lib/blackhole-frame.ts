import {
    IntegratorKind,
    PostFlags,
    buildPostFlags,
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
    /** UCF 色调映射（默认 demo 开启）。 */
    ucfToneMap?: boolean;
    /** UCF 泛光（须开启色调映射）。 */
    bloom?: boolean;
}

export async function renderTonFrame(
    canvas: HTMLCanvasElement,
    options: RenderTonFrameOptions,
): Promise<{ tier: string; statusLine: string }> {
    const spin =
        options.integrator === IntegratorKind.KerrEquatorial ? options.spin : 0;
    const ucfToneMap = options.ucfToneMap ?? true;
    const bloom = ucfToneMap && (options.bloom ?? false);
    const postFlags = buildPostFlags(ucfToneMap, bloom);
    const { ton, tier } = await renderBlackholeFrame(canvas, {
        ...options,
        spin,
        postFlags,
        wasmUrl: WASM_URL,
    });
    const info = ton.info();
    const spinNote =
        options.integrator === IntegratorKind.KerrEquatorial
            ? ` · a=${spin.toFixed(2)}`
            : "";
    const postNote =
        postFlags === PostFlags.Raw
            ? ""
            : bloom
              ? " · UCF tone+bloom"
              : " · UCF tone";
    return {
        tier,
        statusLine:
            `${info.name} v${info.version} · UCF diag=0x${info.ucfDiagMask.toString(16)} · ` +
            `present ${tier} · ${canvas.width}×${canvas.height}${spinNote}${postNote}`,
    };
}

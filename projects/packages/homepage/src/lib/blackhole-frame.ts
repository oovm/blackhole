import {
    IntegratorKind,
    PostFlags,
    buildPostFlags,
    type DiskModelKind,
    type IntegratorKind as IntegratorKindType,
    type ProgressivePassInfo,
    renderBlackholeFrameProgressive,
} from "@doki-land/blackhole/wasm";

const WASM_URL = "/ton_wasm_bg.wasm";

export interface RenderTonFrameOptions {
    mass: number;
    /** 克尔自旋 `a`（几何单位）；仅 `KerrEquatorial` 生效。 */
    spin: number;
    integrator: IntegratorKindType;
    diskModel: DiskModelKind;
    /** UCF 色调映射（仅最终 pass 应用）。 */
    ucfToneMap?: boolean;
    /** UCF 泛光（须开启色调映射，仅最终 pass）。 */
    bloom?: boolean;
}

export interface RenderTonFrameHooks {
    onPass?: (info: ProgressivePassInfo) => void | Promise<void>;
    shouldContinue?: () => boolean;
}

export async function renderTonFrame(
    canvas: HTMLCanvasElement,
    options: RenderTonFrameOptions,
    hooks: RenderTonFrameHooks = {},
): Promise<{ tier: string; statusLine: string }> {
    const spin =
        options.integrator === IntegratorKind.KerrEquatorial ? options.spin : 0;
    const ucfToneMap = options.ucfToneMap ?? true;
    const bloom = ucfToneMap && (options.bloom ?? false);
    const postFlags = buildPostFlags(ucfToneMap, bloom);
    const { ton, tier } = await renderBlackholeFrameProgressive(canvas, {
        mass: options.mass,
        spin,
        integrator: options.integrator,
        diskModel: options.diskModel,
        ucfToneMap,
        bloom,
        wasmUrl: WASM_URL,
        onPass: hooks.onPass,
        shouldContinue: hooks.shouldContinue,
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
            `present ${tier} · progressive · ${canvas.width}×${canvas.height}${spinNote}${postNote}`,
    };
}

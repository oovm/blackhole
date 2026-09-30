import { renderBlackholeFrame } from "@ton/blackhole";
import wasmUrl from "@ton/blackhole/ton_wasm_bg.wasm?url";

const canvas = document.getElementById("view") as HTMLCanvasElement;
const massInput = document.getElementById("mass") as HTMLInputElement;
const massVal = document.getElementById("massVal") as HTMLSpanElement;
const widthInput = document.getElementById("width") as HTMLInputElement;
const heightInput = document.getElementById("height") as HTMLInputElement;
const renderBtn = document.getElementById("render") as HTMLButtonElement;
const status = document.getElementById("status") as HTMLParagraphElement;

function syncCanvasSize(): void {
    canvas.width = Number(widthInput.value);
    canvas.height = Number(heightInput.value);
}

massInput.addEventListener("input", () => {
    massVal.textContent = Number(massInput.value).toFixed(1);
});

async function runRender(): Promise<void> {
    syncCanvasSize();
    renderBtn.disabled = true;
    status.textContent = "渲染中…（CPU 测地线，可能需数秒）";

    const t0 = performance.now();
    try {
        const mass = Number(massInput.value);
        const { ton, tier } = await renderBlackholeFrame(canvas, { mass, wasmUrl });
        const info = ton.info();
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        status.textContent =
            `${info.name} v${info.version} · UCF diag=0x${info.ucfDiagMask.toString(16)} · ` +
            `present ${tier} · ${canvas.width}×${canvas.height} · ${dt}s`;
    } catch (err) {
        status.textContent = `失败: ${err instanceof Error ? err.message : String(err)}`;
    } finally {
        renderBtn.disabled = false;
    }
}

renderBtn.addEventListener("click", () => {
    void runRender();
});

widthInput.addEventListener("change", syncCanvasSize);
heightInput.addEventListener("change", syncCanvasSize);

status.textContent = "点击「渲染一帧」开始";

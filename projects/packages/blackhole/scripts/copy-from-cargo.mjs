/**
 * 将 `ton-wasm` release 产物拷到 `lib/ton_wasm_bg.wasm`。
 */
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "../../..");
const src = join(repoRoot, "target/wasm32-unknown-unknown/release/ton_wasm.wasm");
const dest = join(pkgRoot, "lib/ton_wasm_bg.wasm");

if (!existsSync(src)) {
    console.error(
        `未找到 ${src}\n请先：cargo build -p ton-wasm --target wasm32-unknown-unknown --release`,
    );
    process.exit(1);
}

mkdirSync(join(pkgRoot, "lib"), { recursive: true });
copyFileSync(src, dest);
console.log(`已拷贝 → ${dest}`);

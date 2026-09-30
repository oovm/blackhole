import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(here, "..");
const src = path.resolve(pkgRoot, "../blackhole/lib/ton_wasm_bg.wasm");
const publicDir = path.resolve(pkgRoot, "public");
const dst = path.join(publicDir, "ton_wasm_bg.wasm");

if (!fs.existsSync(src)) {
    console.error(`missing ${src} — run: pnpm --filter @ton/blackhole run build:wasm`);
    process.exit(1);
}

fs.mkdirSync(publicDir, { recursive: true });
fs.copyFileSync(src, dst);
console.log(`copied wasm → ${dst}`);

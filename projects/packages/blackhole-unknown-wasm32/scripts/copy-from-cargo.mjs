import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "../../..");
const src = join(repoRoot, "target/wasm32-unknown-unknown/release/ton_wasm.wasm");
const dest = join(pkgRoot, "lib/ton_wasm_bg.wasm");

if (!existsSync(src)) {
    console.error(`missing ${src}\nrun: cargo build -p ton-wasm --target wasm32-unknown-unknown --release`);
    process.exit(1);
}

mkdirSync(join(pkgRoot, "lib"), { recursive: true });
copyFileSync(src, dest);
console.log(`copied → ${dest}`);

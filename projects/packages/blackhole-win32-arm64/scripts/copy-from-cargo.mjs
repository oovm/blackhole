import { copyFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "../../..");
const src = join(repoRoot, "target/aarch64-pc-windows-msvc/release/ton_napi.dll");
const dest = join(pkgRoot, "ton.win32-arm64-msvc.node");

if (!existsSync(src)) {
    console.error(`missing ${src}\nrun build:napi on win32-arm64`);
    process.exit(1);
}

copyFileSync(src, dest);
console.log(`copied → ${dest}`);

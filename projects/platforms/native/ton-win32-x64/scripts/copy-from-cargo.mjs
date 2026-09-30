/**
 * 将 `ton-napi` release 产物拷到平台 npm 袋根目录。
 */
import { copyFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "../../../..");
const src = join(repoRoot, "target/release/ton_napi.dll");
const dest = join(pkgRoot, "ton.win32-x64-msvc.node");

if (!existsSync(src)) {
    console.error(`missing ${src}\nrun: cargo build -p ton-napi --release`);
    process.exit(1);
}

copyFileSync(src, dest);
console.log(`copied → ${dest}`);

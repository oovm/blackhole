import { copyFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "../../..");
const src = join(repoRoot, "target/aarch64-unknown-linux-gnu/release/libton_napi.so");
const dest = join(pkgRoot, "ton.linux-arm64-gnu.node");

if (!existsSync(src)) {
    console.error(`missing ${src}\nrun build:napi on linux-arm64`);
    process.exit(1);
}

copyFileSync(src, dest);
console.log(`copied → ${dest}`);

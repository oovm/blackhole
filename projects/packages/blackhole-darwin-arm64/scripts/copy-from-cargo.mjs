import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "../../..");
const src = join(repoRoot, "target/aarch64-apple-darwin/release/libton_napi.dylib");
const dest = join(pkgRoot, "lib/ton.darwin-arm64.node");

if (!existsSync(src)) {
    console.error(`missing ${src}\nrun build:napi on darwin-arm64`);
    process.exit(1);
}

mkdirSync(join(pkgRoot, "lib"), { recursive: true });
copyFileSync(src, dest);
console.log(`copied → ${dest}`);

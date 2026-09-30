import { defineConfig } from "vite";

export default defineConfig({
    root: ".",
    server: {
        port: 5188,
        open: true,
    },
    build: {
        outDir: "dist",
        emptyOutDir: true,
    },
    assetsInclude: ["**/*.wasm"],
});

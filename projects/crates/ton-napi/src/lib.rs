//! Node N-API 绑定：native 侧调用 `ton-sim`（不经 wasm）。

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ton_sim::Scene;
use ton_types::BlackHole;

/// 与 `ton-wasm` / `ton-view` 对齐的版本编码。
#[napi]
pub fn ton_version_code() -> u32 {
    0_001_000
}

/// 史瓦西 + 薄盘一帧 RGBA8（CPU 测地线）。
#[napi]
pub fn render_schwarzschild_rgba8(width: u32, height: u32, mass: f64) -> Result<Buffer> {
    if width == 0 || height == 0 {
        return Ok(Buffer::from(vec![]));
    }
    let mass = mass.max(0.1);
    let scene = Scene::builder(BlackHole::schwarzschild(mass))
        .with_default_disk()
        .with_resolution(width, height)
        .build();
    Ok(Buffer::from(scene.render_frame().to_rgba8()))
}

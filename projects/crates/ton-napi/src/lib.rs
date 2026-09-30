//! Node N-API 绑定：native 侧调用 `ton-sim`（不经 wasm）。

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ton_sim::{black_hole_for_render, Scene};
use ton_types::{DiskModelKind, IntegratorKind};

/// 与 `ton-wasm` / `ton-view` 对齐的版本编码。
#[napi]
pub fn ton_version_code() -> u32 {
    0_001_000
}

fn render_frame_rgba8_inner(
    width: u32,
    height: u32,
    mass: f64,
    spin: f64,
    integrator: IntegratorKind,
    disk_model: DiskModelKind,
) -> Result<Buffer> {
    if width == 0 || height == 0 {
        return Ok(Buffer::from(vec![]));
    }
    let black_hole = black_hole_for_render(mass, spin, integrator)
        .map_err(|e| Error::from_reason(format!("{e}")))?;
    let mut builder = Scene::builder(black_hole)
        .with_integrator(integrator)
        .with_disk_model(disk_model)
        .with_resolution(width, height);
    if disk_model != DiskModelKind::None {
        builder = builder.with_default_disk();
    }
    Ok(Buffer::from(builder.build().render_frame().to_rgba8()))
}

/// 积分器 × 盘模型 × 自旋一帧 RGBA8（`integrator` / `disk_model` 为 `u8` 判别值）。
#[napi]
pub fn render_frame_rgba8(
    width: u32,
    height: u32,
    mass: f64,
    spin: f64,
    integrator: u8,
    disk_model: u8,
) -> Result<Buffer> {
    let integrator = IntegratorKind::from_u8(integrator)
        .ok_or_else(|| Error::from_reason("invalid integrator discriminant"))?;
    let disk_model = DiskModelKind::from_u8(disk_model)
        .ok_or_else(|| Error::from_reason("invalid disk_model discriminant"))?;
    render_frame_rgba8_inner(width, height, mass, spin, integrator, disk_model)
}

/// 史瓦西轨道 + Novikov 薄盘（兼容旧 API）。
#[napi]
pub fn render_schwarzschild_rgba8(width: u32, height: u32, mass: f64) -> Result<Buffer> {
    render_frame_rgba8_inner(
        width,
        height,
        mass,
        0.0,
        IntegratorKind::SchwarzschildOrbit,
        DiskModelKind::NovikovThorneThin,
    )
}

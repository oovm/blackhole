//! 史瓦西场景一帧 RGBA8 渲染。

use ton_sim::Scene;
use ton_types::BlackHole;

pub fn render_schwarzschild_rgba8(
    width: u32,
    height: u32,
    mass: f64,
) -> Result<Vec<u8>, i32> {
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let mass = mass.max(0.1);
    let scene = Scene::builder(BlackHole::schwarzschild(mass))
        .with_default_disk()
        .with_resolution(width, height)
        .build();
    let fb = scene.render_frame();
    Ok(fb.to_rgba8())
}

/// 写入线性内存；返回 `0` 或负错误码。
///
/// # Safety
///
/// `out` 须可写且至少 `width * height * 4` 字节。
pub unsafe fn write_schwarzschild_rgba8(
    width: u32,
    height: u32,
    mass: f64,
    out: *mut u8,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let need = (width as usize).saturating_mul(height as usize).saturating_mul(4);
    if need == 0 {
        return 0;
    }
    match render_schwarzschild_rgba8(width, height, mass) {
        Ok(bytes) if bytes.len() == need => {
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), out, need);
            }
            0
        }
        Ok(_) => -2,
        Err(code) => code,
    }
}

//! TON Wasm 绑定（`wasm32-unknown-unknown`）。
//!
//! ```text
//! rustup target add wasm32-unknown-unknown
//! cargo build -p ton-wasm --target wasm32-unknown-unknown --release
//! ```

mod host;
mod render;
mod ucf_probe;

pub use host::present_rgba8_webgpu_via_host;
pub use render::{render_schwarzschild_rgba8, write_schwarzschild_rgba8};
pub use ucf_probe::{error_code_to_i32, run_diag_mask};

pub const TARGET_TRIPLE: &str = "wasm32-unknown-unknown";
pub const VERSION_CODE: u32 = 0_001_000;

static TARGET_TRIPLE_CSTR: &[u8] = b"wasm32-unknown-unknown\0";

#[unsafe(no_mangle)]
pub extern "C" fn ton_version_code() -> u32 {
    VERSION_CODE
}

#[unsafe(no_mangle)]
pub extern "C" fn ton_target_triple() -> *const u8 {
    TARGET_TRIPLE_CSTR.as_ptr()
}

/// 史瓦西 + 薄盘一帧 → RGBA8 写入 `out`。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ton_render_schwarzschild_rgba8(
    width: u32,
    height: u32,
    mass: f64,
    out: *mut u8,
) -> i32 {
    unsafe { write_schwarzschild_rgba8(width, height, mass, out) }
}

/// UCF 应用契约诊断掩码（证明同模块 UCF CPU 后端可跑）。
#[unsafe(no_mangle)]
pub extern "C" fn ton_ucf_application_contract_diag_mask(out: *mut u32) -> i32 {
    if out.is_null() {
        return -1;
    }
    match run_diag_mask() {
        Ok(mask) => {
            unsafe { out.write(mask) };
            0
        }
        Err(code) => error_code_to_i32(code),
    }
}

/// L2：经 `ton_host` WebGPU 上传 `out` 指向的 RGBA8 并上屏。
#[unsafe(no_mangle)]
pub extern "C" fn ton_l2_present_rgba8_webgpu(
    width: u32,
    height: u32,
    ptr: u32,
    byte_len: u32,
) -> i32 {
    present_rgba8_webgpu_via_host(width, height, ptr, byte_len)
}

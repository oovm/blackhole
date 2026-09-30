//! Wasm / N-API 后处理标志位（`post_flags: u8`）。

/// 原始 HDR 帧（无 UCF 后处理）。
pub const POST_RAW: u8 = 0;
/// 走 UCF CPU 参考链并应用 `ton_tone_map`。
pub const POST_UCF_TONE_MAP: u8 = 1;
/// 在色调映射前插入 `ton_bloom`（须与 `POST_UCF_TONE_MAP` 同用）。
pub const POST_UCF_BLOOM: u8 = 2;

/// 是否走 UCF 后处理路径（tone map 或 bloom）。
pub fn use_ucf_post(flags: u8) -> bool {
    flags & (POST_UCF_TONE_MAP | POST_UCF_BLOOM) != 0
}

/// 是否启用泛光（仅 UCF 路径有效）。
pub fn post_bloom_from_flags(flags: u8) -> bool {
    flags & POST_UCF_BLOOM != 0
}

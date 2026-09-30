//! HDR 后处理：泛光与色调映射（UCF `ton_bloom` / `ton_tone_map` CPU 参考）。

use ton_types::Rgb;

use crate::Framebuffer;

/// ACES 拟合色调映射（单通道）。
pub fn aces_filmic(x: f64) -> f64 {
    let x = x.max(0.0);
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// 线性 RGB → 显示 RGB（UCF `ton_tone_map`）。
pub fn tone_map_rgb(color: Rgb) -> Rgb {
    Rgb::new(
        aces_filmic(color.r),
        aces_filmic(color.g),
        aces_filmic(color.b),
    )
}

fn luminance(c: Rgb) -> f64 {
    0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
}

/// 简易泛光：阈值提取 + 3×3 盒模糊 + 加回（UCF `ton_bloom`）。
pub fn bloom_pass(fb: &mut Framebuffer, threshold: f64, intensity: f64) {
    let w = fb.width;
    let h = fb.height;
    if w == 0 || h == 0 {
        return;
    }
    let mut bright = vec![Rgb::BLACK; fb.pixels.len()];
    for (i, px) in fb.pixels.iter().enumerate() {
        let l = luminance(*px);
        if l > threshold {
            let scale = (l - threshold) / l.max(1.0e-9);
            bright[i] = px.scale(scale);
        }
    }
    let blurred = box_blur_3x3(&bright, w, h);
    for (dst, add) in fb.pixels.iter_mut().zip(blurred.iter()) {
        *dst = dst.add(add.scale(intensity)).clamp01();
    }
}

fn box_blur_3x3(src: &[Rgb], width: u32, height: u32) -> Vec<Rgb> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![Rgb::BLACK; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = Rgb::BLACK;
            let mut count = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                        sum = sum.add(src[ny as usize * w + nx as usize]);
                        count += 1;
                    }
                }
            }
            let inv = 1.0 / count as f64;
            out[y * w + x] = Rgb::new(sum.r * inv, sum.g * inv, sum.b * inv);
        }
    }
    out
}

/// 对帧缓冲就地应用色调映射。
pub fn apply_tone_map(fb: &mut Framebuffer) {
    for px in &mut fb.pixels {
        *px = tone_map_rgb(*px);
    }
}

/// 后处理链：可选泛光后色调映射。
pub fn apply_post_process(fb: &mut Framebuffer, bloom: bool) {
    if bloom {
        bloom_pass(fb, 0.85, 0.35);
    }
    apply_tone_map(fb);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_map_clamps_hot_pixel() {
        let mapped = tone_map_rgb(Rgb::new(4.0, 2.0, 0.5));
        assert!(mapped.r <= 1.0 && mapped.g <= 1.0 && mapped.b <= 1.0);
        assert!(mapped.r > 0.9);
    }

    #[test]
    fn bloom_spreads_energy_to_neighbors() {
        let mut fb = Framebuffer::new(8, 8);
        fb.set(4, 4, Rgb::new(1.2, 0.9, 0.4));
        bloom_pass(&mut fb, 0.5, 0.5);
        let center = fb.pixels[4 * 8 + 4];
        let neighbor = fb.pixels[4 * 8 + 3];
        assert!(neighbor.r > 0.0);
        assert!(center.r >= neighbor.r);
    }
}

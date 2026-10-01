//! UCF Program IR 驱动的后处理：per-pixel `CpuInterpreter` 执行 + 宿主多尺度模糊。

use ton_types::Rgb;
use ucf_program::{post_bloom_extract, tone_map_aces, CpuInterpreter, Value};

use crate::framebuffer::{Framebuffer, FramebufferEncoding};
use crate::post::{composite_bloom_from_bright, extract_bright_pass};

/// 默认泛光阈值（与 UCF 帧图 `ton_bloom` 任务一致）。
pub const DEFAULT_BLOOM_THRESHOLD: f64 = 0.85;
/// 默认泛光强度。
pub const DEFAULT_BLOOM_INTENSITY: f64 = 0.35;

fn run_bloom_extract(cpu: &mut CpuInterpreter, color: Rgb, threshold: f64) -> Rgb {
    cpu.with_uniform(0, Value::Vec3([color.r, color.g, color.b]));
    cpu.with_uniform(1, Value::F64(threshold));
    let v = cpu
        .run(&post_bloom_extract())
        .expect("post_bloom_extract")
        .value
        .as_vec3()
        .expect("vec3");
    Rgb::new(v[0], v[1], v[2])
}

fn run_tone_map(cpu: &mut CpuInterpreter, color: Rgb) -> Rgb {
    cpu.with_uniform(0, Value::Vec3([color.r, color.g, color.b]));
    let v = cpu
        .run(&tone_map_aces())
        .expect("tone_map_aces")
        .value
        .as_vec3()
        .expect("vec3");
    Rgb::new(v[0], v[1], v[2])
}

/// 阈值亮部提取：每像素执行 UCF `post_bloom_extract`。
pub fn extract_bright_pass_ucf(pixels: &[Rgb], threshold: f64) -> Vec<Rgb> {
    let mut cpu = CpuInterpreter::new();
    pixels
        .iter()
        .map(|px| run_bloom_extract(&mut cpu, *px, threshold))
        .collect()
}

/// 泛光：UCF 亮部提取 + 宿主多尺度模糊合成。
pub fn bloom_pass_ucf(fb: &mut Framebuffer, threshold: f64, intensity: f64) {
    if fb.width == 0 || fb.height == 0 {
        return;
    }
    let bright = extract_bright_pass_ucf(&fb.pixels, threshold);
    composite_bloom_from_bright(fb, &bright, intensity);
}

/// 色调映射：每像素执行 UCF `tone_map_aces`。
pub fn apply_tone_map_ucf(fb: &mut Framebuffer) {
    let mut cpu = CpuInterpreter::new();
    for px in &mut fb.pixels {
        *px = run_tone_map(&mut cpu, *px);
    }
    fb.encoding = FramebufferEncoding::Display;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::post::{apply_tone_map, bloom_pass};

    #[test]
    fn ucf_tone_map_matches_native_on_synthetic_frame() {
        let mut native = Framebuffer::new(8, 8);
        let mut ucf = Framebuffer::new(8, 8);
        for (i, px) in native.pixels.iter_mut().enumerate() {
            let t = i as f64 * 0.07;
            *px = Rgb::new(t, t * 0.5, t * 0.25);
            ucf.pixels[i] = *px;
        }
        apply_tone_map(&mut native);
        apply_tone_map_ucf(&mut ucf);
        for (a, b) in native.pixels.iter().zip(ucf.pixels.iter()) {
            assert!((a.r - b.r).abs() < 1e-9, "r mismatch");
            assert!((a.g - b.g).abs() < 1e-9, "g mismatch");
            assert!((a.b - b.b).abs() < 1e-9, "b mismatch");
        }
    }

    #[test]
    fn ucf_bloom_extract_matches_native() {
        let samples = [
            Rgb::new(0.2, 0.1, 0.05),
            Rgb::new(1.2, 0.8, 0.3),
            Rgb::new(4.0, 2.0, 0.5),
        ];
        let threshold = DEFAULT_BLOOM_THRESHOLD;
        let native = extract_bright_pass(&samples, threshold);
        let ucf = extract_bright_pass_ucf(&samples, threshold);
        for (a, b) in native.iter().zip(ucf.iter()) {
            assert!((a.r - b.r).abs() < 1e-9);
            assert!((a.g - b.g).abs() < 1e-9);
            assert!((a.b - b.b).abs() < 1e-9);
        }
    }

    #[test]
    fn ucf_bloom_pass_matches_native_on_hot_pixel() {
        let mut native = Framebuffer::new(32, 32);
        let mut ucf = Framebuffer::new(32, 32);
        native.set(16, 16, Rgb::new(4.0, 2.0, 0.5));
        ucf.set(16, 16, Rgb::new(4.0, 2.0, 0.5));
        bloom_pass(&mut native, 0.5, 0.5);
        bloom_pass_ucf(&mut ucf, 0.5, 0.5);
        for (a, b) in native.pixels.iter().zip(ucf.pixels.iter()) {
            assert!((a.r - b.r).abs() < 1e-9);
            assert!((a.g - b.g).abs() < 1e-9);
            assert!((a.b - b.b).abs() < 1e-9);
        }
    }
}

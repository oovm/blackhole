//! UCF Program IR 驱动的后处理：per-pixel `CpuInterpreter` 执行 + 宿主多尺度模糊。

use ton_types::Rgb;
use ton_ucf_frontend::{post_bloom_extract, tone_map_aces, CpuInterpreter, Program, UcfBoxBlur3x3Builder, UcfDownsample2xRgbBuilder, UcfUpsampleRgbBuilder, Value};

use crate::framebuffer::{Framebuffer, FramebufferEncoding};
use crate::post::extract_bright_pass;

/// 默认泛光阈值（与 UCF 帧图 `ton_bloom` 任务一致）。
pub const DEFAULT_BLOOM_THRESHOLD: f64 = 1.15;
/// 默认泛光强度。
pub const DEFAULT_BLOOM_INTENSITY: f64 = 0.18;

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
    let bloom = multi_scale_bloom_ucf(&bright, fb.width, fb.height);
    for (dst, add) in fb.pixels.iter_mut().zip(bloom) {
        *dst = dst.add(add.scale(intensity));
    }
}

fn execute_rgb_stage(program: &Program, input: &[f32], count: u32) -> Vec<f32> {
    let mut cpu = CpuInterpreter::new();
    cpu.with_buffer(1, input.iter().copied().map(Value::F32).collect())
        .with_buffer(2, vec![Value::F32(0.0); count as usize * 3]);
    cpu.run_parallel(program, count).expect("UCF RGB stage");
    cpu.buffer(2).expect("UCF RGB output").iter().map(|value| value.as_f64().expect("UCF F32 output") as f32).collect()
}

fn multi_scale_bloom_ucf(src: &[Rgb], width: u32, height: u32) -> Vec<Rgb> {
    let input = src.iter().flat_map(|pixel| [pixel.r as f32, pixel.g as f32, pixel.b as f32]).collect::<Vec<_>>();
    let count = width.saturating_mul(height);
    let blur = |values: &[f32], w: u32, h: u32| {
        let program = UcfBoxBlur3x3Builder::new(w, h).build().expect("UCF blur builder");
        execute_rgb_stage(&program, values, w.saturating_mul(h))
    };
    let l0 = blur(&input, width, height);
    let (w2, h2) = (width / 2, height / 2);
    let mut l1 = vec![0.0; input.len()];
    let mut l2 = vec![0.0; input.len()];
    if w2 > 0 && h2 > 0 {
        let downsample1 = UcfDownsample2xRgbBuilder::new(width, height).build().expect("UCF downsample builder");
        let down1 = execute_rgb_stage(&downsample1, &input, w2 * h2);
        let blur1 = blur(&down1, w2, h2);
        let upsample1 = UcfUpsampleRgbBuilder::new(w2, h2, width, height, 0.55).build().expect("UCF upsample builder");
        l1 = execute_rgb_stage(&upsample1, &blur1, count);
        let (w4, h4) = (w2 / 2, h2 / 2);
        if w4 > 0 && h4 > 0 {
            let downsample2 = UcfDownsample2xRgbBuilder::new(w2, h2).build().expect("UCF downsample builder");
            let down2 = execute_rgb_stage(&downsample2, &down1, w4 * h4);
            let blur2 = blur(&down2, w4, h4);
            let upsample2_mid = UcfUpsampleRgbBuilder::new(w4, h4, w2, h2, 1.0).build().expect("UCF upsample builder");
            let up2_mid = execute_rgb_stage(&upsample2_mid, &blur2, w2 * h2);
            let upsample2 = UcfUpsampleRgbBuilder::new(w2, h2, width, height, 0.35).build().expect("UCF upsample builder");
            l2 = execute_rgb_stage(&upsample2, &up2_mid, count);
        }
    }
    input.chunks_exact(3).zip(l0.chunks_exact(3)).zip(l1.chunks_exact(3)).zip(l2.chunks_exact(3))
        .map(|(((base, a), b), c)| Rgb::new(
            (base[0] + a[0] + b[0] + c[0]) as f64,
            (base[1] + a[1] + b[1] + c[1]) as f64,
            (base[2] + a[2] + b[2] + c[2]) as f64,
        ))
        .collect()
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
            assert!((a.r - b.r).abs() < 1e-5);
            assert!((a.g - b.g).abs() < 1e-5);
            assert!((a.b - b.b).abs() < 1e-5);
        }
    }
}

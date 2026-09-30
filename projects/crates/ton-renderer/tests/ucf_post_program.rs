//! UCF 后处理程序与 Rust 参考实现对照。

use ton_renderer::{aces_filmic, tone_map_rgb};
use ton_types::Rgb;
use ucf_program::{post_bloom_extract, tone_map_aces, CpuInterpreter, Value};

fn run_tone_map(cpu: &mut CpuInterpreter, color: Rgb) -> [f64; 3] {
    cpu.with_uniform(0, Value::Vec3([color.r, color.g, color.b]));
    cpu.run(&tone_map_aces())
        .expect("run")
        .value
        .as_vec3()
        .expect("vec3")
}

fn run_bloom_extract(cpu: &mut CpuInterpreter, color: Rgb, threshold: f64) -> [f64; 3] {
    cpu.with_uniform(0, Value::Vec3([color.r, color.g, color.b]));
    cpu.with_uniform(1, Value::F64(threshold));
    cpu.run(&post_bloom_extract())
        .expect("run")
        .value
        .as_vec3()
        .expect("vec3")
}

fn bloom_extract_native(color: Rgb, threshold: f64) -> Rgb {
    let l = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    if l > threshold {
        let scale = (l - threshold) / l.max(1.0e-9);
        color.scale(scale)
    } else {
        Rgb::BLACK
    }
}

#[test]
fn ucf_tone_map_matches_aces_filmic() {
    let samples = [
        Rgb::new(0.0, 0.0, 0.0),
        Rgb::new(0.25, 0.5, 1.0),
        Rgb::new(4.0, 2.0, 0.5),
        Rgb::new(12.0, 8.0, 3.0),
    ];
    let mut cpu = CpuInterpreter::new();
    for color in samples {
        let rust = tone_map_rgb(color);
        let ucf = run_tone_map(&mut cpu, color);
        for i in 0..3 {
            assert!(
                (ucf[i] - [rust.r, rust.g, rust.b][i]).abs() < 1e-6,
                "color={color:?} ch={i} ucf={} rust={}",
                ucf[i],
                [rust.r, rust.g, rust.b][i]
            );
        }
        assert_eq!(aces_filmic(color.r), ucf[0]);
    }
}

#[test]
fn ucf_bloom_extract_matches_native_threshold() {
    let threshold = 0.85;
    let samples = [
        Rgb::new(0.2, 0.1, 0.05),
        Rgb::new(1.2, 0.8, 0.3),
        Rgb::new(4.0, 2.0, 0.5),
    ];
    let mut cpu = CpuInterpreter::new();
    for color in samples {
        let rust = bloom_extract_native(color, threshold);
        let ucf = run_bloom_extract(&mut cpu, color, threshold);
        for i in 0..3 {
            assert!(
                (ucf[i] - [rust.r, rust.g, rust.b][i]).abs() < 1e-6,
                "color={color:?} ch={i} ucf={} rust={}",
                ucf[i],
                [rust.r, rust.g, rust.b][i]
            );
        }
    }
}

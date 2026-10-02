//! UCF Program IR 驱动的盘面着色：`disk_shade_novikov` per-sample `CpuInterpreter`。

use ton_disk::{AccretionDisk, DiskSample};
use ton_types::Rgb;
use ton_ucf_frontend::{disk_shade_novikov, disk_shade_novikov_batch, CpuInterpreter, Value};

fn run_disk_shade(
    cpu: &mut CpuInterpreter,
    disk: AccretionDisk,
    radius: f64,
    doppler: f64,
) -> [f64; 3] {
    let cfg = disk.config;
    cpu.with_uniform(0, Value::F64(radius));
    cpu.with_uniform(1, Value::F64(doppler));
    cpu.with_uniform(2, Value::F64(cfg.inner_radius));
    cpu.with_uniform(3, Value::F64(cfg.outer_radius));
    cpu.with_uniform(4, Value::F64(disk.bh.mass));
    cpu.with_uniform(5, Value::F64(cfg.accretion_rate));
    cpu.run(&disk_shade_novikov())
        .expect("disk_shade_novikov")
        .value
        .as_vec3()
        .expect("vec3")
}

/// 用 UCF `disk_shade_novikov` 采样盘面辐射；吸收系数由宿主从峰值推导（与 native 对照一致）。
pub fn sample_disk_radiance_ucf(disk: AccretionDisk, radius: f64, doppler: f64) -> DiskSample {
    let mut cpu = CpuInterpreter::new();
    let rgb = run_disk_shade(&mut cpu, disk, radius, doppler);
    let emissive = Rgb::new(rgb[0], rgb[1], rgb[2]);
    let peak = emissive.r.max(emissive.g).max(emissive.b);
    DiskSample {
        emissive_radiance: emissive,
        absorption: (peak * 0.18).clamp(0.04, 0.55),
        temperature: disk.effective_temperature(radius),
        redshift_factor: doppler,
    }
}

/// 使用并行 UCF Program 批量采样盘面辐射。
///
/// 输入和输出均通过 UCF buffer 传递，便于 CPU 与原生 GPU backend 共享同一
/// 资源契约。输出顺序与 `samples` 一致。
pub fn sample_disk_radiance_ucf_batch(
    disk: AccretionDisk,
    samples: &[(f64, f64)],
) -> Vec<DiskSample> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F32(disk.config.inner_radius as f32));
    cpu.with_uniform(1, Value::F32(disk.config.outer_radius as f32));
    cpu.with_uniform(2, Value::F32(disk.bh.mass as f32));
    cpu.with_uniform(3, Value::F32(disk.config.accretion_rate as f32));
    cpu.with_buffer(
        1,
        samples
            .iter()
            .flat_map(|(radius, doppler)| [Value::F32(*radius as f32), Value::F32(*doppler as f32)])
            .collect(),
    );
    cpu.with_buffer(2, vec![Value::F32(0.0); samples.len() * 3]);
    cpu.run_parallel(&disk_shade_novikov_batch(), samples.len() as u32)
        .expect("disk_shade_novikov_batch");
    let output = cpu.buffer(2).expect("disk shade output buffer");
    samples
        .iter()
        .enumerate()
        .map(|(index, (_, doppler))| {
            let base = index * 3;
            let emissive = Rgb::new(
                output[base].as_f64().expect("red output"),
                output[base + 1].as_f64().expect("green output"),
                output[base + 2].as_f64().expect("blue output"),
            );
            let peak = emissive.r.max(emissive.g).max(emissive.b);
            DiskSample {
                emissive_radiance: emissive,
                absorption: (peak * 0.18).clamp(0.04, 0.55),
                temperature: disk.effective_temperature(samples[index].0),
                redshift_factor: *doppler,
            }
        })
        .collect()
}

/// 复用同一解释器批量采样（帧内热路径）。
pub struct DiskShadeUcf {
    cpu: CpuInterpreter,
}

impl DiskShadeUcf {
    pub fn new() -> Self {
        Self {
            cpu: CpuInterpreter::new(),
        }
    }

    pub fn sample(&mut self, disk: AccretionDisk, radius: f64, doppler: f64) -> DiskSample {
        let rgb = run_disk_shade(&mut self.cpu, disk, radius, doppler);
        let emissive = Rgb::new(rgb[0], rgb[1], rgb[2]);
        let peak = emissive.r.max(emissive.g).max(emissive.b);
        DiskSample {
            emissive_radiance: emissive,
            absorption: (peak * 0.18).clamp(0.04, 0.55),
            temperature: disk.effective_temperature(radius),
            redshift_factor: doppler,
        }
    }
}

impl Default for DiskShadeUcf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_disk::{sample_disk_radiance, DiskConfig};
    use ton_types::BlackHole;

    #[test]
    fn ucf_disk_shade_matches_native_samples() {
        let bh = BlackHole::schwarzschild(1.0);
        let disk = AccretionDisk::new(bh, DiskConfig::schwarzschild_default(bh));
        let mut ucf = DiskShadeUcf::new();
        for (r, d) in [(10.0, 1.0), (12.0, 1.5), (18.0, 2.2), (24.0, 0.8)] {
            let a = sample_disk_radiance(disk, r, d);
            let b = ucf.sample(disk, r, d);
            assert!((a.emissive_radiance.r - b.emissive_radiance.r).abs() < 1e-6);
            assert!((a.emissive_radiance.g - b.emissive_radiance.g).abs() < 1e-6);
            assert!((a.emissive_radiance.b - b.emissive_radiance.b).abs() < 1e-6);
            assert!((a.absorption - b.absorption).abs() < 1e-6);
        }
    }

    #[test]
    fn ucf_disk_shade_batch_matches_scalar_program() {
        let bh = BlackHole::schwarzschild(1.0);
        let disk = AccretionDisk::new(bh, DiskConfig::schwarzschild_default(bh));
        let samples = [(10.0, 1.0), (12.0, 1.5), (18.0, 2.2), (24.0, 0.8)];
        let batch = sample_disk_radiance_ucf_batch(disk, &samples);
        let scalar = samples
            .iter()
            .map(|(radius, doppler)| sample_disk_radiance_ucf(disk, *radius, *doppler))
            .collect::<Vec<_>>();
        for (actual, expected) in batch.iter().zip(scalar.iter()) {
            assert!((actual.emissive_radiance.r - expected.emissive_radiance.r).abs() < 2e-4);
            assert!((actual.emissive_radiance.g - expected.emissive_radiance.g).abs() < 2e-4);
            assert!((actual.emissive_radiance.b - expected.emissive_radiance.b).abs() < 2e-4);
        }
    }
}

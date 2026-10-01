//! UCF Program IR 驱动的盘面着色：`disk_shade_novikov` per-sample `CpuInterpreter`。

use ton_disk::{AccretionDisk, DiskSample};
use ton_types::Rgb;
use ucf_program::{disk_shade_novikov, CpuInterpreter, Value};

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
}

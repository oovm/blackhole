//! UCF Novikov-Thorne 盘面着色与 Rust 参考实现对照。

use ton_disk::{sample_disk_radiance, AccretionDisk, DiskConfig};
use ton_types::BlackHole;
use ton_ucf_frontend::{disk_shade_novikov, CpuInterpreter, Value};

fn run_disk_shade(
    cpu: &mut CpuInterpreter,
    radius: f64,
    doppler: f64,
    disk: AccretionDisk,
) -> [f64; 3] {
    let cfg = disk.config;
    cpu.with_uniform(0, Value::F64(radius));
    cpu.with_uniform(1, Value::F64(doppler));
    cpu.with_uniform(2, Value::F64(cfg.inner_radius));
    cpu.with_uniform(3, Value::F64(cfg.outer_radius));
    cpu.with_uniform(4, Value::F64(disk.bh.mass));
    cpu.with_uniform(5, Value::F64(cfg.accretion_rate));
    cpu.run(&disk_shade_novikov())
        .expect("run")
        .value
        .as_vec3()
        .expect("vec3")
}

#[test]
fn ucf_disk_shade_matches_sample_disk_radiance() {
    let bh = BlackHole::schwarzschild(1.0);
    let disk = AccretionDisk::new(bh, DiskConfig::schwarzschild_default(bh));
    let mut cpu = CpuInterpreter::new();

    let samples = [(10.0, 1.0), (12.0, 1.5), (18.0, 2.2), (24.0, 0.8)];
    for (radius, doppler) in samples {
        let rust = sample_disk_radiance(disk, radius, doppler).emissive_radiance;
        let ucf = run_disk_shade(&mut cpu, radius, doppler, disk);
        for i in 0..3 {
            assert!(
                (ucf[i] - [rust.r, rust.g, rust.b][i]).abs() < 1e-6,
                "r={radius} doppler={doppler} ch={i} ucf={} rust={}",
                ucf[i],
                [rust.r, rust.g, rust.b][i]
            );
        }
    }
}

#[test]
fn ucf_disk_shade_outside_disk_is_black() {
    let bh = BlackHole::schwarzschild(1.0);
    let disk = AccretionDisk::new(bh, DiskConfig::schwarzschild_default(bh));
    let mut cpu = CpuInterpreter::new();
    let ucf = run_disk_shade(&mut cpu, 2.0, 1.0, disk);
    assert!(ucf[0].abs() < 1e-9);
    assert!(ucf[1].abs() < 1e-9);
    assert!(ucf[2].abs() < 1e-9);
}

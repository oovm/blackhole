//! UCF Program IR 驱动的相机射线生成：`generate_camera_ray`。

use ton_types::{Camera, Vec3};
use ton_ucf_frontend::{generate_camera_ray, CpuInterpreter, Value};

fn run_camera_ray(cpu: &mut CpuInterpreter, cam: Camera, u: f64, v: f64) -> Vec3 {
    let scale = (cam.fov_y * 0.5).tan();
    cpu.with_uniform(0, Value::F64(u));
    cpu.with_uniform(1, Value::F64(v));
    cpu.with_uniform(2, Value::Vec3([cam.forward.x, cam.forward.y, cam.forward.z]));
    cpu.with_uniform(3, Value::Vec3([cam.right.x, cam.right.y, cam.right.z]));
    cpu.with_uniform(4, Value::Vec3([cam.up.x, cam.up.y, cam.up.z]));
    cpu.with_uniform(5, Value::F64(scale));
    let d = cpu
        .run(&generate_camera_ray())
        .expect("generate_camera_ray")
        .value
        .as_vec3()
        .expect("vec3");
    Vec3::new(d[0], d[1], d[2])
}

/// 单次调用 UCF 程序生成射线方向。
pub fn camera_ray_direction_ucf(cam: Camera, u: f64, v: f64) -> Vec3 {
    let mut cpu = CpuInterpreter::new();
    run_camera_ray(&mut cpu, cam, u, v)
}

/// 复用解释器的帧内射线生成器。
pub struct CameraRayUcf {
    cpu: CpuInterpreter,
}

impl CameraRayUcf {
    pub fn new() -> Self {
        Self {
            cpu: CpuInterpreter::new(),
        }
    }

    pub fn direction(&mut self, cam: Camera, u: f64, v: f64) -> Vec3 {
        run_camera_ray(&mut self.cpu, cam, u, v)
    }
}

impl Default for CameraRayUcf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::BlackHole;

    #[test]
    fn ucf_camera_ray_matches_native() {
        let cam = Camera::cinematic(BlackHole::schwarzschild(1.0));
        let mut ucf = CameraRayUcf::new();
        for (u, v) in [(0.0, 0.0), (0.5, -0.25), (-0.8, 0.4), (1.2, -0.9)] {
            let a = cam.ray_direction(u, v);
            let b = ucf.direction(cam, u, v);
            assert!((a.x - b.x).abs() < 1e-9, "u={u} v={v} x {a:?} vs {b:?}");
            assert!((a.y - b.y).abs() < 1e-9);
            assert!((a.z - b.z).abs() < 1e-9);
        }
    }
}

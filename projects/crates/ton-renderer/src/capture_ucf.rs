//! UCF `geodesic_capture_classify`：史瓦西解析捕获分类。

use ton_geodesic::schwarzschild_geodesic_impact_parameter;
use ton_types::{BlackHole, Vec3};
use ucf_program::{geodesic_capture_classify, CpuInterpreter, Value};

/// 用 UCF 程序按冲击参数判定是否捕获（与解析 `b < b_c` 一致）。
pub fn analytic_captured_ucf(bh: BlackHole, origin: Vec3, direction: Vec3) -> bool {
    let dir = direction.normalize();
    let Some(b) = schwarzschild_geodesic_impact_parameter(bh.mass, origin, dir) else {
        return false;
    };
    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(b));
    cpu.with_uniform(1, Value::F64(bh.critical_impact_parameter()));
    let flag = cpu
        .run(&geodesic_capture_classify())
        .expect("geodesic_capture_classify")
        .value
        .as_f64()
        .expect("f64");
    flag > 0.5
}

/// 复用解释器的捕获分类器（帧级网格扫描）。
pub struct CaptureClassifyUcf {
    cpu: CpuInterpreter,
    program: ucf_program::Program,
    bc: f64,
}

impl CaptureClassifyUcf {
    pub fn new(bh: BlackHole) -> Self {
        Self {
            cpu: CpuInterpreter::new(),
            program: geodesic_capture_classify(),
            bc: bh.critical_impact_parameter(),
        }
    }

    pub fn captured(&mut self, bh: BlackHole, origin: Vec3, direction: Vec3) -> bool {
        let dir = direction.normalize();
        let Some(b) = schwarzschild_geodesic_impact_parameter(bh.mass, origin, dir) else {
            return false;
        };
        self.cpu.with_uniform(0, Value::F64(b));
        self.cpu.with_uniform(1, Value::F64(self.bc));
        let flag = self
            .cpu
            .run(&self.program)
            .expect("geodesic_capture_classify")
            .value
            .as_f64()
            .expect("f64");
        flag > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_geodesic::schwarzschild_geodesic_analytic_captured;
    use ton_types::Camera;

    use crate::shadow_overlay::pixel_ray_dir;

    #[test]
    fn ucf_capture_grid_matches_analytic() {
        let bh = BlackHole::schwarzschild(1.0);
        let cam = Camera::cinematic(bh);
        let mut ucf = CaptureClassifyUcf::new(bh);
        let (w, h) = (48u32, 27u32);
        let mut agree = 0u32;
        for y in 0..h {
            for x in 0..w {
                let dir = pixel_ray_dir(x, y, w, h, cam);
                let a = schwarzschild_geodesic_analytic_captured(bh, cam.position, dir);
                let b = ucf.captured(bh, cam.position, dir);
                if a == b {
                    agree += 1;
                }
            }
        }
        assert_eq!(agree, w * h);
    }
}

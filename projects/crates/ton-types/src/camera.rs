use crate::{BlackHole, Vec3};

/// 针孔相机（右手系，看向 -Z 或自定义）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub right: Vec3,
    /// 垂直视场（弧度）。
    pub fov_y: f64,
}

impl Camera {
    /// 由位置、目标点与上方向构造相机。
    pub fn look_at(position: Vec3, target: Vec3, up: Vec3, fov_y: f64) -> Self {
        let forward = target.sub(position).normalize();
        let right = forward.cross(up).normalize();
        let up = right.cross(forward).normalize();
        Self {
            position,
            forward,
            up,
            right,
            fov_y,
        }
    }

    /// 归一化设备坐标 (-1..1) 对应的世界空间射线方向。
    pub fn ray_direction(self, u: f64, v: f64) -> Vec3 {
        let aspect_scale = (self.fov_y * 0.5).tan();
        let dir = self.forward
            .add(self.right.scale(u * aspect_scale))
            .add(self.up.scale(v * aspect_scale));
        dir.normalize()
    }

    /// 电影构图预设：距离约 10M、盘面倾角 ~70°、较窄视场，阴影约占画面高度 35%～55%。
    pub fn cinematic(bh: BlackHole) -> Self {
        let m = bh.mass;
        let dist = 10.0 * m;
        let inclination = 70.0 * std::f64::consts::PI / 180.0;
        let azimuth: f64 = 0.35;
        let sin_i = inclination.sin();
        let cos_i = inclination.cos();
        let position = Vec3::new(
            dist * sin_i * azimuth.cos(),
            dist * cos_i,
            dist * sin_i * azimuth.sin(),
        );
        let target = Vec3::ZERO;
        let up = Vec3::new(0.0, 1.0, 0.0);
        // ~49° 垂直视场，配合 10M 距离使 b_c/D 阴影角占画面主体。
        let fov_y = 0.85;
        Self::look_at(position, target, up, fov_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cinematic_looks_at_origin() {
        let bh = BlackHole::schwarzschild(1.0);
        let cam = Camera::cinematic(bh);
        let to_center = Vec3::ZERO.sub(cam.position).normalize();
        let align = cam.forward.dot(to_center);
        assert!(align > 0.99);
    }
}

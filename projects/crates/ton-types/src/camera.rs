use crate::{BlackHole, DiskFrame, Vec3};

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

    /// 相对薄盘参考系的电影构图：`inclination` 为与盘面法线的极角（0=正视盘面，π/2=边视）。
    pub fn cinematic_over_disk(
        bh: BlackHole,
        disk: DiskFrame,
        inclination_rad: f64,
        azimuth_rad: f64,
    ) -> Self {
        let dist = 10.0 * bh.mass;
        let (t1, t2, n) = disk.tangent_basis();
        let sin_i = inclination_rad.sin();
        let cos_i = inclination_rad.cos();
        let position = n
            .scale(dist * cos_i)
            .add(t1.scale(dist * sin_i * azimuth_rad.cos()))
            .add(t2.scale(dist * sin_i * azimuth_rad.sin()));
        Self::look_at(position, Vec3::ZERO, t1, 0.85)
    }

    /// 电影构图预设：赤道 XY 盘面、相对法线倾角约 70°（近边视，可看到透镜抬升）。
    pub fn cinematic(bh: BlackHole) -> Self {
        let inclination = 70.0 * std::f64::consts::PI / 180.0;
        Self::cinematic_over_disk(bh, DiskFrame::equatorial_xy(), inclination, 0.35)
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

    #[test]
    fn cinematic_inclination_relative_to_disk_normal() {
        let bh = BlackHole::schwarzschild(1.0);
        let disk = DiskFrame::equatorial_xy();
        let (_, _, n) = disk.tangent_basis();
        let cam = Camera::cinematic(bh);
        let pos = cam.position.normalize();
        let angle_from_normal = n.dot(pos).acos();
        let expected = 70.0 * std::f64::consts::PI / 180.0;
        assert!((angle_from_normal - expected).abs() < 0.02);
    }
}

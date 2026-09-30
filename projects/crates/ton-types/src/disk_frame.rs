use crate::Vec3;

/// 薄吸积盘参考系：盘面位于 `normal` 过原点的平面内。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskFrame {
    /// 盘面法线（单位向量），默认 +Z。
    pub normal: Vec3,
    /// 盘面内参考上方向（用于方位角），须与 `normal` 正交。
    pub reference_up: Vec3,
}

impl Default for DiskFrame {
    fn default() -> Self {
        Self::equatorial_xy()
    }
}

impl DiskFrame {
    /// 赤道薄盘：盘面为 XY 平面，法线 +Z，参考上方向 +Y。
    pub fn equatorial_xy() -> Self {
        Self {
            normal: Vec3::new(0.0, 0.0, 1.0),
            reference_up: Vec3::new(0.0, 1.0, 0.0),
        }
    }

    /// 盘面内切向基 `tangent1`（参考上方向投影）与 `tangent2`。
    pub fn tangent_basis(self) -> (Vec3, Vec3, Vec3) {
        let n = self.normal.normalize();
        let up_proj = self
            .reference_up
            .sub(n.scale(self.reference_up.dot(n)))
            .normalize();
        let t1 = up_proj;
        let t2 = n.cross(t1).normalize();
        (t1, t2, n)
    }
}

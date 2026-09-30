//! 半经典 Ray March 用加速度场（\(G=c=1\) 几何单位）。

use ton_types::{BlackHole, Vec3};

/// 史瓦西黑洞在原点：\( \vec{a} = M/r^2 \cdot (-\hat{r}) \)（教程同族的半经典近似）。
pub fn semi_classical_acceleration(bh: BlackHole, pos: Vec3) -> Vec3 {
    let r = pos.length();
    if r <= f64::EPSILON || !bh.mass.is_finite() || bh.mass <= 0.0 {
        return Vec3::ZERO;
    }
    let toward = pos.scale(-1.0 / r);
    toward.scale(bh.mass / (r * r))
}

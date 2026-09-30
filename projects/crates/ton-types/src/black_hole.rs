use serde::{Deserialize, Serialize};

/// 黑洞参数（几何单位，G = c = 1）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BlackHole {
    /// 质量 M。
    pub mass: f64,
    /// 克尔自旋参数 a = J / (Mc)，|a| < M。
    pub spin: f64,
}

impl BlackHole {
    /// 史瓦西黑洞（无自旋）。
    pub fn schwarzschild(mass: f64) -> Self {
        Self { mass, spin: 0.0 }
    }

    /// 克尔黑洞；自旋参数 clamp 到 `|a| < M`。
    pub fn kerr(mass: f64, spin: f64) -> Self {
        Self {
            mass,
            spin: spin.clamp(-mass, mass),
        }
    }

    /// TON 618 量级参考（约 6.6×10¹⁰ 太阳质量，取对数刻度演示用）。
    pub fn ton618_scale() -> Self {
        Self::schwarzschild(6.6e10)
    }

    /// 事件视界半径（史瓦西：2M；克尔赤道近似取 r₊）。
    pub fn event_horizon_radius(self) -> f64 {
        if self.spin.abs() < f64::EPSILON {
            2.0 * self.mass
        } else {
            let m = self.mass;
            let a = self.spin.clamp(-m, m);
            m + (m * m - a * a).sqrt()
        }
    }

    /// 光子球半径（史瓦西：3M）。
    pub fn photon_sphere_radius(self) -> f64 {
        if self.spin.abs() < f64::EPSILON {
            3.0 * self.mass
        } else {
            // 赤道平面近似。
            3.0 * self.mass
        }
    }

    /// 临界冲击参数 b_c（史瓦西：3√3 M）。光子球擦边轨道的角动量尺度。
    pub fn critical_impact_parameter(self) -> f64 {
        if self.spin.abs() < f64::EPSILON {
            3.0_f64 * 3.0_f64.sqrt() * self.mass
        } else {
            // 克尔赤道第一版仍用史瓦西近似，后续替换为精确值。
            3.0_f64 * 3.0_f64.sqrt() * self.mass
        }
    }

    /// 远处观察者看到的阴影视角半径 θ ≈ b_c / D（小角近似）。
    pub fn shadow_angular_radius(self, observer_distance: f64) -> f64 {
        let d = observer_distance.max(self.mass * 1e-6);
        self.critical_impact_parameter() / d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schwarzschild_optical_radii() {
        let bh = BlackHole::schwarzschild(1.0);
        assert!((bh.event_horizon_radius() - 2.0).abs() < 1e-12);
        assert!((bh.photon_sphere_radius() - 3.0).abs() < 1e-12);
        assert!((bh.critical_impact_parameter() - 3.0_f64 * 3.0_f64.sqrt()).abs() < 1e-12);
        let theta = bh.shadow_angular_radius(10.0);
        assert!((theta - 3.0_f64 * 3.0_f64.sqrt() / 10.0).abs() < 1e-12);
    }
}

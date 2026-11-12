use ton_types::BlackHole;

use crate::metric::Metric;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kerr {
    pub bh: BlackHole,
}

impl Kerr {
    pub fn new(bh: BlackHole) -> Self {
        Self { bh }
    }

    fn sigma(&self, r: f64) -> f64 {
        let a = self.bh.spin;
        r * r + a * a
    }

    fn delta(&self, r: f64) -> f64 {
        let m = self.bh.mass;
        let a = self.bh.spin;
        r * r - 2.0 * m * r + a * a
    }

    /// 赤道面顺行/逆行圆形光子轨道半径。
    ///
    /// `prograde` 为真时与自旋同向。该解析结果只作为赤道 Kerr 的物理
    /// 参考，不表示完整三维 Kerr 已经实现。
    pub fn equatorial_photon_orbit_radius(&self, prograde: bool) -> f64 {
        let m = self.bh.mass;
        let a = self.bh.spin;
        if !m.is_finite() || m <= 0.0 || !a.is_finite() || a.abs() >= m {
            return f64::NAN;
        }
        if a.abs() <= f64::EPSILON {
            return 3.0 * m;
        }
        let signed_spin = if prograde { a } else { -a };
        let ratio = (signed_spin / m).clamp(-1.0, 1.0);
        2.0 * m * (1.0 + (2.0 / 3.0 * (-ratio).acos()).cos())
    }

    /// 赤道面顺行/逆行临界冲击参数的绝对值。
    pub fn equatorial_critical_impact_parameter(&self, prograde: bool) -> f64 {
        let m = self.bh.mass;
        let a = self.bh.spin;
        let r = self.equatorial_photon_orbit_radius(prograde);
        if !r.is_finite() {
            return f64::NAN;
        }
        if a.abs() <= f64::EPSILON {
            return 3.0_f64.sqrt() * 3.0 * m;
        }
        let numerator = r * r * (r - 3.0 * m) + a * a * (r + m);
        let denominator = a * (m - r);
        (numerator / denominator).abs()
    }
}

impl Metric for Kerr {
    fn black_hole(&self) -> BlackHole {
        self.bh
    }

    fn line_element(&self, r: f64, theta: f64) -> [f64; 4] {
        let _ = theta;
        let m = self.bh.mass;
        let a = self.bh.spin;
        let sigma = self.sigma(r);
        let delta = self.delta(r);
        let omega = 2.0 * m * a * r / sigma;
        let g_tt = -(1.0 - 2.0 * m * r / sigma);
        let g_rr = sigma / delta;
        let g_theta_theta = sigma;
        let g_phi_phi = (r * r + a * a + 2.0 * m * a * a * r / sigma) * (1.0 - omega * omega * g_tt.abs());
        [g_tt, g_rr, g_theta_theta, g_phi_phi]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kerr_photon_orbit_reduces_to_schwarzschild() {
        let kerr = Kerr::new(BlackHole::kerr(1.0, 0.0));
        assert!((kerr.equatorial_photon_orbit_radius(true) - 3.0).abs() < 1.0e-12);
        assert!((kerr.equatorial_photon_orbit_radius(false) - 3.0).abs() < 1.0e-12);
        assert!((kerr.equatorial_critical_impact_parameter(true) - 3.0_f64.sqrt() * 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn kerr_spin_splits_prograde_and_retrograde_photon_orbits() {
        let kerr = Kerr::new(BlackHole::kerr(1.0, 0.5));
        let prograde = kerr.equatorial_photon_orbit_radius(true);
        let retrograde = kerr.equatorial_photon_orbit_radius(false);
        assert!(prograde > 2.0 && prograde < 3.0);
        assert!(retrograde > 3.0 && retrograde < 4.0);
        assert_ne!(
            kerr.equatorial_critical_impact_parameter(true),
            kerr.equatorial_critical_impact_parameter(false)
        );
    }

    #[test]
    fn kerr_reference_rejects_extremal_or_invalid_mass() {
        let extremal = Kerr::new(BlackHole { mass: 1.0, spin: 1.0 });
        assert!(extremal.equatorial_photon_orbit_radius(true).is_nan());
        let invalid = Kerr::new(BlackHole { mass: 0.0, spin: 0.0 });
        assert!(invalid.equatorial_critical_impact_parameter(false).is_nan());
    }
}

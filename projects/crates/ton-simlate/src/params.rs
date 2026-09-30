//! 场景参数：质量、自旋与积分器族的组合规则。

use ton_types::{BlackHole, IntegratorKind, TonError, TonResult};

/// 按积分器族构造 `BlackHole`：仅 `KerrEquatorial` 使用非零自旋。
pub fn black_hole_for_render(
    mass: f64,
    spin: f64,
    integrator: IntegratorKind,
) -> TonResult<BlackHole> {
    let mass = mass.max(0.1);
    if !spin.is_finite() {
        return Err(TonError::Domain("spin must be finite".into()));
    }
    match integrator {
        IntegratorKind::KerrEquatorial => {
            if spin.abs() >= mass {
                return Err(TonError::Domain("spin must satisfy |a| < M".into()));
            }
            Ok(BlackHole::kerr(mass, spin))
        }
        _ => Ok(BlackHole::schwarzschild(mass)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_kerr_integrator_ignores_spin() {
        let bh = black_hole_for_render(1.0, 0.8, IntegratorKind::SchwarzschildOrbit).unwrap();
        assert_eq!(bh.spin, 0.0);
    }

    #[test]
    fn kerr_rejects_excessive_spin() {
        let err = black_hole_for_render(1.0, 1.0, IntegratorKind::KerrEquatorial).unwrap_err();
        assert!(matches!(err, TonError::Domain(_)));
    }
}

//! 算法族枚举：多积分器 / 多盘模型并存，供 UCF 蓝图与场景切换。

/// 光线追迹积分器族（每种对应独立 Rust 参考 + UCF `GraphBlueprint`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IntegratorKind {
    /// 史瓦西赤道平面轨道方程（Starless 同族）。
    #[default]
    SchwarzschildOrbit,
    /// 半经典 Ray March：平直空间 + 牛顿式加速度，速度归一化。
    SemiClassicalRayMarch,
    /// 完整 3D 零测地线（占位，待实现）。
    Geodesic3d,
    /// 克尔赤道近似（占位，待实现）。
    KerrEquatorial,
}

impl IntegratorKind {
    /// UCF 任务 `op` 字符串标签。
    pub fn ucf_op_name(self) -> &'static str {
        match self {
            Self::SchwarzschildOrbit => "ton_trace_schwarzschild_orbit",
            Self::SemiClassicalRayMarch => "ton_trace_sem_classical",
            Self::Geodesic3d => "ton_trace_geodesic_3d",
            Self::KerrEquatorial => "ton_trace_kerr_equatorial",
        }
    }
}

/// 吸积盘材质模型族。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DiskModelKind {
    /// 无盘，仅背景与捕获。
    #[default]
    None,
    /// Novikov-Thorne 薄盘温度近似。
    NovikovThorneThin,
    /// 程序化 FBM 流动盘（占位，待实现）。
    ProceduralFbm,
}

//! 算法族枚举：多积分器 / 多盘模型并存，供 UCF 蓝图与场景切换。

/// 光线追迹积分器族（每种对应独立 Rust 参考 + UCF `GraphBlueprint`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum IntegratorKind {
    /// 史瓦西赤道平面轨道方程（Starless 同族）。
    #[default]
    SchwarzschildOrbit = 0,
    /// 半经典 Ray March：平直空间 + 牛顿式加速度，速度归一化。
    SemiClassicalRayMarch = 1,
    /// 完整 3D 零测地线（Christoffel RK4）。
    Geodesic3d = 2,
    /// 克尔赤道近似（占位，待实现）。
    KerrEquatorial = 3,
}

impl IntegratorKind {
    /// Wasm / N-API 稳定判别值。
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// 从宿主传入的 `u8` 解析；未知值返回 `None`。
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::SchwarzschildOrbit),
            1 => Some(Self::SemiClassicalRayMarch),
            2 => Some(Self::Geodesic3d),
            3 => Some(Self::KerrEquatorial),
            _ => None,
        }
    }

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
#[repr(u8)]
pub enum DiskModelKind {
    /// 无盘，仅背景与捕获。
    #[default]
    None = 0,
    /// Novikov-Thorne 薄盘温度近似。
    NovikovThorneThin = 1,
    /// 程序化 FBM 流动盘。
    ProceduralFbm = 2,
}

impl DiskModelKind {
    /// Wasm / N-API 稳定判别值。
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// 从宿主传入的 `u8` 解析；未知值返回 `None`。
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::NovikovThorneThin),
            2 => Some(Self::ProceduralFbm),
            _ => None,
        }
    }
}

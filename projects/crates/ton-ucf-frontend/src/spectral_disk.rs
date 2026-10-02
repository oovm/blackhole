//! 无内缘力矩温度基线与曝光显示前端。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

/// buffer 1 为频移程序输出 `[radius,g,g4,status]`，buffer 2 为 `[T,g,exposure]`。
/// uniform 0..2 内缘、峰值 Kelvin、曝光。返回 0 无盘、1 有效、2 非法。
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroTorqueDiskInputsBuilder;

impl ZeroTorqueDiskInputsBuilder {
    /// 创建温度前端。
    pub fn new() -> Self { Self }
    /// 不实现相对论盘通量修正，不应称为 Novikov-Thorne 模型。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("zero_torque_disk_temperature_inputs");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(4));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        let load = |index| builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(index)));
        let radius = load(0); let shift = load(1); let input_status = load(3);
        let inner = builder.load_uniform(0); let peak = builder.load_uniform(1); let exposure = builder.load_uniform(2);
        let mut valid = Expr::ConstBool(true);
        for value in [&inner, &peak, &exposure] {
            valid = builder.select(valid, builder.select(builder.unary(UnOp::IsFinite, value.clone()),
                b(&builder, BinOp::Gt, value.clone(), builder.const_f32(0.0)), Expr::ConstBool(false)), Expr::ConstBool(false));
        }
        let no_hit = b(&builder, BinOp::Eq, input_status.clone(), builder.const_f32(0.0));
        let hit = b(&builder, BinOp::Eq, input_status, builder.const_f32(1.0));
        let mut hit_valid = hit;
        for value in [&radius, &shift] { hit_valid = builder.select(hit_valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false)); }
        hit_valid = builder.select(hit_valid, b(&builder, BinOp::Ge, radius.clone(), inner.clone()), Expr::ConstBool(false));
        hit_valid = builder.select(hit_valid, b(&builder, BinOp::Gt, shift.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        let status = builder.select(valid, builder.select(no_hit, builder.const_f32(0.0), builder.select(hit_valid, builder.const_f32(1.0), builder.const_f32(2.0))), builder.const_f32(2.0));
        let status = Expr::LoadLocal(builder.let_value(Ty::f32(), status));
        let active = b(&builder, BinOp::Eq, status.clone(), builder.const_f32(1.0));
        let safe_radius = builder.select(active.clone(), radius, inner.clone());
        let safe_radius = Expr::LoadLocal(builder.let_value(Ty::f32(), safe_radius));
        let ratio = b(&builder, BinOp::Div, inner.clone(), safe_radius.clone());
        let ratio = Expr::LoadLocal(builder.let_value(Ty::f32(), ratio));
        let torque = b(&builder, BinOp::Div, b(&builder, BinOp::Sub, safe_radius.clone(), inner.clone()),
            b(&builder, BinOp::Add, safe_radius.clone(), builder.unary(UnOp::Sqrt, b(&builder, BinOp::Mul, safe_radius, inner))));
        let shape = b(&builder, BinOp::Mul, b(&builder, BinOp::Mul, b(&builder, BinOp::Mul, ratio.clone(), ratio.clone()), ratio.clone()),
            torque);
        let fraction = b(&builder, BinOp::Max, b(&builder, BinOp::Div, shape, builder.const_f32(((36.0f64 / 49.0).powi(3) / 7.0) as f32)), builder.const_f32(0.0));
        let temperature = b(&builder, BinOp::Mul, peak, builder.unary(UnOp::Sqrt, builder.unary(UnOp::Sqrt, fraction)));
        for (index, value) in [builder.select(active.clone(), temperature, builder.const_f32(0.0)),
            builder.select(active, shift, builder.const_f32(1.0)), exposure].into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(index as u32)), value);
        }
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

/// 输入 buffer 1 为七标量光谱结果，输出 buffer 2 为显示 sRGB 三标量。
/// 负线性 RGB 在此显示阶段裁到色域边界，ACES 后才做 sRGB 编码。
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectralDisplayBuilder;

impl SpectralDisplayBuilder {
    /// 创建显示前端。
    pub fn new() -> Self { Self }
    /// 使用通用 F32 运算、Pow 与资源写入。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("spectral_aces_srgb_display");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(7));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        let status = builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(6)));
        let mut valid = b(&builder, BinOp::Eq, status, builder.const_f32(1.0));
        let rgb: [Expr; 3] = std::array::from_fn(|axis| {
            let value = builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(axis as u32 + 3)));
            Expr::LoadLocal(builder.let_value(Ty::f32(), value))
        });
        for value in &rgb { valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false)); }
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(ScalarTy::Bool), valid));
        for (axis, value) in rgb.into_iter().enumerate() {
            let value = b(&builder, BinOp::Max, value, builder.const_f32(0.0));
            let numerator = b(&builder, BinOp::Mul, value.clone(), b(&builder, BinOp::Add,
                b(&builder, BinOp::Mul, builder.const_f32(2.51), value.clone()), builder.const_f32(0.03)));
            let denominator = b(&builder, BinOp::Add, b(&builder, BinOp::Mul, value.clone(), b(&builder, BinOp::Add,
                b(&builder, BinOp::Mul, builder.const_f32(2.43), value), builder.const_f32(0.59))), builder.const_f32(0.14));
            let mapped = b(&builder, BinOp::Min, b(&builder, BinOp::Div, numerator, denominator), builder.const_f32(1.0));
            let mapped = Expr::LoadLocal(builder.let_value(Ty::f32(), mapped));
            let encoded = builder.select(b(&builder, BinOp::Le, mapped.clone(), builder.const_f32(0.0031308)),
                b(&builder, BinOp::Mul, builder.const_f32(12.92), mapped.clone()),
                b(&builder, BinOp::Sub, b(&builder, BinOp::Mul, builder.const_f32(1.055),
                    b(&builder, BinOp::Pow, mapped, builder.const_f32(1.0 / 2.4))), builder.const_f32(0.055)));
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(axis as u32)),
                builder.select(valid.clone(), encoded, builder.const_f32(0.0)));
        }
        let status = builder.select(valid, builder.const_f32(1.0), builder.const_f32(2.0));
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

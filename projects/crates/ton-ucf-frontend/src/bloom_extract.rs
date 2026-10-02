//! 通用 HDR 亮部提取前端。

use super::{b, BinOp, Expr, Program, ProgramBuilder, StructuredFunctionBuilder, Ty, UnOp};

/// 从每实例线性 RGB 中提取超过阈值的亮部。
///
/// 输入 buffer 1 为 `[R,G,B]`，输出 buffer 2 为 `[R,G,B]`。
/// uniform 0 为亮度阈值。该程序不包含纹理、窗口或领域语义。
#[derive(Debug, Clone, Copy, Default)]
pub struct HdrBrightExtractBuilder;

impl HdrBrightExtractBuilder {
    /// 创建 HDR 亮部提取 Builder。
    pub fn new() -> Self { Self }

    /// 展开通用 F32 亮度、阈值和资源写入程序。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("hdr_bright_extract");
        let input = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        let rgb: [Expr; 3] = std::array::from_fn(|index| builder.load_buffer(1,
            b(&builder, BinOp::Add, input.clone(), builder.const_u32(index as u32))));
        let threshold = builder.load_uniform(0);
        let luminance = b(&builder, BinOp::Add,
            b(&builder, BinOp::Add,
                b(&builder, BinOp::Mul, rgb[0].clone(), builder.const_f32(0.2126)),
                b(&builder, BinOp::Mul, rgb[1].clone(), builder.const_f32(0.7152))),
            b(&builder, BinOp::Mul, rgb[2].clone(), builder.const_f32(0.0722)));
        let luminance = Expr::LoadLocal(builder.let_value(Ty::f32(), luminance));
        let finite = rgb.iter().fold(builder.unary(UnOp::IsFinite, threshold.clone()), |valid, value|
            builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false)));
        let active = b(&builder, BinOp::Gt, luminance.clone(), threshold.clone());
        let scale = b(&builder, BinOp::Div,
            b(&builder, BinOp::Sub, luminance.clone(), threshold),
            b(&builder, BinOp::Max, luminance.clone(), builder.const_f32(1.0e-9)));
        for (index, value) in rgb.into_iter().enumerate() {
            let bright = b(&builder, BinOp::Mul, value, scale.clone());
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(index as u32)),
                builder.select(b(&builder, BinOp::Eq, finite.clone(), Expr::ConstBool(true)),
                    builder.select(active.clone(), bright, builder.const_f32(0.0)), builder.const_f32(0.0)));
        }
        let status = builder.select(finite, builder.const_f32(1.0), builder.const_f32(2.0));
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn extracts_only_hdr_values_above_luminance_threshold() {
        let program = HdrBrightExtractBuilder::new().build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_uniform(0, Value::F32(1.0));
        cpu.with_buffer(1, [4.0, 2.0, 0.5].into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 3]);
        cpu.run_parallel(&program, 1).unwrap();
        let output = cpu.buffer(2).unwrap();
        let luminance = 0.2126 * 4.0 + 0.7152 * 2.0 + 0.0722 * 0.5;
        let scale = (luminance - 1.0) / luminance;
        assert!((output[0].as_f64().unwrap() - 4.0 * scale).abs() < 1.0e-5);
        assert!((output[2].as_f64().unwrap() - 0.5 * scale).abs() < 1.0e-6);
    }
}

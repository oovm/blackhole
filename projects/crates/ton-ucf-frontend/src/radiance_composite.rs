//! 通用线性辐射合成前端。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

/// 合成基础 RGB 与标量辐射传输结果。
///
/// 输入 buffer 1 每实例为 `[X,Y,Z,R,G,B,color_status,source,transmittance,count,transfer_status,sky_r,sky_g,sky_b,trace_status]`。
/// 输出 buffer 2 为 `[X,Y,Z,R,G,B,status]`。`source_scale` 将标量源强度映射到 RGB
/// 的线性 HDR 单位，调用方必须显式选择该单位换算。
#[derive(Debug, Clone, Copy, Default)]
pub struct RadianceCompositeBuilder;

impl RadianceCompositeBuilder {
    /// 创建辐射合成 Builder。
    pub fn new() -> Self { Self }

    /// 展开为通用 F32 资源读写和算术程序。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("radiance_composite");
        let input = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(15));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(7));
        let xyz: [Expr; 3] = std::array::from_fn(|index| builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(index as u32))));
        let rgb: [Expr; 3] = std::array::from_fn(|index| builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(index as u32 + 3))));
        let color_status = builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(6)));
        let source = builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(7)));
        let transmittance = builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(8)));
        let transfer_status = builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(10)));
        let count = builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(9)));
        let sky: [Expr; 3] = std::array::from_fn(|index| builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(index as u32 + 11))));
        let trace_status = builder.load_buffer(1, b(&builder, BinOp::Add, input.clone(), builder.const_u32(14)));
        let scale = builder.load_uniform(0);
        let mut valid = builder.unary(UnOp::IsFinite, scale.clone());
        valid = builder.select(valid, b(&builder, BinOp::Ge, scale.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Eq, color_status, builder.const_f32(1.0)), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Eq, transfer_status, builder.const_f32(1.0)), Expr::ConstBool(false));
        valid = builder.select(valid, builder.unary(UnOp::IsFinite, count.clone()), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Ge, count.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        valid = builder.select(valid, builder.unary(UnOp::IsFinite, trace_status.clone()), Expr::ConstBool(false));
        let trace_valid = builder.select(b(&builder, BinOp::Eq, trace_status.clone(), builder.const_f32(1.0)),
            Expr::ConstBool(true), b(&builder, BinOp::Eq, trace_status.clone(), builder.const_f32(2.0)));
        valid = builder.select(valid, trace_valid, Expr::ConstBool(false));
        valid = builder.select(valid, builder.unary(UnOp::IsFinite, source.clone()), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Ge, source.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        valid = builder.select(valid, builder.unary(UnOp::IsFinite, transmittance.clone()), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Ge, transmittance.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Le, transmittance.clone(), builder.const_f32(1.0)), Expr::ConstBool(false));
        for value in &rgb { valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false)); }
        for value in &sky { valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false)); }
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(ScalarTy::Bool), valid));
        let source_rgb = b(&builder, BinOp::Mul, source, scale);
        let has_events = b(&builder, BinOp::Gt, count, builder.const_f32(0.0));
        let use_sky = builder.select(b(&builder, BinOp::Eq, trace_status, builder.const_f32(2.0)),
            b(&builder, BinOp::Eq, has_events, Expr::ConstBool(false)), Expr::ConstBool(false));
        for index in 0..3 {
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(index)),
                builder.select(valid.clone(), xyz[index as usize].clone(), builder.const_f32(0.0)));
        }
        for (index, value) in rgb.into_iter().enumerate() {
            let transmitted = b(&builder, BinOp::Mul, value, transmittance.clone());
            let result = builder.select(use_sky.clone(), sky[index].clone(), b(&builder, BinOp::Add, transmitted, source_rgb.clone()));
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(index as u32 + 3)),
                builder.select(valid.clone(), result, builder.const_f32(0.0)));
        }
        let status = builder.select(valid, builder.const_f32(1.0), builder.const_f32(2.0));
        builder.store_buffer(2, b(&builder, BinOp::Add, output, builder.const_u32(6)), status.clone());
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn composite_builder_matches_linear_reference() {
        let program = RadianceCompositeBuilder::new().build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_uniform(0, Value::F32(0.25));
        cpu.with_buffer(1, [0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 1.0, 8.0, 0.5, 1.0, 1.0, 9.0, 8.0, 7.0, 1.0].into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 7]);
        cpu.run_parallel(&program, 1).unwrap();
        assert_eq!(cpu.buffer(2).unwrap(), &[Value::F32(0.0), Value::F32(0.0), Value::F32(0.0), Value::F32(4.0), Value::F32(3.0), Value::F32(2.5), Value::F32(1.0)]);

        cpu.with_buffer(1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.2, 0.3, 0.5, 2.0].into_iter().map(Value::F32).collect());
        cpu.run_parallel(&program, 1).unwrap();
        assert_eq!(cpu.buffer(2).unwrap(), &[Value::F32(0.0), Value::F32(0.0), Value::F32(0.0), Value::F32(0.2), Value::F32(0.3), Value::F32(0.5), Value::F32(1.0)]);
    }
}

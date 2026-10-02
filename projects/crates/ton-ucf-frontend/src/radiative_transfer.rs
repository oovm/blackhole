//! 通用 UCF 离散辐射传输 Builder。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

/// 输入 buffer 1 每实例为 `[source, opacity]` 的固定事件数组。
/// uniform 0 为背景强度，uniform 1 为事件数量，输出 `[intensity,transmittance,count,status]`。
#[derive(Debug, Clone, Copy)]
pub struct RadiativeTransferBuilder {
    maximum_events: u32,
}

impl Default for RadiativeTransferBuilder { fn default() -> Self { Self { maximum_events: 64 } } }

impl RadiativeTransferBuilder {
    /// 创建默认最多 64 个事件的 Builder。
    pub fn new() -> Self { Self::default() }
    /// 设置固定上限，避免无界循环进入后端。
    pub fn maximum_events(mut self, value: u32) -> Self { self.maximum_events = value; self }
    /// 展开前向离散传输程序。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_events == 0 || self.maximum_events > 64 { return Err("invalid radiative transfer event bound".into()); }
        let mut builder = StructuredFunctionBuilder::new("bounded_radiative_transfer");
        let input_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(self.maximum_events * 2));
        let output_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(4));
        let intensity = builder.let_value(Ty::f32(), builder.load_uniform(0));
        let transmittance = builder.let_value(Ty::f32(), builder.const_f32(1.0));
        let count = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let event_limit = builder.load_uniform(1);
        let event_index = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let valid = builder.let_value(Ty::Scalar(ScalarTy::Bool), builder.select(
            builder.unary(UnOp::IsFinite, event_limit.clone()),
            b(&builder, BinOp::Ge, event_limit, builder.const_f32(0.0)), Expr::ConstBool(false)));
        builder.bounded_for(self.maximum_events, |builder, iteration| {
            let active = b(builder, BinOp::Lt, Expr::LoadLocal(event_index), builder.load_uniform(1));
            let row = b(builder, BinOp::Add, input_base.clone(), b(builder, BinOp::Mul, Expr::LoadLocal(iteration), builder.const_u32(2)));
            let source = builder.load_buffer(1, row.clone());
            let opacity = builder.load_buffer(1, b(builder, BinOp::Add, row, builder.const_u32(1)));
            let finite = builder.select(builder.unary(UnOp::IsFinite, source.clone()), builder.unary(UnOp::IsFinite, opacity.clone()), Expr::ConstBool(false));
            let bounded = builder.select(b(builder, BinOp::Ge, source.clone(), builder.const_f32(0.0)),
                builder.select(b(builder, BinOp::Ge, opacity.clone(), builder.const_f32(0.0)), b(builder, BinOp::Le, opacity.clone(), builder.const_f32(1.0)), Expr::ConstBool(false)), Expr::ConstBool(false));
            let event_valid = builder.select(active.clone(), builder.select(finite, bounded, Expr::ConstBool(false)), Expr::ConstBool(true));
            builder.store_local(valid, builder.select(Expr::LoadLocal(valid), event_valid, Expr::ConstBool(false)));
            let contribution = b(builder, BinOp::Mul, source, opacity.clone());
            let remainder = b(builder, BinOp::Sub, builder.const_f32(1.0), opacity.clone());
            let next_intensity = b(builder, BinOp::Add, contribution, b(builder, BinOp::Mul, remainder.clone(), Expr::LoadLocal(intensity)));
            let next_transmittance = b(builder, BinOp::Mul, remainder, Expr::LoadLocal(transmittance));
            builder.store_local(intensity, builder.select(active.clone(), next_intensity, Expr::LoadLocal(intensity)));
            builder.store_local(transmittance, builder.select(active.clone(), next_transmittance, Expr::LoadLocal(transmittance)));
            builder.store_local(count, builder.select(active, b(builder, BinOp::Add, Expr::LoadLocal(count), builder.const_f32(1.0)), Expr::LoadLocal(count)));
            builder.store_local(event_index, b(builder, BinOp::Add, Expr::LoadLocal(event_index), builder.const_f32(1.0)));
        });
        for (index, value) in [Expr::LoadLocal(intensity), Expr::LoadLocal(transmittance), Expr::LoadLocal(count),
            builder.select(Expr::LoadLocal(valid), builder.const_f32(1.0), builder.const_f32(2.0))].into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)), value);
        }
        let status = builder.select(Expr::LoadLocal(valid), builder.const_f32(1.0), builder.const_f32(2.0));
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

/// 从固定容量事件包执行逐实例辐射传输。
///
/// 输入 buffer 1 为 `[count, overflow, source0, opacity0, ...]`，每实例
/// 固定 `2 + maximum_events * 2` 个 F32。count 来自资源头部而不是 uniform，
/// 因而每个像素可以拥有不同数量的盘面事件。overflow 非零时输出失败。
#[derive(Debug, Clone, Copy)]
pub struct PackedRadiativeTransferBuilder {
    maximum_events: u32,
}

impl Default for PackedRadiativeTransferBuilder {
    fn default() -> Self { Self { maximum_events: 4 } }
}

impl PackedRadiativeTransferBuilder {
    /// 创建默认四事件容量的 Builder。
    pub fn new() -> Self { Self::default() }
    /// 设置固定事件容量。
    pub fn maximum_events(mut self, value: u32) -> Self { self.maximum_events = value; self }
    /// 展开固定包辐射传输程序。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_events == 0 || self.maximum_events > 64 { return Err("invalid packed radiative event bound".into()); }
        let mut builder = StructuredFunctionBuilder::new("packed_radiative_transfer");
        let stride = 2 + self.maximum_events * 2;
        let input_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(stride));
        let output_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(4));
        let intensity = builder.let_value(Ty::f32(), builder.load_uniform(0));
        let transmittance = builder.let_value(Ty::f32(), builder.const_f32(1.0));
        let count = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let valid = builder.let_value(Ty::Scalar(ScalarTy::Bool), Expr::ConstBool(true));
        let event_limit = builder.load_buffer(1, input_base.clone());
        let overflow = builder.load_buffer(1, b(&builder, BinOp::Add, input_base.clone(), builder.const_u32(1)));
        let valid_header = builder.select(builder.unary(UnOp::IsFinite, event_limit.clone()),
            builder.select(b(&builder, BinOp::Ge, event_limit.clone(), builder.const_f32(0.0)),
                builder.select(b(&builder, BinOp::Le, event_limit.clone(), builder.const_f32(self.maximum_events as f32)),
                    builder.select(b(&builder, BinOp::Eq, overflow, builder.const_f32(0.0)), Expr::ConstBool(true), Expr::ConstBool(false)),
                    Expr::ConstBool(false)), Expr::ConstBool(false)), Expr::ConstBool(false));
        builder.store_local(valid, valid_header);
        let event_index = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        builder.bounded_for(self.maximum_events, |builder, iteration| {
            let active = b(builder, BinOp::Lt, Expr::LoadLocal(event_index), event_limit.clone());
            let row = b(builder, BinOp::Add, input_base.clone(), b(builder, BinOp::Add,
                builder.const_u32(2), b(builder, BinOp::Mul, Expr::LoadLocal(iteration), builder.const_u32(2))));
            let source = builder.load_buffer(1, row.clone());
            let opacity = builder.load_buffer(1, b(builder, BinOp::Add, row, builder.const_u32(1)));
            let finite = builder.select(builder.unary(UnOp::IsFinite, source.clone()), builder.unary(UnOp::IsFinite, opacity.clone()), Expr::ConstBool(false));
            let bounded = builder.select(b(builder, BinOp::Ge, source.clone(), builder.const_f32(0.0)),
                builder.select(b(builder, BinOp::Ge, opacity.clone(), builder.const_f32(0.0)), b(builder, BinOp::Le, opacity.clone(), builder.const_f32(1.0)), Expr::ConstBool(false)), Expr::ConstBool(false));
            let event_valid = builder.select(active.clone(), builder.select(finite, bounded, Expr::ConstBool(false)), Expr::ConstBool(true));
            builder.store_local(valid, builder.select(Expr::LoadLocal(valid), event_valid, Expr::ConstBool(false)));
            let contribution = b(builder, BinOp::Mul, source, opacity.clone());
            let remainder = b(builder, BinOp::Sub, builder.const_f32(1.0), opacity);
            builder.store_local(intensity, builder.select(active.clone(),
                b(builder, BinOp::Add, contribution, b(builder, BinOp::Mul, remainder.clone(), Expr::LoadLocal(intensity))), Expr::LoadLocal(intensity)));
            builder.store_local(transmittance, builder.select(active.clone(), b(builder, BinOp::Mul, remainder, Expr::LoadLocal(transmittance)), Expr::LoadLocal(transmittance)));
            builder.store_local(count, builder.select(active, b(builder, BinOp::Add, Expr::LoadLocal(count), builder.const_f32(1.0)), Expr::LoadLocal(count)));
            builder.store_local(event_index, b(builder, BinOp::Add, Expr::LoadLocal(event_index), builder.const_f32(1.0)));
        });
        for (index, value) in [Expr::LoadLocal(intensity), Expr::LoadLocal(transmittance), Expr::LoadLocal(count),
            builder.select(Expr::LoadLocal(valid), builder.const_f32(1.0), builder.const_f32(2.0))].into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)), value);
        }
        let status = builder.select(Expr::LoadLocal(valid), builder.const_f32(1.0), builder.const_f32(2.0));
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod packed_tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn packed_transfer_uses_per_instance_count_and_rejects_overflow() {
        let program = PackedRadiativeTransferBuilder::new().maximum_events(2).build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_uniform(0, Value::F32(1.0));
        cpu.with_buffer(1, vec![Value::F32(1.0), Value::F32(0.0), Value::F32(2.0), Value::F32(0.25), Value::F32(8.0), Value::F32(0.5),
            Value::F32(0.0), Value::F32(1.0), Value::F32(4.0), Value::F32(0.0), Value::F32(7.0), Value::F32(1.1)]);
        cpu.with_buffer(2, vec![Value::F32(0.0); 8]);
        cpu.run_parallel(&program, 2).unwrap();
        let output = cpu.buffer(2).unwrap();
        assert!((output[0].as_f64().unwrap() - 1.25).abs() < 1.0e-5);
        assert_eq!(output[2], Value::F32(1.0));
        assert_eq!(output[7], Value::F32(2.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn radiative_transfer_program_matches_reference() {
        let program = RadiativeTransferBuilder::new().maximum_events(4).build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_uniform(0, Value::F32(1.0)); cpu.with_uniform(1, Value::F32(2.0));
        cpu.with_buffer(1, [2.0, 0.25, 8.0, 0.5, 0.0, 0.0, 0.0, 0.0].into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 4]); cpu.run_parallel(&program, 1).unwrap();
        let values = cpu.buffer(2).unwrap();
        assert!((values[0].as_f64().unwrap() - 4.625).abs() < 1.0e-5);
        assert!((values[1].as_f64().unwrap() - 0.375).abs() < 1.0e-5);
        assert_eq!(values[2], Value::F32(2.0)); assert_eq!(values[3], Value::F32(1.0));
    }

    #[test]
    fn radiative_transfer_program_rejects_invalid_active_events() {
        let program = RadiativeTransferBuilder::new().maximum_events(2).build().unwrap();
        let mut cpu = CpuInterpreter::new(); cpu.with_uniform(0, Value::F32(0.0)); cpu.with_uniform(1, Value::F32(1.0));
        cpu.with_buffer(1, [1.0, 1.1, 0.0, 0.0].into_iter().map(Value::F32).collect()); cpu.with_buffer(2, vec![Value::F32(7.0); 4]);
        cpu.run_parallel(&program, 1).unwrap(); assert_eq!(cpu.buffer(2).unwrap()[3], Value::F32(2.0));
    }
}

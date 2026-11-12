use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

/// 按相机到远处的事件顺序合成线性谱色与背景。
///
/// buffer 1 每实例为 `[count,overflow,(R,G,B,opacity)*capacity,sky_R,sky_G,sky_B,trace_status]`。
/// trace_status 为 `1=captured` 或 `2=escaped`，buffer 2 为 `[0,0,0,R,G,B,status]`。
/// 状态 `1` 表示成功，`2` 表示事件包或输入无效。颜色允许保留线性色域外分量。
#[derive(Debug, Clone, Copy)]
pub struct PackedSpectralRadianceBuilder {
    maximum_events: u32,
}

impl Default for PackedSpectralRadianceBuilder {
    fn default() -> Self { Self { maximum_events: 4 } }
}

impl PackedSpectralRadianceBuilder {
    /// 使用默认四事件容量。
    pub fn new() -> Self { Self::default() }

    /// 设置事件容量，必须与输入资源的固定布局一致。
    pub fn maximum_events(mut self, count: u32) -> Self { self.maximum_events = count; self }

    /// 构造有界、逐事件近到远的谱色传输程序。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_events == 0 || self.maximum_events > 64 {
            return Err("invalid spectral event capacity".into());
        }
        let mut builder = StructuredFunctionBuilder::new("packed_spectral_radiance");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(6 + self.maximum_events * 4));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(7));
        let count = builder.load_buffer(1, base.clone());
        let overflow = builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(1)));
        let background_base = b(&builder, BinOp::Add, base.clone(), builder.const_u32(2 + self.maximum_events * 4));
        let sky: [Expr; 3] = std::array::from_fn(|axis| builder.load_buffer(1,
            b(&builder, BinOp::Add, background_base.clone(), builder.const_u32(axis as u32))));
        let trace_status = builder.load_buffer(1, b(&builder, BinOp::Add, background_base, builder.const_u32(3)));
        let mut header_valid = builder.unary(UnOp::IsFinite, count.clone());
        for condition in [
            b(&builder, BinOp::Ge, count.clone(), builder.const_f32(0.0)),
            b(&builder, BinOp::Le, count.clone(), builder.const_f32(self.maximum_events as f32)),
            b(&builder, BinOp::Eq, b(&builder, BinOp::Mod, count.clone(), builder.const_f32(1.0)), builder.const_f32(0.0)),
            b(&builder, BinOp::Eq, overflow, builder.const_f32(0.0)),
            builder.select(b(&builder, BinOp::Eq, trace_status.clone(), builder.const_f32(1.0)), Expr::ConstBool(true),
                b(&builder, BinOp::Eq, trace_status.clone(), builder.const_f32(2.0))),
        ] {
            header_valid = builder.select(header_valid, condition, Expr::ConstBool(false));
        }
        for value in &sky {
            header_valid = builder.select(header_valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
        }
        let valid = builder.let_value(Ty::Scalar(ScalarTy::Bool), header_valid);
        let transmittance = builder.let_value(Ty::f32(), builder.const_f32(1.0));
        let intensity: [u32; 3] = std::array::from_fn(|_| builder.let_value(Ty::f32(), Expr::ConstF32(0.0)));
        let event_index = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        builder.bounded_for(self.maximum_events, |builder, iteration| {
            let active = b(builder, BinOp::Lt, Expr::LoadLocal(event_index), count.clone());
            let event_base = b(builder, BinOp::Add, base.clone(), b(builder, BinOp::Add,
                builder.const_u32(2), b(builder, BinOp::Mul, Expr::LoadLocal(iteration), builder.const_u32(4))));
            let source: [Expr; 3] = std::array::from_fn(|axis| builder.load_buffer(1,
                b(builder, BinOp::Add, event_base.clone(), builder.const_u32(axis as u32))));
            let opacity = builder.load_buffer(1, b(builder, BinOp::Add, event_base, builder.const_u32(3)));
            let mut event_valid = builder.unary(UnOp::IsFinite, opacity.clone());
            for condition in [b(builder, BinOp::Ge, opacity.clone(), builder.const_f32(0.0)),
                b(builder, BinOp::Le, opacity.clone(), builder.const_f32(1.0))] {
                event_valid = builder.select(event_valid, condition, Expr::ConstBool(false));
            }
            for value in &source {
                event_valid = builder.select(event_valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
            }
            builder.store_local(valid, builder.select(Expr::LoadLocal(valid),
                builder.select(active.clone(), event_valid, Expr::ConstBool(true)), Expr::ConstBool(false)));
            for (axis, value) in source.into_iter().enumerate() {
                let contribution = b(builder, BinOp::Mul, Expr::LoadLocal(transmittance), b(builder, BinOp::Mul, opacity.clone(), value));
                builder.store_local(intensity[axis], builder.select(active.clone(),
                    b(builder, BinOp::Add, Expr::LoadLocal(intensity[axis]), contribution), Expr::LoadLocal(intensity[axis])));
            }
            builder.store_local(transmittance, builder.select(active,
                b(builder, BinOp::Mul, Expr::LoadLocal(transmittance), b(builder, BinOp::Sub, builder.const_f32(1.0), opacity)),
                Expr::LoadLocal(transmittance)));
            builder.store_local(event_index, b(builder, BinOp::Add, Expr::LoadLocal(event_index), builder.const_f32(1.0)));
        });
        for axis in 0..3 {
            let escaped = b(&builder, BinOp::Eq, trace_status.clone(), builder.const_f32(2.0));
            let background = builder.select(escaped, b(&builder, BinOp::Mul, Expr::LoadLocal(transmittance), sky[axis].clone()), builder.const_f32(0.0));
            let color = b(&builder, BinOp::Add, Expr::LoadLocal(intensity[axis]), background);
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(axis as u32)), builder.const_f32(0.0));
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(axis as u32 + 3)),
                builder.select(Expr::LoadLocal(valid), color, builder.const_f32(0.0)));
        }
        let status = builder.select(Expr::LoadLocal(valid), builder.const_f32(1.0), builder.const_f32(2.0));
        builder.store_buffer(2, b(&builder, BinOp::Add, output, builder.const_u32(6)), status.clone());
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status)
            .map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    fn reference_composite(events: &[[f64; 4]], sky: [f64; 3], escaped: bool) -> [f64; 3] {
        let mut transmittance = 1.0f64;
        let mut intensity = [0.0f64; 3];
        for event in events {
            for axis in 0..3 {
                intensity[axis] += transmittance * event[3] * event[axis];
            }
            transmittance *= 1.0 - event[3];
        }
        if escaped {
            for axis in 0..3 {
                intensity[axis] += transmittance * sky[axis];
            }
        }
        intensity
    }

    #[test]
    fn spectral_events_match_independent_reference_formula() {
        let program = PackedSpectralRadianceBuilder::new().maximum_events(4).build().unwrap();
        let input = [
            2.0, 0.0,
            4.0, 1.0, 0.0, 0.25,
            0.0, 2.0, 3.0, 0.5,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 4.0, 2.0,
        ];
        let expected = reference_composite(
            &[[4.0, 1.0, 0.0, 0.25], [0.0, 2.0, 3.0, 0.5]],
            [1.0, 2.0, 4.0],
            true,
        );
        let mut cpu = CpuInterpreter::new();
        cpu.with_buffer(1, input.into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 7]);
        cpu.run_parallel(&program, 1).unwrap();
        let output = cpu.buffer(2).unwrap();
        for (actual, expected) in output[3..6].iter().zip(expected) {
            assert!((actual.as_f64().unwrap() - expected).abs() < 1.0e-6);
        }
        assert_eq!(output[6], Value::F32(1.0));
    }

    #[test]
    fn spectral_events_prioritize_near_emission_and_transmit_background() {
        let program = PackedSpectralRadianceBuilder::new().maximum_events(2).build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_buffer(1, [2.0, 0.0, 8.0, 0.0, 0.0, 0.5, 0.0, 4.0, 0.0, 0.25, 1.0, 2.0, 3.0, 2.0]
            .into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 7]);
        assert_eq!(cpu.run_parallel(&program, 1).unwrap()[0].value, Value::F32(1.0));
        assert_eq!(&cpu.buffer(2).unwrap()[3..], &[Value::F32(4.375), Value::F32(1.25), Value::F32(1.125), Value::F32(1.0)]);
    }

    #[test]
    fn spectral_packets_reject_fractional_counts_overflow_and_invalid_opacity() {
        let program = PackedSpectralRadianceBuilder::new().maximum_events(1).build().unwrap();
        for (count, overflow, opacity) in [(0.5, 0.0, 0.5), (1.0, 1.0, 0.5), (1.0, 0.0, 2.0)] {
            let mut cpu = CpuInterpreter::new();
            cpu.with_buffer(1, [count, overflow, 1.0, 2.0, 3.0, opacity, 0.0, 0.0, 0.0, 1.0]
                .into_iter().map(Value::F32).collect());
            cpu.with_buffer(2, vec![Value::F32(0.0); 7]);
            assert_eq!(cpu.run_parallel(&program, 1).unwrap()[0].value, Value::F32(2.0));
            assert_eq!(&cpu.buffer(2).unwrap()[3..6], &vec![Value::F32(0.0); 3]);
        }
    }
}

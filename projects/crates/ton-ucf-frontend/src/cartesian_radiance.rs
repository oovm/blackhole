//! 静止观测者与圆轨道薄盘的频移前端，不生成占位 RGB。

use super::{b, BinOp, Expr, Program, ProgramBuilder, StructuredFunctionBuilder, Ty, UnOp};

/// 输入 buffer 1 为 29 标量笛卡尔追踪记录，输出 buffer 2 为 `[radius,g,g^4,status]`。
/// status：0 无交点，1 有效，2 非法。uniform 0/1 为质量与静止观测者半径。
#[derive(Debug, Clone, Copy, Default)]
pub struct SchwarzschildDiskFrequencyBuilder;

impl SchwarzschildDiskFrequencyBuilder {
    /// 构造普通 UCF 程序前端。
    pub fn new() -> Self { Self }

    /// 相机逆向追踪的空间角动量与未来指向光子相反，因此分母使用加号。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_disk_frequency");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(29));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(4));
        let load = |offset| builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(offset)));
        let count = load(13);
        let state = std::array::from_fn(|index| load(21 + index as u32));
        let state = super::adaptive::snapshot(&mut builder, state);
        let mass = builder.load_uniform(0);
        let observer = builder.load_uniform(1);
        let radius = super::cartesian_null::radius(&builder, &state);
        let radius = Expr::LoadLocal(builder.let_value(Ty::f32(), radius));
        let mut valid = Expr::ConstBool(true);
        for value in state.iter().chain([&mass, &observer, &count, &radius]) {
            valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
        }
        for condition in [
            b(&builder, BinOp::Gt, mass.clone(), builder.const_f32(0.0)),
            b(&builder, BinOp::Gt, observer.clone(), b(&builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())),
            b(&builder, BinOp::Ge, radius.clone(), b(&builder, BinOp::Mul, builder.const_f32(6.0), mass.clone())),
            b(&builder, BinOp::Gt, state[4].clone(), builder.const_f32(0.0)),
            b(&builder, BinOp::Le, builder.unary(UnOp::Abs, state[3].clone()), b(&builder, BinOp::Mul, builder.const_f32(1.0e-4), mass.clone())),
        ] { valid = builder.select(valid, condition, Expr::ConstBool(false)); }
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(super::ScalarTy::Bool), valid));
        let safe_radius = builder.select(valid.clone(), radius.clone(), builder.const_f32(8.0));
        let safe_mass = builder.select(valid.clone(), mass, builder.const_f32(1.0));
        let safe_observer = builder.select(valid.clone(), observer, builder.const_f32(24.0));
        let safe_energy = builder.select(valid.clone(), state[4].clone(), builder.const_f32(1.0));
        let angular = b(&builder, BinOp::Sub, b(&builder, BinOp::Mul, state[1].clone(), state[6].clone()),
            b(&builder, BinOp::Mul, state[2].clone(), state[5].clone()));
        let angular = builder.select(valid.clone(), angular, builder.const_f32(0.0));
        let cubed = b(&builder, BinOp::Mul, b(&builder, BinOp::Mul, safe_radius.clone(), safe_radius.clone()), safe_radius.clone());
        let omega = builder.unary(UnOp::Sqrt, b(&builder, BinOp::Div, safe_mass.clone(), cubed));
        let lapse = |factor: f32, radius: Expr| builder.unary(UnOp::Sqrt, b(&builder, BinOp::Sub, builder.const_f32(1.0),
            b(&builder, BinOp::Div, b(&builder, BinOp::Mul, builder.const_f32(factor), safe_mass.clone()), radius)));
        let numerator = lapse(3.0, safe_radius);
        let denominator = b(&builder, BinOp::Mul, lapse(2.0, safe_observer), b(&builder, BinOp::Add, builder.const_f32(1.0),
            b(&builder, BinOp::Div, b(&builder, BinOp::Mul, omega, angular), safe_energy)));
        let denominator = Expr::LoadLocal(builder.let_value(Ty::f32(), denominator));
        let positive = builder.select(builder.unary(UnOp::IsFinite, denominator.clone()),
            b(&builder, BinOp::Gt, denominator.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        let valid = builder.select(valid, positive, Expr::ConstBool(false));
        let shift = b(&builder, BinOp::Div, numerator, builder.select(valid.clone(), denominator, builder.const_f32(1.0)));
        let shift = Expr::LoadLocal(builder.let_value(Ty::f32(), shift));
        let squared = b(&builder, BinOp::Mul, shift.clone(), shift.clone());
        let fourth = b(&builder, BinOp::Mul, squared.clone(), squared);
        let valid = builder.select(valid, builder.unary(UnOp::IsFinite, fourth.clone()), Expr::ConstBool(false));
        let status = builder.select(b(&builder, BinOp::Gt, count, builder.const_f32(0.0)),
            builder.select(valid, builder.const_f32(1.0), builder.const_f32(2.0)), builder.const_f32(0.0));
        let status = Expr::LoadLocal(builder.let_value(Ty::f32(), status));
        let active = b(&builder, BinOp::Eq, status.clone(), builder.const_f32(1.0));
        for (index, value) in [radius, shift, fourth, status.clone()].into_iter().enumerate() {
            let value = if index < 3 { builder.select(active.clone(), value, builder.const_f32(0.0)) } else { value };
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(index as u32)), value);
        }
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status)
            .map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn disk_frequency_checks_layout_sign_and_invalid_inputs() {
        let program = SchwarzschildDiskFrequencyBuilder::new().build().unwrap();
        for (count, radius, velocity, observer) in [(1.0, 8.0, 0.0, 24.0), (1.0, 8.0, 0.5, 24.0),
            (1.0, 8.0, -0.5, 24.0), (0.0, 0.0, 0.0, 24.0), (1.0, 5.0, 0.0, 24.0), (1.0, 8.0, 0.0, 2.0)] {
            let mut cpu = CpuInterpreter::new();
            cpu.with_uniform(0, Value::F32(1.0)); cpu.with_uniform(1, Value::F32(observer));
            let mut record = vec![Value::F32(0.0); 29];
            for (index, value) in [(13, count), (21, 123.0), (22, radius), (25, 1.0), (27, velocity)] { record[index] = Value::F32(value); }
            cpu.with_buffer(1, record); cpu.with_buffer(2, vec![Value::F32(0.0); 4]);
            cpu.run_parallel(&program, 1).unwrap();
            let output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert!(output.iter().all(|value| value.is_finite()));
            let status = if count == 0.0 { 0.0 } else if radius < 6.0 || observer <= 2.0 { 2.0 } else { 1.0 };
            assert_eq!(output[3], status);
            if status == 1.0 {
                let expected = ((1.0f64 - 3.0 / radius as f64) / (1.0 - 2.0 / observer as f64)).sqrt()
                    / (1.0 + (1.0 / (radius as f64).powi(3)).sqrt() * radius as f64 * velocity as f64);
                assert!((output[1] - expected).abs() < 1.0e-6);
                assert!((output[2] - expected.powi(4)).abs() < 2.0e-6);
            } else { assert_eq!(&output[..3], &[0.0; 3]); }
        }
    }
}

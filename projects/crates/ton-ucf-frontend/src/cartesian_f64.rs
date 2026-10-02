//! 独立 F64 笛卡尔 RK4 前端，不把 F32 常量重新标记为高精度。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

fn snapshot(builder: &mut StructuredFunctionBuilder, state: [Expr; 8]) -> [Expr; 8] {
    state.map(|value| Expr::LoadLocal(builder.let_value(Ty::f64(), value)))
}

fn derivative(builder: &mut StructuredFunctionBuilder, state: &[Expr; 8], mass: &Expr, angular: &Expr) -> ([Expr; 8], Expr) {
    let radius_squared = (1..=3).fold(builder.const_f64(0.0), |sum, index|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, state[index].clone(), state[index].clone())));
    let radius = builder.unary(UnOp::Sqrt, radius_squared.clone());
    let radius = Expr::LoadLocal(builder.let_value(Ty::f64(), radius));
    let fifth = b(builder, BinOp::Mul, b(builder, BinOp::Mul, radius_squared.clone(), radius_squared), radius.clone());
    let coefficient = b(builder, BinOp::Div,
        b(builder, BinOp::Mul, b(builder, BinOp::Mul, builder.const_f64(-3.0), mass.clone()), angular.clone()), fifth);
    let coefficient = Expr::LoadLocal(builder.let_value(Ty::f64(), coefficient));
    let lapse = b(builder, BinOp::Div, b(builder, BinOp::Sub, radius.clone(),
        b(builder, BinOp::Mul, builder.const_f64(2.0), mass.clone())), radius.clone());
    let slopes = [b(builder, BinOp::Div, state[4].clone(), lapse), state[5].clone(), state[6].clone(), state[7].clone(),
        builder.const_f64(0.0), b(builder, BinOp::Mul, coefficient.clone(), state[1].clone()),
        b(builder, BinOp::Mul, coefficient.clone(), state[2].clone()), b(builder, BinOp::Mul, coefficient, state[3].clone())];
    let mut valid = b(builder, BinOp::Gt, radius,
        b(builder, BinOp::Mul, builder.const_f64(2.0), mass.clone()));
    for value in state.iter().chain(slopes.iter()) {
        valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
    }
    (snapshot(builder, slopes), valid)
}

/// 输入 buffer 1 为 `[t,x,y,z,E,vx,vy,vz,L0_squared]`，输出 buffer 2 为 `[state(8),status]`。
/// uniform 0/1 为质量与固定步长。status 0 完成、1 输入或 RK4 中间状态非法。
/// 失败保留最后有效状态，不做速度归一化或守恒量投影。事件与自适应积分另行验证。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildCartesianF64Rk4Builder { maximum_steps: u32 }

impl Default for SchwarzschildCartesianF64Rk4Builder {
    fn default() -> Self { Self { maximum_steps: 1 } }
}

impl SchwarzschildCartesianF64Rk4Builder {
    /// 默认一个固定步长。
    pub fn new() -> Self { Self::default() }
    /// 设置静态循环预算，不允许无界执行。
    pub fn maximum_steps(mut self, value: u32) -> Self { self.maximum_steps = value; self }
    /// 使用通用 F64 算术、局部状态和有界循环构造程序。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_steps == 0 || self.maximum_steps > 65536 { return Err("invalid F64 RK4 step budget".into()); }
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_cartesian_f64_rk4");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(9));
        let state: [Expr; 8] = std::array::from_fn(|index| builder.load_buffer(1,
            b(&builder, BinOp::Add, base.clone(), builder.const_u32(index as u32))));
        let state: [u32; 8] = state.map(|value| builder.let_value(Ty::f64(), value));
        let angular = builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(8)));
        let angular = Expr::LoadLocal(builder.let_value(Ty::f64(), angular));
        let mass = builder.load_uniform(0);
        let step = builder.load_uniform(1);
        let mut valid = b(&builder, BinOp::Ge, angular.clone(), builder.const_f64(0.0));
        for value in [&mass, &step, &angular] {
            valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
        }
        for value in [&mass, &step] {
            valid = builder.select(valid, b(&builder, BinOp::Gt, value.clone(), builder.const_f64(0.0)), Expr::ConstBool(false));
        }
        valid = builder.select(valid, b(&builder, BinOp::Gt, Expr::LoadLocal(state[4]), builder.const_f64(0.0)), Expr::ConstBool(false));
        let status = builder.let_value(Ty::f64(), builder.select(valid, builder.const_f64(0.0), builder.const_f64(1.0)));
        let running = b(&builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f64(0.0));
        builder.bounded_while(running, self.maximum_steps, |builder| {
            let initial = snapshot(builder, state.map(Expr::LoadLocal));
            let half = b(builder, BinOp::Mul, step.clone(), builder.const_f64(0.5));
            let (first, mut valid) = derivative(builder, &initial, &mass, &angular);
            let middle = snapshot(builder, std::array::from_fn(|index| b(builder, BinOp::Add, initial[index].clone(),
                b(builder, BinOp::Mul, half.clone(), first[index].clone()))));
            let (second, usable) = derivative(builder, &middle, &mass, &angular);
            valid = builder.select(valid, usable, Expr::ConstBool(false));
            let middle = snapshot(builder, std::array::from_fn(|index| b(builder, BinOp::Add, initial[index].clone(),
                b(builder, BinOp::Mul, half.clone(), second[index].clone()))));
            let (third, usable) = derivative(builder, &middle, &mass, &angular);
            valid = builder.select(valid, usable, Expr::ConstBool(false));
            let end = snapshot(builder, std::array::from_fn(|index| b(builder, BinOp::Add, initial[index].clone(),
                b(builder, BinOp::Mul, step.clone(), third[index].clone()))));
            let (fourth, usable) = derivative(builder, &end, &mass, &angular);
            valid = builder.select(valid, usable, Expr::ConstBool(false));
            let next = snapshot(builder, std::array::from_fn(|index| {
                let weighted = b(builder, BinOp::Add,
                    b(builder, BinOp::Add, first[index].clone(), b(builder, BinOp::Mul, builder.const_f64(2.0), second[index].clone())),
                    b(builder, BinOp::Add, b(builder, BinOp::Mul, builder.const_f64(2.0), third[index].clone()), fourth[index].clone()));
                b(builder, BinOp::Add, initial[index].clone(), b(builder, BinOp::Mul, step.clone(), b(builder, BinOp::Div, weighted, builder.const_f64(6.0))))
            }));
            let (_, usable) = derivative(builder, &next, &mass, &angular);
            valid = builder.select(valid, usable, Expr::ConstBool(false));
            let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(ScalarTy::Bool), valid));
            for index in 0..8 { builder.store_local(state[index], builder.select(valid.clone(), next[index].clone(), initial[index].clone())); }
            builder.store_local(status, builder.select(valid, builder.const_f64(0.0), builder.const_f64(1.0)));
        });
        for index in 0..9 {
            let value = Expr::LoadLocal(if index < 8 { state[index] } else { status });
            builder.store_buffer(2, b(&builder, BinOp::Add, base.clone(), builder.const_u32(index as u32)), value);
        }
        ProgramBuilder::new().structured_entry(builder, Ty::f64(), Expr::LoadLocal(status))
            .map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn f64_rk4_builder_preserves_radial_motion_and_failure_state() {
        assert!(SchwarzschildCartesianF64Rk4Builder::new().maximum_steps(0).build().is_err());
        assert!(SchwarzschildCartesianF64Rk4Builder::new().maximum_steps(65537).build().is_err());
        for (radius, step, steps, status) in [(10.0, 0.02, 64, 0.0), (2.01, 0.08, 1, 1.0), (10.0, -0.02, 1, 1.0)] {
            let program = SchwarzschildCartesianF64Rk4Builder::new().maximum_steps(steps).build().unwrap();
            let initial = [0.0f64, 0.0, 0.0, radius, 1.0, 0.0, 0.0, -1.0, 0.0];
            let mut cpu = CpuInterpreter::new();
            cpu.with_uniform(0, Value::F64(1.0)); cpu.with_uniform(1, Value::F64(step));
            cpu.with_buffer(1, initial.into_iter().map(Value::F64).collect());
            cpu.with_buffer(2, vec![Value::F64(0.0); 9]);
            cpu.run_parallel(&program, 1).unwrap();
            let output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert_eq!(output[8], status);
            if status == 0.0 {
                assert!((output[3] - (radius - step * steps as f64)).abs() < 1.0e-12);
                assert_eq!(output[4], 1.0); assert_eq!(output[7], -1.0);
            } else { assert_eq!(&output[..8], &initial[..8]); }
        }
    }
}

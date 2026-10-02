//! 接受步事件定位的通用程序构造，不注册 UCF 领域能力。

use super::{b, BinOp, Expr, Program, StructuredFunctionBuilder, Ty, UnOp};
use super::adaptive::{rk4, snapshot};

/// 通过两个半步重积分与有界二分定位括住的事件。
///
/// 输入 buffer 1 每实例为 `[start(8), accepted(8), step_used]`。
/// 输出 buffer 2 每实例为 `[event_state(8), fraction, residual, status]`。
/// uniform 0..3 为质量、事件面种类（0=球面，1=赤道面）、球面半径、残差容差。
/// status 为 1=定位成功、2=未括住穿越、3=定位预算耗尽、4=非法输入。
/// 失败保留起点状态，不把失败候选写作物理命中。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildEventLocalizationBuilder {
    maximum_bisections: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn localization_failures_preserve_state() {
        assert!(SchwarzschildEventLocalizationBuilder::new().maximum_bisections(0).build().is_err());
        assert!(SchwarzschildEventLocalizationBuilder::new().maximum_bisections(65).build().is_err());
        let program = SchwarzschildEventLocalizationBuilder::new().maximum_bisections(1).build().unwrap();
        let start = [0.0f32, 10.0, 1.2, 0.3, 1.25, -1.0, 0.0, 0.0];
        let mut input = start.to_vec();
        input.extend([0.05, 9.96, 1.2, 0.3, 1.25, -1.0, 0.0, 0.0, 0.04]);
        for (kind, radius, tolerance, status) in [
            (0.0, 8.0, 1.0e-5, 2.0),
            (2.0, 9.98, 1.0e-5, 4.0),
            (0.0, 9.971, 1.0e-9, 3.0),
            (0.0, 9.98, -1.0, 4.0),
        ] {
            let mut cpu = CpuInterpreter::new();
            cpu.with_buffer(1, input.iter().copied().map(Value::F32).collect());
            cpu.with_buffer(2, vec![Value::F32(0.0); 11]);
            for (slot, value) in [1.0, kind, radius, tolerance].into_iter().enumerate() {
                cpu.with_uniform(slot as u32, Value::F32(value));
            }
            cpu.run_parallel(&program, 1).unwrap();
            let output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert_eq!(output[10], status);
            for (actual, expected) in output[..8].iter().zip(start) { assert_eq!(*actual, expected as f64); }
            assert_eq!(output[8], 0.0);
            assert_eq!(output[9], 0.0);
        }
    }
}

impl Default for SchwarzschildEventLocalizationBuilder {
    fn default() -> Self { Self { maximum_bisections: 48 } }
}

fn abs(builder: &StructuredFunctionBuilder, value: Expr) -> Expr {
    b(builder, BinOp::Max, value.clone(), builder.unary(UnOp::Neg, value))
}

fn and(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.select(left, right, Expr::ConstBool(false))
}

fn surface(builder: &StructuredFunctionBuilder, state: &[Expr; 8], kind: &Expr, radius: &Expr) -> Expr {
    builder.select(b(builder, BinOp::Eq, kind.clone(), builder.const_f32(0.0)),
        b(builder, BinOp::Sub, state[1].clone(), radius.clone()),
        b(builder, BinOp::Mul, state[1].clone(), builder.unary(UnOp::Cos, state[2].clone())))
}

impl SchwarzschildEventLocalizationBuilder {
    /// 默认允许四十八次二分。
    pub fn new() -> Self { Self::default() }

    /// 设置静态二分预算。
    pub fn maximum_bisections(mut self, iterations: u32) -> Self { self.maximum_bisections = iterations; self }

    /// 构造仅含通用算术、资源读写和有界控制流的程序。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_bisections == 0 || self.maximum_bisections > 64 {
            return Err("event bisection budget must be between 1 and 64".into());
        }
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_event_localization");
        let gid = builder.global_id();
        let input_base = b(&builder, BinOp::Mul, gid.clone(), builder.const_u32(17));
        let output_base = b(&builder, BinOp::Mul, gid, builder.const_u32(11));
        let load = |index| Expr::LoadBuffer { resource: 1, index: Box::new(Expr::BinOp {
            op: BinOp::Add, lhs: Box::new(input_base.clone()), rhs: Box::new(Expr::ConstU32(index)),
        }) };
        let start = snapshot(&mut builder, std::array::from_fn(|index| load(index as u32)));
        let end = snapshot(&mut builder, std::array::from_fn(|index| load(index as u32 + 8)));
        let step = load(16);
        let [mass, kind, radius, tolerance]: [Expr; 4] = std::array::from_fn(|slot| builder.load_uniform(slot as u32));
        let mut valid = Expr::ConstBool(true);
        for value in start.iter().chain(end.iter()).chain([&step, &mass, &kind, &radius, &tolerance]) {
            valid = and(&builder, valid, builder.unary(UnOp::IsFinite, value.clone()));
        }
        for value in [&mass, &step, &tolerance] {
            valid = and(&builder, valid, b(&builder, BinOp::Gt, value.clone(), builder.const_f32(0.0)));
        }
        for state in [&start, &end] {
            valid = and(&builder, valid, b(&builder, BinOp::Gt, state[1].clone(), b(&builder, BinOp::Mul, mass.clone(), builder.const_f32(2.0))));
        }
        let sphere_valid = and(&builder, b(&builder, BinOp::Eq, kind.clone(), builder.const_f32(0.0)),
            b(&builder, BinOp::Gt, radius.clone(), b(&builder, BinOp::Mul, mass.clone(), builder.const_f32(2.0))));
        valid = and(&builder, valid, builder.select(b(&builder, BinOp::Eq, kind.clone(), builder.const_f32(1.0)), Expr::ConstBool(true), sphere_valid));
        let before = surface(&builder, &start, &kind, &radius);
        let after = surface(&builder, &end, &kind, &radius);
        let opposite = b(&builder, BinOp::Ne,
            b(&builder, BinOp::Ge, before.clone(), builder.const_f32(0.0)),
            b(&builder, BinOp::Ge, after.clone(), builder.const_f32(0.0)));
        let bracketed = and(&builder, b(&builder, BinOp::Gt, abs(&builder, before.clone()), tolerance.clone()),
            builder.select(b(&builder, BinOp::Eq, after.clone(), builder.const_f32(0.0)), Expr::ConstBool(true), opposite));
        let status = builder.let_value(Ty::f32(), builder.select(valid,
            builder.select(bracketed, builder.const_f32(0.0), builder.const_f32(2.0)), builder.const_f32(4.0)));
        let endpoint = builder.let_value(Ty::Scalar(super::ScalarTy::Bool), and(&builder,
            b(&builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(0.0)),
            b(&builder, BinOp::Le, abs(&builder, after.clone()), tolerance.clone())));
        let output: [u32; 8] = std::array::from_fn(|index| builder.let_value(Ty::f32(), builder.select(Expr::LoadLocal(endpoint), end[index].clone(), start[index].clone())));
        let fraction = builder.let_value(Ty::f32(), builder.select(Expr::LoadLocal(endpoint), builder.const_f32(1.0), builder.const_f32(0.0)));
        let residual = builder.let_value(Ty::f32(), builder.select(Expr::LoadLocal(endpoint), abs(&builder, after), builder.const_f32(0.0)));
        builder.store_local(status, builder.select(Expr::LoadLocal(endpoint), builder.const_f32(1.0), Expr::LoadLocal(status)));
        let lower = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let upper = builder.let_value(Ty::f32(), builder.const_f32(1.0));
        let lower_value = builder.let_value(Ty::f32(), before);
        let running = b(&builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(0.0));
        builder.bounded_while(running, self.maximum_bisections, |builder| {
            let middle = builder.let_value(Ty::f32(), b(builder, BinOp::Mul,
                b(builder, BinOp::Add, Expr::LoadLocal(lower), Expr::LoadLocal(upper)), builder.const_f32(0.5)));
            let half_step = b(builder, BinOp::Mul, b(builder, BinOp::Mul, step.clone(), Expr::LoadLocal(middle)), builder.const_f32(0.5));
            let half = rk4(builder, &start, &mass, half_step.clone());
            let state = rk4(builder, &half, &mass, half_step);
            let value = builder.let_value(Ty::f32(), surface(builder, &state, &kind, &radius));
            let mut finite = Expr::ConstBool(true);
            for component in &state { finite = and(builder, finite, builder.unary(UnOp::IsFinite, component.clone())); }
            let found = builder.let_value(Ty::Scalar(super::ScalarTy::Bool), and(builder, finite.clone(),
                b(builder, BinOp::Le, abs(builder, Expr::LoadLocal(value)), tolerance.clone())));
            for index in 0..8 {
                builder.store_local(output[index], builder.select(Expr::LoadLocal(found), state[index].clone(), Expr::LoadLocal(output[index])));
            }
            builder.store_local(fraction, builder.select(Expr::LoadLocal(found), Expr::LoadLocal(middle), Expr::LoadLocal(fraction)));
            builder.store_local(residual, builder.select(Expr::LoadLocal(found), abs(builder, Expr::LoadLocal(value)), Expr::LoadLocal(residual)));
            builder.store_local(status, builder.select(finite, builder.select(Expr::LoadLocal(found), builder.const_f32(1.0), builder.const_f32(0.0)), builder.const_f32(4.0)));
            let same_side = builder.let_value(Ty::Scalar(super::ScalarTy::Bool), b(builder, BinOp::Eq,
                b(builder, BinOp::Ge, Expr::LoadLocal(value), builder.const_f32(0.0)),
                b(builder, BinOp::Ge, Expr::LoadLocal(lower_value), builder.const_f32(0.0))));
            builder.store_local(lower, builder.select(Expr::LoadLocal(same_side), Expr::LoadLocal(middle), Expr::LoadLocal(lower)));
            builder.store_local(upper, builder.select(Expr::LoadLocal(same_side), Expr::LoadLocal(upper), Expr::LoadLocal(middle)));
            builder.store_local(lower_value, builder.select(Expr::LoadLocal(same_side), Expr::LoadLocal(value), Expr::LoadLocal(lower_value)));
        });
        builder.store_local(status, builder.select(b(&builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(0.0)), builder.const_f32(3.0), Expr::LoadLocal(status)));
        let values = output.into_iter().map(Expr::LoadLocal).chain([fraction, residual, status].into_iter().map(Expr::LoadLocal));
        for (index, value) in values.enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)), value);
        }
        let program = Program { version: 1, entry: 0, functions: vec![builder.finish(Ty::f32(), Expr::LoadLocal(status))] };
        super::validate_program(&program).map_err(|error| error.to_string())?;
        Ok(program)
    }
}

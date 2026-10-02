//! 自适应参考算法的通用 UCF 单步前端。

use super::{b, schwarzschild_derivative, BinOp, Expr, Program, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

/// 构造有界 Schwarzschild 自适应 RK4 单步程序。
///
/// buffer 1 每实例为八个状态分量加请求步长。buffer 2 每实例为
/// `[state(8), status, step_used, next_step, error_ratio, rejections]`。
/// status 为 `1=accepted`、`2=minimum_step`、`3=retry_budget`、`4=invalid_input`。
/// uniform 0..4 为质量、绝对容差、相对容差、最小步长、最大步长。
/// 非有限候选使用 `f32::MAX` 作为误差哨兵，保证程序 JSON 可往返。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildAdaptiveStepBuilder {
    maximum_retries: u32,
}

impl Default for SchwarzschildAdaptiveStepBuilder {
    fn default() -> Self { Self { maximum_retries: 16 } }
}

/// 有界多步自适应积分构造器，资源布局与单步构造器相同。
///
/// `status=1` 表示所有请求步均接受，拒绝次数为整个 chunk 的累计值。
/// 该程序只推进积分，不替代捕获、逃逸、盘面事件定位或终止分类。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildAdaptiveChunkBuilder {
    maximum_steps: u32,
    maximum_retries: u32,
}

impl Default for SchwarzschildAdaptiveChunkBuilder {
    fn default() -> Self { Self { maximum_steps: 64, maximum_retries: 16 } }
}

impl SchwarzschildAdaptiveChunkBuilder {
    /// 使用默认的六十四个积分步与十六次单步重试。
    pub fn new() -> Self { Self::default() }

    /// 设置 chunk 请求的接受步数。
    pub fn maximum_steps(mut self, steps: u32) -> Self { self.maximum_steps = steps; self }

    /// 设置每个接受步的重试上限。
    pub fn maximum_retries(mut self, retries: u32) -> Self { self.maximum_retries = retries; self }

    /// 构造通用程序，拒绝零预算与过大静态循环上限。
    pub fn build(self) -> Result<Program, String> {
        SchwarzschildAdaptiveStepBuilder::new().maximum_retries(self.maximum_retries).build_steps(self.maximum_steps)
    }
}

pub(crate) fn snapshot(builder: &mut StructuredFunctionBuilder, values: [Expr; 8]) -> [Expr; 8] {
    values.map(|value| Expr::LoadLocal(builder.let_value(Ty::f32(), value)))
}

pub(crate) fn rk4(builder: &mut StructuredFunctionBuilder, values: &[Expr; 8], mass: &Expr, step: Expr) -> [Expr; 8] {
    let half_step = b(builder, BinOp::Mul, step.clone(), builder.const_f32(0.5));
    let k1 = snapshot(builder, schwarzschild_derivative(builder, values, mass));
    let middle = snapshot(builder, std::array::from_fn(|index| b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, k1[index].clone(), half_step.clone()))));
    let k2 = snapshot(builder, schwarzschild_derivative(builder, &middle, mass));
    let middle = snapshot(builder, std::array::from_fn(|index| b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, k2[index].clone(), half_step.clone()))));
    let k3 = snapshot(builder, schwarzschild_derivative(builder, &middle, mass));
    let end = snapshot(builder, std::array::from_fn(|index| b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, k3[index].clone(), step.clone()))));
    let k4 = snapshot(builder, schwarzschild_derivative(builder, &end, mass));
    let next = std::array::from_fn(|index| {
        let weighted = b(builder, BinOp::Add,
            b(builder, BinOp::Add, k1[index].clone(), b(builder, BinOp::Mul, builder.const_f32(2.0), k2[index].clone())),
            b(builder, BinOp::Add, b(builder, BinOp::Mul, builder.const_f32(2.0), k3[index].clone()), k4[index].clone()));
        b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, step.clone(), b(builder, BinOp::Div, weighted, builder.const_f32(6.0))))
    });
    snapshot(builder, next)
}

fn abs(builder: &StructuredFunctionBuilder, value: Expr) -> Expr {
    b(builder, BinOp::Max, value.clone(), builder.unary(UnOp::Neg, value))
}

fn and(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.select(left, right, Expr::ConstBool(false))
}

impl SchwarzschildAdaptiveStepBuilder {
    /// 使用默认的十六次重试上限。
    pub fn new() -> Self { Self::default() }

    /// 设置拒绝候选后允许的重试次数。
    pub fn maximum_retries(mut self, retries: u32) -> Self { self.maximum_retries = retries; self }

    /// 构造和校验程序，不注册 UCF 领域算子。
    pub fn build(self) -> Result<Program, String> {
        self.build_steps(1)
    }

    fn build_steps(self, maximum_steps: u32) -> Result<Program, String> {
        if self.maximum_retries > 64 { return Err("adaptive RK4 retry limit exceeds 64".into()); }
        if maximum_steps == 0 || maximum_steps > 65_536 { return Err("adaptive RK4 step count must be between 1 and 65536".into()); }
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_adaptive_rk4_step");
        let gid = builder.global_id();
        let input_base = b(&builder, BinOp::Mul, gid.clone(), builder.const_u32(9));
        let output_base = b(&builder, BinOp::Mul, gid, builder.const_u32(13));
        let initial = snapshot(&mut builder, std::array::from_fn(|index| Expr::LoadBuffer {
            resource: 1, index: Box::new(Expr::BinOp { op: BinOp::Add, lhs: Box::new(input_base.clone()), rhs: Box::new(Expr::ConstU32(index as u32)) }),
        }));
        let request = builder.load_buffer(1, b(&builder, BinOp::Add, input_base, builder.const_u32(8)));
        let uniforms: [Expr; 5] = std::array::from_fn(|slot| builder.load_uniform(slot as u32));
        let [mass, absolute, relative, minimum, maximum] = uniforms;
        let mut valid = Expr::ConstBool(true);
        for value in initial.iter().chain([&request, &mass, &absolute, &relative, &minimum, &maximum]) {
            valid = and(&builder, valid, builder.unary(UnOp::IsFinite, value.clone()));
        }
        for value in [&request, &mass, &absolute, &minimum] {
            valid = and(&builder, valid, b(&builder, BinOp::Gt, value.clone(), builder.const_f32(0.0)));
        }
        valid = and(&builder, valid, b(&builder, BinOp::Ge, relative.clone(), builder.const_f32(0.0)));
        valid = and(&builder, valid, b(&builder, BinOp::Ge, maximum.clone(), minimum.clone()));
        valid = and(&builder, valid, b(&builder, BinOp::Gt, initial[1].clone(), b(&builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())));
        let status = builder.let_value(Ty::f32(), builder.select(valid, builder.const_f32(0.0), builder.const_f32(4.0)));
        let step = builder.let_value(Ty::f32(), b(&builder, BinOp::Min, maximum.clone(), b(&builder, BinOp::Max, minimum.clone(), request)));
        let used = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let next = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let error = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let rejections = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let output: [u32; 8] = std::array::from_fn(|index| builder.let_value(Ty::f32(), initial[index].clone()));
        let can_advance = b(&builder, BinOp::Le, Expr::LoadLocal(status), builder.const_f32(1.0));
        builder.bounded_while(can_advance, maximum_steps, |builder| {
        let initial = snapshot(builder, output.map(Expr::LoadLocal));
        builder.store_local(status, builder.select(
            b(builder, BinOp::Gt, initial[1].clone(), b(builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())),
            builder.const_f32(0.0), builder.const_f32(4.0),
        ));
        let running = b(builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(0.0));
        builder.bounded_while(running, self.maximum_retries + 1, |builder| {
            let current_step = Expr::LoadLocal(step);
            let full = rk4(builder, &initial, &mass, current_step.clone());
            let half = rk4(builder, &initial, &mass, b(builder, BinOp::Mul, current_step.clone(), builder.const_f32(0.5)));
            let refined = rk4(builder, &half, &mass, b(builder, BinOp::Mul, current_step.clone(), builder.const_f32(0.5)));
            let mut ratio = builder.const_f32(0.0);
            let mut finite = Expr::ConstBool(true);
            for index in 0..8 {
                let scale = b(builder, BinOp::Add, absolute.clone(), b(builder, BinOp::Mul, relative.clone(), b(builder, BinOp::Max, abs(builder, initial[index].clone()), abs(builder, refined[index].clone()))));
                let component = b(builder, BinOp::Div, abs(builder, b(builder, BinOp::Div, b(builder, BinOp::Sub, refined[index].clone(), full[index].clone()), builder.const_f32(15.0))), scale);
                finite = and(builder, finite, builder.unary(UnOp::IsFinite, full[index].clone()));
                finite = and(builder, finite, builder.unary(UnOp::IsFinite, refined[index].clone()));
                finite = and(builder, finite, builder.unary(UnOp::IsFinite, component.clone()));
                ratio = b(builder, BinOp::Max, ratio, component);
            }
            let ratio = builder.let_value(Ty::f32(), builder.select(finite, ratio, builder.const_f32(f32::MAX)));
            let ratio = Expr::LoadLocal(ratio);
            let accepted = builder.let_value(Ty::Scalar(ScalarTy::Bool), b(builder, BinOp::Le, ratio.clone(), builder.const_f32(1.0)));
            let accepted = Expr::LoadLocal(accepted);
            let factor = b(builder, BinOp::Mul, builder.const_f32(0.9), b(builder, BinOp::Pow, ratio.clone(), builder.const_f32(-0.2)));
            let accept_factor = builder.select(b(builder, BinOp::Eq, ratio.clone(), builder.const_f32(0.0)), builder.const_f32(2.0), b(builder, BinOp::Min, builder.const_f32(2.0), b(builder, BinOp::Max, builder.const_f32(0.5), factor.clone())));
            let next_step = b(builder, BinOp::Min, maximum.clone(), b(builder, BinOp::Max, minimum.clone(), b(builder, BinOp::Mul, current_step.clone(), accept_factor)));
            builder.store_local(error, ratio);
            for index in 0..8 { builder.store_local(output[index], builder.select(accepted.clone(), refined[index].clone(), Expr::LoadLocal(output[index]))); }
            builder.store_local(used, builder.select(accepted.clone(), current_step.clone(), Expr::LoadLocal(used)));
            builder.store_local(next, builder.select(accepted.clone(), next_step, Expr::LoadLocal(next)));
            builder.store_local(status, builder.select(accepted.clone(), builder.const_f32(1.0), builder.select(b(builder, BinOp::Le, current_step.clone(), minimum.clone()), builder.const_f32(2.0), builder.const_f32(0.0))));
            builder.store_local(rejections, builder.select(accepted.clone(), Expr::LoadLocal(rejections), b(builder, BinOp::Add, Expr::LoadLocal(rejections), builder.const_f32(1.0))));
            let reject_factor = b(builder, BinOp::Min, builder.const_f32(0.5), b(builder, BinOp::Max, builder.const_f32(0.1), factor));
            builder.store_local(step, builder.select(accepted, current_step.clone(), b(builder, BinOp::Max, minimum.clone(), b(builder, BinOp::Mul, current_step, reject_factor))));
        });
        builder.store_local(status, builder.select(b(builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(0.0)), builder.const_f32(3.0), Expr::LoadLocal(status)));
        builder.store_local(step, builder.select(b(builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(1.0)), Expr::LoadLocal(next), Expr::LoadLocal(step)));
        });
        let values = output.into_iter().map(Expr::LoadLocal).chain([status, used, next, error, rejections].into_iter().map(Expr::LoadLocal));
        for (index, value) in values.enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)), value);
        }
        let program = Program { version: 1, entry: 0, functions: vec![builder.finish(Ty::f32(), Expr::LoadLocal(status))] };
        super::validate_program(&program).map_err(|error| error.to_string())?;
        Ok(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn adaptive_program_is_bounded_and_executes_a_step() {
        assert!(SchwarzschildAdaptiveChunkBuilder::new().maximum_steps(0).build().is_err());
        assert!(SchwarzschildAdaptiveChunkBuilder::new().maximum_steps(65_537).build().is_err());
        assert!(SchwarzschildAdaptiveStepBuilder::new().maximum_retries(65).build().is_err());
        let program = SchwarzschildAdaptiveStepBuilder::new().build().unwrap();
        let mut interpreter = CpuInterpreter::new();
        for (slot, value) in [1.0, 1.0e-9, 1.0e-7, 1.0e-8, 0.08].into_iter().enumerate() {
            interpreter.with_uniform(slot as u32, Value::F32(value));
        }
        interpreter.with_buffer(1, [0.0, 12.0, 1.0, 0.3, 1.2, -0.7, 0.22, 0.08, 0.01].into_iter().map(Value::F32).collect());
        interpreter.with_buffer(2, vec![Value::F32(0.0); 13]);
        interpreter.run_parallel(&program, 1).unwrap();
        let output = interpreter.buffer(2).unwrap();
        assert_eq!(output[8].as_f64().unwrap(), 1.0);
        assert!(output[11].as_f64().unwrap() <= 1.0);
        assert!(output[1].as_f64().unwrap() < 12.0);
    }
}

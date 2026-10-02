//! 笛卡尔零测地线算法的通用算术与有界循环展开。

use super::{b, BinOp, Expr, Program, StructuredFunctionBuilder, Ty, UnOp};
use super::adaptive::{emit_integration_model, snapshot, IntegrationModel};

pub(crate) fn radius(builder: &StructuredFunctionBuilder, state: &[Expr; 8]) -> Expr {
    let square = |index: usize| b(builder, BinOp::Mul, state[index].clone(), state[index].clone());
    builder.unary(UnOp::Sqrt, b(builder, BinOp::Add, b(builder, BinOp::Add, square(1), square(2)), square(3)))
}

pub(crate) fn derivative(builder: &StructuredFunctionBuilder, state: &[Expr; 8], mass: &Expr, conserved_angular: Option<Expr>) -> [Expr; 8] {
    let radius = radius(builder, state);
    let cross = |first: usize, second: usize, third: usize, fourth: usize| b(builder, BinOp::Sub,
        b(builder, BinOp::Mul, state[first].clone(), state[second].clone()),
        b(builder, BinOp::Mul, state[third].clone(), state[fourth].clone()));
    let angular = [cross(2, 7, 3, 6), cross(3, 5, 1, 7), cross(1, 6, 2, 5)];
    let square = |index: usize| b(builder, BinOp::Mul, angular[index].clone(), angular[index].clone());
    let angular_squared = conserved_angular.unwrap_or_else(|| b(builder, BinOp::Add, b(builder, BinOp::Add, square(0), square(1)), square(2)));
    let radius_squared = b(builder, BinOp::Mul, radius.clone(), radius.clone());
    let radius_fifth = b(builder, BinOp::Mul, b(builder, BinOp::Mul, radius_squared.clone(), radius_squared), radius.clone());
    let coefficient = b(builder, BinOp::Div, b(builder, BinOp::Mul,
        b(builder, BinOp::Mul, builder.const_f32(-3.0), mass.clone()), angular_squared), radius_fifth);
    let f = b(builder, BinOp::Div, b(builder, BinOp::Sub, radius.clone(),
        b(builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())), radius.clone());
    let time = builder.select(b(builder, BinOp::Gt, radius,
        b(builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())),
        b(builder, BinOp::Div, state[4].clone(), f), builder.unary(UnOp::Sqrt, builder.const_f32(-1.0)));
    [time, state[5].clone(), state[6].clone(), state[7].clone(),
        builder.const_f32(0.0), b(builder, BinOp::Mul, coefficient.clone(), state[1].clone()),
        b(builder, BinOp::Mul, coefficient.clone(), state[2].clone()), b(builder, BinOp::Mul, coefficient, state[3].clone())]
}

/// 输入 `[t,x,y,z,E,vx,vy,vz,step]`，输出八标量状态及 status、已用步长、下步步长、误差、累计拒绝次数、最大零约束残差、最大角动量平方漂移、初始角动量平方。
/// 角动量平方误差按 `max(L0²,(M E0)²)` 归一化，L0 与 E0 为当前 chunk 输入。
/// 此程序只做积分，不负责视界或盘面事件，调用方必须独立检查终止与守恒门禁。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildCartesianAdaptiveChunkBuilder {
    maximum_steps: u32,
    maximum_retries: u32,
}

impl Default for SchwarzschildCartesianAdaptiveChunkBuilder {
    fn default() -> Self { Self { maximum_steps: 1, maximum_retries: 16 } }
}

impl SchwarzschildCartesianAdaptiveChunkBuilder {
    /// 默认单个接受步，最多十六次重试。
    pub fn new() -> Self { Self::default() }
    /// 静态接受步上限。
    pub fn maximum_steps(mut self, steps: u32) -> Self { self.maximum_steps = steps; self }
    /// 每步静态重试上限。
    pub fn maximum_retries(mut self, retries: u32) -> Self { self.maximum_retries = retries; self }
    /// 生成通用 UCF 程序，与球坐标模型共用 RK4 与误差控制实现。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_steps == 0 || self.maximum_steps > 65536 || self.maximum_retries > 64 {
            return Err("invalid Cartesian adaptive chunk budget".into());
        }
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_cartesian_adaptive_chunk");
        let input_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(9));
        let output_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(16));
        let values = std::array::from_fn(|index| builder.load_buffer(1,
            b(&builder, BinOp::Add, input_base.clone(), builder.const_u32(index as u32))));
        let initial = snapshot(&mut builder, values);
        let request = builder.load_buffer(1, b(&builder, BinOp::Add, input_base, builder.const_u32(8)));
        let uniforms: [Expr; 5] = std::array::from_fn(|slot| builder.load_uniform(slot as u32));
        let initial_invariants = invariants(&builder, &initial, &uniforms[0]);
        let initial_null = builder.let_value(Ty::f32(), initial_invariants[0].clone());
        let initial_angular = builder.let_value(Ty::f32(), initial_invariants[1].clone());
        let valid = builder.select(b(&builder, BinOp::Gt, initial[4].clone(), builder.const_f32(0.0)),
            b(&builder, BinOp::Le, Expr::LoadLocal(initial_null), builder.const_f32(1.0e-4)), Expr::ConstBool(false));
        let request = builder.select(valid, request, builder.const_f32(0.0));
        let locals: [u32; 13] = std::array::from_fn(|index| builder.let_value(Ty::f32(),
            if index < 8 { initial[index].clone() } else if index == 10 { request.clone() } else { builder.const_f32(0.0) }));
        let max_null = builder.let_value(Ty::f32(), Expr::LoadLocal(initial_null));
        let max_angular = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        let running = b(&builder, BinOp::Le, Expr::LoadLocal(locals[8]), builder.const_f32(1.0));
        builder.bounded_while(running, self.maximum_steps, |builder| {
            let current = std::array::from_fn(|index| Expr::LoadLocal(locals[index]));
            let integrated = emit_integration_model(builder, current, Expr::LoadLocal(locals[10]), uniforms.clone(),
                1, self.maximum_retries, IntegrationModel::CartesianNull);
            let integrated = integrated.map(|value| Expr::LoadLocal(builder.let_value(Ty::f32(), value)));
            let state = std::array::from_fn(|index| integrated[index].clone());
            let [null, angular] = invariants(builder, &state, &uniforms[0]);
            let difference = b(builder, BinOp::Sub, angular, Expr::LoadLocal(initial_angular));
            let absolute = builder.unary(UnOp::Abs, difference);
            let energy_scale = b(builder, BinOp::Mul, uniforms[0].clone(), initial[4].clone());
            let angular_scale = b(builder, BinOp::Max, Expr::LoadLocal(initial_angular),
                b(builder, BinOp::Mul, energy_scale.clone(), energy_scale));
            let drift = b(builder, BinOp::Div, absolute, angular_scale);
            builder.store_local(max_null, b(builder, BinOp::Max, Expr::LoadLocal(max_null), null));
            builder.store_local(max_angular, b(builder, BinOp::Max, Expr::LoadLocal(max_angular), drift));
            for index in 0..13 {
                let value = if index == 12 { b(builder, BinOp::Add, Expr::LoadLocal(locals[index]), integrated[index].clone()) }
                    else { integrated[index].clone() };
                builder.store_local(locals[index], value);
            }
        });
        let mut output = locals.map(Expr::LoadLocal).to_vec();
        output.extend([Expr::LoadLocal(max_null), Expr::LoadLocal(max_angular), Expr::LoadLocal(initial_angular)]);
        for (index, value) in output.iter().cloned().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)), value);
        }
        let program = Program { version: 1, entry: 0, functions: vec![builder.finish(Ty::f32(), output[8].clone())] };
        super::validate_program(&program).map_err(|error| error.to_string())?;
        Ok(program)
    }
}

pub(crate) fn invariants(builder: &StructuredFunctionBuilder, state: &[Expr; 8], mass: &Expr) -> [Expr; 2] {
    let radius = radius(builder, state);
    let cross = |first: usize, second: usize, third: usize, fourth: usize| b(builder, BinOp::Sub,
        b(builder, BinOp::Mul, state[first].clone(), state[second].clone()),
        b(builder, BinOp::Mul, state[third].clone(), state[fourth].clone()));
    let angular = [cross(2, 7, 3, 6), cross(3, 5, 1, 7), cross(1, 6, 2, 5)];
    let squared = angular.into_iter().fold(builder.const_f32(0.0), |sum, value|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, value.clone(), value)));
    let dot = (1..=3).fold(builder.const_f32(0.0), |sum, index|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, state[index].clone(), state[index + 4].clone())));
    let radial = b(builder, BinOp::Div, dot, radius.clone());
    let f = b(builder, BinOp::Div, b(builder, BinOp::Sub, radius.clone(),
        b(builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())), radius.clone());
    let spatial = b(builder, BinOp::Add, b(builder, BinOp::Mul, radial.clone(), radial),
        b(builder, BinOp::Mul, f, b(builder, BinOp::Div, squared.clone(), b(builder, BinOp::Mul, radius.clone(), radius))));
    let temporal = b(builder, BinOp::Mul, state[4].clone(), state[4].clone());
    let null = b(builder, BinOp::Div, builder.unary(UnOp::Abs, b(builder, BinOp::Sub, spatial.clone(), temporal.clone())),
        b(builder, BinOp::Add, spatial, temporal));
    [null, squared]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    fn execute(input: [f32; 9]) -> Vec<f64> {
        let program = SchwarzschildCartesianAdaptiveChunkBuilder::new().build().unwrap();
        let mut interpreter = CpuInterpreter::new();
        for (slot, value) in [1.0, 1.0e-6, 1.0e-5, 1.0e-8, 0.08].into_iter().enumerate() {
            interpreter.with_uniform(slot as u32, Value::F32(value));
        }
        interpreter.with_buffer(1, input.into_iter().map(Value::F32).collect());
        interpreter.with_buffer(2, vec![Value::F32(0.0); 16]);
        interpreter.run_parallel(&program, 1).unwrap();
        interpreter.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect()
    }

    #[test]
    fn cartesian_builder_rejects_invalid_budgets_and_non_null_input() {
        assert!(SchwarzschildCartesianAdaptiveChunkBuilder::new().maximum_steps(0).build().is_err());
        assert!(SchwarzschildCartesianAdaptiveChunkBuilder::new().maximum_steps(65537).build().is_err());
        assert!(SchwarzschildCartesianAdaptiveChunkBuilder::new().maximum_retries(65).build().is_err());
        let input = [0.0, 0.0, 0.0, 10.0, 2.0, 0.0, 0.0, -1.0, 0.04];
        let output = execute(input);
        assert_eq!(output[8], 4.0);
        for (actual, expected) in output[..8].iter().zip(&input[..8]) { assert_eq!(*actual, *expected as f64); }
    }

    #[test]
    fn cartesian_axis_radial_program_has_finite_exact_spatial_motion() {
        let output = execute([0.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, -1.0, 0.04]);
        assert_eq!(output[8], 1.0);
        assert!((output[3] - 9.96).abs() < 2.0e-6);
        assert_eq!(output[4], 1.0);
        assert_eq!(output[7], -1.0);
        assert_eq!(output[13], 0.0);
        assert_eq!(output[14], 0.0);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn cartesian_program_rejects_horizon_crossing_candidate_and_retries() {
        let output = execute([0.0, 0.0, 0.0, 2.01, 1.0, 0.0, 0.0, -1.0, 0.08]);
        assert_eq!(output[8], 1.0);
        assert!(output[9] > 0.0 && output[9] < 0.01);
        assert!(output[3] > 2.0);
        assert!(output[12] > 0.0);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}

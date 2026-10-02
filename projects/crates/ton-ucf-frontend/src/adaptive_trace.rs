//! 自适应积分与事件定位在同一个通用 UCF 程序中的组合。

use super::{b, BinOp, Expr, Program, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};
use super::adaptive::{emit_integration_model, snapshot};
use super::events::emit_localization_model;

/// 构造逐接受步检查事件的追踪 chunk，不执行参考函数或 shader 模板。
///
/// 输入 buffer 1、输出 buffer 2 的每实例布局均为二十九个标量：
/// `[state(8), status, next_step, accepted_steps, rejections, affine,
/// disk_count, first_disk_radius, first_disk_affine, max_null, max_energy_drift,
/// max_angular_squared_drift, initial_energy, initial_angular_squared, first_disk_state(8)]`。
/// status 为 0=运行中、1=捕获、2=逃逸、3=积分失败、4=事件定位失败、5=非法输入。
/// uniform 0..9 为质量、绝对容差、相对容差、最小步长、最大步长、
/// 捕获半径、逃逸半径、盘内缘、盘外缘、事件面残差容差。
/// chunk 预算用尽保持运行中，调用方必须独立报告全局预算耗尽。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildAdaptiveTraceChunkBuilder {
    maximum_steps: u32,
    maximum_retries: u32,
    maximum_bisections: u32,
    null_energy: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn null_energy_formulation_preserves_null_input_and_rejects_non_null_input() {
        let program = SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(1).conserved_null_energy().build().unwrap();
        let mut input = [0.0f32; 29];
        input[..8].copy_from_slice(&[0.0, 10.0, 1.2, 0.3, 1.25, -1.0, 0.0, 0.0]);
        input[9] = 0.04;
        for (utheta, status) in [(0.0, 0.0), (0.1, 5.0)] {
            input[6] = utheta;
            let mut cpu = CpuInterpreter::new();
            for (slot, value) in [1.0, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 80.0, 6.0, 20.0, 1.0e-5].into_iter().enumerate() {
                cpu.with_uniform(slot as u32, Value::F32(value));
            }
            cpu.with_buffer(1, input.into_iter().map(Value::F32).collect());
            cpu.with_buffer(2, vec![Value::F32(0.0); 29]);
            cpu.run_parallel(&program, 1).unwrap();
            let output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert_eq!(output[8], status);
            if status == 0.0 {
                assert_eq!(output[10], 1.0);
                assert!(output[16] < 1.0e-6);
                assert!(output[17] < 1.0e-6);
            } else {
                assert_eq!(output[10], 0.0);
                for (actual, expected) in output[..8].iter().zip(&input[..8]) { assert_eq!(*actual, *expected as f64); }
            }
        }
    }

    #[test]
    fn trace_chunk_distinguishes_budget_integration_event_and_input_failures() {
        assert!(SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(0).build().is_err());
        assert!(SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_retries(65).build().is_err());
        assert!(SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_bisections(0).build().is_err());
        let mut input = [0.0f32; 29];
        input[..8].copy_from_slice(&[0.0, 10.0, std::f32::consts::FRAC_PI_2 - 0.001, 0.3, 1.25, 1.0, 0.1, 0.0]);
        input[9] = 0.04;
        let defaults = [1.0f32, 1.0e-6, 1.0e-5, 1.0e-8, 0.08, 2.0002, 80.0, 6.0, 20.0, 1.0e-5];
        for (case, expected) in [(0, 0.0), (1, 3.0), (2, 4.0), (3, 5.0)] {
            let mut uniforms = defaults;
            let mut current = input;
            let mut config = SchwarzschildAdaptiveTraceChunkBuilder::new().maximum_steps(1);
            match case {
                1 => { uniforms[1] = 1.0e-30; uniforms[2] = 0.0; uniforms[3] = 0.08; },
                2 => { config = config.maximum_bisections(1); uniforms[9] = 1.0e-9; },
                3 => { current[8] = 42.0; },
                _ => {},
            }
            let program = config.build().unwrap();
            let mut cpu = CpuInterpreter::new();
            for (slot, value) in uniforms.into_iter().enumerate() { cpu.with_uniform(slot as u32, Value::F32(value)); }
            cpu.with_buffer(1, current.into_iter().map(Value::F32).collect());
            cpu.with_buffer(2, vec![Value::F32(0.0); 29]);
            cpu.run_parallel(&program, 1).unwrap();
            let output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert_eq!(output[8], expected, "case={case}, output={output:?}");
            if case == 0 {
                assert_eq!(output[10], 1.0);
                assert_eq!(output[13], 1.0);
            } else {
                assert_eq!(output[10], 0.0);
                assert_eq!(output[13], 0.0);
                for (actual, expected) in output[..8].iter().zip(input[..8].iter()) { assert_eq!(*actual, *expected as f64); }
            }
        }
    }
}

impl Default for SchwarzschildAdaptiveTraceChunkBuilder {
    fn default() -> Self { Self { maximum_steps: 16, maximum_retries: 16, maximum_bisections: 48, null_energy: false } }
}

fn and(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.select(left, right, Expr::ConstBool(false))
}

fn abs(builder: &StructuredFunctionBuilder, value: Expr) -> Expr {
    b(builder, BinOp::Max, value.clone(), builder.unary(UnOp::Neg, value))
}

fn square(builder: &StructuredFunctionBuilder, value: Expr) -> Expr {
    b(builder, BinOp::Mul, value.clone(), value)
}

fn invariants(builder: &StructuredFunctionBuilder, state: &[Expr; 8], mass: &Expr, null_energy: bool) -> [Expr; 3] {
    let f = b(builder, BinOp::Div, b(builder, BinOp::Sub, state[1].clone(),
        b(builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())), state[1].clone());
    let angular = b(builder, BinOp::Add, square(builder, state[6].clone()),
        b(builder, BinOp::Mul, square(builder, builder.unary(UnOp::Sin, state[2].clone())), square(builder, state[7].clone())));
    let radius_squared = square(builder, state[1].clone());
    let energy = if null_energy { state[4].clone() } else { b(builder, BinOp::Mul, f.clone(), state[4].clone()) };
    let temporal = square(builder, energy.clone());
    let spatial = b(builder, BinOp::Add, square(builder, state[5].clone()),
        b(builder, BinOp::Mul, f, b(builder, BinOp::Mul, radius_squared.clone(), angular.clone())));
    [energy, b(builder, BinOp::Mul, square(builder, radius_squared), angular),
        b(builder, BinOp::Div, abs(builder, b(builder, BinOp::Sub, spatial.clone(), temporal.clone())),
            b(builder, BinOp::Add, abs(builder, spatial), abs(builder, temporal)))]
}

fn observe(builder: &mut StructuredFunctionBuilder, locals: &[u32; 29], state: &[Expr; 8], mass: &Expr, enabled: Expr, null_energy: bool) {
    let [energy, angular, null] = invariants(builder, state, mass, null_energy);
    let energy_drift = abs(builder, b(builder, BinOp::Div, b(builder, BinOp::Sub, energy, Expr::LoadLocal(locals[19])), Expr::LoadLocal(locals[19])));
    let angular_drift = builder.select(b(builder, BinOp::Eq, Expr::LoadLocal(locals[20]), builder.const_f32(0.0)), abs(builder, angular.clone()),
        abs(builder, b(builder, BinOp::Div, b(builder, BinOp::Sub, angular, Expr::LoadLocal(locals[20])), Expr::LoadLocal(locals[20]))));
    for (index, value) in [(16, null), (17, energy_drift), (18, angular_drift)] {
        builder.store_local(locals[index], builder.select(enabled.clone(), b(builder, BinOp::Max, Expr::LoadLocal(locals[index]), value), Expr::LoadLocal(locals[index])));
    }
}

impl SchwarzschildAdaptiveTraceChunkBuilder {
    /// 默认每个 chunk 十六个接受步。
    pub fn new() -> Self { Self::default() }
    /// 设置静态接受步预算。
    pub fn maximum_steps(mut self, steps: u32) -> Self { self.maximum_steps = steps; self }
    /// 设置每步拒绝候选的重试预算。
    pub fn maximum_retries(mut self, retries: u32) -> Self { self.maximum_retries = retries; self }
    /// 设置每个事件的二分定位预算。
    pub fn maximum_bisections(mut self, iterations: u32) -> Self { self.maximum_bisections = iterations; self }

    /// 使用守恒能量降阶零测地线模型，拒绝初始归一化零约束残差大于 `1e-4` 的输入。
    /// 对外状态仍是时间四速度布局。它不适用于一般时类或非零测地线。
    pub fn conserved_null_energy(mut self) -> Self { self.null_energy = true; self }

    /// 组合通用程序片段，保留终止状态与首次盘命中跨 chunk 的一致性。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_steps == 0 || self.maximum_steps > 65_536 || self.maximum_retries > 64
            || self.maximum_bisections == 0 || self.maximum_bisections > 64
        { return Err("invalid adaptive trace chunk loop bounds".into()); }
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_adaptive_trace_chunk");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(29));
        let locals: [u32; 29] = std::array::from_fn(|index| builder.let_value(Ty::f32(), builder.load_buffer(1,
            b(&builder, BinOp::Add, base.clone(), builder.const_u32(index as u32)))));
        let uniforms: [Expr; 10] = std::array::from_fn(|slot| builder.load_uniform(slot as u32));
        let mass = uniforms[0].clone();
        let mut valid = Expr::ConstBool(true);
        for value in locals.iter().map(|id| Expr::LoadLocal(*id)).chain(uniforms.iter().cloned()) {
            valid = and(&builder, valid, builder.unary(UnOp::IsFinite, value));
        }
        for value in [uniforms[0].clone(), uniforms[1].clone(), uniforms[3].clone(), uniforms[9].clone(), Expr::LoadLocal(locals[9])] {
            valid = and(&builder, valid, b(&builder, BinOp::Gt, value, builder.const_f32(0.0)));
        }
        for value in [uniforms[2].clone(), Expr::LoadLocal(locals[10]), Expr::LoadLocal(locals[11]), Expr::LoadLocal(locals[12]), Expr::LoadLocal(locals[13])] {
            valid = and(&builder, valid, b(&builder, BinOp::Ge, value, builder.const_f32(0.0)));
        }
        valid = and(&builder, valid, b(&builder, BinOp::Ge, uniforms[4].clone(), uniforms[3].clone()));
        for (left, right) in [(uniforms[5].clone(), b(&builder, BinOp::Mul, mass.clone(), builder.const_f32(2.0))),
            (uniforms[6].clone(), uniforms[5].clone()), (uniforms[7].clone(), uniforms[5].clone()), (uniforms[8].clone(), uniforms[7].clone())] {
            valid = and(&builder, valid, b(&builder, BinOp::Gt, left, right));
        }
        valid = and(&builder, valid, b(&builder, BinOp::Gt, Expr::LoadLocal(locals[1]), b(&builder, BinOp::Mul, mass.clone(), builder.const_f32(2.0))));
        valid = and(&builder, valid, b(&builder, BinOp::Gt, Expr::LoadLocal(locals[4]), builder.const_f32(0.0)));
        let status_valid = (0..=5).fold(Expr::ConstBool(false), |found, status| builder.select(found, Expr::ConstBool(true),
            b(&builder, BinOp::Eq, Expr::LoadLocal(locals[8]), builder.const_f32(status as f32))));
        valid = and(&builder, valid, status_valid);
        let valid_local = builder.let_value(Ty::Scalar(ScalarTy::Bool), valid);
        valid = Expr::LoadLocal(valid_local);
        let initial: [Expr; 8] = std::array::from_fn(|index| Expr::LoadLocal(locals[index]));
        let [physical_energy, _, initial_null] = invariants(&builder, &initial, &mass, false);
        if self.null_energy {
            let null_valid = and(&builder, builder.unary(UnOp::IsFinite, initial_null.clone()),
                b(&builder, BinOp::Le, initial_null, builder.const_f32(1.0e-4)));
            valid = and(&builder, valid, builder.select(b(&builder, BinOp::Eq, Expr::LoadLocal(locals[10]), builder.const_f32(0.0)),
                null_valid, Expr::ConstBool(true)));
        }
        valid = and(&builder, valid, builder.select(b(&builder, BinOp::Eq, Expr::LoadLocal(locals[10]), builder.const_f32(0.0)),
            Expr::ConstBool(true), b(&builder, BinOp::Gt, Expr::LoadLocal(locals[19]), builder.const_f32(0.0))));
        builder.store_local(locals[8], builder.select(valid, Expr::LoadLocal(locals[8]), builder.const_f32(5.0)));
        let converted = builder.let_value(Ty::Scalar(ScalarTy::Bool), b(&builder, BinOp::Eq, Expr::LoadLocal(locals[8]), builder.const_f32(0.0)));
        if self.null_energy {
            builder.store_local(locals[4], builder.select(Expr::LoadLocal(converted), physical_energy, Expr::LoadLocal(locals[4])));
        }
        let initialized = and(&builder, b(&builder, BinOp::Eq, Expr::LoadLocal(locals[10]), builder.const_f32(0.0)),
            b(&builder, BinOp::Eq, Expr::LoadLocal(locals[8]), builder.const_f32(0.0)));
        let initialized = builder.let_value(Ty::Scalar(ScalarTy::Bool), initialized);
        let initial: [Expr; 8] = std::array::from_fn(|index| Expr::LoadLocal(locals[index]));
        let [energy, angular, null] = invariants(&builder, &initial, &mass, self.null_energy);
        for (index, value) in [(19, energy), (20, angular), (16, null)] {
            builder.store_local(locals[index], builder.select(Expr::LoadLocal(initialized), value, Expr::LoadLocal(locals[index])));
        }
        let running = b(&builder, BinOp::Eq, Expr::LoadLocal(locals[8]), builder.const_f32(0.0));
        let captured = b(&builder, BinOp::Le, Expr::LoadLocal(locals[1]), uniforms[5].clone());
        let escaped = and(&builder, b(&builder, BinOp::Ge, Expr::LoadLocal(locals[1]), uniforms[6].clone()),
            b(&builder, BinOp::Gt, Expr::LoadLocal(locals[5]), builder.const_f32(0.0)));
        builder.store_local(locals[8], builder.select(running, builder.select(captured, builder.const_f32(1.0),
            builder.select(escaped, builder.const_f32(2.0), builder.const_f32(0.0))), Expr::LoadLocal(locals[8])));
        let running = b(&builder, BinOp::Eq, Expr::LoadLocal(locals[8]), builder.const_f32(0.0));
        builder.bounded_while(running, self.maximum_steps, |builder| {
            let start = snapshot(builder, std::array::from_fn(|index| Expr::LoadLocal(locals[index])));
            let integrated = emit_integration_model(builder, start.clone(), Expr::LoadLocal(locals[9]), std::array::from_fn(|index| uniforms[index].clone()), 1, self.maximum_retries, self.null_energy);
            let end: [Expr; 8] = std::array::from_fn(|index| integrated[index].clone());
            let capture = emit_localization_model(builder, start.clone(), end.clone(), integrated[9].clone(),
                [mass.clone(), builder.const_f32(0.0), uniforms[5].clone(), uniforms[9].clone()], self.maximum_bisections, self.null_energy);
            let escape = emit_localization_model(builder, start.clone(), end.clone(), integrated[9].clone(),
                [mass.clone(), builder.const_f32(0.0), uniforms[6].clone(), uniforms[9].clone()], self.maximum_bisections, self.null_energy);
            let disk = emit_localization_model(builder, start, end.clone(), integrated[9].clone(),
                [mass.clone(), builder.const_f32(1.0), uniforms[5].clone(), uniforms[9].clone()], self.maximum_bisections, self.null_energy);
            let accepted = b(builder, BinOp::Eq, integrated[8].clone(), builder.const_f32(1.0));
            let event_failure = [capture[10].clone(), escape[10].clone(), disk[10].clone()].into_iter().fold(Expr::ConstBool(false), |failed, status|
                builder.select(failed, Expr::ConstBool(true), b(builder, BinOp::Gt, status, builder.const_f32(2.0))));
            let usable = builder.let_value(Ty::Scalar(ScalarTy::Bool), and(builder, accepted.clone(), builder.select(event_failure.clone(), Expr::ConstBool(false), Expr::ConstBool(true))));
            let captured = builder.let_value(Ty::Scalar(ScalarTy::Bool), and(builder, b(builder, BinOp::Eq, capture[10].clone(), builder.const_f32(1.0)),
                b(builder, BinOp::Le, end[1].clone(), uniforms[5].clone())));
            let escaped = builder.let_value(Ty::Scalar(ScalarTy::Bool), and(builder, b(builder, BinOp::Eq, escape[10].clone(), builder.const_f32(1.0)),
                b(builder, BinOp::Gt, end[5].clone(), builder.const_f32(0.0))));
            let fraction = builder.let_value(Ty::f32(), builder.select(Expr::LoadLocal(captured), capture[8].clone(),
                builder.select(Expr::LoadLocal(escaped), escape[8].clone(), builder.const_f32(1.0))));
            let hit = and(builder, Expr::LoadLocal(usable), and(builder,
                b(builder, BinOp::Eq, disk[10].clone(), builder.const_f32(1.0)), and(builder,
                    b(builder, BinOp::Ge, disk[1].clone(), uniforms[7].clone()), and(builder,
                        b(builder, BinOp::Le, disk[1].clone(), uniforms[8].clone()), b(builder, BinOp::Le, disk[8].clone(), Expr::LoadLocal(fraction))))));
            let hit = builder.let_value(Ty::Scalar(ScalarTy::Bool), hit);
            let first_hit = builder.let_value(Ty::Scalar(ScalarTy::Bool), and(builder, Expr::LoadLocal(hit),
                b(builder, BinOp::Eq, Expr::LoadLocal(locals[13]), builder.const_f32(0.0))));
            let event_affine = b(builder, BinOp::Add, Expr::LoadLocal(locals[12]), b(builder, BinOp::Mul, integrated[9].clone(), disk[8].clone()));
            for (index, value) in [(14, disk[1].clone()), (15, event_affine)] {
                builder.store_local(locals[index], builder.select(Expr::LoadLocal(first_hit), value, Expr::LoadLocal(locals[index])));
            }
            for index in 0..8 {
                let value = if self.null_energy && index == 4 {
                    b(builder, BinOp::Div, disk[4].clone(), b(builder, BinOp::Div,
                        b(builder, BinOp::Sub, disk[1].clone(), b(builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())), disk[1].clone()))
                } else { disk[index].clone() };
                builder.store_local(locals[21 + index], builder.select(Expr::LoadLocal(first_hit), value, Expr::LoadLocal(locals[21 + index])));
            }
            builder.store_local(locals[13], builder.select(Expr::LoadLocal(hit), b(builder, BinOp::Add, Expr::LoadLocal(locals[13]), builder.const_f32(1.0)), Expr::LoadLocal(locals[13])));
            let final_state = snapshot(builder, std::array::from_fn(|index| builder.select(Expr::LoadLocal(captured), capture[index].clone(),
                builder.select(Expr::LoadLocal(escaped), escape[index].clone(), end[index].clone()))));
            observe(builder, &locals, &final_state, &mass, Expr::LoadLocal(usable), self.null_energy);
            observe(builder, &locals, &std::array::from_fn(|index| disk[index].clone()), &mass, Expr::LoadLocal(hit), self.null_energy);
            for index in 0..8 { builder.store_local(locals[index], builder.select(Expr::LoadLocal(usable), final_state[index].clone(), Expr::LoadLocal(locals[index]))); }
            for (index, value) in [(9, integrated[10].clone()),
                (10, b(builder, BinOp::Add, Expr::LoadLocal(locals[10]), builder.const_f32(1.0))),
                (11, b(builder, BinOp::Add, Expr::LoadLocal(locals[11]), integrated[12].clone())),
                (12, b(builder, BinOp::Add, Expr::LoadLocal(locals[12]), b(builder, BinOp::Mul, integrated[9].clone(), Expr::LoadLocal(fraction))))] {
                builder.store_local(locals[index], builder.select(Expr::LoadLocal(usable), value, Expr::LoadLocal(locals[index])));
            }
            builder.store_local(locals[8], builder.select(accepted, builder.select(event_failure, builder.const_f32(4.0),
                builder.select(Expr::LoadLocal(captured), builder.const_f32(1.0), builder.select(Expr::LoadLocal(escaped), builder.const_f32(2.0), builder.const_f32(0.0)))), builder.const_f32(3.0)));
        });
        if self.null_energy {
            let f = b(&builder, BinOp::Div, b(&builder, BinOp::Sub, Expr::LoadLocal(locals[1]),
                b(&builder, BinOp::Mul, builder.const_f32(2.0), mass)), Expr::LoadLocal(locals[1]));
            builder.store_local(locals[4], builder.select(Expr::LoadLocal(converted),
                b(&builder, BinOp::Div, Expr::LoadLocal(locals[4]), f), Expr::LoadLocal(locals[4])));
        }
        for (index, id) in locals.into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, base.clone(), builder.const_u32(index as u32)), Expr::LoadLocal(id));
        }
        let program = Program { version: 1, entry: 0, functions: vec![builder.finish(Ty::f32(), Expr::LoadLocal(locals[8]))] };
        super::validate_program(&program).map_err(|error| error.to_string())?;
        Ok(program)
    }
}

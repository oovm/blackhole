//! F64 笛卡尔有界追踪，固定最大步长并限制近视界空间位移。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};
use super::cartesian_f64::{rk4, snapshot};

fn saved(builder: &mut StructuredFunctionBuilder, value: Expr, ty: Ty) -> Expr {
    Expr::LoadLocal(builder.let_value(ty, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    fn execute(program: &Program, input: &[f64; 29], uniforms: [f64; 6]) -> [f64; 29] {
        let mut cpu = CpuInterpreter::new();
        for (slot, value) in uniforms.into_iter().enumerate() { cpu.with_uniform(slot as u32, Value::F64(value)); }
        cpu.with_buffer(1, input.iter().copied().map(Value::F64).collect());
        cpu.with_buffer(2, vec![Value::F64(-99.0); 29]);
        cpu.run_parallel(program, 1).unwrap();
        cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>().try_into().unwrap()
    }

    fn radial(radius: f64, velocity: f64) -> [f64; 29] {
        let mut record = [0.0; 29];
        record[3] = radius; record[4] = 1.0; record[7] = velocity; record[9] = 0.04; record[19] = 1.0;
        record
    }

    #[test]
    fn f64_trace_localizes_radial_events_and_preserves_terminal_chunks() {
        let program = SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(4).build().unwrap();
        for (initial, uniforms, status, expected_radius) in [
            (radial(10.0, -1.0), [1.0, 0.04, 9.98, 80.0, 10.1, 20.0], 1.0, 9.98),
            (radial(79.99, 1.0), [1.0, 0.04, 2.0002, 80.0, 6.0, 20.0], 2.0, 80.0),
        ] {
            let result = execute(&program, &initial, uniforms);
            assert_eq!(result[8], status);
            assert_eq!(result[10], 1.0);
            assert!((result[3] - expected_radius).abs() <= 1.0e-11);
            assert!((result[12] - (initial[3] - expected_radius).abs()).abs() <= 1.0e-11);
            assert!(result[16] < 1.0e-14);
            assert_eq!(execute(&program, &result, uniforms), result);
        }
    }

    #[test]
    fn f64_trace_keeps_budget_distinct_and_rejects_invalid_or_unlocalized_steps() {
        assert!(SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(0).build().is_err());
        assert!(SchwarzschildCartesianF64TraceBuilder::new().maximum_bisections(65).build().is_err());
        let program = SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(1).build().unwrap();
        let initial = radial(10.0, -1.0);
        let uniforms = [1.0, 0.04, 2.0002, 80.0, 6.0, 20.0];
        let budget = execute(&program, &initial, uniforms);
        assert_eq!(budget[8], 0.0); assert_eq!(budget[10], 1.0);
        let mut invalid = initial; invalid[4] = 2.0;
        let result = execute(&program, &invalid, uniforms);
        assert_eq!(result[8], 5.0); assert_eq!(&result[..8], &invalid[..8]); assert_eq!(result[10], 0.0);
        let limited = SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(1).maximum_bisections(1).build().unwrap();
        let failure = execute(&limited, &initial, [1.0, 0.04, 9.977, 80.0, 10.1, 20.0]);
        assert_eq!(failure[8], 4.0); assert_eq!(&failure[..8], &initial[..8]); assert_eq!(failure[10], 0.0);
        let program = SchwarzschildCartesianF64TraceBuilder::new().maximum_steps(256).build().unwrap();
        let capture = execute(&program, &radial(2.001, -1.0), uniforms);
        assert_eq!(capture[8], 1.0); assert!(capture.iter().all(|value| value.is_finite()));
        assert!((capture[3] - 2.0002).abs() < 1.0e-11);
        let expected_time = 2.001 - capture[3] + 2.0 * ((2.001 - 2.0) / (capture[3] - 2.0)).ln();
        assert!((capture[0] - expected_time).abs() < 1.0e-8, "radial coordinate time error={}", (capture[0] - expected_time).abs());
    }
}

fn both(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.select(left, right, Expr::ConstBool(false))
}

fn squared(builder: &StructuredFunctionBuilder, values: &[Expr]) -> Expr {
    values.iter().fold(builder.const_f64(0.0), |sum, value|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, value.clone(), value.clone())))
}

fn radius(builder: &StructuredFunctionBuilder, state: &[Expr; 8]) -> Expr {
    builder.unary(UnOp::Sqrt, squared(builder, &state[1..4]))
}

fn angular_squared(builder: &StructuredFunctionBuilder, state: &[Expr; 8]) -> Expr {
    let angular: [Expr; 3] = std::array::from_fn(|index| {
        let next = (index + 1) % 3;
        let last = (index + 2) % 3;
        b(builder, BinOp::Sub, b(builder, BinOp::Mul, state[next + 1].clone(), state[last + 5].clone()),
            b(builder, BinOp::Mul, state[last + 1].clone(), state[next + 5].clone()))
    });
    squared(builder, &angular)
}

fn radial_dot(builder: &StructuredFunctionBuilder, state: &[Expr; 8]) -> Expr {
    (0..3).fold(builder.const_f64(0.0), |sum, index|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, state[index + 1].clone(), state[index + 5].clone())))
}

fn null_residual(builder: &StructuredFunctionBuilder, state: &[Expr; 8], mass: &Expr) -> Expr {
    let radius = radius(builder, state);
    let radius_squared = b(builder, BinOp::Mul, radius.clone(), radius.clone());
    let radial = b(builder, BinOp::Div, radial_dot(builder, state), radius.clone());
    let lapse = b(builder, BinOp::Div, b(builder, BinOp::Sub, radius.clone(),
        b(builder, BinOp::Mul, builder.const_f64(2.0), mass.clone())), radius);
    let spatial = b(builder, BinOp::Add, b(builder, BinOp::Mul, radial.clone(), radial),
        b(builder, BinOp::Div, b(builder, BinOp::Mul, lapse, angular_squared(builder, state)), radius_squared));
    let temporal = b(builder, BinOp::Mul, state[4].clone(), state[4].clone());
    b(builder, BinOp::Div, builder.unary(UnOp::Abs, b(builder, BinOp::Sub, spatial.clone(), temporal.clone())),
        b(builder, BinOp::Add, builder.unary(UnOp::Abs, spatial), builder.unary(UnOp::Abs, temporal)))
}

fn surface(builder: &StructuredFunctionBuilder, state: &[Expr; 8], boundary: Option<&Expr>) -> Expr {
    match boundary { Some(value) => b(builder, BinOp::Sub, radius(builder, state), value.clone()), None => state[3].clone() }
}

fn locate(builder: &mut StructuredFunctionBuilder, start: &[Expr; 8], end: &[Expr; 8],
    mass: &Expr, angular: &Expr, step: &Expr, boundary: Option<&Expr>, crossing: Expr,
    tolerance: f64, budget: u32,
) -> ([Expr; 8], Expr, Expr) {
    let before = surface(builder, start, boundary);
    let before = saved(builder, before, Ty::f64());
    let lower = builder.let_value(Ty::f64(), builder.const_f64(0.0));
    let upper = builder.let_value(Ty::f64(), builder.const_f64(1.0));
    let fraction = builder.let_value(Ty::f64(), builder.const_f64(1.0));
    let endpoint_met = b(builder, BinOp::Le, builder.unary(UnOp::Abs, surface(builder, end, boundary)), builder.const_f64(tolerance));
    let located = builder.let_value(Ty::Scalar(ScalarTy::Bool), endpoint_met);
    let usable = builder.let_value(Ty::Scalar(ScalarTy::Bool), Expr::ConstBool(true));
    let event_state = end.clone().map(|value| builder.let_value(Ty::f64(), value));
    let running = both(builder, crossing.clone(), both(builder, Expr::LoadLocal(usable),
        b(builder, BinOp::Eq, Expr::LoadLocal(located), Expr::ConstBool(false))));
    builder.bounded_while(running, budget, |builder| {
        let midpoint = b(builder, BinOp::Mul, builder.const_f64(0.5),
            b(builder, BinOp::Add, Expr::LoadLocal(lower), Expr::LoadLocal(upper)));
        let midpoint = saved(builder, midpoint, Ty::f64());
        let trial_step = b(builder, BinOp::Mul, step.clone(), midpoint.clone());
        let (trial, valid) = rk4(builder, start, mass, angular, &trial_step);
        let valid = saved(builder, valid, Ty::Scalar(ScalarTy::Bool));
        let value = surface(builder, &trial, boundary);
        let value = saved(builder, value, Ty::f64());
        builder.store_local(usable, valid.clone());
        builder.store_local(located, both(builder, valid,
            b(builder, BinOp::Le, builder.unary(UnOp::Abs, value.clone()), builder.const_f64(tolerance))));
        builder.store_local(fraction, midpoint.clone());
        for index in 0..8 { builder.store_local(event_state[index], trial[index].clone()); }
        let same_side = builder.select(b(builder, BinOp::Gt, before.clone(), builder.const_f64(0.0)),
            b(builder, BinOp::Gt, value.clone(), builder.const_f64(0.0)),
            b(builder, BinOp::Lt, value, builder.const_f64(0.0)));
        let same_side = saved(builder, same_side, Ty::Scalar(ScalarTy::Bool));
        builder.store_local(lower, builder.select(same_side.clone(), midpoint.clone(), Expr::LoadLocal(lower)));
        builder.store_local(upper, builder.select(same_side, Expr::LoadLocal(upper), midpoint));
    });
    let success = both(builder, crossing, both(builder, Expr::LoadLocal(usable), Expr::LoadLocal(located)));
    let success = saved(builder, success, Ty::Scalar(ScalarTy::Bool));
    (event_state.map(Expr::LoadLocal), Expr::LoadLocal(fraction), success)
}

fn observe(builder: &mut StructuredFunctionBuilder, record: &[u32; 29], state: &[Expr; 8], mass: &Expr, accept: Expr) {
    let null = null_residual(builder, state, mass);
    let energy = builder.unary(UnOp::Abs, b(builder, BinOp::Sub,
        b(builder, BinOp::Div, state[4].clone(), Expr::LoadLocal(record[19])), builder.const_f64(1.0)));
    let scale = b(builder, BinOp::Mul, mass.clone(), Expr::LoadLocal(record[19]));
    let scale = b(builder, BinOp::Max, Expr::LoadLocal(record[20]), b(builder, BinOp::Mul, scale.clone(), scale));
    let angular = b(builder, BinOp::Div, builder.unary(UnOp::Abs,
        b(builder, BinOp::Sub, angular_squared(builder, state), Expr::LoadLocal(record[20]))), scale);
    for (index, value) in [(16, null), (17, energy), (18, angular)] {
        builder.store_local(record[index], builder.select(accept.clone(),
            b(builder, BinOp::Max, Expr::LoadLocal(record[index]), value), Expr::LoadLocal(record[index])));
    }
}

/// F64 二十九标量布局与 F32 Cartesian trace 相同，绝不能按 F32 字节布局绑定。
/// status 0 运行中、1 捕获、2 逃逸、3 积分失败、4 定位失败、5 非法输入。
/// uniform 0..5 为 M、最大仿射步长、捕获半径、逃逸半径、盘内缘、盘外缘。
/// buffer 1 输入、buffer 2 输出。chunk 预算耗尽保持 0，由宿主报告总预算耗尽。
/// 此版本不自适应估计局部误差，只限近视界位移；精度须通过步长收敛证明。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildCartesianF64TraceBuilder {
    maximum_steps: u32,
    maximum_bisections: u32,
    surface_tolerance: f64,
}

impl Default for SchwarzschildCartesianF64TraceBuilder {
    fn default() -> Self { Self { maximum_steps: 64, maximum_bisections: 48, surface_tolerance: 1.0e-11 } }
}

impl SchwarzschildCartesianF64TraceBuilder {
    /// 默认静态 chunk 和事件定位预算。
    pub fn new() -> Self { Self::default() }
    /// 设置每实例 chunk 的积分步数。
    pub fn maximum_steps(mut self, value: u32) -> Self { self.maximum_steps = value; self }
    /// 设置每事件最大二分次数。
    pub fn maximum_bisections(mut self, value: u32) -> Self { self.maximum_bisections = value; self }
    /// 设置未投影事件面的绝对残差容差。
    pub fn surface_tolerance(mut self, value: f64) -> Self { self.surface_tolerance = value; self }
    /// 生成通用 UCF F64 程序。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_steps == 0 || self.maximum_steps > 65536 || self.maximum_bisections == 0 || self.maximum_bisections > 64
            || !self.surface_tolerance.is_finite() || self.surface_tolerance <= 0.0 {
            return Err("invalid F64 trace budget or tolerance".into());
        }
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_cartesian_f64_trace");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(29));
        let record: [u32; 29] = std::array::from_fn(|index| {
            let value = builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(index as u32)));
            builder.let_value(Ty::f64(), value)
        });
        let mass = builder.load_uniform(0);
        let maximum_step = builder.load_uniform(1);
        let capture = builder.load_uniform(2);
        let escape = builder.load_uniform(3);
        let inner = builder.load_uniform(4);
        let outer = builder.load_uniform(5);
        let initial: [Expr; 8] = std::array::from_fn(|index| Expr::LoadLocal(record[index]));
        let mut valid = Expr::ConstBool(true);
        for value in (0..6).map(|slot| builder.load_uniform(slot)).chain(record.map(Expr::LoadLocal)) {
            valid = both(&builder, valid, builder.unary(UnOp::IsFinite, value));
        }
        for value in [mass.clone(), maximum_step.clone(), Expr::LoadLocal(record[4]), Expr::LoadLocal(record[19])] {
            valid = both(&builder, valid, b(&builder, BinOp::Gt, value, builder.const_f64(0.0)));
        }
        valid = both(&builder, valid, b(&builder, BinOp::Ge, Expr::LoadLocal(record[20]), builder.const_f64(0.0)));
        valid = both(&builder, valid, b(&builder, BinOp::Gt, capture.clone(), b(&builder, BinOp::Mul, builder.const_f64(2.0), mass.clone())));
        valid = both(&builder, valid, b(&builder, BinOp::Ge, inner.clone(), b(&builder, BinOp::Mul, builder.const_f64(6.0), mass.clone())));
        valid = both(&builder, valid, b(&builder, BinOp::Gt, inner.clone(), capture.clone()));
        valid = both(&builder, valid, b(&builder, BinOp::Gt, outer.clone(), inner.clone()));
        valid = both(&builder, valid, b(&builder, BinOp::Gt, escape.clone(), outer.clone()));
        valid = both(&builder, valid, b(&builder, BinOp::Le, maximum_step.clone(), b(&builder, BinOp::Mul, builder.const_f64(0.1), mass.clone())));
        valid = both(&builder, valid, b(&builder, BinOp::Gt, radius(&builder, &initial), b(&builder, BinOp::Mul, builder.const_f64(2.0), mass.clone())));
        valid = both(&builder, valid, b(&builder, BinOp::Le, null_residual(&builder, &initial, &mass), builder.const_f64(1.0e-6)));
        for index in [8, 10, 11, 12, 13, 14, 15, 16, 17, 18] {
            valid = both(&builder, valid, b(&builder, BinOp::Ge, Expr::LoadLocal(record[index]), builder.const_f64(0.0)));
        }
        let known_status = (0..=5).fold(Expr::ConstBool(false), |known, status|
            builder.select(b(&builder, BinOp::Eq, Expr::LoadLocal(record[8]), builder.const_f64(status as f64)), Expr::ConstBool(true), known));
        valid = both(&builder, valid, known_status);
        let valid = saved(&mut builder, valid, Ty::Scalar(ScalarTy::Bool));
        builder.store_local(record[8], builder.select(valid, Expr::LoadLocal(record[8]), builder.const_f64(5.0)));
        let initial_running = b(&builder, BinOp::Eq, Expr::LoadLocal(record[8]), builder.const_f64(0.0));
        let initial_capture = b(&builder, BinOp::Le, radius(&builder, &initial), capture.clone());
        let initial_escape = both(&builder, b(&builder, BinOp::Ge, radius(&builder, &initial), escape.clone()),
            b(&builder, BinOp::Gt, radial_dot(&builder, &initial), builder.const_f64(0.0)));
        builder.store_local(record[8], builder.select(initial_running,
            builder.select(initial_capture, builder.const_f64(1.0), builder.select(initial_escape, builder.const_f64(2.0), builder.const_f64(0.0))), Expr::LoadLocal(record[8])));
        let running = b(&builder, BinOp::Eq, Expr::LoadLocal(record[8]), builder.const_f64(0.0));
        builder.bounded_while(running, self.maximum_steps, |builder| {
            let start = snapshot(builder, std::array::from_fn(|index| Expr::LoadLocal(record[index])));
            let gap = b(builder, BinOp::Sub, radius(builder, &start), b(builder, BinOp::Mul, builder.const_f64(2.0), mass.clone()));
            let speed = builder.unary(UnOp::Sqrt, squared(builder, &start[5..8]));
            let near_fraction = b(builder, BinOp::Min, builder.const_f64(0.02), b(builder, BinOp::Div, maximum_step.clone(), mass.clone()));
            let near_step = b(builder, BinOp::Div, b(builder, BinOp::Mul, near_fraction, gap),
                b(builder, BinOp::Max, speed, builder.const_f64(1.0e-12)));
            let step = b(builder, BinOp::Min, maximum_step.clone(), near_step);
            let step = saved(builder, step, Ty::f64());
            let angular = Expr::LoadLocal(record[20]);
            let (end, usable) = rk4(builder, &start, &mass, &angular, &step);
            let usable = saved(builder, usable, Ty::Scalar(ScalarTy::Bool));
            let captured = both(builder, usable.clone(), both(builder,
                b(builder, BinOp::Gt, radius(builder, &start), capture.clone()), b(builder, BinOp::Le, radius(builder, &end), capture.clone())));
            let captured = saved(builder, captured, Ty::Scalar(ScalarTy::Bool));
            let escaped = both(builder, usable.clone(), both(builder,
                both(builder, b(builder, BinOp::Lt, radius(builder, &start), escape.clone()), b(builder, BinOp::Ge, radius(builder, &end), escape.clone())),
                b(builder, BinOp::Gt, radial_dot(builder, &end), builder.const_f64(0.0))));
            let escaped = saved(builder, escaped, Ty::Scalar(ScalarTy::Bool));
            let crossing = both(builder, usable.clone(), both(builder,
                b(builder, BinOp::Gt, builder.unary(UnOp::Abs, start[3].clone()), builder.const_f64(self.surface_tolerance)),
                builder.select(b(builder, BinOp::Gt, start[3].clone(), builder.const_f64(0.0)),
                    b(builder, BinOp::Le, end[3].clone(), builder.const_f64(0.0)), b(builder, BinOp::Ge, end[3].clone(), builder.const_f64(0.0)))));
            let crossing = saved(builder, crossing, Ty::Scalar(ScalarTy::Bool));
            let (capture_state, capture_fraction, capture_ok) = locate(builder, &start, &end, &mass, &angular, &step, Some(&capture),
                captured.clone(), self.surface_tolerance, self.maximum_bisections);
            let (escape_state, escape_fraction, escape_ok) = locate(builder, &start, &end, &mass, &angular, &step, Some(&escape),
                escaped.clone(), self.surface_tolerance, self.maximum_bisections);
            let (disk_state, disk_fraction, disk_ok) = locate(builder, &start, &end, &mass, &angular, &step, None,
                crossing.clone(), self.surface_tolerance, self.maximum_bisections);
            let mut event_valid = Expr::ConstBool(true);
            for (crossing, success) in [(captured.clone(), capture_ok.clone()), (escaped.clone(), escape_ok.clone()), (crossing, disk_ok.clone())] {
                event_valid = both(builder, event_valid, builder.select(crossing, success, Expr::ConstBool(true)));
            }
            let event_valid = saved(builder, event_valid, Ty::Scalar(ScalarTy::Bool));
            let accepted = both(builder, usable.clone(), event_valid.clone());
            let accepted = saved(builder, accepted, Ty::Scalar(ScalarTy::Bool));
            let terminal_fraction = builder.select(capture_ok.clone(), capture_fraction,
                builder.select(escape_ok.clone(), escape_fraction, builder.const_f64(1.0)));
            let terminal_fraction = saved(builder, terminal_fraction, Ty::f64());
            let disk_radius = radius(builder, &disk_state);
            let disk_radius = saved(builder, disk_radius, Ty::f64());
            let hit = both(builder, accepted.clone(), both(builder, disk_ok, both(builder,
                b(builder, BinOp::Le, disk_fraction.clone(), terminal_fraction.clone()), both(builder,
                    b(builder, BinOp::Ge, disk_radius.clone(), inner.clone()), b(builder, BinOp::Le, disk_radius.clone(), outer.clone())))));
            let hit = saved(builder, hit, Ty::Scalar(ScalarTy::Bool));
            let first_hit = both(builder, hit.clone(), b(builder, BinOp::Eq, Expr::LoadLocal(record[13]), builder.const_f64(0.0)));
            let first_hit = saved(builder, first_hit, Ty::Scalar(ScalarTy::Bool));
            for index in 0..8 { builder.store_local(record[index + 21], builder.select(first_hit.clone(), disk_state[index].clone(), Expr::LoadLocal(record[index + 21]))); }
            builder.store_local(record[14], builder.select(first_hit.clone(), disk_radius, Expr::LoadLocal(record[14])));
            builder.store_local(record[15], builder.select(first_hit,
                b(builder, BinOp::Add, Expr::LoadLocal(record[12]), b(builder, BinOp::Mul, step.clone(), disk_fraction)), Expr::LoadLocal(record[15])));
            builder.store_local(record[13], builder.select(hit.clone(), b(builder, BinOp::Add, Expr::LoadLocal(record[13]), builder.const_f64(1.0)), Expr::LoadLocal(record[13])));
            let terminal: [Expr; 8] = std::array::from_fn(|index| builder.select(capture_ok.clone(), capture_state[index].clone(),
                builder.select(escape_ok.clone(), escape_state[index].clone(), end[index].clone())));
            observe(builder, &record, &end, &mass, accepted.clone());
            observe(builder, &record, &terminal, &mass, accepted.clone());
            observe(builder, &record, &disk_state, &mass, hit);
            for index in 0..8 { builder.store_local(record[index], builder.select(accepted.clone(), terminal[index].clone(), start[index].clone())); }
            builder.store_local(record[9], builder.select(accepted.clone(), step.clone(), Expr::LoadLocal(record[9])));
            builder.store_local(record[10], builder.select(accepted.clone(), b(builder, BinOp::Add, Expr::LoadLocal(record[10]), builder.const_f64(1.0)), Expr::LoadLocal(record[10])));
            builder.store_local(record[12], builder.select(accepted,
                b(builder, BinOp::Add, Expr::LoadLocal(record[12]), b(builder, BinOp::Mul, step, terminal_fraction)), Expr::LoadLocal(record[12])));
            builder.store_local(record[8], builder.select(usable, builder.select(event_valid,
                builder.select(captured, builder.const_f64(1.0), builder.select(escaped, builder.const_f64(2.0), builder.const_f64(0.0))), builder.const_f64(4.0)), builder.const_f64(3.0)));
        });
        for (index, local) in record.into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, base.clone(), builder.const_u32(index as u32)), Expr::LoadLocal(local));
        }
        ProgramBuilder::new().structured_entry(builder, Ty::f64(), Expr::LoadLocal(record[8]))
            .map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

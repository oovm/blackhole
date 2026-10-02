//! F64 固定步长首盘面交点验证前端，不作为完整捕获/逃逸追踪器。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};
use super::cartesian_f64::{rk4, snapshot};

fn saved(builder: &mut StructuredFunctionBuilder, value: Expr, ty: Ty) -> Expr {
    Expr::LoadLocal(builder.let_value(ty, value))
}

fn radius(builder: &StructuredFunctionBuilder, state: &[Expr; 8]) -> Expr {
    let squared = (1..=3).fold(builder.const_f64(0.0), |sum, index|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, state[index].clone(), state[index].clone())));
    builder.unary(UnOp::Sqrt, squared)
}

fn both(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.select(left, right, Expr::ConstBool(false))
}

/// 输入 buffer 1 每实例 `[state(8),L0_squared]`，输出 buffer 2 每实例 12 个 F64。
/// 输出 `[state(8),status,disk_radius,affine,surface_residual]`。
/// status 0 步数耗尽、1 非法输入/积分失败、2 首盘交点、3 定位预算耗尽。
/// uniform 0..3 为质量、固定步长、盘内缘、盘外缘。
/// 调用方提供满足零约束的初态与对应初始 L²，程序不投影或修复输入。
/// 仅检测端点括住的 z=0 穿越，不保证同一步内多穿越或切触。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildCartesianF64DiskBuilder {
    maximum_steps: u32,
    maximum_bisections: u32,
    surface_tolerance: f64,
}

impl Default for SchwarzschildCartesianF64DiskBuilder {
    fn default() -> Self {
        Self { maximum_steps: 4096, maximum_bisections: 48, surface_tolerance: 1.0e-11 }
    }
}

impl SchwarzschildCartesianF64DiskBuilder {
    /// 默认固定步长预算及二分精度。
    pub fn new() -> Self { Self::default() }
    /// 设置有界积分步数。
    pub fn maximum_steps(mut self, value: u32) -> Self { self.maximum_steps = value; self }
    /// 设置有界交点定位次数。
    pub fn maximum_bisections(mut self, value: u32) -> Self { self.maximum_bisections = value; self }
    /// 设置未投影交点的 z 坐标绝对误差。
    pub fn surface_tolerance(mut self, value: f64) -> Self { self.surface_tolerance = value; self }
    /// 展开为通用 F64 程序，无领域后端或回调。
    pub fn build(self) -> Result<Program, String> {
        if self.maximum_steps == 0 || self.maximum_steps > 65536
            || self.maximum_bisections == 0 || self.maximum_bisections > 64
            || !self.surface_tolerance.is_finite() || self.surface_tolerance <= 0.0 {
            return Err("invalid F64 disk localization budget or tolerance".into());
        }
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_cartesian_f64_first_disk");
        let input_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(9));
        let output_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(12));
        let initial: [Expr; 8] = std::array::from_fn(|index| builder.load_buffer(1,
            b(&builder, BinOp::Add, input_base.clone(), builder.const_u32(index as u32))));
        let state = initial.map(|value| builder.let_value(Ty::f64(), value));
        let angular = builder.load_buffer(1, b(&builder, BinOp::Add, input_base, builder.const_u32(8)));
        let angular = saved(&mut builder, angular, Ty::f64());
        let mass = builder.load_uniform(0);
        let step = builder.load_uniform(1);
        let inner = builder.load_uniform(2);
        let outer = builder.load_uniform(3);
        let mut valid = b(&builder, BinOp::Ge, angular.clone(), builder.const_f64(0.0));
        for value in [&mass, &step, &inner, &outer, &angular] {
            valid = both(&builder, valid, builder.unary(UnOp::IsFinite, value.clone()));
        }
        for value in [&mass, &step] {
            valid = both(&builder, valid, b(&builder, BinOp::Gt, value.clone(), builder.const_f64(0.0)));
        }
        valid = both(&builder, valid, b(&builder, BinOp::Ge, inner.clone(), b(&builder, BinOp::Mul, builder.const_f64(6.0), mass.clone())));
        valid = both(&builder, valid, b(&builder, BinOp::Gt, outer.clone(), inner.clone()));
        valid = both(&builder, valid, b(&builder, BinOp::Gt, Expr::LoadLocal(state[4]), builder.const_f64(0.0)));
        let status = builder.let_value(Ty::f64(), builder.select(valid, builder.const_f64(0.0), builder.const_f64(1.0)));
        let affine = builder.let_value(Ty::f64(), builder.const_f64(0.0));
        let disk_radius = builder.let_value(Ty::f64(), builder.const_f64(0.0));
        let residual = builder.let_value(Ty::f64(), builder.const_f64(0.0));
        let running = b(&builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f64(0.0));
        builder.bounded_while(running, self.maximum_steps, |builder| {
            let start = snapshot(builder, state.map(Expr::LoadLocal));
            let (end, valid) = rk4(builder, &start, &mass, &angular, &step);
            let valid = saved(builder, valid, Ty::Scalar(ScalarTy::Bool));
            let crossing = both(builder, valid.clone(), both(builder,
                b(builder, BinOp::Ne, start[3].clone(), builder.const_f64(0.0)),
                builder.select(b(builder, BinOp::Gt, start[3].clone(), builder.const_f64(0.0)),
                    b(builder, BinOp::Le, end[3].clone(), builder.const_f64(0.0)),
                    b(builder, BinOp::Ge, end[3].clone(), builder.const_f64(0.0)))));
            let crossing = saved(builder, crossing, Ty::Scalar(ScalarTy::Bool));
            let lower = builder.let_value(Ty::f64(), builder.const_f64(0.0));
            let upper = builder.let_value(Ty::f64(), builder.const_f64(1.0));
            let fraction = builder.let_value(Ty::f64(), builder.const_f64(1.0));
            let located = builder.let_value(Ty::Scalar(ScalarTy::Bool), Expr::ConstBool(false));
            let usable = builder.let_value(Ty::Scalar(ScalarTy::Bool), Expr::ConstBool(true));
            let event_state = end.clone().map(|value| builder.let_value(Ty::f64(), value));
            let searching = both(builder, crossing.clone(), both(builder, Expr::LoadLocal(usable),
                b(builder, BinOp::Eq, Expr::LoadLocal(located), Expr::ConstBool(false))));
            builder.bounded_while(searching, self.maximum_bisections, |builder| {
                let midpoint = b(builder, BinOp::Mul, builder.const_f64(0.5),
                    b(builder, BinOp::Add, Expr::LoadLocal(lower), Expr::LoadLocal(upper)));
                let midpoint = saved(builder, midpoint, Ty::f64());
                let trial_step = b(builder, BinOp::Mul, step.clone(), midpoint.clone());
                let (trial, trial_valid) = rk4(builder, &start, &mass, &angular, &trial_step);
                let trial_valid = saved(builder, trial_valid, Ty::Scalar(ScalarTy::Bool));
                builder.store_local(usable, trial_valid.clone());
                let tolerance_met = both(builder, trial_valid,
                    b(builder, BinOp::Le, builder.unary(UnOp::Abs, trial[3].clone()), builder.const_f64(self.surface_tolerance)));
                builder.store_local(located, tolerance_met);
                builder.store_local(fraction, midpoint.clone());
                for index in 0..8 { builder.store_local(event_state[index], trial[index].clone()); }
                let same_side = builder.select(b(builder, BinOp::Gt, start[3].clone(), builder.const_f64(0.0)),
                    b(builder, BinOp::Gt, trial[3].clone(), builder.const_f64(0.0)),
                    b(builder, BinOp::Lt, trial[3].clone(), builder.const_f64(0.0)));
                let same_side = saved(builder, same_side, Ty::Scalar(ScalarTy::Bool));
                builder.store_local(lower, builder.select(same_side.clone(), midpoint.clone(), Expr::LoadLocal(lower)));
                builder.store_local(upper, builder.select(same_side, Expr::LoadLocal(upper), midpoint));
            });
            let localized = both(builder, crossing.clone(), both(builder, Expr::LoadLocal(usable), Expr::LoadLocal(located)));
            let localized = saved(builder, localized, Ty::Scalar(ScalarTy::Bool));
            let event_radius = radius(builder, &event_state.map(Expr::LoadLocal));
            let event_radius = saved(builder, event_radius, Ty::f64());
            let hit = both(builder, localized.clone(), both(builder,
                b(builder, BinOp::Ge, event_radius.clone(), inner.clone()),
                b(builder, BinOp::Le, event_radius.clone(), outer.clone())));
            let hit = saved(builder, hit, Ty::Scalar(ScalarTy::Bool));
            let failure = both(builder, crossing, b(builder, BinOp::Eq, localized, Expr::ConstBool(false)));
            let failure = saved(builder, failure, Ty::Scalar(ScalarTy::Bool));
            let accepted = both(builder, valid.clone(), b(builder, BinOp::Eq, failure.clone(), Expr::ConstBool(false)));
            let accepted = saved(builder, accepted, Ty::Scalar(ScalarTy::Bool));
            for index in 0..8 {
                let candidate = builder.select(hit.clone(), Expr::LoadLocal(event_state[index]), end[index].clone());
                builder.store_local(state[index], builder.select(accepted.clone(), candidate, start[index].clone()));
            }
            let advance = b(builder, BinOp::Mul, step.clone(), builder.select(hit.clone(), Expr::LoadLocal(fraction), builder.const_f64(1.0)));
            builder.store_local(affine, builder.select(accepted,
                b(builder, BinOp::Add, Expr::LoadLocal(affine), advance), Expr::LoadLocal(affine)));
            builder.store_local(disk_radius, builder.select(hit.clone(), event_radius, Expr::LoadLocal(disk_radius)));
            builder.store_local(residual, builder.select(hit.clone(), builder.unary(UnOp::Abs, Expr::LoadLocal(event_state[3])), Expr::LoadLocal(residual)));
            builder.store_local(status, builder.select(valid,
                builder.select(failure, builder.const_f64(3.0), builder.select(hit, builder.const_f64(2.0), builder.const_f64(0.0))),
                builder.const_f64(1.0)));
        });
        let output = [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], status, disk_radius, affine, residual];
        for (index, local) in output.into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)), Expr::LoadLocal(local));
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
    fn f64_disk_builder_localizes_crossing_and_preserves_failures() {
        assert!(SchwarzschildCartesianF64DiskBuilder::new().maximum_bisections(0).build().is_err());
        assert!(SchwarzschildCartesianF64DiskBuilder::new().surface_tolerance(f64::NAN).build().is_err());
        for (maximum_steps, bisections, expected_status) in [(64, 48, 2.0), (1, 48, 0.0), (64, 1, 3.0)] {
            let program = SchwarzschildCartesianF64DiskBuilder::new().maximum_steps(maximum_steps)
                .maximum_bisections(bisections).build().unwrap();
            let mut cpu = CpuInterpreter::new();
            for (slot, value) in [1.0, 0.3, 6.0, 20.0].into_iter().enumerate() { cpu.with_uniform(slot as u32, Value::F64(value)); }
            let energy = (1.0 - 200.0 / (101.0 * 101.0f64.sqrt())).sqrt();
            cpu.with_buffer(1, [0.0, 10.0, 0.0, 1.0, energy, 0.0, 0.0, -1.0, 100.0].into_iter().map(Value::F64).collect());
            cpu.with_buffer(2, vec![Value::F64(0.0); 12]);
            cpu.run_parallel(&program, 1).unwrap();
            let output: Vec<_> = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect();
            assert_eq!(output[8], expected_status);
            if expected_status == 2.0 {
                assert!((6.0..20.0).contains(&output[9]));
                assert!((0.9..1.1).contains(&output[10]));
                assert!(output[11] <= 1.0e-11);
            } else if expected_status == 3.0 {
                assert!(output[3] > 0.0);
                assert!((output[10] - 0.9).abs() < 1.0e-12);
                assert_eq!(output[9], 0.0);
            }
        }
    }

    #[test]
    fn f64_disk_batch_keeps_instance_status_and_resource_stride_independent() {
        let program = SchwarzschildCartesianF64DiskBuilder::new().maximum_steps(64).build().unwrap();
        let energy = (1.0 - 200.0 / (101.0 * 101.0f64.sqrt())).sqrt();
        let seeds = [
            [0.0, 10.0, 0.0, 1.0, energy, 0.0, 0.0, -1.0, 100.0],
            [0.0, 10.0, 0.0, 1.0, energy, 0.0, 0.0, 1.0, 100.0],
            [0.0, 0.0, 0.0, 2.01, 1.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 10.0, 0.0, 1.0, -energy, 0.0, 0.0, -1.0, 100.0],
        ];
        let mut cpu = CpuInterpreter::new();
        for (slot, value) in [1.0, 0.3, 6.0, 20.0].into_iter().enumerate() { cpu.with_uniform(slot as u32, Value::F64(value)); }
        cpu.with_buffer(1, seeds.into_iter().flatten().map(Value::F64).collect());
        cpu.with_buffer(2, vec![Value::F64(-999.0); 48]);
        cpu.run_parallel(&program, 4).unwrap();
        let output: Vec<_> = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect();
        assert_eq!(output.chunks_exact(12).map(|record| record[8]).collect::<Vec<_>>(), [2.0, 0.0, 1.0, 1.0]);
        assert!(output[11] <= 1.0e-11);
        for index in [2, 3] { assert_eq!(&output[index * 12..index * 12 + 8], &seeds[index][..8]); }
        assert!(output.iter().all(|value| value.is_finite() && *value != -999.0));
    }
}

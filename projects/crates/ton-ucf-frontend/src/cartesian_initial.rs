//! 从通用相机射线 buffer 初始化笛卡尔零测地线状态。

use super::{b, BinOp, Expr, Program, StructuredFunctionBuilder, Ty, UnOp};

/// 输入 buffer 1 为每实例 `[origin(3), direction(3)]`，输出 buffer 2 是融合追踪的二十九标量记录。
/// uniform 0/1 为质量和初始仿射步长。状态为 `[t,x,y,z,E,vx,vy,vz]`，status 位于索引 8。
/// 有效输入 status=0，非法输入 status=5，后者不写入非有限状态或伪造有效射线。
#[derive(Debug, Clone, Copy, Default)]
pub struct SchwarzschildCartesianInitialStateBuilder;

fn norm(builder: &StructuredFunctionBuilder, values: &[Expr; 3]) -> Expr {
    let squared = values.iter().fold(builder.const_f32(0.0), |sum, value|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, value.clone(), value.clone())));
    builder.unary(UnOp::Sqrt, squared)
}

fn snapshot(builder: &mut StructuredFunctionBuilder, value: Expr) -> Expr {
    Expr::LoadLocal(builder.let_value(Ty::f32(), value))
}

impl SchwarzschildCartesianInitialStateBuilder {
    /// 构造独立初态程序，不调用 Rust 参考函数。
    pub fn new() -> Self { Self }

    /// 以通用标量表达式构造方向归一化与零约束能量，不投影积分后的状态。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_cartesian_initial_state");
        let input_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(6));
        let output_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(29));
        let input: [Expr; 6] = std::array::from_fn(|index| {
            let value = builder.load_buffer(1, b(&builder, BinOp::Add, input_base.clone(), builder.const_u32(index as u32)));
            snapshot(&mut builder, value)
        });
        let origin: [Expr; 3] = std::array::from_fn(|index| input[index].clone());
        let direction: [Expr; 3] = std::array::from_fn(|index| input[index + 3].clone());
        let mass = builder.load_uniform(0);
        let step = builder.load_uniform(1);
        let radius_expression = norm(&builder, &origin);
        let radius = snapshot(&mut builder, radius_expression);
        let length_expression = norm(&builder, &direction);
        let length = snapshot(&mut builder, length_expression);
        let velocity: [Expr; 3] = std::array::from_fn(|index| {
            let value = b(&builder, BinOp::Div, direction[index].clone(), length.clone());
            snapshot(&mut builder, value)
        });
        let dot = (0..3).fold(builder.const_f32(0.0), |sum, index|
            b(&builder, BinOp::Add, sum, b(&builder, BinOp::Mul, origin[index].clone(), velocity[index].clone())));
        let radial_expression = b(&builder, BinOp::Div, dot, radius.clone());
        let radial = snapshot(&mut builder, radial_expression);
        let tangent: [Expr; 3] = std::array::from_fn(|index| {
            let value = b(&builder, BinOp::Sub, velocity[index].clone(),
                b(&builder, BinOp::Mul, origin[index].clone(), b(&builder, BinOp::Div, radial.clone(), radius.clone())));
            snapshot(&mut builder, value)
        });
        let tangent_squared = tangent.iter().fold(builder.const_f32(0.0), |sum, value|
            b(&builder, BinOp::Add, sum, b(&builder, BinOp::Mul, value.clone(), value.clone())));
        let f = b(&builder, BinOp::Div, b(&builder, BinOp::Sub, radius.clone(),
            b(&builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())), radius.clone());
        let energy_expression = builder.unary(UnOp::Sqrt, b(&builder, BinOp::Add,
            b(&builder, BinOp::Mul, radial.clone(), radial), b(&builder, BinOp::Mul, f, tangent_squared)));
        let energy = snapshot(&mut builder, energy_expression);
        let mut valid = Expr::ConstBool(true);
        for value in input.iter().chain([&mass, &step, &radius, &length, &energy]).chain(velocity.iter()) {
            valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
        }
        for value in [&mass, &step, &length, &energy] {
            valid = builder.select(valid, b(&builder, BinOp::Gt, value.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        }
        valid = builder.select(valid, b(&builder, BinOp::Gt, radius,
            b(&builder, BinOp::Mul, builder.const_f32(2.0), mass)), Expr::ConstBool(false));
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(super::ScalarTy::Bool), valid));
        let state = [builder.const_f32(0.0), origin[0].clone(), origin[1].clone(), origin[2].clone(), energy,
            velocity[0].clone(), velocity[1].clone(), velocity[2].clone()];
        for index in 0..29 {
            let value = match index {
                0..=7 => builder.select(valid.clone(), state[index].clone(), builder.const_f32(0.0)),
                8 => builder.select(valid.clone(), builder.const_f32(0.0), builder.const_f32(5.0)),
                9 => builder.select(valid.clone(), step.clone(), builder.const_f32(0.0)),
                _ => builder.const_f32(0.0),
            };
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)), value);
        }
        let result = builder.select(valid, builder.const_f32(0.0), builder.const_f32(5.0));
        let program = Program { version: 1, entry: 0, functions: vec![builder.finish(Ty::f32(), result)] };
        super::validate_program(&program).map_err(|error| error.to_string())?;
        Ok(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    fn execute(input: [f32; 6], mass: f32, step: f32) -> Vec<f64> {
        let program = SchwarzschildCartesianInitialStateBuilder::new().build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_uniform(0, Value::F32(mass));
        cpu.with_uniform(1, Value::F32(step));
        cpu.with_buffer(1, input.into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 29]);
        cpu.run_parallel(&program, 1).unwrap();
        cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect()
    }

    #[test]
    fn cartesian_initial_program_supports_axis_and_non_unit_direction() {
        let output = execute([0.0, 0.0, 10.0, 0.0, 0.0, -3.0], 1.0, 0.08);
        assert_eq!(&output[..9], &[0.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, -1.0, 0.0]);
        assert!((output[9] - 0.08).abs() < 1.0e-8);
        assert!(output[10..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn cartesian_initial_program_reports_invalid_input_with_finite_record() {
        for (input, mass, step) in [([0.0, 0.0, 10.0, 0.0, 0.0, 0.0], 1.0, 0.08),
            ([0.0, 0.0, 2.0, 0.0, 0.0, -1.0], 1.0, 0.08),
            ([f32::NAN, 0.0, 10.0, 0.0, 0.0, -1.0], 1.0, 0.08),
            ([0.0, 0.0, 10.0, 0.0, 0.0, -1.0], f32::NAN, 0.08),
            ([0.0, 0.0, 10.0, 0.0, 0.0, -1.0], 1.0, 0.0)] {
            let output = execute(input, mass, step);
            assert_eq!(output[8], 5.0);
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output[..8].iter().all(|value| *value == 0.0));
        }
    }
}

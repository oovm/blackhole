//! 从像素坐标到零测地线初态的 F64 程序，不上传宿主计算的逐射线初态。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

fn snapshot(builder: &mut StructuredFunctionBuilder, value: Expr) -> Expr {
    Expr::LoadLocal(builder.let_value(Ty::f64(), value))
}

fn squared_length(builder: &StructuredFunctionBuilder, value: &[Expr; 3]) -> Expr {
    value.iter().fold(builder.const_f64(0.0), |sum, component|
        b(builder, BinOp::Add, sum, b(builder, BinOp::Mul, component.clone(), component.clone())))
}

/// 输入 buffer 1 每实例 `[u,v]`，u 已包含宽高比缩放。
/// uniform 0..2 位置、3..5 forward、6..8 right、9..11 up、12 tan(fov_y/2)、13 质量。
/// 输出 buffer 2 每实例 `[t,x,y,z,E,vx,vy,vz,L0_squared]`，适配 F64 积分输入。
/// 返回值 0 有效、1 非法。非法输出全零，不能伪装成有效射线。
#[derive(Debug, Clone, Copy, Default)]
pub struct SchwarzschildCartesianF64CameraBuilder;

impl SchwarzschildCartesianF64CameraBuilder {
    /// 创建相机初态构造器。
    pub fn new() -> Self { Self }

    /// 只使用通用 F64 标量算术、sqrt、比较与资源写入。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("schwarzschild_cartesian_f64_camera_initial");
        let input_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(2));
        let output_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(9));
        let coordinates: [Expr; 2] = std::array::from_fn(|index| {
            let value = builder.load_buffer(1, b(&builder, BinOp::Add, input_base.clone(), builder.const_u32(index as u32)));
            snapshot(&mut builder, value)
        });
        let origin: [Expr; 3] = std::array::from_fn(|index| builder.load_uniform(index as u32));
        let scale = builder.load_uniform(12);
        let mass = builder.load_uniform(13);
        let direction: [Expr; 3] = std::array::from_fn(|index| {
            let horizontal = b(&builder, BinOp::Mul, builder.load_uniform(6 + index as u32),
                b(&builder, BinOp::Mul, coordinates[0].clone(), scale.clone()));
            let vertical = b(&builder, BinOp::Mul, builder.load_uniform(9 + index as u32),
                b(&builder, BinOp::Mul, coordinates[1].clone(), scale.clone()));
            let value = b(&builder, BinOp::Add, b(&builder, BinOp::Add, builder.load_uniform(3 + index as u32), horizontal), vertical);
            snapshot(&mut builder, value)
        });
        let radius = builder.unary(UnOp::Sqrt, squared_length(&builder, &origin));
        let radius = snapshot(&mut builder, radius);
        let length = builder.unary(UnOp::Sqrt, squared_length(&builder, &direction));
        let length = snapshot(&mut builder, length);
        let velocity: [Expr; 3] = std::array::from_fn(|index| {
            let value = b(&builder, BinOp::Div, direction[index].clone(), length.clone());
            snapshot(&mut builder, value)
        });
        let dot = (0..3).fold(builder.const_f64(0.0), |sum, index|
            b(&builder, BinOp::Add, sum, b(&builder, BinOp::Mul, origin[index].clone(), velocity[index].clone())));
        let radial = b(&builder, BinOp::Div, dot, radius.clone());
        let radial = snapshot(&mut builder, radial);
        let tangent: [Expr; 3] = std::array::from_fn(|index| {
            let value = b(&builder, BinOp::Sub, velocity[index].clone(),
                b(&builder, BinOp::Mul, origin[index].clone(), b(&builder, BinOp::Div, radial.clone(), radius.clone())));
            snapshot(&mut builder, value)
        });
        let lapse = b(&builder, BinOp::Div, b(&builder, BinOp::Sub, radius.clone(),
            b(&builder, BinOp::Mul, builder.const_f64(2.0), mass.clone())), radius.clone());
        let energy = builder.unary(UnOp::Sqrt, b(&builder, BinOp::Add,
            b(&builder, BinOp::Mul, radial.clone(), radial), b(&builder, BinOp::Mul, lapse, squared_length(&builder, &tangent))));
        let energy = snapshot(&mut builder, energy);
        let angular: [Expr; 3] = std::array::from_fn(|index| {
            let next = (index + 1) % 3;
            let last = (index + 2) % 3;
            let value = b(&builder, BinOp::Sub, b(&builder, BinOp::Mul, origin[next].clone(), velocity[last].clone()),
                b(&builder, BinOp::Mul, origin[last].clone(), velocity[next].clone()));
            snapshot(&mut builder, value)
        });
        let angular_squared = squared_length(&builder, &angular);
        let angular_squared = snapshot(&mut builder, angular_squared);
        let mut valid = Expr::ConstBool(true);
        for value in (0..14).map(|slot| builder.load_uniform(slot)).chain(coordinates)
            .chain(velocity.clone()).chain([radius.clone(), length.clone(), energy.clone(), angular_squared.clone()]) {
            valid = builder.select(valid, builder.unary(UnOp::IsFinite, value), Expr::ConstBool(false));
        }
        for value in [mass.clone(), scale, length, energy.clone()] {
            valid = builder.select(valid, b(&builder, BinOp::Gt, value, builder.const_f64(0.0)), Expr::ConstBool(false));
        }
        valid = builder.select(valid, b(&builder, BinOp::Gt, radius,
            b(&builder, BinOp::Mul, builder.const_f64(2.0), mass)), Expr::ConstBool(false));
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(ScalarTy::Bool), valid));
        let output = [builder.const_f64(0.0), origin[0].clone(), origin[1].clone(), origin[2].clone(), energy,
            velocity[0].clone(), velocity[1].clone(), velocity[2].clone(), angular_squared];
        for (index, value) in output.into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output_base.clone(), builder.const_u32(index as u32)),
                builder.select(valid.clone(), value, builder.const_f64(0.0)));
        }
        let status = builder.select(valid, builder.const_f64(0.0), builder.const_f64(1.0));
        ProgramBuilder::new().structured_entry(builder, Ty::f64(), status)
            .map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn f64_camera_initial_rejects_invalid_inputs_without_nonfinite_output() {
        let program = SchwarzschildCartesianF64CameraBuilder::new().build().unwrap();
        for (height, scale, coordinate, status) in [(10.0, 0.3, 0.0, 0.0), (2.0, 0.3, 0.0, 1.0),
            (10.0, -0.3, 0.0, 1.0), (10.0, 0.3, f64::NAN, 1.0)] {
            let mut cpu = CpuInterpreter::new();
            for (slot, value) in [0.0, 0.0, height, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, scale, 1.0].into_iter().enumerate() {
                cpu.with_uniform(slot as u32, Value::F64(value));
            }
            cpu.with_buffer(1, vec![Value::F64(coordinate), Value::F64(0.0)]);
            cpu.with_buffer(2, vec![Value::F64(-99.0); 9]);
            let result = cpu.run_parallel(&program, 1).unwrap();
            assert_eq!(result[0].value.as_f64().unwrap(), status);
            let output: Vec<_> = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect();
            if status == 0.0 { assert_eq!(output, [0.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, -1.0, 0.0]); }
            else { assert_eq!(output, [0.0; 9]); }
        }
    }
}

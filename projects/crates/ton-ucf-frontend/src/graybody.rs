//! 由真实盘面交点驱动的有界灰体发射前端。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

/// 输入 buffer 1 为 `[radius,g,g4,status,x,y,t]`，输出为 `[T,g,exposure]`。
/// `exposure` 被灰体发射系数缩放，温度和频移保持不变。
/// uniform 0..7 为 `[inner,peak,exposure,mass,contrast,arms,pitch,radial_frequency]`。
#[derive(Debug, Clone, Copy, Default)]
pub struct GraybodyEmissionBuilder;

impl GraybodyEmissionBuilder {
    /// 创建灰体发射程序 Builder。
    pub fn new() -> Self { Self }

    /// 展开有界圆轨道平流纹理为通用 F32 程序。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("graybody_advected_emission");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(7));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        let load = |index| builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(index)));
        let radius = load(0); let shift = load(1); let g4 = load(2); let input_status = load(3);
        let x = load(4); let y = load(5); let time = load(6);
        let inner = builder.load_uniform(0); let peak = builder.load_uniform(1); let exposure = builder.load_uniform(2);
        let mass = builder.load_uniform(3); let contrast = builder.load_uniform(4); let arms = builder.load_uniform(5);
        let pitch = builder.load_uniform(6); let radial_frequency = builder.load_uniform(7);
        let mut valid = Expr::ConstBool(true);
        for value in [&inner, &peak, &exposure, &mass, &contrast, &arms, &pitch, &radial_frequency] {
            valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
        }
        let active = b(&builder, BinOp::Eq, input_status.clone(), builder.const_f32(1.0));
        let no_hit = b(&builder, BinOp::Eq, input_status.clone(), builder.const_f32(0.0));
        for value in [&radius, &shift, &g4, &x, &y, &time] {
            valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false));
        }
        valid = builder.select(valid, builder.select(active.clone(), Expr::ConstBool(true), no_hit.clone()), Expr::ConstBool(false));
        let physical_hit = builder.select(active.clone(), b(&builder, BinOp::Ge, radius.clone(), inner.clone()), Expr::ConstBool(true));
        let physical_hit = builder.select(physical_hit, builder.select(active.clone(), b(&builder, BinOp::Gt, shift.clone(), builder.const_f32(0.0)), Expr::ConstBool(true)), Expr::ConstBool(false));
        valid = builder.select(valid, physical_hit, Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Gt, mass.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Ge, contrast.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        valid = builder.select(valid, b(&builder, BinOp::Le, contrast.clone(), builder.const_f32(1.0)), Expr::ConstBool(false));
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(ScalarTy::Bool), valid));
        let safe_radius = Expr::LoadLocal(builder.let_value(Ty::f32(), builder.select(valid.clone(), radius.clone(), inner.clone())));
        let ratio = b(&builder, BinOp::Div, inner.clone(), safe_radius.clone());
        let torque = b(&builder, BinOp::Div, b(&builder, BinOp::Sub, safe_radius.clone(), inner.clone()),
            b(&builder, BinOp::Add, safe_radius.clone(), builder.unary(UnOp::Sqrt, b(&builder, BinOp::Mul, safe_radius.clone(), inner.clone()))));
        let shape = b(&builder, BinOp::Mul, b(&builder, BinOp::Mul, b(&builder, BinOp::Mul, ratio.clone(), ratio.clone()), ratio), torque);
        let fraction = b(&builder, BinOp::Max, b(&builder, BinOp::Div, shape, builder.const_f32(((36.0f64 / 49.0).powi(3) / 7.0) as f32)), builder.const_f32(0.0));
        let temperature = Expr::LoadLocal(builder.let_value(Ty::f32(), b(&builder, BinOp::Mul, peak, builder.unary(UnOp::Sqrt, builder.unary(UnOp::Sqrt, fraction)))));

        let radius_dimless = b(&builder, BinOp::Div, safe_radius.clone(), mass.clone());
        let log_radius = builder.unary(UnOp::Ln, b(&builder, BinOp::Div, radius_dimless.clone(), b(&builder, BinOp::Div, inner.clone(), mass.clone())));
        let orbit_phase = b(&builder, BinOp::Div, b(&builder, BinOp::Div, time, mass.clone()),
            builder.unary(UnOp::Sqrt, b(&builder, BinOp::Mul, b(&builder, BinOp::Mul, radius_dimless.clone(), radius_dimless.clone()), radius_dimless)));
        let epsilon = builder.const_f32(1.0e-4);
        let x_dimensionless = b(&builder, BinOp::Div, x, mass.clone());
        let x_safe = b(&builder, BinOp::Max, builder.unary(UnOp::Abs, x_dimensionless.clone()), epsilon);
        let tangent = b(&builder, BinOp::Div, b(&builder, BinOp::Div, y, mass), x_safe.clone());
        let mut angle = builder.unary(UnOp::Atan, tangent);
        let x_negative = b(&builder, BinOp::Lt, x_dimensionless, builder.const_f32(0.0));
        angle = builder.select(x_negative, b(&builder, BinOp::Add, angle.clone(), builder.const_f32(std::f32::consts::PI)), angle);
        let phase = b(&builder, BinOp::Add, b(&builder, BinOp::Mul, arms, b(&builder, BinOp::Sub, angle, orbit_phase)),
            b(&builder, BinOp::Mul, pitch, log_radius.clone()));
        let spiral = b(&builder, BinOp::Mul, builder.const_f32(0.5), b(&builder, BinOp::Add, builder.const_f32(1.0), builder.unary(UnOp::Cos, phase.clone())));
        let rings = b(&builder, BinOp::Mul, builder.const_f32(0.5), b(&builder, BinOp::Add, builder.const_f32(1.0), builder.unary(UnOp::Cos,
            b(&builder, BinOp::Add, b(&builder, BinOp::Mul, radial_frequency.clone(), log_radius.clone()), b(&builder, BinOp::Mul, builder.const_f32(2.0), builder.unary(UnOp::Sin, phase.clone()))))));
        let fine = b(&builder, BinOp::Mul, builder.const_f32(0.5), b(&builder, BinOp::Add, builder.const_f32(1.0), builder.unary(UnOp::Cos,
            b(&builder, BinOp::Sub, b(&builder, BinOp::Mul, builder.const_f32(2.73), b(&builder, BinOp::Mul, radial_frequency, log_radius.clone())),
                b(&builder, BinOp::Mul, builder.const_f32(3.0), builder.unary(UnOp::Cos, phase))))));
        let signal = b(&builder, BinOp::Add, b(&builder, BinOp::Add, b(&builder, BinOp::Mul, builder.const_f32(0.5), spiral),
            b(&builder, BinOp::Mul, builder.const_f32(0.3), rings)), b(&builder, BinOp::Mul, builder.const_f32(0.2), fine));
        let emissivity = b(&builder, BinOp::Add, b(&builder, BinOp::Sub, builder.const_f32(1.0), contrast.clone()),
            b(&builder, BinOp::Mul, contrast, b(&builder, BinOp::Mul, signal.clone(), signal)));
        let emission_exposure = builder.select(active.clone(), b(&builder, BinOp::Mul, exposure.clone(), emissivity), exposure);
        for (index, value) in [builder.select(active.clone(), temperature, builder.const_f32(0.0)),
            builder.select(active, shift, builder.const_f32(1.0)),
            builder.select(valid.clone(), emission_exposure, builder.const_f32(0.0))].into_iter().enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(index as u32)), value);
        }
        let status = builder.select(valid.clone(), builder.const_f32(1.0), builder.const_f32(0.0));
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn graybody_program_matches_zero_contrast_and_changes_only_exposure() {
        let program = GraybodyEmissionBuilder::new().build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_uniform(0, Value::F32(6.0)); cpu.with_uniform(1, Value::F32(6500.0)); cpu.with_uniform(2, Value::F32(1.0e-2));
        cpu.with_uniform(3, Value::F32(1.0)); cpu.with_uniform(4, Value::F32(0.0)); cpu.with_uniform(5, Value::F32(3.0));
        cpu.with_uniform(6, Value::F32(14.0)); cpu.with_uniform(7, Value::F32(48.0));
        cpu.with_buffer(1, vec![Value::F32(8.0), Value::F32(0.8), Value::F32(0.4096), Value::F32(1.0), Value::F32(8.0), Value::F32(0.0), Value::F32(-100.0)]);
        cpu.with_buffer(2, vec![Value::F32(0.0); 3]);
        cpu.run_parallel(&program, 1).unwrap();
        let output = cpu.buffer(2).unwrap();
        let ratio: f64 = 6.0 / 8.0;
        let fraction = ratio.powi(3) * (8.0f64 - 6.0) / (8.0 + (8.0f64 * 6.0).sqrt()) / ((36.0f64 / 49.0).powi(3) / 7.0);
        let expected = 6500.0 * fraction.max(0.0).sqrt().sqrt();
        assert!((output[0].as_f64().unwrap() - expected).abs() < 0.5);
        assert!((output[1].as_f64().unwrap() - 0.8).abs() < 1.0e-6);
        assert!((output[2].as_f64().unwrap() - 1.0e-2).abs() < 1.0e-8);
        cpu.with_uniform(4, Value::F32(0.8));
        cpu.run_parallel(&program, 1).unwrap();
        let patterned = cpu.buffer(2).unwrap()[2].as_f64().unwrap();
        assert!(patterned >= 2.0e-3 && patterned <= 1.0e-2);
        assert_ne!(patterned, 1.0e-2);
    }

    #[test]
    fn graybody_program_rejects_bad_status_without_success_output() {
        let program = GraybodyEmissionBuilder::new().build().unwrap();
        let mut cpu = CpuInterpreter::new();
        for (slot, value) in [6.0, 6500.0, 1.0e-2, 1.0, 0.5, 3.0, 14.0, 48.0].into_iter().enumerate() {
            cpu.with_uniform(slot as u32, Value::F32(value));
        }
        cpu.with_buffer(1, vec![Value::F32(8.0), Value::F32(0.8), Value::F32(0.4096), Value::F32(2.0), Value::F32(8.0), Value::F32(0.0), Value::F32(-100.0)]);
        cpu.with_buffer(2, vec![Value::F32(9.0); 3]); cpu.run_parallel(&program, 1).unwrap();
        assert_eq!(cpu.buffer(2).unwrap(), &[Value::F32(0.0), Value::F32(1.0), Value::F32(0.0)]);
    }

    #[test]
    fn graybody_program_keeps_no_hit_row_finite() {
        let program = GraybodyEmissionBuilder::new().build().unwrap();
        let mut cpu = CpuInterpreter::new();
        for (slot, value) in [6.0, 6500.0, 1.0e-2, 1.0, 0.5, 3.0, 14.0, 48.0].into_iter().enumerate() {
            cpu.with_uniform(slot as u32, Value::F32(value));
        }
        cpu.with_buffer(1, vec![Value::F32(6.0), Value::F32(1.0), Value::F32(1.0), Value::F32(0.0), Value::F32(6.0), Value::F32(0.0), Value::F32(0.0)]);
        cpu.with_buffer(2, vec![Value::F32(9.0); 3]);
        cpu.run_parallel(&program, 1).unwrap();
        let output = cpu.buffer(2).unwrap();
        assert!(output.iter().all(|value| value.as_f64().unwrap().is_finite()));
        assert_eq!(output, &[Value::F32(0.0), Value::F32(1.0), Value::F32(1.0e-2)]);
    }
}

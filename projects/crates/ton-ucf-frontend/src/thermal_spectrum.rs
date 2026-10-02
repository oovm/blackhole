//! Planck 谱的通用算术展开，光谱与显示 RGB 分离。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

pub(super) fn intensity(builder: &mut StructuredFunctionBuilder, temperature: Expr, frequency: Expr) -> Expr {
    let exponent = b(builder, BinOp::Mul, builder.const_f32(47.9924307337), b(builder, BinOp::Div, frequency.clone(), temperature));
    let exponent = Expr::LoadLocal(builder.let_value(Ty::f32(), exponent));
    let safe = b(builder, BinOp::Min, exponent.clone(), builder.const_f32(80.0));
    let polynomial = b(builder, BinOp::Mul, safe.clone(), b(builder, BinOp::Add, builder.const_f32(1.0),
        b(builder, BinOp::Mul, safe.clone(), b(builder, BinOp::Add, builder.const_f32(0.5),
            b(builder, BinOp::Mul, safe.clone(), b(builder, BinOp::Add, builder.const_f32(1.0 / 6.0),
                b(builder, BinOp::Mul, safe.clone(), builder.const_f32(1.0 / 24.0))))))));
    let exponential = b(builder, BinOp::Sub, b(builder, BinOp::Pow, builder.const_f32(std::f32::consts::E), safe), builder.const_f32(1.0));
    let denominator = builder.select(b(builder, BinOp::Lt, exponent.clone(), builder.const_f32(0.01)), polynomial, exponential);
    let numerator = b(builder, BinOp::Mul, b(builder, BinOp::Mul, b(builder, BinOp::Mul,
        builder.const_f32(1.47449946476e-14), frequency.clone()), frequency.clone()), frequency);
    let value = builder.select(b(builder, BinOp::Gt, exponent, builder.const_f32(80.0)), builder.const_f32(0.0), b(builder, BinOp::Div, numerator, denominator));
    Expr::LoadLocal(builder.let_value(Ty::f32(), value))
}

/// 输入 buffer 1 为 `[temperature_kelvin, observer_frequency_thz, g]`。
/// 输出 buffer 2 为 `[I_nu,status]`，I_nu 单位 W m^-2 sr^-1 Hz^-1。
/// status=1 有效，status=2 非法。函数返回值必须绑定独立资源。
#[derive(Debug, Clone, Copy, Default)]
pub struct BlackbodySpectrumBuilder;

impl BlackbodySpectrumBuilder {
    /// 构造普通 UCF 程序前端。
    pub fn new() -> Self { Self }

    /// 展开 Planck 公式。频移通过有效温度 g*T 表达，不另乘 g^3 或 g^4。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("blackbody_spectral_radiance");
        let input_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        let output_base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(2));
        let input: [Expr; 3] = std::array::from_fn(|index| builder.load_buffer(1,
            b(&builder, BinOp::Add, input_base.clone(), builder.const_u32(index as u32))));
        let input = input.map(|value| Expr::LoadLocal(builder.let_value(Ty::f32(), value)));
        let mut valid = Expr::ConstBool(true);
        for value in &input {
            let positive = builder.select(builder.unary(UnOp::IsFinite, value.clone()),
                b(&builder, BinOp::Gt, value.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
            valid = builder.select(valid, positive, Expr::ConstBool(false));
        }
        let effective = b(&builder, BinOp::Mul, input[0].clone(), input[2].clone());
        valid = builder.select(valid, builder.select(builder.unary(UnOp::IsFinite, effective.clone()),
            b(&builder, BinOp::Gt, effective.clone(), builder.const_f32(0.0)), Expr::ConstBool(false)), Expr::ConstBool(false));
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(ScalarTy::Bool), valid));
        let temperature = builder.select(valid.clone(), effective, builder.const_f32(6000.0));
        let frequency = builder.select(valid.clone(), input[1].clone(), builder.const_f32(500.0));
        let intensity = intensity(&mut builder, temperature, frequency);
        let finite = builder.select(builder.unary(UnOp::IsFinite, intensity.clone()),
            b(&builder, BinOp::Ge, intensity.clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        let valid = builder.select(valid, finite, Expr::ConstBool(false));
        let valid = Expr::LoadLocal(builder.let_value(Ty::Scalar(ScalarTy::Bool), valid));
        let status = builder.select(valid.clone(), builder.const_f32(1.0), builder.const_f32(2.0));
        builder.store_buffer(2, output_base.clone(), builder.select(valid, intensity, builder.const_f32(0.0)));
        builder.store_buffer(2, b(&builder, BinOp::Add, output_base, builder.const_u32(1)), status.clone());
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status)
            .map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn spectrum_builder_checks_physical_units_low_frequency_and_failure_status() {
        let program = BlackbodySpectrumBuilder::new().build().unwrap();
        for input in [[6000.0f32, 500.0, 1.0], [6000.0, 500.0, 0.7], [6000.0, 0.000001, 1.0],
            [1.0, 1000000.0, 1.0], [0.0, 500.0, 1.0], [6000.0, 500.0, -1.0]] {
            let mut cpu = CpuInterpreter::new();
            cpu.with_buffer(1, input.into_iter().map(Value::F32).collect());
            cpu.with_buffer(2, vec![Value::F32(0.0); 2]);
            cpu.run_parallel(&program, 1).unwrap();
            let output = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert!(output.iter().all(|value| value.is_finite()));
            if input[0] <= 0.0 || input[2] <= 0.0 { assert_eq!(output, [0.0, 2.0]); continue; }
            assert_eq!(output[1], 1.0);
            let frequency = input[1] as f64;
            let exponent = 47.99243073366221 * frequency / (input[0] as f64 * input[2] as f64);
            let expected = if exponent > 700.0 { 0.0 } else { 1.474499464762542e-14 * frequency.powi(3) / exponent.exp_m1() };
            if expected == 0.0 { assert_eq!(output[0], 0.0); }
            else { assert!((output[0] / expected - 1.0).abs() < 2.0e-5, "input={input:?}, output={output:?}, expected={expected}"); }
        }
    }
}

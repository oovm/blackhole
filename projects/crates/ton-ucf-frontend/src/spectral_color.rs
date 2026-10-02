//! 有界可见光积分前端，观察者响应作为只读资源而非 shader 源文件。

use super::{b, BinOp, Expr, Program, ProgramBuilder, ScalarTy, StructuredFunctionBuilder, Ty, UnOp};

/// buffer 1 每实例 `[T_kelvin,g,exposure]`，buffer 2 输出 `[X,Y,Z,R,G,B,status]`。
/// XYZ 是绝对光度值，RGB 为曝光后的线性 HDR，保留负分量。status 1 有效、2 非法。
/// buffer 3 是 response_table() 的公共波长响应，不能与输入输出资源混用。
#[derive(Debug, Clone, Copy)]
pub struct VisibleSpectrumColorBuilder { step_nm: u32 }

impl Default for VisibleSpectrumColorBuilder { fn default() -> Self { Self { step_nm: 5 } } }

impl VisibleSpectrumColorBuilder {
    /// 默认 81 个波长样本，以 5nm 梯形积分。
    pub fn new() -> Self { Self::default() }
    /// 设置可见光采样步长，须整除 400nm，范围 1..20nm。
    pub fn wavelength_step_nm(mut self, value: u32) -> Self { self.step_nm = value; self }
    fn validate(self) -> Result<(), String> {
        if self.step_nm == 0 || self.step_nm > 20 || 400 % self.step_nm != 0 { return Err("invalid visible wavelength sampling".into()); }
        Ok(())
    }
    /// 每行 `[frequency_THz,weight_X,weight_Y,weight_Z]`，含 B_nu→B_lambda 雅可比和 683 光度因子。
    /// 使用 CIE1931 非对称高斯近似，不宣称官方标准表精度。
    pub fn response_table(self) -> Result<Vec<f32>, String> {
        self.validate()?;
        let mut table = Vec::new();
        for wavelength in (380..=780).step_by(self.step_nm as usize) {
            let wavelength = wavelength as f64;
            let response = |center: f64, below: f64, above: f64| {
                let argument = (wavelength - center) * if wavelength < center { below } else { above };
                (-0.5 * argument * argument).exp()
            };
            let xyz = [1.056 * response(599.8, 0.0264, 0.0323) + 0.362 * response(442.0, 0.0624, 0.0374) - 0.065 * response(501.1, 0.0490, 0.0382),
                0.821 * response(568.8, 0.0213, 0.0247) + 0.286 * response(530.9, 0.0613, 0.0322),
                1.217 * response(437.0, 0.0845, 0.0278) + 0.681 * response(459.0, 0.0385, 0.0725)];
            let meters = wavelength * 1.0e-9;
            let endpoint = if wavelength == 380.0 || wavelength == 780.0 { 0.5 } else { 1.0 };
            let scale = 299792458.0 / (meters * meters) * self.step_nm as f64 * 1.0e-9 * 683.0 * endpoint;
            table.push((299792458.0 / meters * 1.0e-12) as f32);
            table.extend(xyz.map(|value| (value * scale) as f32));
        }
        Ok(table)
    }
    /// 展开为通用 F32 数学与有界积分，不调用参考渲染。
    pub fn build(self) -> Result<Program, String> {
        self.validate()?;
        let mut builder = StructuredFunctionBuilder::new("visible_blackbody_xyz_rgb");
        let base = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(7));
        let input: [Expr; 3] = std::array::from_fn(|index| {
            let value = builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(index as u32)));
            Expr::LoadLocal(builder.let_value(Ty::f32(), value))
        });
        let mut valid = Expr::ConstBool(true);
        for value in &input { valid = builder.select(valid, builder.unary(UnOp::IsFinite, value.clone()), Expr::ConstBool(false)); }
        valid = builder.select(valid, b(&builder, BinOp::Ge, input[0].clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        for index in [1, 2] { valid = builder.select(valid, b(&builder, BinOp::Gt, input[index].clone(), builder.const_f32(0.0)), Expr::ConstBool(false)); }
        let effective = b(&builder, BinOp::Mul, input[0].clone(), input[1].clone());
        valid = builder.select(valid, builder.unary(UnOp::IsFinite, effective.clone()), Expr::ConstBool(false));
        let valid = builder.let_value(Ty::Scalar(ScalarTy::Bool), valid);
        let temperature = builder.select(Expr::LoadLocal(valid), b(&builder, BinOp::Max, effective, builder.const_f32(1.0)), builder.const_f32(6000.0));
        let temperature = Expr::LoadLocal(builder.let_value(Ty::f32(), temperature));
        let xyz: [u32; 3] = std::array::from_fn(|_| builder.let_value(Ty::f32(), builder.const_f32(0.0)));
        builder.bounded_for(400 / self.step_nm + 1, |builder, iteration| {
            let row = b(builder, BinOp::Mul, Expr::LoadLocal(iteration), builder.const_u32(4));
            let frequency = builder.load_buffer(3, row.clone());
            let value = super::thermal_spectrum::intensity(builder, temperature.clone(), frequency.clone());
            let mut row_valid = builder.select(builder.unary(UnOp::IsFinite, frequency.clone()),
                b(builder, BinOp::Gt, frequency, builder.const_f32(0.0)), Expr::ConstBool(false));
            for axis in 0..3 {
                let weight = builder.load_buffer(3, b(builder, BinOp::Add, row.clone(), builder.const_u32(axis as u32 + 1)));
                row_valid = builder.select(row_valid, builder.select(builder.unary(UnOp::IsFinite, weight.clone()),
                    b(builder, BinOp::Ge, weight.clone(), builder.const_f32(0.0)), Expr::ConstBool(false)), Expr::ConstBool(false));
                builder.store_local(xyz[axis], b(builder, BinOp::Add, Expr::LoadLocal(xyz[axis]), b(builder, BinOp::Mul, value.clone(), weight)));
            }
            builder.store_local(valid, builder.select(Expr::LoadLocal(valid), row_valid, Expr::ConstBool(false)));
        });
        let rgb: [Expr; 3] = std::array::from_fn(|axis| {
            let coefficients = [[3.2406f32, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
            let value = (0..3).fold(builder.const_f32(0.0), |sum, index| b(&builder, BinOp::Add, sum,
                b(&builder, BinOp::Mul, builder.const_f32(coefficients[axis][index]), Expr::LoadLocal(xyz[index]))));
            let value = b(&builder, BinOp::Mul, value, input[2].clone());
            Expr::LoadLocal(builder.let_value(Ty::f32(), value))
        });
        for value in xyz.map(Expr::LoadLocal).into_iter().chain(rgb.clone()) {
            builder.store_local(valid, builder.select(Expr::LoadLocal(valid), builder.unary(UnOp::IsFinite, value), Expr::ConstBool(false)));
        }
        let valid = Expr::LoadLocal(valid);
        let active = builder.select(valid.clone(), b(&builder, BinOp::Gt, input[0].clone(), builder.const_f32(0.0)), Expr::ConstBool(false));
        for (index, value) in xyz.map(Expr::LoadLocal).into_iter().chain(rgb).enumerate() {
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(index as u32)),
                builder.select(active.clone(), value, builder.const_f32(0.0)));
        }
        let status = builder.select(valid, builder.const_f32(1.0), builder.const_f32(2.0));
        builder.store_buffer(2, b(&builder, BinOp::Add, output, builder.const_u32(6)), status.clone());
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};
    #[test]
    fn spectral_color_builder_preserves_hdr_and_rejects_bad_inputs() {
        let config = VisibleSpectrumColorBuilder::new();
        assert!(config.wavelength_step_nm(3).build().is_err());
        let program = config.build().unwrap();
        for input in [[6000.0, 1.0, 1.0e-8], [0.0, 1.0, 1.0e-8], [-1.0, 1.0, 1.0e-8], [6000.0, -1.0, 1.0e-8]] {
            let mut cpu = CpuInterpreter::new();
            cpu.with_buffer(1, input.into_iter().map(Value::F32).collect()); cpu.with_buffer(2, vec![Value::F32(0.0); 7]);
            cpu.with_buffer(3, config.response_table().unwrap().into_iter().map(Value::F32).collect()); cpu.run_parallel(&program, 1).unwrap();
            let result = cpu.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect::<Vec<_>>();
            assert!(result.iter().all(|value| value.is_finite()));
            assert_eq!(result[6], if input[0] < 0.0 || input[1] < 0.0 { 2.0 } else { 1.0 });
            if input[0] > 0.0 && input[1] > 0.0 { assert!(result[1] > 1.0e7 && result[3] > 1.0); }
            else { assert_eq!(&result[..6], &[0.0; 6]); }
        }
    }
}

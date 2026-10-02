//! 通用二维 F32 3×3 盒式模糊前端。

use super::{b, BinOp, Expr, Program, ProgramBuilder, StructuredFunctionBuilder, Ty};

/// 对线性 RGB 图像执行边界复制的 3×3 平均滤波。
///
/// 输入 buffer 1 和输出 buffer 2 每实例均为 `[R,G,B]`。
/// Builder 配置图像宽、高。边界采样复制最近的合法像素。
#[derive(Debug, Clone, Copy)]
pub struct UcfBoxBlur3x3Builder {
    width: u32,
    height: u32,
}

impl UcfBoxBlur3x3Builder {
    /// 创建 3×3 盒式模糊 Builder。
    pub fn new(width: u32, height: u32) -> Self { Self { width, height } }

    /// 展开通用 U32 索引算术和 F32 资源读写程序。
    pub fn build(self) -> Result<Program, String> {
        let mut builder = StructuredFunctionBuilder::new("box_blur_3x3_rgb");
        let gid = builder.global_id();
        if self.width == 0 || self.height == 0 { return Err("blur dimensions must be non-zero".into()); }
        let width = builder.const_u32(self.width);
        let height = builder.const_u32(self.height);
        let x = b(&builder, BinOp::Mod, gid.clone(), width.clone());
        let y = b(&builder, BinOp::Div, gid.clone(), width.clone());
        let output = b(&builder, BinOp::Mul, gid, builder.const_u32(3));
        let sum: [u32; 3] = std::array::from_fn(|_| builder.let_value(Ty::f32(), builder.const_f32(0.0)));
        let count = builder.let_value(Ty::f32(), builder.const_f32(0.0));
        for (row, col) in [(0u32, 0u32), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)] {
            let last_x = b(&builder, BinOp::Sub, width.clone(), builder.const_u32(1));
            let left = builder.select(
                b(&builder, BinOp::Gt, x.clone(), builder.const_u32(0)),
                b(&builder, BinOp::Sub, x.clone(), builder.const_u32(1)),
                builder.const_u32(0),
            );
            let right = builder.select(
                b(&builder, BinOp::Lt, b(&builder, BinOp::Add, x.clone(), builder.const_u32(1)), width.clone()),
                b(&builder, BinOp::Add, x.clone(), builder.const_u32(1)),
                last_x,
            );
            let nx = match col {
                0 => left,
                1 => x.clone(),
                _ => right,
            };
            let last_y = b(&builder, BinOp::Sub, height.clone(), builder.const_u32(1));
            let top = builder.select(
                b(&builder, BinOp::Gt, y.clone(), builder.const_u32(0)),
                b(&builder, BinOp::Sub, y.clone(), builder.const_u32(1)),
                builder.const_u32(0),
            );
            let bottom = builder.select(
                b(&builder, BinOp::Lt, b(&builder, BinOp::Add, y.clone(), builder.const_u32(1)), height.clone()),
                b(&builder, BinOp::Add, y.clone(), builder.const_u32(1)),
                last_y,
            );
            let ny = match row {
                0 => top,
                1 => y.clone(),
                _ => bottom,
            };
            let pixel = b(&builder, BinOp::Add, b(&builder, BinOp::Mul, ny, width.clone()), nx);
            let base = b(&builder, BinOp::Mul, pixel, builder.const_u32(3));
            for channel in 0usize..3 {
                let value = builder.load_buffer(1, b(&builder, BinOp::Add, base.clone(), builder.const_u32(channel as u32)));
                builder.store_local(sum[channel], b(&builder, BinOp::Add, Expr::LoadLocal(sum[channel]), value));
            }
            builder.store_local(count, b(&builder, BinOp::Add, Expr::LoadLocal(count), builder.const_f32(1.0)));
        }
        for channel in 0usize..3 {
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(channel as u32)),
                b(&builder, BinOp::Div, Expr::LoadLocal(sum[channel]), Expr::LoadLocal(count)));
        }
        let valid = builder.select(b(&builder, BinOp::Gt, width.clone(), builder.const_u32(0)),
            b(&builder, BinOp::Gt, height, builder.const_u32(0)), Expr::ConstBool(false));
        let status = builder.select(valid, builder.const_f32(1.0), builder.const_f32(2.0));
        ProgramBuilder::new().structured_entry(builder, Ty::f32(), status).map_err(|error| error.to_string())?.finish().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn blur_builder_matches_edge_clamped_reference() {
        let program = UcfBoxBlur3x3Builder::new(2, 1).build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_buffer(1, [1.0, 0.0, 0.0, 3.0, 0.0, 0.0].into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 6]);
        cpu.run_parallel(&program, 2).unwrap();
        let output = cpu.buffer(2).unwrap();
        assert!((output[0].as_f64().unwrap() - 1.6666667).abs() < 1.0e-5);
        assert!((output[3].as_f64().unwrap() - 2.3333333).abs() < 1.0e-5);
    }

    #[test]
    fn blur_builder_matches_reference_for_non_degenerate_image() {
        let program = UcfBoxBlur3x3Builder::new(3, 2).build().unwrap();
        let input = [
            1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
            4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0,
        ];
        let mut cpu = CpuInterpreter::new();
        cpu.with_buffer(1, input.into_iter().map(Value::F32).collect());
        cpu.with_buffer(2, vec![Value::F32(0.0); 18]);
        cpu.run_parallel(&program, 6).unwrap();
        let output = cpu.buffer(2).unwrap();
        let expected = [
            2.3333333, 3.0, 3.6666667,
            3.3333333, 4.0, 4.6666667,
        ];
        for (pixel, expected_red) in expected.into_iter().enumerate() {
            assert!((output[pixel * 3].as_f64().unwrap() - expected_red).abs() < 1.0e-5);
        }
        assert!(output.iter().skip(1).step_by(3).all(|value| value.as_f64().unwrap() == 0.0));
        assert!(output.iter().skip(2).step_by(3).all(|value| value.as_f64().unwrap() == 0.0));
    }
}

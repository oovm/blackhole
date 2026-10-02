//! 通用 RGB 图像尺度变换前端。

use super::{b, BinOp, Expr, Program, ProgramBuilder, StructuredFunctionBuilder, Ty};

fn channel_sum(builder: &StructuredFunctionBuilder, values: [Expr; 4]) -> Expr {
    let first = b(builder, BinOp::Add, values[0].clone(), values[1].clone());
    let second = b(builder, BinOp::Add, values[2].clone(), values[3].clone());
    b(builder, BinOp::Mul, b(builder, BinOp::Add, first, second), builder.const_f32(0.25))
}

/// 将 RGB 图像按 2×2 样本平均降采样。
#[derive(Debug, Clone, Copy)]
pub struct UcfDownsample2xRgbBuilder {
    source_width: u32,
    source_height: u32,
}

impl UcfDownsample2xRgbBuilder {
    /// 创建固定源尺寸的 2×2 降采样 Builder。
    pub fn new(source_width: u32, source_height: u32) -> Self {
        Self { source_width, source_height }
    }

    /// 构建通用 UCF Program。输入 buffer 1，输出 buffer 2。
    pub fn build(self) -> Result<Program, String> {
        if self.source_width < 2 || self.source_height < 2 {
            return Err("downsample source dimensions must be at least 2".into());
        }
        let destination_width = self.source_width / 2;
        let mut builder = StructuredFunctionBuilder::new("downsample_2x_rgb");
        let gid = builder.global_id();
        let source_width = builder.const_u32(self.source_width);
        let destination_width_expr = builder.const_u32(destination_width);
        let x = b(&builder, BinOp::Mod, gid.clone(), destination_width_expr.clone());
        let y = b(&builder, BinOp::Div, gid, destination_width_expr);
        let source_x = b(&builder, BinOp::Mul, x, builder.const_u32(2));
        let source_y = b(&builder, BinOp::Mul, y, builder.const_u32(2));
        let source_row = b(&builder, BinOp::Mul, source_y, source_width.clone());
        let next_row = b(&builder, BinOp::Add, source_row.clone(), source_width);
        let source_pixel = b(&builder, BinOp::Add, source_row, source_x.clone());
        let next_pixel = b(&builder, BinOp::Add, next_row, source_x);
        let output = b(&builder, BinOp::Mul, builder.global_id(), builder.const_u32(3));
        for channel in 0..3u32 {
            let offset = builder.const_u32(channel);
            let indices = [
                b(&builder, BinOp::Add, b(&builder, BinOp::Mul, source_pixel.clone(), builder.const_u32(3)), offset.clone()),
                b(&builder, BinOp::Add, b(&builder, BinOp::Mul, b(&builder, BinOp::Add, source_pixel.clone(), builder.const_u32(1)), builder.const_u32(3)), offset.clone()),
                b(&builder, BinOp::Add, b(&builder, BinOp::Mul, next_pixel.clone(), builder.const_u32(3)), offset.clone()),
                b(&builder, BinOp::Add, b(&builder, BinOp::Mul, b(&builder, BinOp::Add, next_pixel.clone(), builder.const_u32(1)), builder.const_u32(3)), offset),
            ];
            let values = indices.map(|index| builder.load_buffer(1, index));
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(channel)), channel_sum(&builder, values));
        }
        ProgramBuilder::new()
            .structured_entry(builder, Ty::f32(), Expr::ConstF32(0.0))
            .map_err(|error| error.to_string())?
            .finish()
            .map_err(|error| error.to_string())
    }

    /// 返回降采样后的宽度和高度。
    pub fn destination_size(self) -> (u32, u32) {
        (self.source_width / 2, self.source_height / 2)
    }
}

/// 将 RGB 图像按最近邻采样放大，并乘以固定权重。
#[derive(Debug, Clone, Copy)]
pub struct UcfUpsampleRgbBuilder {
    source_width: u32,
    source_height: u32,
    destination_width: u32,
    destination_height: u32,
    weight: f32,
}

impl UcfUpsampleRgbBuilder {
    /// 创建固定源尺寸、目标尺寸和权重的放大 Builder。
    pub fn new(source_width: u32, source_height: u32, destination_width: u32, destination_height: u32, weight: f32) -> Self {
        Self { source_width, source_height, destination_width, destination_height, weight }
    }

    /// 构建通用 UCF Program。输入 buffer 1，输出 buffer 2。
    pub fn build(self) -> Result<Program, String> {
        if self.source_width == 0 || self.source_height == 0 || self.destination_width == 0 || self.destination_height == 0 {
            return Err("upsample dimensions must be non-zero".into());
        }
        if !self.weight.is_finite() {
            return Err("upsample weight must be finite".into());
        }
        let mut builder = StructuredFunctionBuilder::new("upsample_rgb");
        let gid = builder.global_id();
        let destination_width = builder.const_u32(self.destination_width);
        let destination_height = builder.const_u32(self.destination_height);
        let x = b(&builder, BinOp::Mod, gid.clone(), destination_width.clone());
        let y = b(&builder, BinOp::Div, gid.clone(), destination_width.clone());
        let source_x_raw = b(&builder, BinOp::Div, b(&builder, BinOp::Mul, x, builder.const_u32(self.source_width)), destination_width);
        let source_y_raw = b(&builder, BinOp::Div, b(&builder, BinOp::Mul, y, builder.const_u32(self.source_height)), destination_height);
        let source_x = builder.select(
            b(&builder, BinOp::Lt, source_x_raw.clone(), builder.const_u32(self.source_width)),
            source_x_raw,
            builder.const_u32(self.source_width - 1),
        );
        let source_y = builder.select(
            b(&builder, BinOp::Lt, source_y_raw.clone(), builder.const_u32(self.source_height)),
            source_y_raw,
            builder.const_u32(self.source_height - 1),
        );
        let source_pixel = b(&builder, BinOp::Add, b(&builder, BinOp::Mul, source_y, builder.const_u32(self.source_width)), source_x);
        let source_base = b(&builder, BinOp::Mul, source_pixel, builder.const_u32(3));
        let output = b(&builder, BinOp::Mul, gid, builder.const_u32(3));
        for channel in 0..3u32 {
            let value = builder.load_buffer(1, b(&builder, BinOp::Add, source_base.clone(), builder.const_u32(channel)));
            builder.store_buffer(2, b(&builder, BinOp::Add, output.clone(), builder.const_u32(channel)),
                b(&builder, BinOp::Mul, value, builder.const_f32(self.weight)));
        }
        ProgramBuilder::new()
            .structured_entry(builder, Ty::f32(), Expr::ConstF32(0.0))
            .map_err(|error| error.to_string())?
            .finish()
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn downsample_builder_matches_four_pixel_average() {
        let program = UcfDownsample2xRgbBuilder::new(4, 2).build().unwrap();
        let input = (1..=24).map(|value| Value::F32(value as f32)).collect();
        let mut cpu = CpuInterpreter::new();
        cpu.with_buffer(1, input).with_buffer(2, vec![Value::F32(0.0); 6]);
        cpu.run_parallel(&program, 2).unwrap();
        let output = cpu.buffer(2).unwrap();
        assert_eq!(output[0], Value::F32(8.5));
        assert_eq!(output[1], Value::F32(9.5));
        assert_eq!(output[2], Value::F32(10.5));
        assert_eq!(output[3], Value::F32(14.5));
        assert_eq!(output[4], Value::F32(15.5));
        assert_eq!(output[5], Value::F32(16.5));
    }

    #[test]
    fn upsample_builder_matches_nearest_neighbor_and_weight() {
        let program = UcfUpsampleRgbBuilder::new(2, 1, 4, 2, 0.5).build().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_buffer(1, vec![Value::F32(2.0), Value::F32(4.0), Value::F32(6.0), Value::F32(8.0), Value::F32(10.0), Value::F32(12.0)])
            .with_buffer(2, vec![Value::F32(0.0); 24]);
        cpu.run_parallel(&program, 8).unwrap();
        let output = cpu.buffer(2).unwrap();
        assert_eq!(&output[0..3], &[Value::F32(1.0), Value::F32(2.0), Value::F32(3.0)]);
        assert_eq!(&output[9..12], &[Value::F32(4.0), Value::F32(5.0), Value::F32(6.0)]);
        assert_eq!(&output[12..15], &[Value::F32(1.0), Value::F32(2.0), Value::F32(3.0)]);
    }
}

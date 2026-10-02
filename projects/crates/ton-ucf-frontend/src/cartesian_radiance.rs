//! 从笛卡尔追踪记录展开盘面辐射程序。

use super::{BinOp, Expr, Program, ProgramBuilder, StructuredFunctionBuilder, Ty, UnOp};

fn add(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.binary(BinOp::Add, left, right)
}

fn sub(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.binary(BinOp::Sub, left, right)
}

fn mul(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.binary(BinOp::Mul, left, right)
}

fn div(builder: &StructuredFunctionBuilder, left: Expr, right: Expr) -> Expr {
    builder.binary(BinOp::Div, left, right)
}

fn load(builder: &StructuredFunctionBuilder, base: Expr, offset: u32) -> Expr {
    builder.load_buffer(1, add(builder, base, builder.const_u32(offset)))
}

/// 输入资源 `1` 每实例为笛卡尔融合追踪的 29 标量记录，输出资源 `2` 为线性 RGB。
/// 首个盘面状态位于偏移 `21..29`，频移由 `E` 与 `Lz` 推导。
/// uniform `0..3` 依次为内半径、外半径、质量和吸积率。
pub fn disk_radiance_from_cartesian_trace() -> Program {
    let mut builder = StructuredFunctionBuilder::new("disk_radiance_from_cartesian_trace");
    let global = builder.global_id();
    let input_base = mul(&builder, global.clone(), builder.const_u32(29));
    let output_base = mul(&builder, global, builder.const_u32(3));
    let inner = builder.load_uniform(0);
    let outer = builder.load_uniform(1);
    let mass = builder.load_uniform(2);
    let accretion = builder.load_uniform(3);
    let disk_count = load(&builder, input_base.clone(), 13);
    let x = load(&builder, input_base.clone(), 21);
    let y = load(&builder, input_base.clone(), 22);
    let z = load(&builder, input_base.clone(), 23);
    let energy = load(&builder, input_base.clone(), 25);
    let vx = load(&builder, input_base.clone(), 26);
    let vy = load(&builder, input_base.clone(), 27);
    let radius = builder.unary(UnOp::Sqrt, add(&builder,
        add(&builder, mul(&builder, x.clone(), x.clone()), mul(&builder, y.clone(), y.clone())),
        mul(&builder, z.clone(), z.clone())));
    let angular_z = sub(&builder, mul(&builder, x, vy), mul(&builder, y, vx));
    let safe_energy = builder.binary(BinOp::Max, energy, builder.const_f32(1.0e-6));
    let impact_z = div(&builder, angular_z, safe_energy);
    let three_mass = mul(&builder, builder.const_f32(3.0), mass.clone());
    let orbit_factor = builder.unary(UnOp::Sqrt, div(&builder, mass.clone(), radius.clone()));
    let time_denominator = builder.unary(UnOp::Sqrt, builder.binary(BinOp::Max,
        builder.const_f32(1.0e-5), sub(&builder, builder.const_f32(1.0), div(&builder, three_mass, radius.clone()))));
    let emitter_time = div(&builder, builder.const_f32(1.0), time_denominator.clone());
    let emitter_phi = div(&builder, orbit_factor, mul(&builder, radius.clone(), time_denominator));
    let frequency_denominator = sub(&builder, emitter_time, mul(&builder, impact_z, emitter_phi));
    let doppler = builder.binary(BinOp::Min, builder.const_f32(4.0), builder.binary(BinOp::Max,
        builder.const_f32(0.05), div(&builder, builder.const_f32(1.0), frequency_denominator)));
    let temperature_base = div(&builder, accretion, mul(&builder, mass.clone(), mass));
    let temperature_ratio = div(&builder, inner.clone(), radius.clone());
    let temperature = mul(&builder, mul(&builder, temperature_base,
        builder.binary(BinOp::Pow, temperature_ratio, builder.const_f32(0.75))), builder.const_f32(28.0));
    let red = builder.binary(BinOp::Max, builder.const_f32(0.0), temperature.clone());
    let green = mul(&builder, temperature.clone(), builder.const_f32(0.58));
    let blue = mul(&builder, temperature, builder.const_f32(0.18));
    let scale = mul(&builder, mul(&builder, doppler.clone(), doppler.clone()), doppler);
    let in_annulus = builder.select(builder.binary(BinOp::Ge, radius.clone(), inner),
        builder.binary(BinOp::Le, radius, outer), Expr::ConstBool(false));
    let active = builder.select(builder.binary(BinOp::Gt, disk_count, builder.const_f32(0.0)), in_annulus, Expr::ConstBool(false));
    for (offset, channel) in [(0, red), (1, green), (2, blue)] {
        builder.store_buffer(2, add(&builder, output_base.clone(), builder.const_u32(offset)),
            builder.select(active.clone(), div(&builder, mul(&builder, channel, scale.clone()), builder.const_f32(255.0)), builder.const_f32(0.0)));
    }
    ProgramBuilder::new().structured_entry(builder, Ty::f32(), scale)
        .expect("valid Cartesian radiance entry").finish()
        .expect("valid Cartesian radiance program")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    fn execute(disk_count: f32) -> Vec<f64> {
        let mut interpreter = CpuInterpreter::new();
        for (slot, value) in [6.0, 20.0, 1.0, 100.0].into_iter().enumerate() {
            interpreter.with_uniform(slot as u32, Value::F32(value));
        }
        let mut state = vec![Value::F32(0.0); 29];
        state[13] = Value::F32(disk_count);
        state[21] = Value::F32(8.0);
        state[25] = Value::F32(1.0);
        state[27] = Value::F32(0.25);
        interpreter.with_buffer(1, state);
        interpreter.with_buffer(2, vec![Value::F32(1.0); 3]);
        interpreter.run_parallel(&disk_radiance_from_cartesian_trace(), 1).unwrap();
        interpreter.buffer(2).unwrap().iter().map(|value| value.as_f64().unwrap()).collect()
    }

    #[test]
    fn cartesian_radiance_uses_disk_hit_and_produces_finite_rgb() {
        let output = execute(1.0);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn cartesian_radiance_rejects_no_disk_hit_with_black_output() {
        assert!(execute(0.0).iter().all(|value| *value == 0.0));
    }
}

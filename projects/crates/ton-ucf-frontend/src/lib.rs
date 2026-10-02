//! TON 领域前端。
//!
//! 本 crate 负责把 TON 的领域算法展开为通用 UCF Program。UCF 核心只
//! 接收 `Program`、校验并执行，不依赖黑洞类型或物理名称。

mod adaptive;
pub use adaptive::{SchwarzschildAdaptiveChunkBuilder, SchwarzschildAdaptiveStepBuilder};
mod events;
pub use events::SchwarzschildEventLocalizationBuilder;
mod adaptive_trace;
mod null_energy;
mod cartesian_null;
mod cartesian_initial;
pub use cartesian_initial::SchwarzschildCartesianInitialStateBuilder;
pub use cartesian_null::SchwarzschildCartesianAdaptiveChunkBuilder;
mod cartesian_f64;
pub use cartesian_f64::SchwarzschildCartesianF64Rk4Builder;
mod cartesian_f64_disk;
pub use cartesian_f64_disk::SchwarzschildCartesianF64DiskBuilder;
mod cartesian_f64_camera;
pub use cartesian_f64_camera::SchwarzschildCartesianF64CameraBuilder;
mod cartesian_f64_trace;
pub use cartesian_f64_trace::{SchwarzschildCartesianF64EventTraceBuilder, SchwarzschildCartesianF64TraceBuilder};
mod cartesian_radiance;
pub use cartesian_radiance::SchwarzschildDiskFrequencyBuilder;
mod thermal_spectrum;
pub use thermal_spectrum::BlackbodySpectrumBuilder;
mod spectral_color;
pub use spectral_color::VisibleSpectrumColorBuilder;
mod spectral_disk;
pub use spectral_disk::{ZeroTorqueDiskInputsBuilder, SpectralDisplayBuilder};
mod graybody;
pub use graybody::GraybodyEmissionBuilder;
mod radiative_transfer;
pub use radiative_transfer::{PackedRadiativeTransferBuilder, RadiativeTransferBuilder};
mod radiance_composite;
pub use radiance_composite::RadianceCompositeBuilder;
pub use adaptive_trace::SchwarzschildAdaptiveTraceChunkBuilder;

pub use ucf_program::{
    program_hash, validate_program, BinOp, CpuExecResult, CpuInterpreter, Expr, Function,
    FunctionBuilder, ProgramBuilder,
    StructuredFunctionBuilder,
    ParallelDim, ParallelInstance, Program, ProgramError, ScalarTy, Stmt, Ty, UnOp, ValidateError,
    Value,
};

fn uni(slot: u32) -> Expr {
    Expr::LoadUniform(slot)
}

fn c(v: f64) -> Expr {
    Expr::ConstF64(v)
}

fn bin(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
}

fn un(op: UnOp, arg: Expr) -> Expr {
    Expr::UnOp { op, arg: Box::new(arg) }
}

fn v3(x: Expr, y: Expr, z: Expr) -> Expr {
    Expr::Vec3Compose { x: Box::new(x), y: Box::new(y), z: Box::new(z) }
}

fn select(cond: Expr, then_: Expr, else_: Expr) -> Expr {
    Expr::Select { cond: Box::new(cond), then_: Box::new(then_), else_: Box::new(else_) }
}

fn f(name: &str, body: Vec<Stmt>, locals: Vec<(u32, Ty)>, ret: Ty) -> Program {
    Program {
        version: 1,
        functions: vec![Function { name: name.into(), params: Vec::new(), locals, body, ret }],
        entry: 0,
    }
}

/// 构建向量归一化程序。
pub fn vector_math_normalize() -> Program {
    let mut builder = FunctionBuilder::new("vector_math_normalize");
    let value = builder.emit_load_uniform(0, Ty::vec3()).expect("entry block is open");
    let normalized = builder.emit_unary(UnOp::Normalize, value, Ty::vec3()).expect("vector normalize");
    builder.return_value(normalized).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON vector program must build").finish().expect("TON vector program must validate")
}

/// 构建单步标量积分程序。
pub fn orbit_step_once() -> Program {
    let mut builder = FunctionBuilder::new("orbit_step_once");
    let position = builder.emit_load_uniform(0, Ty::f64()).expect("entry block is open");
    let velocity = builder.emit_load_uniform(1, Ty::f64()).expect("entry block is open");
    let step = builder.emit_load_uniform(2, Ty::f64()).expect("entry block is open");
    let delta = builder.emit_binary(BinOp::Mul, velocity, step).expect("f64 arithmetic");
    let next = builder.emit_binary(BinOp::Add, position, delta).expect("f64 arithmetic");
    builder.return_value(next).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON orbit program must build").finish().expect("TON orbit program must validate")
}

/// 构建赤道面零测地线 RK4 单步程序。
///
/// 状态 buffer `0` 每实例为 `[u, du_dphi, phi]`，其中 `u = 1 / r`。
/// uniform `0` 为质量，uniform `1` 为自旋，uniform `2` 为 `dphi`。
/// 自旋为零时得到 Schwarzschild 赤道轨道，非零时得到 Kerr 赤道轨道。
pub fn equatorial_orbit_rk4_step() -> Program {
    let mut builder = FunctionBuilder::new("equatorial_orbit_rk4_step");
    let index = builder.emit_global_id().expect("open block");
    let stride = builder.emit_const_u32(3).expect("open block");
    let base = builder.emit_mul_u32(index, stride).expect("u32 index");
    let offset_u = builder.emit_const_u32(0).expect("open block");
    let offset_du = builder.emit_const_u32(1).expect("open block");
    let offset_phi = builder.emit_const_u32(2).expect("open block");
    let index_u = builder.emit_add_u32(base, offset_u).expect("u32 index");
    let index_du = builder.emit_add_u32(base, offset_du).expect("u32 index");
    let index_phi = builder.emit_add_u32(base, offset_phi).expect("u32 index");
    let u = builder.emit_load_buffer(0, index_u, Ty::f64()).expect("u state");
    let du = builder.emit_load_buffer(0, index_du, Ty::f64()).expect("du state");
    let phi = builder.emit_load_buffer(0, index_phi, Ty::f64()).expect("phi state");
    let mass = builder.emit_load_uniform(0, Ty::f64()).expect("mass uniform");
    let spin = builder.emit_load_uniform(1, Ty::f64()).expect("spin uniform");
    let dphi = builder.emit_load_uniform(2, Ty::f64()).expect("step uniform");
    let c3 = builder.emit_const_f64(3.0).expect("constant");
    let c2 = builder.emit_const_f64(2.0).expect("constant");
    let c6 = builder.emit_const_f64(6.0).expect("constant");
    let c05 = builder.emit_const_f64(0.5).expect("constant");
    let rhs = |builder: &mut FunctionBuilder, value: ucf_program::ValueId| {
        let value2 = builder.emit_binary(BinOp::Mul, value, value).expect("rhs square");
        let mass_value = builder.emit_binary(BinOp::Mul, mass, value2).expect("rhs mass");
        let three_mass_value = builder.emit_binary(BinOp::Mul, c3, mass_value).expect("rhs mass term");
        let schwarzschild = builder.emit_binary(BinOp::Sub, three_mass_value, value).expect("rhs Schwarzschild");
        let spin2 = builder.emit_binary(BinOp::Mul, spin, spin).expect("spin square");
        let value4 = builder.emit_binary(BinOp::Mul, value2, value2).expect("rhs fourth power");
        let mass_u = builder.emit_binary(BinOp::Mul, mass, value).expect("rhs mass u");
        let three_mass_u = builder.emit_binary(BinOp::Mul, c3, mass_u).expect("rhs 3Mu");
        let correction_factor = builder.emit_binary(BinOp::Sub, three_mass_u, c2).expect("rhs correction factor");
        let value_correction = builder.emit_binary(BinOp::Mul, value4, correction_factor).expect("rhs correction");
        let correction = builder.emit_binary(BinOp::Mul, spin2, value_correction).expect("rhs correction");
        builder.emit_binary(BinOp::Sub, schwarzschild, correction).expect("Kerr rhs")
    };
    let half_step = builder.emit_binary(BinOp::Mul, dphi, c05).expect("half step");
    let k1_u = du;
    let k1_du = rhs(&mut builder, u);
    let k1_u_step = builder.emit_binary(BinOp::Mul, half_step, k1_u).expect("k1 u");
    let u2 = builder.emit_binary(BinOp::Add, u, k1_u_step).expect("u2");
    let k1_du_step = builder.emit_binary(BinOp::Mul, half_step, k1_du).expect("k1 du");
    let du2 = builder.emit_binary(BinOp::Add, du, k1_du_step).expect("du2");
    let k2_u = du2;
    let k2_du = rhs(&mut builder, u2);
    let k2_u_step = builder.emit_binary(BinOp::Mul, half_step, k2_u).expect("k2 u");
    let u3 = builder.emit_binary(BinOp::Add, u, k2_u_step).expect("u3");
    let k2_du_step = builder.emit_binary(BinOp::Mul, half_step, k2_du).expect("k2 du");
    let du3 = builder.emit_binary(BinOp::Add, du, k2_du_step).expect("du3");
    let k3_u = du3;
    let k3_du = rhs(&mut builder, u3);
    let k3_u_step = builder.emit_binary(BinOp::Mul, dphi, k3_u).expect("k3 u");
    let u4 = builder.emit_binary(BinOp::Add, u, k3_u_step).expect("u4");
    let k3_du_step = builder.emit_binary(BinOp::Mul, dphi, k3_du).expect("k3 du");
    let du4 = builder.emit_binary(BinOp::Add, du, k3_du_step).expect("du4");
    let k4_u = du4;
    let k4_du = rhs(&mut builder, u4);
    let two_k2_u = builder.emit_binary(BinOp::Mul, c2, k2_u).expect("2k2");
    let first_u = builder.emit_binary(BinOp::Add, k1_u, two_k2_u).expect("weighted u");
    let two_k3_u = builder.emit_binary(BinOp::Mul, c2, k3_u).expect("2k3");
    let second_u = builder.emit_binary(BinOp::Add, two_k3_u, k4_u).expect("weighted u");
    let weighted_u = builder.emit_binary(BinOp::Add, first_u, second_u).expect("weighted u");
    let two_k2_du = builder.emit_binary(BinOp::Mul, c2, k2_du).expect("2k2");
    let first_du = builder.emit_binary(BinOp::Add, k1_du, two_k2_du).expect("weighted du");
    let two_k3_du = builder.emit_binary(BinOp::Mul, c2, k3_du).expect("2k3");
    let second_du = builder.emit_binary(BinOp::Add, two_k3_du, k4_du).expect("weighted du");
    let weighted_du = builder.emit_binary(BinOp::Add, first_du, second_du).expect("weighted du");
    let scaled_u = builder.emit_binary(BinOp::Div, weighted_u, c6).expect("rk4 scale");
    let increment_u = builder.emit_binary(BinOp::Mul, dphi, scaled_u).expect("u increment");
    let next_u = builder.emit_binary(BinOp::Add, u, increment_u).expect("next u");
    let scaled_du = builder.emit_binary(BinOp::Div, weighted_du, c6).expect("rk4 scale");
    let increment_du = builder.emit_binary(BinOp::Mul, dphi, scaled_du).expect("du increment");
    let next_du = builder.emit_binary(BinOp::Add, du, increment_du).expect("next du");
    let next_phi = builder.emit_binary(BinOp::Add, phi, dphi).expect("next phi");
    for (offset, value) in [(offset_u, next_u), (offset_du, next_du), (offset_phi, next_phi)] {
        let target = builder.emit_add_u32(base, offset).expect("u32 index");
        builder.emit_store_buffer(0, target, value).expect("store orbit state");
    }
    builder.return_value(next_u).expect("open block");
    ProgramBuilder::new().entry(builder).expect("valid orbit builder").finish().expect("valid orbit program")
}

/// 构建有界径向推进程序。
pub fn geodesic_radial_infall_steps() -> Program {
    f(
        "geodesic_radial_infall_steps",
        vec![
            Stmt::Let { id: 0, ty: Ty::f64(), init: uni(0) },
            Stmt::Let { id: 1, ty: Ty::f64(), init: c(0.0) },
            Stmt::While {
                cond: bin(BinOp::Gt, Expr::LoadLocal(0), uni(1)),
                limit: 512,
                body: vec![
                    Stmt::StoreLocal { id: 0, value: bin(BinOp::Sub, Expr::LoadLocal(0), uni(2)) },
                    Stmt::StoreLocal { id: 1, value: bin(BinOp::Add, Expr::LoadLocal(1), c(1.0)) },
                ],
            },
            Stmt::If {
                cond: bin(BinOp::Le, Expr::LoadLocal(0), uni(1)),
                then_body: vec![Stmt::Return(Expr::LoadLocal(1))],
                else_body: vec![Stmt::Return(c(-1.0))],
            },
        ],
        vec![(0, Ty::f64()), (1, Ty::f64())],
        Ty::f64(),
    )
}

/// 构建史瓦西临界冲击参数分类程序。
pub fn geodesic_capture_classify() -> Program {
    let mut builder = FunctionBuilder::new("geodesic_capture_classify");
    let impact = builder.emit_load_uniform(0, Ty::f32()).expect("entry block is open");
    let critical = builder.emit_load_uniform(1, Ty::f32()).expect("entry block is open");
    let captured = builder.emit_binary(BinOp::Lt, impact, critical).expect("matching scalar types");
    let one = builder.emit_const_f32(1.0).expect("entry block is open");
    let zero = builder.emit_const_f32(0.0).expect("entry block is open");
    let result = builder.emit_select(captured, one, zero).expect("matching branch types");
    builder.return_value(result).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON UCF frontend must build a valid capture classifier").finish().expect("TON UCF frontend program must validate")
}

/// 构建并行捕获分类程序。资源 `0` 是冲击参数输入，返回值是分类结果。
pub fn geodesic_capture_classify_batch() -> Program {
    let mut builder = FunctionBuilder::new("geodesic_capture_classify_batch");
    let index = builder.emit_global_id().expect("entry block is open");
    let impact = builder.emit_load_buffer(0, index, Ty::f32()).expect("global id is a u32 index");
    let critical = builder.emit_load_uniform(0, Ty::f32()).expect("entry block is open");
    let captured = builder.emit_binary(BinOp::Lt, impact, critical).expect("matching scalar types");
    let one = builder.emit_const_f32(1.0).expect("entry block is open");
    let zero = builder.emit_const_f32(0.0).expect("entry block is open");
    let result = builder.emit_select(captured, one, zero).expect("matching branch types");
    builder.return_value(result).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON UCF frontend must build a valid batch classifier").finish().expect("TON UCF frontend program must validate")
}

/// 构建 Schwarzschild 径向零测地线的四阶 Runge-Kutta 单步程序。
/// 从位置和方向计算守恒冲击参数。输入资源 0 每实例为六个 f32。
/// uniform 0 是质量，返回值为总角动量与能量之比。
pub fn schwarzschild_impact_batch() -> Program {
    use ucf_program::ValueId;
    fn binary(builder: &mut FunctionBuilder, op: BinOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        builder.emit_binary(op, lhs, rhs).expect("matching scalar operands")
    }
    fn dot(builder: &mut FunctionBuilder, lhs: &[ValueId], rhs: &[ValueId]) -> ValueId {
        let products = lhs.iter().zip(rhs).map(|(left, right)| binary(builder, BinOp::Mul, *left, *right)).collect::<Vec<_>>();
        let sum = binary(builder, BinOp::Add, products[0], products[1]);
        binary(builder, BinOp::Add, sum, products[2])
    }
    let mut builder = FunctionBuilder::new("schwarzschild_impact_batch");
    let global = builder.emit_global_id().expect("open block");
    let stride = builder.emit_const_u32(6).expect("open block");
    let base = builder.emit_mul_u32(global, stride).expect("u32 indices");
    let values = (0..6).map(|offset| {
        let offset = builder.emit_const_u32(offset).expect("open block");
        let index = builder.emit_add_u32(base, offset).expect("u32 indices");
        builder.emit_load_buffer(0, index, Ty::f32()).expect("f32 input")
    }).collect::<Vec<_>>();
    let position = &values[..3];
    let direction = &values[3..];
    let radius2 = dot(&mut builder, position, position);
    let radius = builder.emit_unary(UnOp::Sqrt, radius2, Ty::f32()).expect("scalar sqrt");
    let direction2 = dot(&mut builder, direction, direction);
    let direction_length = builder.emit_unary(UnOp::Sqrt, direction2, Ty::f32()).expect("scalar sqrt");
    let normalized = direction.iter().map(|value| binary(&mut builder, BinOp::Div, *value, direction_length)).collect::<Vec<_>>();
    let radial_dot = dot(&mut builder, position, &normalized);
    let radial = binary(&mut builder, BinOp::Div, radial_dot, radius);
    let angular = [(1, 2), (2, 0), (0, 1)].map(|(left, right)| {
        let first = binary(&mut builder, BinOp::Mul, position[left], normalized[right]);
        let second = binary(&mut builder, BinOp::Mul, position[right], normalized[left]);
        binary(&mut builder, BinOp::Sub, first, second)
    });
    let angular2 = dot(&mut builder, &angular, &angular);
    let mass = builder.emit_load_uniform(0, Ty::f32()).expect("mass uniform");
    let two = builder.emit_const_f32(2.0).expect("open block");
    let one = builder.emit_const_f32(1.0).expect("open block");
    let twice_mass = binary(&mut builder, BinOp::Mul, two, mass);
    let compactness = binary(&mut builder, BinOp::Div, twice_mass, radius);
    let metric = binary(&mut builder, BinOp::Sub, one, compactness);
    let radial2 = binary(&mut builder, BinOp::Mul, radial, radial);
    let transverse2 = binary(&mut builder, BinOp::Div, angular2, radius2);
    let transverse_energy = binary(&mut builder, BinOp::Mul, metric, transverse2);
    let energy2 = binary(&mut builder, BinOp::Add, radial2, transverse_energy);
    let ratio2 = binary(&mut builder, BinOp::Div, angular2, energy2);
    let impact = builder.emit_unary(UnOp::Sqrt, ratio2, Ty::f32()).expect("scalar sqrt");
    builder.return_value(impact).expect("open block");
    ProgramBuilder::new().entry(builder).expect("valid impact builder").finish().expect("valid impact program")
}

/// 从 UV buffer 生成相机射线，并将 xyz 写入输出 buffer。
/// 输入资源 1 每实例为 `[u, v]`，输出资源 2 每实例为 `[x, y, z]`。
/// uniform 0..2 为 forward，3..5 为 right，6..8 为 up，9 为 tan(fov/2)。
pub fn camera_ray_batch() -> Program {
    let mut builder = FunctionBuilder::new("camera_ray_batch");
    let global = builder.emit_global_id().expect("open block");
    let two = builder.emit_const_u32(2).expect("open block");
    let three = builder.emit_const_u32(3).expect("open block");
    let input_base = builder.emit_mul_u32(global, two).expect("u32 indices");
    let output_base = builder.emit_mul_u32(global, three).expect("u32 indices");
    let input_offset = |builder: &mut FunctionBuilder, base, offset| {
        let offset = builder.emit_const_u32(offset).expect("open block");
        let index = builder.emit_add_u32(base, offset).expect("u32 indices");
        builder.emit_load_buffer(1, index, Ty::f32()).expect("f32 UV input")
    };
    let u = input_offset(&mut builder, input_base, 0);
    let v = input_offset(&mut builder, input_base, 1);
    let uniform = |builder: &mut FunctionBuilder, slot| builder.emit_load_uniform(slot, Ty::f32()).expect("camera uniform");
    let forward = [uniform(&mut builder, 0), uniform(&mut builder, 1), uniform(&mut builder, 2)];
    let right = [uniform(&mut builder, 3), uniform(&mut builder, 4), uniform(&mut builder, 5)];
    let up = [uniform(&mut builder, 6), uniform(&mut builder, 7), uniform(&mut builder, 8)];
    let tan_half = uniform(&mut builder, 9);
    let horizontal = builder.emit_binary(BinOp::Mul, u, tan_half).expect("scalar multiply");
    let vertical = builder.emit_binary(BinOp::Mul, v, tan_half).expect("scalar multiply");
    let direction = [0, 1, 2].map(|axis| {
        let horizontal_offset = builder.emit_binary(BinOp::Mul, horizontal, right[axis]).expect("scalar scale");
        let vertical_offset = builder.emit_binary(BinOp::Mul, vertical, up[axis]).expect("scalar scale");
        let partial = builder.emit_binary(BinOp::Add, forward[axis], horizontal_offset).expect("scalar add");
        builder.emit_binary(BinOp::Add, partial, vertical_offset).expect("scalar add")
    });
    let square = direction.map(|value| builder.emit_binary(BinOp::Mul, value, value).expect("scalar multiply"));
    let sum = builder.emit_binary(BinOp::Add, square[0], square[1]).expect("scalar add");
    let sum = builder.emit_binary(BinOp::Add, sum, square[2]).expect("scalar add");
    let length = builder.emit_unary(UnOp::Sqrt, sum, Ty::f32()).expect("scalar sqrt");
    let direction = direction.map(|value| builder.emit_binary(BinOp::Div, value, length).expect("scalar divide"));
    for component in 0u32..3 {
        let offset = builder.emit_const_u32(component).expect("open block");
        let index = builder.emit_add_u32(output_base, offset).expect("u32 indices");
        builder.emit_store_buffer(2, index, direction[component as usize]).expect("direction output");
    }
    builder.return_value(direction[0]).expect("open block");
    ProgramBuilder::new().entry(builder).expect("valid camera builder").finish().expect("valid camera program")
}

/// 从笛卡尔光线生成 Schwarzschild 球坐标零测地线初态。
///
/// 输入资源 `1` 每实例为 `[origin.xyz, direction.xyz]`，输出资源 `2` 每实例为：
/// `[t, r, theta, phi, ut, ur, utheta, uphi, status, bound_infall]`。
/// uniform `0` 为质量，uniform `1` 为临界冲击参数。所有计算均由通用 UCF
/// 算术、比较、选择和有限控制流表达，UCF 不知道这些字段的物理含义。
pub fn schwarzschild_initial_state_batch() -> Program {
    fn local(id: u32) -> Expr { Expr::LoadLocal(id) }
    fn f32(value: f32) -> Expr { Expr::ConstF32(value) }
    fn load(resource: u32, index: Expr) -> Expr { Expr::LoadBuffer { resource, index: Box::new(index) } }
    fn index(base: Expr, offset: u32) -> Expr { bin(BinOp::Add, base, Expr::ConstU32(offset)) }
    fn unary(op: UnOp, arg: Expr) -> Expr { un(op, arg) }
    fn square(value: Expr) -> Expr { bin(BinOp::Mul, value.clone(), value) }
    let gid = Expr::GlobalId;
    let input_base = bin(BinOp::Mul, gid.clone(), Expr::ConstU32(6));
    let output_base = bin(BinOp::Mul, gid, Expr::ConstU32(10));
    let input = |offset| load(1, index(input_base.clone(), offset));
    let mut body = Vec::new();
    for offset in 0..6 { body.push(Stmt::Let { id: offset, ty: Ty::f32(), init: input(offset) }); }
    let x = local(0); let y = local(1); let z = local(2);
    body.push(Stmt::Let { id: 6, ty: Ty::f32(), init: bin(BinOp::Add, bin(BinOp::Add, square(x.clone()), square(y.clone())), square(z.clone())) });
    body.push(Stmt::Let { id: 7, ty: Ty::f32(), init: unary(UnOp::Sqrt, local(6)) });
    body.push(Stmt::Let { id: 8, ty: Ty::f32(), init: bin(BinOp::Add, square(x.clone()), square(y.clone())) });
    body.push(Stmt::Let { id: 9, ty: Ty::f32(), init: unary(UnOp::Sqrt, local(8)) });
    let eps = f32(1.0e-6); let neg_eps = unary(UnOp::Neg, eps.clone());
    let theta_pos = unary(UnOp::Atan, bin(BinOp::Div, local(9), bin(BinOp::Max, z.clone(), eps.clone())));
    let theta_neg = bin(BinOp::Add, unary(UnOp::Atan, bin(BinOp::Div, local(9), bin(BinOp::Min, z.clone(), neg_eps.clone()))), f32(std::f32::consts::PI));
    let theta = select(bin(BinOp::Gt, z.clone(), eps.clone()), theta_pos, select(bin(BinOp::Lt, z.clone(), neg_eps.clone()), theta_neg, f32(std::f32::consts::FRAC_PI_2)));
    body.push(Stmt::Let { id: 10, ty: Ty::f32(), init: theta });
    let phi_pos = unary(UnOp::Atan, bin(BinOp::Div, y.clone(), bin(BinOp::Max, x.clone(), eps.clone())));
    let phi_neg_base = unary(UnOp::Atan, bin(BinOp::Div, y.clone(), bin(BinOp::Min, x.clone(), neg_eps.clone())));
    let phi_neg = select(bin(BinOp::Ge, y.clone(), f32(0.0)), bin(BinOp::Add, phi_neg_base.clone(), f32(std::f32::consts::PI)), bin(BinOp::Sub, phi_neg_base, f32(std::f32::consts::PI)));
    let phi_axis = f32(0.0);
    body.push(Stmt::Let { id: 11, ty: Ty::f32(), init: select(bin(BinOp::Gt, x.clone(), eps.clone()), phi_pos, select(bin(BinOp::Lt, x.clone(), neg_eps), phi_neg, phi_axis)) });
    let direction_length = unary(UnOp::Sqrt, bin(BinOp::Add, bin(BinOp::Add, square(local(3)), square(local(4))), square(local(5))));
    body.push(Stmt::Let { id: 12, ty: Ty::f32(), init: bin(BinOp::Div, local(3), direction_length.clone()) });
    body.push(Stmt::Let { id: 13, ty: Ty::f32(), init: bin(BinOp::Div, local(4), direction_length.clone()) });
    body.push(Stmt::Let { id: 14, ty: Ty::f32(), init: bin(BinOp::Div, local(5), direction_length) });
    let sin_t = unary(UnOp::Sin, local(10));
    body.push(Stmt::Let { id: 15, ty: Ty::f32(), init: bin(BinOp::Max, sin_t, eps.clone()) });
    body.push(Stmt::Let { id: 16, ty: Ty::f32(), init: unary(UnOp::Sin, local(10)) });
    body.push(Stmt::Let { id: 17, ty: Ty::f32(), init: unary(UnOp::Cos, local(10)) });
    body.push(Stmt::Let { id: 18, ty: Ty::f32(), init: unary(UnOp::Sin, local(11)) });
    body.push(Stmt::Let { id: 19, ty: Ty::f32(), init: unary(UnOp::Cos, local(11)) });
    let radial = bin(
        BinOp::Div,
        bin(BinOp::Add, bin(BinOp::Add, bin(BinOp::Mul, local(12), x.clone()), bin(BinOp::Mul, local(13), y.clone())), bin(BinOp::Mul, local(14), z.clone())),
        local(7),
    );
    body.push(Stmt::Let { id: 20, ty: Ty::f32(), init: radial });
    let theta_dot = bin(
        BinOp::Add,
        bin(BinOp::Add, bin(BinOp::Mul, local(12), bin(BinOp::Mul, local(17), local(19))), bin(BinOp::Mul, local(13), bin(BinOp::Mul, local(17), local(18)))),
        bin(BinOp::Mul, local(14), unary(UnOp::Neg, local(16))),
    );
    body.push(Stmt::Let { id: 21, ty: Ty::f32(), init: bin(BinOp::Div, theta_dot, local(7)) });
    let phi_dot = bin(BinOp::Add, bin(BinOp::Mul, local(12), unary(UnOp::Neg, local(18))), bin(BinOp::Mul, local(13), local(19)));
    body.push(Stmt::Let { id: 22, ty: Ty::f32(), init: bin(BinOp::Div, phi_dot, bin(BinOp::Mul, local(7), local(15))) });
    let mass = Expr::LoadUniform(0); let metric = bin(BinOp::Sub, f32(1.0), bin(BinOp::Div, bin(BinOp::Mul, f32(2.0), mass), local(7)));
    body.push(Stmt::Let { id: 23, ty: Ty::f32(), init: metric });
    let spatial = bin(BinOp::Add, bin(BinOp::Add, bin(BinOp::Div, square(local(20)), local(23)), bin(BinOp::Mul, square(local(7)), square(local(21)))), bin(BinOp::Mul, square(local(7)), bin(BinOp::Mul, square(local(15)), square(local(22)))));
    body.push(Stmt::Let { id: 24, ty: Ty::f32(), init: bin(BinOp::Div, unary(UnOp::Sqrt, spatial), unary(UnOp::Sqrt, local(23))) });
    let angular = unary(UnOp::Sqrt, bin(BinOp::Add, bin(BinOp::Mul, square(local(7)), square(local(21))), bin(BinOp::Mul, square(local(7)), bin(BinOp::Mul, square(local(15)), square(local(22))))));
    let impact = bin(BinOp::Div, bin(BinOp::Mul, local(7), angular), bin(BinOp::Mul, local(23), local(24)));
    body.push(Stmt::Let { id: 25, ty: Ty::f32(), init: impact });
    body.push(Stmt::Let { id: 26, ty: Ty::f32(), init: select(bin(BinOp::Lt, local(20), f32(0.0)), select(bin(BinOp::Lt, local(25), Expr::LoadUniform(1)), f32(1.0), f32(0.0)), f32(0.0)) });
    let outputs = [f32(0.0), local(7), local(10), local(11), local(24), local(20), local(21), local(22), f32(0.0), local(26)];
    for (offset, value) in outputs.into_iter().enumerate() {
        body.push(Stmt::StoreBuffer { resource: 2, index: index(output_base.clone(), offset as u32), value });
    }
    body.push(Stmt::Return(local(7)));
    f("schwarzschild_initial_state_batch", body, (0..27).map(|id| (id, Ty::f32())).collect(), Ty::f32())
}

/// 构建 Schwarzschild 径向零测地线的四阶 Runge-Kutta 单步程序。
///
/// 状态 buffer 的每个实例占四个 `f32`，布局为 `[t, r, u^t, u^r]`。
/// uniform `0` 是质量，uniform `1` 是仿射步长。程序原地更新状态 buffer。
pub fn schwarzschild_radial_rk4_step() -> Program {
    fn binary(builder: &mut FunctionBuilder, op: BinOp, lhs: ucf_program::ValueId, rhs: ucf_program::ValueId) -> ucf_program::ValueId {
        builder.emit_binary(op, lhs, rhs).expect("valid f32 arithmetic")
    }
    fn unary(builder: &mut FunctionBuilder, op: UnOp, arg: ucf_program::ValueId) -> ucf_program::ValueId {
        builder.emit_unary(op, arg, Ty::f32()).expect("valid f32 arithmetic")
    }
    let mut builder = FunctionBuilder::new("schwarzschild_radial_rk4_step");
    let gid = builder.emit_global_id().expect("entry block is open");
    let stride = builder.emit_const_u32(4).expect("entry block is open");
    let base = builder.emit_mul_u32(gid, stride).expect("u32 indices");
    let state = (0..4).map(|offset| {
        let offset = builder.emit_const_u32(offset).expect("entry block is open");
        let index = builder.emit_add_u32(base, offset).expect("u32 indices");
        builder.emit_load_buffer(0, index, Ty::f32()).expect("state buffer is f32")
    }).collect::<Vec<_>>();
    let mass = builder.emit_load_uniform(0, Ty::f32()).expect("entry block is open");
    let step = builder.emit_load_uniform(1, Ty::f32()).expect("entry block is open");

    let deriv = |builder: &mut FunctionBuilder, values: &[ucf_program::ValueId]| {
        let t = values[0];
        let r = values[1];
        let ut = values[2];
        let ur = values[3];
        let two = builder.emit_const_f32(2.0).expect("open block");
        let one = builder.emit_const_f32(1.0).expect("open block");
        let r2 = builder.emit_binary(BinOp::Mul, r, r).expect("f32 arithmetic");
        let inv_r2 = builder.emit_binary(BinOp::Div, one, r2).expect("f32 arithmetic");
        let two_mass = builder.emit_binary(BinOp::Mul, two, mass).expect("f32 arithmetic");
        let inv_r = builder.emit_binary(BinOp::Div, one, r).expect("f32 arithmetic");
        let compactness = binary(builder, BinOp::Mul, two_mass, inv_r);
        let f = binary(builder, BinOp::Sub, one, compactness);
        let dt = ut;
        let dr = ur;
        let m_over_r2 = builder.emit_binary(BinOp::Mul, mass, inv_r2).expect("f32 arithmetic");
        let gamma_t = builder.emit_binary(BinOp::Div, m_over_r2, f).expect("f32 arithmetic");
        let two_gamma_t = builder.emit_binary(BinOp::Mul, two, gamma_t).expect("f32 arithmetic");
        let dut_product = binary(builder, BinOp::Mul, two_gamma_t, ut);
        let dut_product = binary(builder, BinOp::Mul, dut_product, ur);
        let dut = unary(builder, UnOp::Neg, dut_product);
        let gamma_r_tt = builder.emit_binary(BinOp::Mul, f, m_over_r2).expect("f32 arithmetic");
        let ut2 = binary(builder, BinOp::Mul, ut, ut);
        let first_product = binary(builder, BinOp::Mul, gamma_r_tt, ut2);
        let first = unary(builder, UnOp::Neg, first_product);
        let gamma_r_rr = builder.emit_binary(BinOp::Div, m_over_r2, f).expect("f32 arithmetic");
        let ur2 = binary(builder, BinOp::Mul, ur, ur);
        let second = binary(builder, BinOp::Mul, gamma_r_rr, ur2);
        let dur = binary(builder, BinOp::Add, first, second);
        let _ = t;
        [dt, dr, dut, dur]
    };
    let half = builder.emit_const_f32(0.5).expect("open block");
    let sixth = builder.emit_const_f32(1.0 / 6.0).expect("open block");
    let half_step = builder.emit_binary(BinOp::Mul, step, half).expect("f32 arithmetic");
    let k1 = deriv(&mut builder, &state);
    let s2 = state.iter().zip(k1.iter()).map(|(value, slope)| {
        let delta = binary(&mut builder, BinOp::Mul, *slope, half_step);
        binary(&mut builder, BinOp::Add, *value, delta)
    }).collect::<Vec<_>>();
    let k2 = deriv(&mut builder, &s2);
    let s3 = state.iter().zip(k2.iter()).map(|(value, slope)| {
        let delta = binary(&mut builder, BinOp::Mul, *slope, half_step);
        binary(&mut builder, BinOp::Add, *value, delta)
    }).collect::<Vec<_>>();
    let k3 = deriv(&mut builder, &s3);
    let s4 = state.iter().zip(k3.iter()).map(|(value, slope)| {
        let delta = binary(&mut builder, BinOp::Mul, *slope, step);
        binary(&mut builder, BinOp::Add, *value, delta)
    }).collect::<Vec<_>>();
    let k4 = deriv(&mut builder, &s4);
    let two = builder.emit_const_f32(2.0).expect("open block");
    let output = (0..4).map(|index| {
        let two_k2 = builder.emit_binary(BinOp::Mul, k2[index], two).expect("f32 arithmetic");
        let two_k3 = builder.emit_binary(BinOp::Mul, k3[index], two).expect("f32 arithmetic");
        let sum = builder.emit_binary(BinOp::Add, k1[index], two_k2).expect("f32 arithmetic");
        let sum = builder.emit_binary(BinOp::Add, sum, two_k3).expect("f32 arithmetic");
        let sum = builder.emit_binary(BinOp::Add, sum, k4[index]).expect("f32 arithmetic");
        let weighted = binary(&mut builder, BinOp::Mul, sixth, sum);
        let delta = binary(&mut builder, BinOp::Mul, step, weighted);
        binary(&mut builder, BinOp::Add, state[index], delta)
    }).collect::<Vec<_>>();
    for (offset, value) in output.iter().enumerate() {
        let offset = builder.emit_const_u32(offset as u32).expect("entry block is open");
        let index = builder.emit_add_u32(base, offset).expect("u32 indices");
        builder.emit_store_buffer(0, index, *value).expect("state buffer is writable");
    }
    builder.return_value(output[1]).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON RK4 frontend must build a valid program").finish().expect("TON RK4 program must validate")
}

/// 构建完整球坐标 Schwarzschild 零测地线的 RK4 单步程序。
///
/// 状态 buffer 的每个实例占八个 `f32`，布局为
/// `[t, r, theta, phi, ut, ur, utheta, uphi]`。uniform `0` 是质量，
/// uniform `1` 是仿射步长。该程序只使用 UCF 通用算术和三角原语。
pub fn schwarzschild_geodesic_rk4_step() -> Program {
    fn binary(builder: &mut FunctionBuilder, op: BinOp, lhs: ucf_program::ValueId, rhs: ucf_program::ValueId) -> ucf_program::ValueId {
        builder.emit_binary(op, lhs, rhs).expect("valid f32 arithmetic")
    }
    fn unary(builder: &mut FunctionBuilder, op: UnOp, arg: ucf_program::ValueId) -> ucf_program::ValueId {
        builder.emit_unary(op, arg, Ty::f32()).expect("valid f32 arithmetic")
    }

    let mut builder = FunctionBuilder::new("schwarzschild_geodesic_rk4_step");
    let gid = builder.emit_global_id().expect("entry block is open");
    let stride = builder.emit_const_u32(8).expect("entry block is open");
    let base = builder.emit_mul_u32(gid, stride).expect("u32 indices");
    let state = (0..8).map(|offset| {
        let offset = builder.emit_const_u32(offset).expect("entry block is open");
        let index = builder.emit_add_u32(base, offset).expect("u32 indices");
        builder.emit_load_buffer(0, index, Ty::f32()).expect("state buffer is f32")
    }).collect::<Vec<_>>();
    let mass = builder.emit_load_uniform(0, Ty::f32()).expect("mass uniform");
    let step = builder.emit_load_uniform(1, Ty::f32()).expect("step uniform");

    let deriv = |builder: &mut FunctionBuilder, values: &[ucf_program::ValueId]| {
        let r = values[1];
        let theta = values[2];
        let ut = values[4];
        let ur = values[5];
        let utheta = values[6];
        let uphi = values[7];
        let one = builder.emit_const_f32(1.0).expect("f32 constant");
        let two = builder.emit_const_f32(2.0).expect("f32 constant");
        let r2 = binary(builder, BinOp::Mul, r, r);
        let inv_r = binary(builder, BinOp::Div, one, r);
        let inv_r2 = binary(builder, BinOp::Div, one, r2);
        let two_mass = binary(builder, BinOp::Mul, two, mass);
        let compactness = binary(builder, BinOp::Mul, two_mass, inv_r);
        let f = binary(builder, BinOp::Sub, one, compactness);
        let fp = binary(builder, BinOp::Mul, two_mass, inv_r2);
        let two_f = binary(builder, BinOp::Mul, two, f);
        let gamma_t_tr = binary(builder, BinOp::Div, fp, two_f);
        let ut_ur = binary(builder, BinOp::Mul, ut, ur);
        let gamma_ut_ur = binary(builder, BinOp::Mul, gamma_t_tr, ut_ur);
        let two_gamma_ut_ur = binary(builder, BinOp::Mul, two, gamma_ut_ur);
        let du_t = unary(builder, UnOp::Neg, two_gamma_ut_ur);
        let m_inv_r2 = binary(builder, BinOp::Mul, mass, inv_r2);
        let gamma_r_tt = binary(builder, BinOp::Mul, f, m_inv_r2);
        let fp_two_f = binary(builder, BinOp::Div, fp, two_f);
        let gamma_r_rr = unary(builder, UnOp::Neg, fp_two_f);
        let sin_theta = unary(builder, UnOp::Sin, theta);
        let cos_theta = unary(builder, UnOp::Cos, theta);
        let sin2 = binary(builder, BinOp::Mul, sin_theta, sin_theta);
        let f_r = binary(builder, BinOp::Mul, f, r);
        let gamma_r_tt_theta = unary(builder, UnOp::Neg, f_r);
        let f_r_sin2 = binary(builder, BinOp::Mul, f_r, sin2);
        let gamma_r_pp = unary(builder, UnOp::Neg, f_r_sin2);
        let ur2 = binary(builder, BinOp::Mul, ur, ur);
        let utheta2 = binary(builder, BinOp::Mul, utheta, utheta);
        let uphi2 = binary(builder, BinOp::Mul, uphi, uphi);
        let ut2 = binary(builder, BinOp::Mul, ut, ut);
        let term_tt = binary(builder, BinOp::Mul, gamma_r_tt, ut2);
        let term_rr = binary(builder, BinOp::Mul, gamma_r_rr, ur2);
        let term_theta = binary(builder, BinOp::Mul, gamma_r_tt_theta, utheta2);
        let term_phi = binary(builder, BinOp::Mul, gamma_r_pp, uphi2);
        let radial_a = binary(builder, BinOp::Add, term_tt, term_rr);
        let radial_b = binary(builder, BinOp::Add, term_theta, term_phi);
        let radial_terms = binary(builder, BinOp::Add, radial_a, radial_b);
        let du_r = unary(builder, UnOp::Neg, radial_terms);
        let gamma_theta_rtheta = inv_r;
        let sin_cos = binary(builder, BinOp::Mul, sin_theta, cos_theta);
        let gamma_theta_pp = unary(builder, UnOp::Neg, sin_cos);
        let two_gamma_theta = binary(builder, BinOp::Mul, two, gamma_theta_rtheta);
        let ur_utheta = binary(builder, BinOp::Mul, ur, utheta);
        let angular_a = binary(builder, BinOp::Mul, two_gamma_theta, ur_utheta);
        let angular_b = binary(builder, BinOp::Mul, gamma_theta_pp, uphi2);
        let angular_terms = binary(builder, BinOp::Add, angular_a, angular_b);
        let du_theta = unary(builder, UnOp::Neg, angular_terms);
        let cot = binary(builder, BinOp::Div, cos_theta, sin_theta);
        let two_inv_r = binary(builder, BinOp::Mul, two, inv_r);
        let ur_uphi = binary(builder, BinOp::Mul, ur, uphi);
        let phi_a = binary(builder, BinOp::Mul, two_inv_r, ur_uphi);
        let two_cot = binary(builder, BinOp::Mul, two, cot);
        let utheta_uphi = binary(builder, BinOp::Mul, utheta, uphi);
        let phi_b = binary(builder, BinOp::Mul, two_cot, utheta_uphi);
        let phi_terms = binary(builder, BinOp::Add, phi_a, phi_b);
        let du_phi = unary(builder, UnOp::Neg, phi_terms);
        [ut, ur, utheta, uphi, du_t, du_r, du_theta, du_phi]
    };
    let state: [ucf_program::ValueId; 8] = state.try_into().expect("eight state values");
    let half = builder.emit_const_f32(0.5).expect("f32 constant");
    let half_step = binary(&mut builder, BinOp::Mul, step, half);
    let add_scaled = |builder: &mut FunctionBuilder, values: &[ucf_program::ValueId; 8], slope: &[ucf_program::ValueId; 8], scale| {
        std::array::from_fn(|index| {
            let delta = binary(builder, BinOp::Mul, slope[index], scale);
            binary(builder, BinOp::Add, values[index], delta)
        })
    };
    let k1: [ucf_program::ValueId; 8] = deriv(&mut builder, &state);
    let mid1: [ucf_program::ValueId; 8] = add_scaled(&mut builder, &state, &k1, half_step);
    let k2: [ucf_program::ValueId; 8] = deriv(&mut builder, &mid1);
    let mid2: [ucf_program::ValueId; 8] = add_scaled(&mut builder, &state, &k2, half_step);
    let k3: [ucf_program::ValueId; 8] = deriv(&mut builder, &mid2);
    let end: [ucf_program::ValueId; 8] = add_scaled(&mut builder, &state, &k3, step);
    let k4: [ucf_program::ValueId; 8] = deriv(&mut builder, &end);
    let two = builder.emit_const_f32(2.0).expect("f32 constant");
    let six = builder.emit_const_f32(6.0).expect("f32 constant");
    for index in 0..8 {
        let two_k2 = binary(&mut builder, BinOp::Mul, k2[index], two);
        let two_k3 = binary(&mut builder, BinOp::Mul, k3[index], two);
        let sum = binary(&mut builder, BinOp::Add, k1[index], two_k2);
        let sum = binary(&mut builder, BinOp::Add, sum, two_k3);
        let sum = binary(&mut builder, BinOp::Add, sum, k4[index]);
        let average = binary(&mut builder, BinOp::Div, sum, six);
        let delta = binary(&mut builder, BinOp::Mul, step, average);
        let next = binary(&mut builder, BinOp::Add, state[index], delta);
        let offset = builder.emit_const_u32(index as u32).expect("state index");
        let index = builder.emit_add_u32(base, offset).expect("state index");
        builder.emit_store_buffer(0, index, next).expect("state buffer is writable");
    }
    builder.return_value(state[0]).expect("open block");
    ProgramBuilder::new().entry(builder).expect("valid geodesic builder").finish().expect("valid geodesic program")
}

/// 构建在单个 UCF 实例内执行有界 RK4 积分的 Schwarzschild 程序。
///
/// 状态资源布局为九个 `f32`：前八个元素是
/// `[t, r, theta, phi, ut, ur, utheta, uphi]`，第九个元素是终止码，
/// 第十个元素是束缚入射标记。非零表示不能仅凭到达逃逸半径判定逃逸。
/// uniform `0` 是质量，`1` 是步长，`2` 是逃逸半径。终止码为
/// `1=captured`、`2=escaped`、`3=budget_exhausted`、`4=non_finite`。
pub fn schwarzschild_geodesic_rk4_integrate(max_steps: u32) -> Program {
    schwarzschild_geodesic_rk4_integrate_mode(max_steps, true, false)
}

/// 构建一个有界积分 chunk。未达到物理终止条件时返回状态码 `0=running`，
/// 由外部任务图决定是否提交下一个 chunk。
pub fn schwarzschild_geodesic_rk4_chunk(max_steps: u32) -> Program {
    schwarzschild_geodesic_rk4_integrate_mode(max_steps, false, false)
}

/// 构建带赤道薄盘交点输出的有界积分 chunk。
///
/// 输入状态仍为十个 `f32`，输出扩展为十四个 `f32`，尾部依次为
/// `[disk_hit, disk_radius, disk_doppler, reserved]`。交点判定和半径
/// 插值在 UCF 程序内完成，宿主只负责绑定资源和调度。
pub fn schwarzschild_geodesic_rk4_chunk_with_disk(max_steps: u32) -> Program {
    schwarzschild_geodesic_rk4_integrate_mode(max_steps, false, true)
}

    fn b(builder: &StructuredFunctionBuilder, op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        builder.binary(op, lhs, rhs)
    }
    fn u(builder: &StructuredFunctionBuilder, op: UnOp, arg: Expr) -> Expr {
        builder.unary(op, arg)
    }
    fn schwarzschild_derivative(builder: &StructuredFunctionBuilder, values: &[Expr; 8], mass: &Expr) -> [Expr; 8] {
        let r = values[1].clone();
        let theta = values[2].clone();
        let ut = values[4].clone();
        let ur = values[5].clone();
        let utheta = values[6].clone();
        let uphi = values[7].clone();
        let one = builder.const_f32(1.0);
        let two = builder.const_f32(2.0);
        let r2 = b(builder, BinOp::Mul, r.clone(), r.clone());
        let inv_r = b(builder, BinOp::Div, one.clone(), r.clone());
        let inv_r2 = b(builder, BinOp::Div, one.clone(), r2);
        let two_mass = b(builder, BinOp::Mul, two.clone(), mass.clone());
        let f = b(builder, BinOp::Div, b(builder, BinOp::Sub, r.clone(), two_mass), r.clone());
        let fp = b(builder, BinOp::Mul, builder.binary(BinOp::Mul, builder.const_f32(2.0), mass.clone()), inv_r2.clone());
        let two_f = b(builder, BinOp::Mul, two.clone(), f.clone());
        let gamma_t_tr = b(builder, BinOp::Div, fp.clone(), two_f);
        let du_t = u(builder, UnOp::Neg, b(builder, BinOp::Mul, two.clone(), b(builder, BinOp::Mul, gamma_t_tr, b(builder, BinOp::Mul, ut.clone(), ur.clone()))));
        let gamma_r_tt = b(builder, BinOp::Mul, f.clone(), b(builder, BinOp::Mul, mass.clone(), inv_r2.clone()));
        let gamma_r_rr = u(builder, UnOp::Neg, b(builder, BinOp::Div, fp, b(builder, BinOp::Mul, builder.const_f32(2.0), f.clone())));
        let sin_theta = u(builder, UnOp::Sin, theta.clone());
        let cos_theta = u(builder, UnOp::Cos, theta);
        let sin2 = b(builder, BinOp::Mul, sin_theta.clone(), sin_theta.clone());
        let fr = b(builder, BinOp::Mul, f, r);
        let gamma_r_tt_theta = u(builder, UnOp::Neg, fr.clone());
        let gamma_r_pp = u(builder, UnOp::Neg, b(builder, BinOp::Mul, fr, sin2));
        let ur2 = b(builder, BinOp::Mul, ur.clone(), ur.clone());
        let utheta2 = b(builder, BinOp::Mul, utheta.clone(), utheta.clone());
        let uphi2 = b(builder, BinOp::Mul, uphi.clone(), uphi.clone());
        let ut2 = b(builder, BinOp::Mul, ut.clone(), ut.clone());
        let radial = b(builder, BinOp::Add,
            b(builder, BinOp::Add, b(builder, BinOp::Mul, gamma_r_tt, ut2), b(builder, BinOp::Mul, gamma_r_rr, ur2)),
            b(builder, BinOp::Add, b(builder, BinOp::Mul, gamma_r_tt_theta, utheta2), b(builder, BinOp::Mul, gamma_r_pp, uphi2.clone())));
        let du_r = u(builder, UnOp::Neg, radial);
        let gamma_theta_rtheta = inv_r;
        let gamma_theta_pp = u(builder, UnOp::Neg, b(builder, BinOp::Mul, sin_theta.clone(), cos_theta));
        let angular = b(builder, BinOp::Add,
            b(builder, BinOp::Mul, b(builder, BinOp::Mul, builder.const_f32(2.0), gamma_theta_rtheta), b(builder, BinOp::Mul, ur.clone(), utheta.clone())),
            b(builder, BinOp::Mul, gamma_theta_pp, uphi2));
        let du_theta = u(builder, UnOp::Neg, angular);
        let abs_sin = b(builder, BinOp::Max, sin_theta.clone(), u(builder, UnOp::Neg, sin_theta.clone()));
        let cot_theta = builder.select(
            b(builder, BinOp::Gt, abs_sin, builder.const_f32(1.0e-12)),
            b(builder, BinOp::Div, u(builder, UnOp::Cos, values[2].clone()), sin_theta),
            builder.const_f32(0.0),
        );
        let phi = b(builder, BinOp::Add,
            b(builder, BinOp::Mul, b(builder, BinOp::Mul, builder.const_f32(2.0), builder.binary(BinOp::Div, builder.const_f32(1.0), values[1].clone())), b(builder, BinOp::Mul, ur.clone(), uphi.clone())),
            b(builder, BinOp::Mul, b(builder, BinOp::Mul, builder.const_f32(2.0), cot_theta), b(builder, BinOp::Mul, utheta.clone(), uphi.clone())));
        [ut, ur, utheta, uphi, du_t, du_r, du_theta, u(builder, UnOp::Neg, phi)]
    }

fn schwarzschild_geodesic_rk4_integrate_mode(
    max_steps: u32,
    report_budget_exhaustion: bool,
    detect_disk: bool,
) -> Program {
    let mut builder = StructuredFunctionBuilder::new(if detect_disk {
        "schwarzschild_geodesic_rk4_chunk_with_disk"
    } else {
        "schwarzschild_geodesic_rk4_integrate"
    });
    let gid = builder.global_id();
    let stride = if detect_disk { 14 } else { 10 };
    let base = b(&builder, BinOp::Mul, gid, builder.const_u32(stride));
    let state: [u32; 8] = std::array::from_fn(|offset| {
        let index = b(&builder, BinOp::Add, base.clone(), builder.const_u32(offset as u32));
        builder.let_value(ucf_program::Ty::f32(), builder.load_buffer(0, index))
    });
    let bound_infall = builder.let_value(
        ucf_program::Ty::f32(),
        builder.load_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(9))),
    );
    let input_status = builder.let_value(
        ucf_program::Ty::f32(),
        builder.load_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(8))),
    );
    let disk_hit = detect_disk.then(|| {
        builder.let_value(
            ucf_program::Ty::f32(),
            builder.load_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(10))),
        )
    });
    let disk_radius = detect_disk.then(|| {
        builder.let_value(ucf_program::Ty::f32(), builder.load_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(11))))
    });
    let disk_doppler = detect_disk.then(|| {
        builder.let_value(ucf_program::Ty::f32(), builder.load_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(12))))
    });
    let mass = builder.load_uniform(0);
    let step = builder.load_uniform(1);
    let horizon = b(&builder, BinOp::Mul, builder.const_f32(2.0), mass.clone());
    let mut initial_finite = builder.unary(UnOp::IsFinite, Expr::LoadLocal(state[0]));
    for index in 1..8 {
        initial_finite = builder.select(
            initial_finite,
            builder.unary(UnOp::IsFinite, Expr::LoadLocal(state[index])),
            Expr::ConstBool(false),
        );
    }
    let status = builder.let_value(
        ucf_program::Ty::f32(),
        builder.select(
            b(&builder, BinOp::Eq, Expr::LoadLocal(input_status), builder.const_f32(0.0)),
            builder.select(initial_finite, builder.const_f32(0.0), builder.const_f32(4.0)),
            Expr::LoadLocal(input_status),
        ),
    );
    let active = |builder: &StructuredFunctionBuilder| {
        let outside_capture = builder.binary(
            BinOp::Gt,
            Expr::LoadLocal(state[1]),
            b(builder, BinOp::Mul, horizon.clone(), builder.const_f32(1.0001)),
        );
        let not_escaped = builder.select(
            b(builder, BinOp::Eq, Expr::LoadLocal(bound_infall), builder.const_f32(0.0)),
            b(builder, BinOp::Lt, Expr::LoadLocal(state[1]), builder.load_uniform(2)),
            Expr::ConstBool(true),
        );
        builder.select(
            builder.binary(BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(0.0)),
            builder.select(outside_capture, not_escaped, Expr::ConstBool(false)),
            Expr::ConstBool(false),
        )
    };
    builder.bounded_while(active(&builder), max_steps, |builder| {
        let values: [Expr; 8] = std::array::from_fn(|index| {
            let previous = builder.let_value(ucf_program::Ty::f32(), Expr::LoadLocal(state[index]));
            Expr::LoadLocal(previous)
        });
        let half_step = b(builder, BinOp::Mul, step.clone(), builder.const_f32(0.5));
        let add_scaled = |builder: &StructuredFunctionBuilder, values: &[Expr; 8], slope: &[Expr; 8], scale: Expr| {
            std::array::from_fn(|index| b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, slope[index].clone(), scale.clone())))
        };
        let k1 = schwarzschild_derivative(builder, &values, &mass);
        let mid1 = add_scaled(builder, &values, &k1, half_step.clone());
        let k2 = schwarzschild_derivative(builder, &mid1, &mass);
        let mid2 = add_scaled(builder, &values, &k2, half_step);
        let k3 = schwarzschild_derivative(builder, &mid2, &mass);
        let end = add_scaled(builder, &values, &k3, step.clone());
        let k4 = schwarzschild_derivative(builder, &end, &mass);
        for index in 0..8 {
            let weighted = b(builder, BinOp::Add, b(builder, BinOp::Add, k1[index].clone(), b(builder, BinOp::Mul, k2[index].clone(), builder.const_f32(2.0))), b(builder, BinOp::Add, b(builder, BinOp::Mul, k3[index].clone(), builder.const_f32(2.0)), k4[index].clone()));
            let next = b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, step.clone(), b(builder, BinOp::Div, weighted, builder.const_f32(6.0))));
            builder.store_local(state[index], next);
        }
        if detect_disk {
            let previous_z = b(
                builder,
                BinOp::Mul,
                values[1].clone(),
                u(builder, UnOp::Cos, values[2].clone()),
            );
            let next_z = b(
                builder,
                BinOp::Mul,
                Expr::LoadLocal(state[1]),
                u(builder, UnOp::Cos, Expr::LoadLocal(state[2])),
            );
            let product = b(builder, BinOp::Mul, previous_z.clone(), next_z.clone());
            let crossed = b(builder, BinOp::Le, product, builder.const_f32(0.0));
            let previous_nonzero = b(builder, BinOp::Ne, previous_z.clone(), builder.const_f32(0.0));
            let next_nonzero = b(builder, BinOp::Ne, next_z.clone(), builder.const_f32(0.0));
            let previous_abs = b(builder, BinOp::Max, previous_z.clone(), u(builder, UnOp::Neg, previous_z));
            let next_abs = b(builder, BinOp::Max, next_z.clone(), u(builder, UnOp::Neg, next_z));
            let alpha = b(builder, BinOp::Div, previous_abs.clone(), b(builder, BinOp::Add, previous_abs, next_abs));
            let radius_delta = b(builder, BinOp::Sub, Expr::LoadLocal(state[1]), values[1].clone());
            let crossing_radius = b(builder, BinOp::Add, values[1].clone(), b(builder, BinOp::Mul, alpha.clone(), radius_delta));
            let prev_sin_theta = u(builder, UnOp::Sin, values[2].clone());
            let prev_cos_theta = u(builder, UnOp::Cos, values[2].clone());
            let prev_sin_phi = u(builder, UnOp::Sin, values[3].clone());
            let prev_cos_phi = u(builder, UnOp::Cos, values[3].clone());
            let dx = b(builder, BinOp::Add,
                b(builder, BinOp::Mul, values[5].clone(), b(builder, BinOp::Mul, prev_sin_theta.clone(), prev_cos_phi.clone())),
                b(builder, BinOp::Sub,
                    b(builder, BinOp::Mul, b(builder, BinOp::Mul, values[1].clone(), values[6].clone()), b(builder, BinOp::Mul, prev_cos_theta.clone(), prev_cos_phi.clone())),
                    b(builder, BinOp::Mul, b(builder, BinOp::Mul, b(builder, BinOp::Mul, values[1].clone(), prev_sin_theta.clone()), prev_sin_phi.clone()), values[7].clone())));
            let dy = b(builder, BinOp::Add,
                b(builder, BinOp::Mul, values[5].clone(), b(builder, BinOp::Mul, prev_sin_theta.clone(), prev_sin_phi.clone())),
                b(builder, BinOp::Add,
                    b(builder, BinOp::Mul, b(builder, BinOp::Mul, values[1].clone(), values[6].clone()), b(builder, BinOp::Mul, prev_cos_theta, prev_sin_phi)),
                    b(builder, BinOp::Mul, b(builder, BinOp::Mul, b(builder, BinOp::Mul, values[1].clone(), prev_sin_theta.clone()), prev_cos_phi.clone()), values[7].clone())));
            let dz = b(builder, BinOp::Sub,
                b(builder, BinOp::Mul, values[5].clone(), u(builder, UnOp::Cos, values[2].clone())),
                b(builder, BinOp::Mul, b(builder, BinOp::Mul, values[1].clone(), u(builder, UnOp::Sin, values[2].clone())), values[6].clone()));
            let tangent_length = u(builder, UnOp::Sqrt, b(builder, BinOp::Add,
                b(builder, BinOp::Add, b(builder, BinOp::Mul, dx.clone(), dx.clone()), b(builder, BinOp::Mul, dy.clone(), dy.clone())),
                b(builder, BinOp::Mul, dz.clone(), dz.clone())));
            let phi_delta = b(builder, BinOp::Sub, Expr::LoadLocal(state[3]), values[3].clone());
            let crossing_phi = b(builder, BinOp::Add, values[3].clone(), b(builder, BinOp::Mul, alpha.clone(), phi_delta));
            let incoming_x = b(builder, BinOp::Div, u(builder, UnOp::Neg, dx), tangent_length.clone());
            let incoming_y = b(builder, BinOp::Div, u(builder, UnOp::Neg, dy), tangent_length);
            let cos_view = b(builder, BinOp::Add,
                b(builder, BinOp::Mul, u(builder, UnOp::Neg, u(builder, UnOp::Sin, crossing_phi.clone())), incoming_x),
                b(builder, BinOp::Mul, u(builder, UnOp::Cos, crossing_phi), incoming_y));
            let denominator = b(builder, BinOp::Max, builder.const_f32(0.05), b(builder, BinOp::Sub, builder.const_f32(1.0), b(builder, BinOp::Mul, cos_view, builder.const_f32(0.35))));
            let computed_doppler = b(builder, BinOp::Min, builder.const_f32(4.0), b(builder, BinOp::Max, builder.const_f32(0.25), b(builder, BinOp::Div, builder.const_f32(1.0), denominator)));
            let in_range = builder.select(
                b(builder, BinOp::Ge, crossing_radius.clone(), builder.load_uniform(3)),
                b(builder, BinOp::Le, crossing_radius.clone(), builder.load_uniform(4)),
                Expr::ConstBool(false),
            );
            let first_hit = b(builder, BinOp::Eq, Expr::LoadLocal(*disk_hit.as_ref().expect("disk hit local")), builder.const_f32(0.0));
            let record = builder.select(crossed, builder.select(previous_nonzero, builder.select(next_nonzero, builder.select(in_range, first_hit, Expr::ConstBool(false)), Expr::ConstBool(false)), Expr::ConstBool(false)), Expr::ConstBool(false));
            let record = builder.let_value(ucf_program::Ty::Scalar(ucf_program::ScalarTy::Bool), record);
            let record = Expr::LoadLocal(record);
            builder.store_local(
                *disk_hit.as_ref().expect("disk hit local"),
                builder.select(record.clone(), builder.const_f32(1.0), Expr::LoadLocal(*disk_hit.as_ref().expect("disk hit local"))),
            );
            builder.store_local(
                *disk_radius.as_ref().expect("disk radius local"),
                builder.select(record.clone(), crossing_radius, Expr::LoadLocal(*disk_radius.as_ref().expect("disk radius local"))),
            );
            builder.store_local(
                *disk_doppler.as_ref().expect("disk doppler local"),
                builder.select(record, computed_doppler, Expr::LoadLocal(*disk_doppler.as_ref().expect("disk doppler local"))),
            );
        }
        let mut finite = builder.unary(UnOp::IsFinite, Expr::LoadLocal(state[0]));
        for index in 1..8 {
            finite = builder.select(
                finite,
                builder.unary(UnOp::IsFinite, Expr::LoadLocal(state[index])),
                Expr::ConstBool(false),
            );
        }
        let finite = builder.let_value(ucf_program::Ty::Scalar(ucf_program::ScalarTy::Bool), finite);
        let finite = Expr::LoadLocal(finite);
        builder.store_local(
            status,
            builder.select(finite.clone(), Expr::LoadLocal(status), builder.const_f32(4.0)),
        );
        for index in 0..8 {
            builder.store_local(state[index], builder.select(finite.clone(), Expr::LoadLocal(state[index]), values[index].clone()));
        }
    });
    let r = Expr::LoadLocal(state[1]);
    let rs = b(&builder, BinOp::Mul, builder.const_f32(2.0), mass.clone());
    let captured = b(&builder, BinOp::Le, r.clone(), b(&builder, BinOp::Mul, rs, builder.const_f32(1.0001)));
    let escaped = builder.select(
        b(&builder, BinOp::Eq, Expr::LoadLocal(bound_infall), builder.const_f32(0.0)),
        b(&builder, BinOp::Ge, r, builder.load_uniform(2)),
        Expr::ConstBool(false),
    );
    let terminal_status = builder.select(
        b(&builder, BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(4.0)),
        builder.const_f32(4.0),
        builder.select(captured, builder.const_f32(1.0), builder.select(escaped, builder.const_f32(2.0), builder.const_f32(0.0))),
    );
    let terminal_status = builder.select(
        b(&builder, BinOp::Ne, Expr::LoadLocal(status), builder.const_f32(0.0)),
        Expr::LoadLocal(status),
        terminal_status,
    );
    let final_status = if report_budget_exhaustion {
        builder.select(
            b(&builder, BinOp::Eq, terminal_status.clone(), builder.const_f32(0.0)),
            builder.const_f32(3.0),
            terminal_status,
        )
    } else {
        terminal_status
    };
    for offset in 0..8usize {
        builder.store_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(offset as u32)), Expr::LoadLocal(state[offset]));
    }
    builder.store_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(8)), final_status);
    builder.store_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(9)), Expr::LoadLocal(bound_infall));
    if detect_disk {
        builder.store_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(10)), Expr::LoadLocal(*disk_hit.as_ref().expect("disk hit local")));
        builder.store_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(11)), Expr::LoadLocal(*disk_radius.as_ref().expect("disk radius local")));
        builder.store_buffer(0, b(&builder, BinOp::Add, base.clone(), builder.const_u32(12)), Expr::LoadLocal(*disk_doppler.as_ref().expect("disk doppler local")));
        builder.store_buffer(0, b(&builder, BinOp::Add, base, builder.const_u32(13)), builder.const_f32(0.0));
    }
    ProgramBuilder::new().structured_entry(builder, ucf_program::Ty::f32(), Expr::LoadLocal(state[0])).expect("valid structured integration builder").finish().expect("valid structured integration program")
}

/// 构建针孔相机射线程序。
pub fn generate_camera_ray() -> Program {
    let mut builder = FunctionBuilder::new("generate_camera_ray");
    let x = builder.emit_load_uniform(0, Ty::f64()).expect("entry block is open");
    let y = builder.emit_load_uniform(1, Ty::f64()).expect("entry block is open");
    let forward = builder.emit_load_uniform(2, Ty::vec3()).expect("entry block is open");
    let right = builder.emit_load_uniform(3, Ty::vec3()).expect("entry block is open");
    let up = builder.emit_load_uniform(4, Ty::vec3()).expect("entry block is open");
    let tan_half_fov = builder.emit_load_uniform(5, Ty::f64()).expect("entry block is open");
    let horizontal = builder.emit_binary(BinOp::Mul, x, tan_half_fov).expect("f64 arithmetic");
    let vertical = builder.emit_binary(BinOp::Mul, y, tan_half_fov).expect("f64 arithmetic");
    let horizontal_offset = builder.emit_binary(BinOp::Mul, horizontal, right).expect("vector arithmetic");
    let vertical_offset = builder.emit_binary(BinOp::Mul, vertical, up).expect("vector arithmetic");
    let direction = builder.emit_binary(BinOp::Add, forward, horizontal_offset).expect("vector arithmetic");
    let direction = builder.emit_binary(BinOp::Add, direction, vertical_offset).expect("vector arithmetic");
    let normalized = builder.emit_unary(UnOp::Normalize, direction, Ty::vec3()).expect("vector normalize");
    builder.return_value(normalized).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON ray program must build").finish().expect("TON ray program must validate")
}

/// 构建线性 HDR ACES 拟合程序。
pub fn tone_map_aces() -> Program {
    let mut builder = FunctionBuilder::new("tone_map_aces");
    let color = builder.emit_load_uniform(0, Ty::vec3()).expect("entry block is open");
    let channel = |builder: &mut FunctionBuilder, input: ucf_program::ValueId| {
        let zero = builder.emit_const_f64(0.0).expect("open block");
        let one = builder.emit_const_f64(1.0).expect("open block");
        let x = builder.emit_binary(BinOp::Max, input, zero).expect("f64 arithmetic");
        let a = builder.emit_const_f64(2.51).expect("open block");
        let b = builder.emit_const_f64(0.03).expect("open block");
        let c = builder.emit_const_f64(2.43).expect("open block");
        let d = builder.emit_const_f64(0.59).expect("open block");
        let e = builder.emit_const_f64(0.14).expect("open block");
        let ax = builder.emit_binary(BinOp::Mul, a, x).expect("f64 arithmetic");
        let ax_plus_b = builder.emit_binary(BinOp::Add, ax, b).expect("f64 arithmetic");
        let numerator = builder.emit_binary(BinOp::Mul, x, ax_plus_b).expect("f64 arithmetic");
        let cx = builder.emit_binary(BinOp::Mul, c, x).expect("f64 arithmetic");
        let cx_plus_d = builder.emit_binary(BinOp::Add, cx, d).expect("f64 arithmetic");
        let x_term = builder.emit_binary(BinOp::Mul, x, cx_plus_d).expect("f64 arithmetic");
        let denominator = builder.emit_binary(BinOp::Add, x_term, e).expect("f64 arithmetic");
        let mapped = builder.emit_binary(BinOp::Div, numerator, denominator).expect("f64 arithmetic");
        let mapped = builder.emit_binary(BinOp::Max, zero, mapped).expect("f64 arithmetic");
        builder.emit_binary(BinOp::Min, one, mapped).expect("f64 arithmetic")
    };
    let red = builder.emit_vec3_component(color, 0, Ty::f64()).expect("vec3 component");
    let green = builder.emit_vec3_component(color, 1, Ty::f64()).expect("vec3 component");
    let blue = builder.emit_vec3_component(color, 2, Ty::f64()).expect("vec3 component");
    let red = channel(&mut builder, red);
    let green = channel(&mut builder, green);
    let blue = channel(&mut builder, blue);
    let mapped = builder.emit_vec3(red, green, blue).expect("matching channel types");
    builder.return_value(mapped).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON tone map program must build").finish().expect("TON tone map program must validate")
}

/// 构建并行线性 HDR 到显示 RGB 的 ACES tone-map 程序。
///
/// 输入 buffer `1` 和输出 buffer `2` 均按每实例三个 `f32` 通道布局。
pub fn tone_map_aces_batch() -> Program {
    let mut builder = StructuredFunctionBuilder::new("tone_map_aces_batch");
    let global = builder.global_id();
    let input_base = builder.binary(BinOp::Mul, global.clone(), builder.const_u32(3));
    let output_base = builder.binary(BinOp::Mul, global, builder.const_u32(3));
    let zero = builder.const_f32(0.0);
    let one = builder.const_f32(1.0);
    let channel = |builder: &mut StructuredFunctionBuilder, input: Expr| {
        let x = builder.binary(BinOp::Max, input, zero.clone());
        let numerator = builder.binary(
            BinOp::Mul,
            x.clone(),
            builder.binary(BinOp::Add, builder.binary(BinOp::Mul, builder.const_f32(2.51), x.clone()), builder.const_f32(0.03)),
        );
        let denominator = builder.binary(
            BinOp::Add,
            builder.binary(BinOp::Mul, x.clone(), builder.binary(BinOp::Add, builder.binary(BinOp::Mul, builder.const_f32(2.43), x), builder.const_f32(0.59))),
            builder.const_f32(0.14),
        );
        builder.binary(BinOp::Min, one.clone(), builder.binary(BinOp::Max, zero.clone(), builder.binary(BinOp::Div, numerator, denominator)))
    };
    for component in 0..3u32 {
        let input = builder.load_buffer(1, builder.binary(BinOp::Add, input_base.clone(), builder.const_u32(component)));
        let mapped = channel(&mut builder, input);
        builder.store_buffer(2, builder.binary(BinOp::Add, output_base.clone(), builder.const_u32(component)), mapped);
    }
    ProgramBuilder::new().structured_entry(builder, Ty::f32(), zero).expect("valid batch tone map builder").finish().expect("valid batch tone map program")
}

/// 构建并行天空合成程序。
///
/// 输入 buffer `1` 为射线方向，buffer `2` 为 geodesic 状态，buffer `3` 为
/// 盘面 radiance。状态偏移 `8` 为终止码，只有逃逸码 `2` 叠加天空颜色。
/// 输出 buffer `4` 为合成后的线性 RGB。天空模型使用通用算术表达式展开。
pub fn compose_sky_batch() -> Program {
    let mut builder = StructuredFunctionBuilder::new("compose_sky_batch");
    let global = builder.global_id();
    let direction_base = builder.binary(BinOp::Mul, global.clone(), builder.const_u32(3));
    let state_base = builder.binary(BinOp::Mul, global.clone(), builder.const_u32(14));
    let radiance_base = builder.binary(BinOp::Mul, global.clone(), builder.const_u32(3));
    let output_base = builder.binary(BinOp::Mul, global, builder.const_u32(3));
    let _dx = builder.load_buffer(1, builder.binary(BinOp::Add, direction_base.clone(), builder.const_u32(0)));
    let dy = builder.load_buffer(1, builder.binary(BinOp::Add, direction_base.clone(), builder.const_u32(1)));
    let _dz = builder.load_buffer(1, builder.binary(BinOp::Add, direction_base, builder.const_u32(2)));
    let status = builder.load_buffer(2, builder.binary(BinOp::Add, state_base, builder.const_u32(8)));
    let radiance_r = builder.load_buffer(3, builder.binary(BinOp::Add, radiance_base.clone(), builder.const_u32(0)));
    let radiance_g = builder.load_buffer(3, builder.binary(BinOp::Add, radiance_base.clone(), builder.const_u32(1)));
    let radiance_b = builder.load_buffer(3, builder.binary(BinOp::Add, radiance_base, builder.const_u32(2)));
    let t = builder.binary(BinOp::Max, builder.const_f32(0.0), builder.binary(BinOp::Min, builder.const_f32(1.0), builder.binary(BinOp::Add, builder.binary(BinOp::Mul, dy, builder.const_f32(0.5)), builder.const_f32(0.5))));
    let sky_r = builder.binary(BinOp::Mul, builder.const_f32(0.02), t.clone());
    let sky_g = builder.binary(BinOp::Mul, builder.const_f32(0.03), t.clone());
    let sky_b = builder.binary(BinOp::Add, builder.const_f32(0.08), builder.binary(BinOp::Mul, builder.const_f32(0.12), t));
    let escaped = builder.binary(BinOp::Eq, status, builder.const_f32(2.0));
    let out_r = builder.select(escaped.clone(), builder.binary(BinOp::Add, sky_r, radiance_r.clone()), radiance_r);
    let out_g = builder.select(escaped.clone(), builder.binary(BinOp::Add, sky_g, radiance_g.clone()), radiance_g);
    let out_b = builder.select(escaped, builder.binary(BinOp::Add, sky_b, radiance_b.clone()), radiance_b);
    builder.store_buffer(4, builder.binary(BinOp::Add, output_base.clone(), builder.const_u32(0)), out_r);
    builder.store_buffer(4, builder.binary(BinOp::Add, output_base.clone(), builder.const_u32(1)), out_g);
    builder.store_buffer(4, builder.binary(BinOp::Add, output_base, builder.const_u32(2)), out_b);
    ProgramBuilder::new().structured_entry(builder, Ty::f32(), Expr::ConstF32(0.0)).expect("valid sky composite builder").finish().expect("valid sky composite program")
}

/// 构建泛光亮部提取程序。
pub fn post_bloom_extract() -> Program {
    let mut builder = FunctionBuilder::new("post_bloom_extract");
    let color = builder.emit_load_uniform(0, Ty::vec3()).expect("entry block is open");
    let threshold = builder.emit_load_uniform(1, Ty::f64()).expect("entry block is open");
    let red = builder.emit_vec3_component(color, 0, Ty::f64()).expect("vec3 component");
    let green = builder.emit_vec3_component(color, 1, Ty::f64()).expect("vec3 component");
    let blue = builder.emit_vec3_component(color, 2, Ty::f64()).expect("vec3 component");
    let lum_r_weight = builder.emit_const_f64(0.2126).expect("open block");
    let lum_g_weight = builder.emit_const_f64(0.7152).expect("open block");
    let lum_b_weight = builder.emit_const_f64(0.0722).expect("open block");
    let lum_r = builder.emit_binary(BinOp::Mul, lum_r_weight, red).expect("f64 arithmetic");
    let lum_g = builder.emit_binary(BinOp::Mul, lum_g_weight, green).expect("f64 arithmetic");
    let lum_b = builder.emit_binary(BinOp::Mul, lum_b_weight, blue).expect("f64 arithmetic");
    let lum_rg = builder.emit_binary(BinOp::Add, lum_r, lum_g).expect("f64 arithmetic");
    let lum = builder.emit_binary(BinOp::Add, lum_rg, lum_b).expect("f64 arithmetic");
    let zero = builder.emit_const_f64(0.0).expect("open block");
    let epsilon = builder.emit_const_f64(1.0e-9).expect("open block");
    let scale_numerator = builder.emit_binary(BinOp::Sub, lum, threshold).expect("f64 arithmetic");
    let scale_denominator = builder.emit_binary(BinOp::Max, lum, epsilon).expect("f64 arithmetic");
    let scale = builder.emit_binary(BinOp::Div, scale_numerator, scale_denominator).expect("f64 arithmetic");
    let bright_red = builder.emit_binary(BinOp::Mul, red, scale).expect("f64 arithmetic");
    let bright_green = builder.emit_binary(BinOp::Mul, green, scale).expect("f64 arithmetic");
    let bright_blue = builder.emit_binary(BinOp::Mul, blue, scale).expect("f64 arithmetic");
    let bright = builder.emit_vec3(bright_red, bright_green, bright_blue).expect("matching channel types");
    let black = builder.emit_vec3(zero, zero, zero).expect("matching channel types");
    let active = builder.emit_binary(BinOp::Gt, lum, threshold).expect("f64 comparison");
    let result = builder.emit_select(active, bright, black).expect("matching vector types");
    builder.return_value(result).expect("entry block is open");
    ProgramBuilder::new().entry(builder).expect("TON bloom program must build").finish().expect("TON bloom program must validate")
}

/// 构建 Novikov-Thorne 工程近似的盘面颜色程序。
pub fn disk_shade_novikov() -> Program {
    let radius = uni(0);
    let temp = bin(
        BinOp::Mul,
        bin(
            BinOp::Mul,
            bin(BinOp::Div, uni(5), bin(BinOp::Mul, uni(4), uni(4))),
            bin(BinOp::Pow, bin(BinOp::Div, uni(2), radius.clone()), c(0.75)),
        ),
        c(2800.0),
    );
    let t = bin(BinOp::Mul, temp.clone(), c(0.01));
    let cool_b = select(bin(BinOp::Le, t.clone(), c(19.0)), c(0.0), bin(BinOp::Sub, bin(BinOp::Mul, c(138.5177312231), un(UnOp::Ln, bin(BinOp::Sub, t.clone(), c(10.0)))), c(305.0447927307)));
    let cool = v3(c(255.0), bin(BinOp::Sub, bin(BinOp::Mul, c(99.4708025861), un(UnOp::Ln, t.clone())), c(161.1195681661)), cool_b);
    let warm = v3(bin(BinOp::Mul, c(329.698727446), bin(BinOp::Pow, bin(BinOp::Sub, t.clone(), c(60.0)), c(-0.1332047592))), bin(BinOp::Mul, c(288.1221695283), bin(BinOp::Pow, bin(BinOp::Sub, t.clone(), c(60.0)), c(-0.0755148492))), c(255.0));
    let rgb = select(bin(BinOp::Le, t, c(66.0)), cool, warm);
    let scale = bin(
        BinOp::Mul,
        bin(BinOp::Pow, bin(BinOp::Max, c(0.0), bin(BinOp::Min, uni(1), c(3.0))), c(3.0)),
        c(0.5),
    );
    f("disk_shade_novikov", vec![Stmt::If { cond: bin(BinOp::Lt, radius.clone(), uni(2)), then_body: vec![Stmt::Return(v3(c(0.0), c(0.0), c(0.0)))], else_body: Vec::new() }, Stmt::If { cond: bin(BinOp::Gt, radius, uni(3)), then_body: vec![Stmt::Return(v3(c(0.0), c(0.0), c(0.0)))], else_body: Vec::new() }, Stmt::Return(bin(BinOp::Mul, v3(bin(BinOp::Div, un(UnOp::Vec3Comp(0), rgb.clone()), c(255.0)), bin(BinOp::Div, un(UnOp::Vec3Comp(1), rgb.clone()), c(255.0)), bin(BinOp::Div, un(UnOp::Vec3Comp(2), rgb), c(255.0))), scale))], Vec::new(), Ty::vec3())
}

/// 构建并行 Novikov-Thorne 盘面着色程序。
///
/// 输入资源 `1` 每实例为 `[radius, doppler]`，输出资源 `2` 每实例为线性
/// `[r, g, b]`。uniform `0..3` 依次为内半径、外半径、质量和吸积率。
/// 该程序是 TON 前端生成的普通 UCF 程序，不是 UCF 领域算子。
pub fn disk_shade_novikov_batch() -> Program {
    disk_shade_novikov_from_state_batch_with_name(
        "disk_shade_novikov_batch",
        2,
        0,
        1,
    )
}

/// 构建从测地线状态资源读取盘面交点的批量着色程序。
///
/// 输入每实例布局为十四个 `f32`，radius 位于偏移 11，Doppler 位于偏移 12。
pub fn disk_shade_novikov_from_geodesic_batch() -> Program {
    disk_shade_novikov_from_state_batch_with_name(
        "disk_shade_novikov_from_geodesic_batch",
        14,
        11,
        12,
    )
}

fn disk_shade_novikov_from_state_batch_with_name(
    name: &str,
    input_stride: u32,
    radius_offset: u32,
    doppler_offset: u32,
) -> Program {
    let mut builder = StructuredFunctionBuilder::new(name);
    let global = builder.global_id();
    let index = |base: Expr, offset: u32| bin(BinOp::Add, base, Expr::ConstU32(offset));
    let input_base = builder.binary(BinOp::Mul, global.clone(), builder.const_u32(input_stride));
    let output_base = builder.binary(BinOp::Mul, global, builder.const_u32(3));
    let radius = builder.load_buffer(1, index(input_base.clone(), radius_offset));
    let doppler = builder.load_buffer(1, index(input_base, doppler_offset));
    let inner = builder.load_uniform(0);
    let outer = builder.load_uniform(1);
    let mass = builder.load_uniform(2);
    let accretion = builder.load_uniform(3);
    let mass_squared = builder.binary(BinOp::Mul, mass.clone(), mass.clone());
    let temperature_base = builder.binary(BinOp::Div, accretion, mass_squared);
    let radius_ratio = builder.binary(BinOp::Div, inner.clone(), radius.clone());
    let radial_power = builder.binary(BinOp::Pow, radius_ratio, builder.const_f32(0.75));
    let temperature_scaled = builder.binary(BinOp::Mul, temperature_base, radial_power);
    let temperature = builder.binary(BinOp::Mul, temperature_scaled, builder.const_f32(2800.0));
    let t = builder.binary(BinOp::Mul, temperature, builder.const_f32(0.01));
    let t_minus_10 = builder.binary(BinOp::Sub, t.clone(), builder.const_f32(10.0));
    let cool_blue_raw = builder.binary(
        BinOp::Sub,
        builder.binary(BinOp::Mul, builder.const_f32(138.5177), builder.unary(UnOp::Ln, t_minus_10)),
        builder.const_f32(305.0448),
    );
    let cool_blue = builder.select(
        builder.binary(BinOp::Le, t.clone(), builder.const_f32(19.0)),
        builder.const_f32(0.0),
        cool_blue_raw,
    );
    let cool_green = builder.binary(
        BinOp::Sub,
        builder.binary(BinOp::Mul, builder.const_f32(99.4708), builder.unary(UnOp::Ln, t.clone())),
        builder.const_f32(161.1196),
    );
    let cool = v3(builder.const_f32(255.0), cool_green, cool_blue);
    let t_minus_60 = builder.binary(BinOp::Sub, t.clone(), builder.const_f32(60.0));
    let warm_red = builder.binary(
        BinOp::Mul,
        builder.const_f32(329.6987),
        builder.binary(BinOp::Pow, t_minus_60.clone(), builder.const_f32(-0.1332)),
    );
    let warm_green = builder.binary(
        BinOp::Mul,
        builder.const_f32(288.1222),
        builder.binary(BinOp::Pow, t_minus_60, builder.const_f32(-0.0755)),
    );
    let warm = v3(warm_red, warm_green, builder.const_f32(255.0));
    let rgb = builder.select(builder.binary(BinOp::Le, t.clone(), builder.const_f32(66.0)), cool, warm);
    let clamped_low = builder.binary(BinOp::Max, builder.const_f32(0.0), doppler);
    let clamped = builder.binary(BinOp::Min, clamped_low, builder.const_f32(3.0));
    let doppler_squared = builder.binary(BinOp::Mul, clamped.clone(), clamped.clone());
    let doppler_cubed = builder.binary(BinOp::Mul, doppler_squared, clamped);
    let scale = builder.binary(BinOp::Mul, doppler_cubed, builder.const_f32(0.5));
    let zero_color = v3(builder.const_f32(0.0), builder.const_f32(0.0), builder.const_f32(0.0));
    let false_value = builder.binary(BinOp::Eq, builder.const_f32(0.0), builder.const_f32(1.0));
    let inside = builder.select(builder.binary(BinOp::Lt, radius.clone(), inner), false_value, builder.binary(BinOp::Le, radius, outer));
    let rgb = builder.select(inside, rgb, zero_color);
    for component in 0u8..3 {
        let channel = un(UnOp::Vec3Comp(component), rgb.clone());
        let normalized = builder.binary(BinOp::Div, channel, builder.const_f32(255.0));
        let value = builder.binary(BinOp::Mul, normalized, scale.clone());
        builder.store_buffer(2, index(output_base.clone(), component as u32), value);
    }
    ProgramBuilder::new().structured_entry(builder, Ty::f32(), scale).expect("valid batch disk builder").finish().expect("valid batch disk program")
}

/// 构建并行步数验证程序。
pub fn ray_batch_steps() -> Program {
    f("ray_batch_steps", vec![Stmt::Let { id: 0, ty: Ty::f64(), init: c(0.0) }, Stmt::Let { id: 1, ty: Ty::f64(), init: bin(BinOp::Add, bin(BinOp::Mod, Expr::GlobalId, c(16.0)), c(1.0)) }, Stmt::For { iter: 2, limit: 32, body: vec![Stmt::If { cond: bin(BinOp::Lt, Expr::LoadLocal(2), Expr::LoadLocal(1)), then_body: vec![Stmt::StoreLocal { id: 0, value: bin(BinOp::Add, Expr::LoadLocal(0), c(1.0)) }], else_body: Vec::new() }] }, Stmt::Return(Expr::LoadLocal(0))], vec![(0, Ty::f64()), (1, Ty::f64())], Ty::f64())
}

/// 构建通用有界累加程序。
pub fn orbit_loop_accumulate() -> Program {
    f("orbit_loop_accumulate", vec![Stmt::Let { id: 0, ty: Ty::f64(), init: c(0.0) }, Stmt::For { iter: 1, limit: 64, body: vec![Stmt::StoreLocal { id: 0, value: bin(BinOp::Add, Expr::LoadLocal(0), Expr::LoadLocal(1)) }] }, Stmt::Return(Expr::LoadLocal(0))], vec![(0, Ty::f64())], Ty::f64())
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn schwarzschild_initial_state_program_is_valid_and_stable() {
        let program = schwarzschild_initial_state_batch();
        validate_program(&program).expect("initial state program must satisfy UCF validation");
        assert_eq!(program.functions[0].name, "schwarzschild_initial_state_batch");
        assert_ne!(program_hash(&program), 0);
    }

    #[test]
    fn structured_geodesic_integration_writes_state_and_status() {
        let program = schwarzschild_geodesic_rk4_integrate(4);
        let state = vec![
            Value::F32(0.0),
            Value::F32(10.0),
            Value::F32(1.1),
            Value::F32(0.3),
            Value::F32(1.2),
            Value::F32(-0.7),
            Value::F32(0.02),
            Value::F32(0.01),
            Value::F32(0.0),
            Value::F32(0.0),
        ];
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F32(1.0));
        interpreter.with_uniform(1, Value::F32(0.01));
        interpreter.with_uniform(2, Value::F32(100.0));
        interpreter.with_buffer(0, state);
        let result = interpreter.run_parallel(&program, 1).expect("structured integration CPU execution");
        assert!(result[0].value.as_f64().expect("return value").is_finite());
        let output = interpreter.buffer(0).expect("state output");
        assert_eq!(output.len(), 10);
        assert_eq!(output[8], Value::F32(3.0));
        assert!((output[1].as_f64().expect("radius") - 9.972).abs() < 0.1);
    }

    #[test]
    fn structured_geodesic_integration_reports_non_finite_initial_state() {
        let program = schwarzschild_geodesic_rk4_integrate(4);
        let mut state = vec![Value::F32(0.0); 10];
        state[1] = Value::F32(f32::NAN);
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F32(1.0));
        interpreter.with_uniform(1, Value::F32(0.01));
        interpreter.with_uniform(2, Value::F32(100.0));
        interpreter.with_buffer(0, state);
        interpreter.run_parallel(&program, 1).expect("non-finite state execution");
        assert_eq!(interpreter.buffer(0).expect("state output")[8], Value::F32(4.0));
    }

    #[test]
    fn disk_geodesic_chunk_validates_and_preserves_extended_layout() {
        let program = schwarzschild_geodesic_rk4_chunk_with_disk(4);
        validate_program(&program).expect("disk geodesic program must satisfy UCF validation");
        let state = vec![
            Value::F32(0.0),
            Value::F32(12.0),
            Value::F32(1.3),
            Value::F32(0.3),
            Value::F32(1.2),
            Value::F32(-0.7),
            Value::F32(0.22),
            Value::F32(0.01),
            Value::F32(0.0),
            Value::F32(0.0),
            Value::F32(0.0),
            Value::F32(0.0),
            Value::F32(0.0),
            Value::F32(0.0),
        ];
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F32(1.0));
        interpreter.with_uniform(1, Value::F32(0.02));
        interpreter.with_uniform(2, Value::F32(100.0));
        interpreter.with_uniform(3, Value::F32(6.0));
        interpreter.with_uniform(4, Value::F32(40.0));
        interpreter.with_buffer(0, state);
        interpreter.run_parallel(&program, 1).expect("disk geodesic execution");
        assert_eq!(interpreter.buffer(0).expect("extended state output").len(), 14);
    }

    #[test]
    fn disk_geodesic_chunk_preserves_terminal_metadata_across_dispatches() {
        let program = schwarzschild_geodesic_rk4_chunk_with_disk(4);
        for status in [1.0, 2.0, 3.0, 4.0] {
            let state = [1.0, 20.0, 1.0, 0.2, 1.0, 0.5, 0.0, 0.0, status, 0.0, 1.0, 8.0, 1.1, 0.0]
                .into_iter().map(Value::F32).collect::<Vec<_>>();
            let mut interpreter = CpuInterpreter::new();
            for (slot, value) in [1.0, 0.01, 100.0, 6.0, 30.0].into_iter().enumerate() {
                interpreter.with_uniform(slot as u32, Value::F32(value));
            }
            interpreter.with_buffer(0, state.clone());
            for _ in 0..2 {
                interpreter.run_parallel(&program, 1).expect("terminal disk chunk execution");
                assert_eq!(interpreter.buffer(0).expect("terminal metadata"), &state);
            }
        }
    }

    #[test]
    fn disk_geodesic_chunk_detects_crossing_between_distinct_states() {
        let mut interpreter = CpuInterpreter::new();
        let input = [0.0, 12.0, 1.56, 0.3, 1.2, -0.7, 0.22, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            .into_iter().map(Value::F32).collect::<Vec<_>>();
        for (slot, value) in [1.0, 0.1, 100.0, 6.0, 40.0].into_iter().enumerate() {
            interpreter.with_uniform(slot as u32, Value::F32(value));
        }
        interpreter.with_buffer(0, input);
        interpreter.run_parallel(&schwarzschild_geodesic_rk4_chunk_with_disk(1), 1).unwrap();
        let output = interpreter.buffer(0).unwrap();
        assert!(output[2].as_f64().unwrap() > std::f64::consts::FRAC_PI_2);
        assert_eq!(output[10], Value::F32(1.0));
        let radius = output[11].as_f64().unwrap();
        assert!(radius > 11.8 && radius < 12.0, "crossing radius={radius}");
    }

    #[test]
    fn geodesic_chunk_stops_at_capture_and_escape_boundaries() {
        for (radius, radial_velocity, expected_status) in [(2.0001, -1.0, 1.0), (100.0, 1.0, 2.0)] {
            let state = [0.0, radius, 1.0, 0.0, 1.0, radial_velocity, 0.0, 0.0, 0.0, 0.0]
                .into_iter().map(Value::F32).collect::<Vec<_>>();
            let mut interpreter = CpuInterpreter::new();
            for (slot, value) in [1.0, 0.1, 100.0].into_iter().enumerate() {
                interpreter.with_uniform(slot as u32, Value::F32(value));
            }
            interpreter.with_buffer(0, state.clone());
            interpreter.run_parallel(&schwarzschild_geodesic_rk4_chunk(64), 1).unwrap();
            let output = interpreter.buffer(0).unwrap();
            assert_eq!(&output[..8], &state[..8]);
            assert_eq!(output[8], Value::F32(expected_status));
        }
    }

    #[test]
    fn geodesic_chunk_preserves_last_finite_state_on_step_failure() {
        let state = [0.0, 4.0, 1.0, 0.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0]
            .into_iter().map(Value::F32).collect::<Vec<_>>();
        let mut interpreter = CpuInterpreter::new();
        for (slot, value) in [1.0, f32::INFINITY, 100.0].into_iter().enumerate() {
            interpreter.with_uniform(slot as u32, Value::F32(value));
        }
        interpreter.with_buffer(0, state.clone());
        interpreter.run_parallel(&schwarzschild_geodesic_rk4_chunk(4), 1).unwrap();
        let output = interpreter.buffer(0).unwrap();
        assert_eq!(output[8], Value::F32(4.0));
        assert_eq!(&output[..8], &state[..8]);
    }

    #[test]
    fn initial_state_distinguishes_capture_scattering_and_outgoing_rays() {
        let input = [
            0.0, 0.0, 20.0, 0.0, 0.0, -1.0,
            0.0, 0.0, 20.0, 0.4, 0.0, -1.0,
            0.0, 0.0, 20.0, 0.0, 0.0, 1.0,
        ].into_iter().map(Value::F32).collect::<Vec<_>>();
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F32(1.0));
        interpreter.with_uniform(1, Value::F32(3.0 * 3.0f32.sqrt()));
        interpreter.with_buffer(1, input);
        interpreter.with_buffer(2, vec![Value::F32(0.0); 30]);
        interpreter.run_parallel(&schwarzschild_initial_state_batch(), 3).unwrap();
        let output = interpreter.buffer(2).unwrap();
        assert_eq!(output[9], Value::F32(1.0));
        assert_eq!(output[19], Value::F32(0.0));
        assert_eq!(output[29], Value::F32(0.0));
    }

    #[test]
    fn geodesic_state_disk_shade_reads_extended_offsets() {
        let program = disk_shade_novikov_from_geodesic_batch();
        validate_program(&program).expect("state disk shade must satisfy UCF validation");
        let mut input = vec![Value::F32(0.0); 14];
        input[11] = Value::F32(12.0);
        input[12] = Value::F32(1.5);
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F32(6.0));
        interpreter.with_uniform(1, Value::F32(40.0));
        interpreter.with_uniform(2, Value::F32(1.0));
        interpreter.with_uniform(3, Value::F32(1.0));
        interpreter.with_buffer(1, input);
        interpreter.with_buffer(2, vec![Value::F32(0.0); 3]);
        interpreter.run_parallel(&program, 1).expect("state disk shade execution");
        assert!(interpreter.buffer(2).expect("radiance output").iter().any(|value| value.as_f64().expect("f32") > 0.0));
    }

    #[test]
    fn batch_tone_map_validates_and_maps_hdr_channels() {
        let program = tone_map_aces_batch();
        validate_program(&program).expect("batch tone map must satisfy UCF validation");
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_buffer(1, vec![Value::F32(4.0), Value::F32(2.0), Value::F32(0.5)]);
        interpreter.with_buffer(2, vec![Value::F32(0.0); 3]);
        interpreter.run_parallel(&program, 1).expect("batch tone map execution");
        let output = interpreter.buffer(2).expect("display output");
        assert!(output.iter().all(|value| {
            let value = value.as_f64().expect("f32 output");
            (0.0..=1.0).contains(&value)
        }));
        assert!(output[0].as_f64().expect("red") > 0.9);
    }

    #[test]
    fn sky_composite_validates_and_adds_escape_background() {
        let program = compose_sky_batch();
        validate_program(&program).expect("sky composite must satisfy UCF validation");
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_buffer(1, vec![Value::F32(0.0), Value::F32(1.0), Value::F32(0.0)]);
        interpreter.with_buffer(2, vec![Value::F32(0.0); 14]);
        interpreter.with_buffer(3, vec![Value::F32(0.1), Value::F32(0.2), Value::F32(0.3)]);
        let mut state = interpreter.buffer(2).expect("state").to_vec();
        state[8] = Value::F32(2.0);
        interpreter.with_buffer(2, state);
        interpreter.with_buffer(4, vec![Value::F32(0.0); 3]);
        interpreter.run_parallel(&program, 1).expect("sky composite execution");
        let output = interpreter.buffer(4).expect("composite output");
        assert!(output[0].as_f64().expect("red") > 0.1);
        assert!(output[2].as_f64().expect("blue") > 0.3);
    }

    #[test]
    fn geodesic_chunk_keeps_running_state_distinct_from_budget_failure() {
        let program = schwarzschild_geodesic_rk4_chunk(1);
        let state = vec![
            Value::F32(0.0), Value::F32(10.0), Value::F32(1.1), Value::F32(0.3),
            Value::F32(1.2), Value::F32(-0.7), Value::F32(0.02), Value::F32(0.01),
            Value::F32(0.0), Value::F32(0.0),
        ];
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F32(1.0));
        interpreter.with_uniform(1, Value::F32(0.01));
        interpreter.with_uniform(2, Value::F32(100.0));
        interpreter.with_buffer(0, state);
        interpreter.run_parallel(&program, 1).expect("chunk execution");
        assert_eq!(interpreter.buffer(0).expect("state output")[8], Value::F32(0.0));
    }

    #[test]
    fn geodesic_chunk_preserves_terminal_instance_state() {
        let program = schwarzschild_geodesic_rk4_chunk(4);
        let state = vec![
            Value::F32(0.0), Value::F32(10.0), Value::F32(1.1), Value::F32(0.3),
            Value::F32(1.2), Value::F32(-0.7), Value::F32(0.02), Value::F32(0.01),
            Value::F32(2.0), Value::F32(0.0),
        ];
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F32(1.0));
        interpreter.with_uniform(1, Value::F32(0.01));
        interpreter.with_uniform(2, Value::F32(100.0));
        interpreter.with_buffer(0, state);
        interpreter.run_parallel(&program, 1).expect("terminal chunk execution");
        let output = interpreter.buffer(0).expect("state output");
        assert_eq!(output[8], Value::F32(2.0));
        assert_eq!(output[1], Value::F32(10.0));
    }

    #[test]
    fn equatorial_orbit_program_matches_independent_rk4_reference() {
        fn rhs(mass: f64, spin: f64, value: f64) -> f64 {
            3.0 * mass * value * value
                - value
                - spin * spin * value.powi(4) * (3.0 * mass * value - 2.0)
        }
        fn reference(mass: f64, spin: f64, u: f64, du: f64, phi: f64, step: f64) -> [f64; 3] {
            let half = step * 0.5;
            let k1_u = du;
            let k1_du = rhs(mass, spin, u);
            let k2_u = du + half * k1_du;
            let k2_du = rhs(mass, spin, u + half * k1_u);
            let k3_u = du + half * k2_du;
            let k3_du = rhs(mass, spin, u + half * k2_u);
            let k4_u = du + step * k3_du;
            let k4_du = rhs(mass, spin, u + step * k3_u);
            [
                u + step * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u) / 6.0,
                du + step * (k1_du + 2.0 * k2_du + 2.0 * k3_du + k4_du) / 6.0,
                phi + step,
            ]
        }

        let state = [0.08, -0.01, 0.3];
        let mass = 1.0;
        let spin = 0.5;
        let step = 0.02;
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_uniform(0, Value::F64(mass));
        interpreter.with_uniform(1, Value::F64(spin));
        interpreter.with_uniform(2, Value::F64(step));
        interpreter.with_buffer(0, state.into_iter().map(Value::F64).collect());
        interpreter
            .run_parallel(&equatorial_orbit_rk4_step(), 1)
            .expect("equatorial orbit execution");
        let output = interpreter.buffer(0).expect("orbit state output");
        let expected = reference(mass, spin, state[0], state[1], state[2], step);
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual.as_f64().expect("f64 output") - expected).abs() < 1.0e-12);
        }
    }
}

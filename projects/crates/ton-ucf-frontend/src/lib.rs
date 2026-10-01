//! TON 领域前端。
//!
//! 本 crate 负责把 TON 的领域算法展开为通用 UCF Program。UCF 核心只
//! 接收 `Program`、校验并执行，不依赖黑洞类型或物理名称。

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
    schwarzschild_geodesic_rk4_integrate_mode(max_steps, true)
}

/// 构建一个有界积分 chunk。未达到物理终止条件时返回状态码 `0=running`，
/// 由外部任务图决定是否提交下一个 chunk。
pub fn schwarzschild_geodesic_rk4_chunk(max_steps: u32) -> Program {
    schwarzschild_geodesic_rk4_integrate_mode(max_steps, false)
}

fn schwarzschild_geodesic_rk4_integrate_mode(max_steps: u32, report_budget_exhaustion: bool) -> Program {
    fn b(builder: &StructuredFunctionBuilder, op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        builder.binary(op, lhs, rhs)
    }
    fn u(builder: &StructuredFunctionBuilder, op: UnOp, arg: Expr) -> Expr {
        builder.unary(op, arg)
    }
    fn derivative(builder: &StructuredFunctionBuilder, values: &[Expr; 8], mass: &Expr) -> [Expr; 8] {
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
        let compactness = b(builder, BinOp::Mul, two_mass, inv_r.clone());
        let f = b(builder, BinOp::Sub, one.clone(), compactness);
        let fp = b(builder, BinOp::Mul, builder.binary(BinOp::Mul, builder.const_f32(2.0), mass.clone()), inv_r2.clone());
        let two_f = b(builder, BinOp::Mul, two.clone(), f.clone());
        let gamma_t_tr = b(builder, BinOp::Div, fp.clone(), two_f);
        let du_t = u(builder, UnOp::Neg, b(builder, BinOp::Mul, two.clone(), b(builder, BinOp::Mul, gamma_t_tr, b(builder, BinOp::Mul, ut.clone(), ur.clone()))));
        let gamma_r_tt = b(builder, BinOp::Mul, f.clone(), b(builder, BinOp::Mul, mass.clone(), inv_r2.clone()));
        let gamma_r_rr = u(builder, UnOp::Neg, b(builder, BinOp::Div, fp, b(builder, BinOp::Mul, builder.const_f32(2.0), f.clone())));
        let sin_theta = b(builder, BinOp::Max, u(builder, UnOp::Sin, theta.clone()), builder.const_f32(1.0e-6));
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
        let phi = b(builder, BinOp::Add,
            b(builder, BinOp::Mul, b(builder, BinOp::Mul, builder.const_f32(2.0), builder.binary(BinOp::Div, builder.const_f32(1.0), values[1].clone())), b(builder, BinOp::Mul, ur.clone(), uphi.clone())),
            b(builder, BinOp::Mul, b(builder, BinOp::Mul, builder.const_f32(2.0), b(builder, BinOp::Div, u(builder, UnOp::Cos, values[2].clone()), b(builder, BinOp::Max, u(builder, UnOp::Sin, values[2].clone()), builder.const_f32(1.0e-6)))), b(builder, BinOp::Mul, utheta.clone(), uphi.clone())));
        [ut, ur, utheta, uphi, du_t, du_r, du_theta, u(builder, UnOp::Neg, phi)]
    }
    let mut builder = StructuredFunctionBuilder::new("schwarzschild_geodesic_rk4_integrate");
    let gid = builder.global_id();
    let base = b(&builder, BinOp::Mul, gid, builder.const_u32(10));
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
        builder.select(
            builder.binary(BinOp::Eq, Expr::LoadLocal(status), builder.const_f32(0.0)),
            builder.binary(BinOp::Gt, Expr::LoadLocal(state[1]), horizon.clone()),
            Expr::ConstBool(false),
        )
    };
    builder.bounded_while(active(&builder), max_steps, |builder| {
        let values: [Expr; 8] = std::array::from_fn(|index| Expr::LoadLocal(state[index]));
        let half_step = b(builder, BinOp::Mul, step.clone(), builder.const_f32(0.5));
        let add_scaled = |builder: &StructuredFunctionBuilder, values: &[Expr; 8], slope: &[Expr; 8], scale: Expr| {
            std::array::from_fn(|index| b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, slope[index].clone(), scale.clone())))
        };
        let k1 = derivative(builder, &values, &mass);
        let mid1 = add_scaled(builder, &values, &k1, half_step.clone());
        let k2 = derivative(builder, &mid1, &mass);
        let mid2 = add_scaled(builder, &values, &k2, half_step);
        let k3 = derivative(builder, &mid2, &mass);
        let end = add_scaled(builder, &values, &k3, step.clone());
        let k4 = derivative(builder, &end, &mass);
        for index in 0..8 {
            let weighted = b(builder, BinOp::Add, b(builder, BinOp::Add, k1[index].clone(), b(builder, BinOp::Mul, k2[index].clone(), builder.const_f32(2.0))), b(builder, BinOp::Add, b(builder, BinOp::Mul, k3[index].clone(), builder.const_f32(2.0)), k4[index].clone()));
            let next = b(builder, BinOp::Add, values[index].clone(), b(builder, BinOp::Mul, step.clone(), b(builder, BinOp::Div, weighted, builder.const_f32(6.0))));
            builder.store_local(state[index], next);
        }
        let mut finite = builder.unary(UnOp::IsFinite, Expr::LoadLocal(state[0]));
        for index in 1..8 {
            finite = builder.select(
                finite,
                builder.unary(UnOp::IsFinite, Expr::LoadLocal(state[index])),
                Expr::ConstBool(false),
            );
        }
        builder.store_local(
            status,
            builder.select(finite, Expr::LoadLocal(status), builder.const_f32(4.0)),
        );
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
    builder.store_buffer(0, b(&builder, BinOp::Add, base, builder.const_u32(9)), Expr::LoadLocal(bound_infall));
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
}

//! 守恒能量零测地线模型的通用表达式展开。

use super::{b, BinOp, Expr, StructuredFunctionBuilder, UnOp};

pub(crate) fn derivative(builder: &StructuredFunctionBuilder, state: &[Expr; 8], mass: &Expr) -> [Expr; 8] {
    let radius = state[1].clone();
    let theta = state[2].clone();
    let ur = state[5].clone();
    let utheta = state[6].clone();
    let uphi = state[7].clone();
    let f = b(builder, BinOp::Div, b(builder, BinOp::Sub, radius.clone(),
        b(builder, BinOp::Mul, builder.const_f32(2.0), mass.clone())), radius.clone());
    let sin = builder.unary(UnOp::Sin, theta.clone());
    let cos = builder.unary(UnOp::Cos, theta);
    let uphi2 = b(builder, BinOp::Mul, uphi.clone(), uphi.clone());
    let angular = b(builder, BinOp::Add, b(builder, BinOp::Mul, utheta.clone(), utheta.clone()),
        b(builder, BinOp::Mul, b(builder, BinOp::Mul, sin.clone(), sin.clone()), uphi2.clone()));
    let radial = b(builder, BinOp::Mul, b(builder, BinOp::Sub, radius.clone(),
        b(builder, BinOp::Mul, builder.const_f32(3.0), mass.clone())), angular);
    let transport = b(builder, BinOp::Div, b(builder, BinOp::Mul, builder.const_f32(2.0), ur.clone()), radius);
    let theta_acceleration = b(builder, BinOp::Sub,
        b(builder, BinOp::Mul, b(builder, BinOp::Mul, sin.clone(), cos.clone()), uphi2),
        b(builder, BinOp::Mul, transport.clone(), utheta.clone()));
    let phi_acceleration = builder.unary(UnOp::Neg, b(builder, BinOp::Add,
        b(builder, BinOp::Mul, transport, uphi.clone()),
        b(builder, BinOp::Mul, b(builder, BinOp::Mul, builder.const_f32(2.0), b(builder, BinOp::Div, cos, sin)),
            b(builder, BinOp::Mul, utheta.clone(), uphi.clone()))));
    [b(builder, BinOp::Div, state[4].clone(), f), ur, utheta, uphi, builder.const_f32(0.0), radial, theta_acceleration, phi_acceleration]
}

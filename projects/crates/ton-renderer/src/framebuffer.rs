use ton_types::Rgb;

use crate::post::tone_map_rgb;

/// 帧缓冲颜色空间：Beauty 输出为线性 HDR，色调映射后标记为显示空间。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FramebufferEncoding {
    /// 线性 HDR 辐射（可含 >1 能量）。
    #[default]
    LinearHdr,
    /// 已色调映射，可直接编码为 8 位显示。
    Display,
}

#[derive(Debug, Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Rgb>,
    pub encoding: FramebufferEncoding,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        Self {
            width,
            height,
            pixels: vec![Rgb::BLACK; len],
            encoding: FramebufferEncoding::LinearHdr,
        }
    }

    pub fn set(&mut self, x: u32, y: u32, color: Rgb) {
        if x < self.width && y < self.height {
            self.pixels[y as usize * self.width as usize + x as usize] = color;
        }
    }

    /// 编码为 RGBA8：线性 HDR 帧会先执行一次色调映射。
    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for px in &self.pixels {
            let display = match self.encoding {
                FramebufferEncoding::LinearHdr => tone_map_rgb(*px),
                FramebufferEncoding::Display => *px,
            };
            let [r, g, b] = display.to_u8();
            out.extend_from_slice(&[r, g, b, 255]);
        }
        out
    }

    /// 线性 HDR 原始值（不色调映射），用于 HDR 诊断输出。
    pub fn to_rgba8_linear(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for px in &self.pixels {
            let c = px.clamp01();
            out.extend_from_slice(&[
                (c.r * 255.0).round() as u8,
                (c.g * 255.0).round() as u8,
                (c.b * 255.0).round() as u8,
                255,
            ]);
        }
        out
    }
}

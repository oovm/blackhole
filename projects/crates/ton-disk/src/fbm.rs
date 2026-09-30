//! 程序化 FBM 薄盘着色（教程式流动盘，非流体模拟）。

use ton_types::Rgb;

/// 有界螺旋相位，避免 `time * omega(r)` 无界拉丝。
#[inline]
fn spiral_phase(r: f64, phi: f64, inner: f64, outer: f64, global_phase: f64) -> (f64, f64) {
    let t = ((r - inner) / (outer - inner).max(1.0e-6)).clamp(0.0, 1.0);
    let twist = 2.5 * std::f64::consts::PI * t + phi * 0.35;
    let u = r * 0.15 + global_phase.cos() * 0.4;
    let v = twist + global_phase.sin() * 0.25;
    (u, v)
}

#[inline]
fn hash21(x: f64, y: f64) -> f64 {
    let s = (x * 127.1 + y * 311.7).sin() * 43758.5453;
    s.fract()
}

#[inline]
fn value_noise(x: f64, y: f64) -> f64 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let a = hash21(xi, yi);
    let b = hash21(xi + 1.0, yi);
    let c = hash21(xi, yi + 1.0);
    let d = hash21(xi + 1.0, yi + 1.0);
    let ux = xf * xf * (3.0 - 2.0 * xf);
    let uy = yf * yf * (3.0 - 2.0 * yf);
    a.lerp(b, ux).lerp(c.lerp(d, ux), uy)
}

trait Lerp {
    fn lerp(self, other: Self, t: f64) -> Self;
}

impl Lerp for f64 {
    fn lerp(self, other: Self, t: f64) -> Self {
        self + (other - self) * t
    }
}

fn fbm2(u: f64, v: f64, octaves: u32) -> f64 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += amp * value_noise(u * freq, v * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// 吸积盘交点处的 FBM 颜色；`global_phase` 用于动画（秒或帧相位）。
pub fn procedural_fbm_color(
    r: f64,
    phi: f64,
    inner: f64,
    outer: f64,
    global_phase: f64,
) -> Rgb {
    if r < inner || r > outer {
        return Rgb::BLACK;
    }
    let (u, v) = spiral_phase(r, phi, inner, outer, global_phase);
    let n = fbm2(u, v, 4);
    let hot = Rgb::new(1.0, 0.55, 0.12);
    let cool = Rgb::new(0.35, 0.08, 0.02);
    let t = (n * 0.85 + 0.1).clamp(0.0, 1.0);
    Rgb::new(
        hot.r.lerp(cool.r, t),
        hot.g.lerp(cool.g, t),
        hot.b.lerp(cool.b, t),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbm_is_deterministic() {
        let a = procedural_fbm_color(12.0, 0.5, 6.0, 30.0, 0.0);
        let b = procedural_fbm_color(12.0, 0.5, 6.0, 30.0, 0.0);
        assert_eq!(a, b);
    }

    #[test]
    fn outside_disk_is_black() {
        let c = procedural_fbm_color(2.0, 0.0, 6.0, 30.0, 0.0);
        assert_eq!(c, Rgb::BLACK);
    }
}

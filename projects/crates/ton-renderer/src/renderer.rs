use ton_disk::{procedural_fbm_color, temperature_to_rgb, AccretionDisk};
use ton_geodesic::{trace_ray, RayHit, TraceConfig};
use ton_types::{BlackHole, Camera, DiskModelKind, IntegratorKind, Rgb, Vec3};

use crate::framebuffer::Framebuffer;
use crate::sky::sky_color;

#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub black_hole: BlackHole,
    pub camera: Camera,
    pub disk: Option<AccretionDisk>,
    pub trace: TraceConfig,
    pub integrator: IntegratorKind,
    pub disk_model: DiskModelKind,
}

pub struct Renderer {
    config: RenderConfig,
}

impl Renderer {
    pub fn new(config: RenderConfig) -> Self {
        Self { config }
    }

    pub fn render(&self) -> Framebuffer {
        let w = self.config.width;
        let h = self.config.height;
        let mut fb = Framebuffer::new(w, h);
        let cam = self.config.camera;
        let bh = self.config.black_hole;
        let disk_trace = self
            .config
            .disk
            .filter(|_| self.config.disk_model != DiskModelKind::None);
        let trace_cfg = TraceConfig {
            disk_inner: disk_trace.map(|d| d.config.inner_radius),
            disk_outer: disk_trace.map(|d| d.config.outer_radius),
            ..self.config.trace
        };
        for y in 0..h {
            for x in 0..w {
                let u = (x as f64 / w as f64) * 2.0 - 1.0;
                let v = 1.0 - (y as f64 / h as f64) * 2.0;
                let aspect = w as f64 / h as f64;
                let dir = cam.ray_direction(u * aspect, v);
                let color = match trace_ray(
                    self.config.integrator,
                    bh,
                    cam.position,
                    dir,
                    trace_cfg,
                ) {
                    Ok(RayHit::Captured { .. }) => Rgb::BLACK,
                    Ok(RayHit::Escaped { direction }) => sky_color(direction),
                    Ok(RayHit::Disk { radius, position, .. }) => self
                        .shade_disk(radius, position),
                    Err(_) => sky_color(dir),
                };
                fb.set(x, y, color);
            }
        }
        fb
    }

    fn shade_disk(&self, radius: f64, position: Vec3) -> Rgb {
        let disk = self.config.disk;
        match self.config.disk_model {
            DiskModelKind::None => Rgb::BLACK,
            DiskModelKind::NovikovThorneThin => disk
                .map(|d| temperature_to_rgb(d.effective_temperature(radius)))
                .unwrap_or(Rgb::new(1.0, 0.6, 0.2)),
            DiskModelKind::ProceduralFbm => disk
                .map(|d| {
                    let phi = position.y.atan2(position.x);
                    procedural_fbm_color(
                        radius,
                        phi,
                        d.config.inner_radius,
                        d.config.outer_radius,
                        0.0,
                    )
                })
                .unwrap_or(Rgb::new(1.0, 0.45, 0.1)),
        }
    }
}

pub fn default_camera(bh: BlackHole) -> Camera {
    let m = bh.mass;
    // 在薄盘外缘之外、赤道平面内（当前测地线为赤道 2D 近似）。
    let dist = 55.0 * m;
    Camera::look_at(
        Vec3::new(dist, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0),
        std::f64::consts::FRAC_PI_4,
    )
}

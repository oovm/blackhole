use ton_disk::{procedural_fbm_color, sample_disk_radiance, AccretionDisk};
use ton_geodesic::{trace_ray_radiance, TraceConfig};
use ton_types::{BlackHole, Camera, DiskModelKind, IntegratorKind, Rgb, Vec3};

use crate::framebuffer::Framebuffer;
use crate::post::tone_map_rgb;
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
                let hdr = trace_ray_radiance(
                    self.config.integrator,
                    bh,
                    cam.position,
                    dir,
                    trace_cfg,
                    |radius, position| self.emit_disk(radius, position),
                    sky_color,
                )
                .map(|r| r.radiance)
                .unwrap_or_else(|_| sky_color(dir));
                let color = display_map(hdr);
                fb.set(x, y, color);
            }
        }
        fb
    }

    fn emit_disk(&self, radius: f64, position: Vec3) -> Rgb {
        let disk = self.config.disk;
        let cam = self.config.camera;
        let bh = self.config.black_hole;
        match self.config.disk_model {
            DiskModelKind::None => Rgb::BLACK,
            DiskModelKind::NovikovThorneThin => disk
                .map(|d| {
                    let doppler = simplified_doppler(position, cam, radius, bh);
                    sample_disk_radiance(d, radius, doppler).emissive_radiance
                })
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
    Camera::cinematic(bh)
}

/// 默认 cinematic 场景使用 3D 测地线（赤道 2D 积分器在倾角相机下会失真）。
pub fn default_integrator(_bh: BlackHole) -> IntegratorKind {
    IntegratorKind::Geodesic3d
}

fn display_map(hdr: Rgb) -> Rgb {
    tone_map_rgb(hdr.scale(2.8))
}

/// 工程化 Doppler 近似：盘面切向速度 × 视线夹角。
fn simplified_doppler(position: Vec3, camera: Camera, radius: f64, bh: BlackHole) -> f64 {
    let m = bh.mass.max(1e-12);
    let speed = (m / radius.max(2.0 * m)).sqrt();
    let tangent = Vec3::new(-position.y, position.x, 0.0);
    let vel = if tangent.length() > 1e-12 {
        tangent.normalize().scale(speed)
    } else {
        Vec3::new(speed, 0.0, 0.0)
    };
    let view = camera.position.sub(position).normalize();
    let cos_theta = vel.dot(view) / speed.max(1e-12);
    (1.0 / (1.0 - cos_theta * 0.35).max(0.05)).clamp(0.25, 4.0)
}

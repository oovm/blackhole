use ton_disk::{procedural_fbm_color, sample_disk_radiance, AccretionDisk};
use ton_geodesic::{
    trace_ray_diagnostic, trace_ray_radiance_outcome, DiagnosticColorMode, DiskEmission, DiskHit,
    TerminationHistogram, TraceConfig, TraceDiagnostic,
};
use ton_types::{BlackHole, Camera, DiskFrame, DiskModelKind, IntegratorKind, Rgb, Vec3};

use crate::framebuffer::{Framebuffer, FramebufferEncoding};
use crate::post::DEFAULT_EXPOSURE;
use crate::rays_ucf::CameraRayUcf;
use crate::shade_ucf::DiskShadeUcf;
use crate::shadow_overlay::draw_analytic_shadow_ring;
use crate::sky::sky_color;

/// 渲染输出模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// 线性 HDR 辐射（曝光后、无色调映射）；失败像素为黑。
    #[default]
    Beauty,
    /// 按终止原因着色（捕获 / 逃逸 / 预算耗尽 / 数值失败等）。
    TerminationMap,
    /// 关闭盘与星空，仅显示黑洞阴影轮廓与失败像素。
    CaptureSilhouette,
    /// 仅高亮积分失败像素，成功路径为中性底。
    FailureMap,
}

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
    pub mode: RenderMode,
    pub exposure: f64,
    /// 在捕获/失败诊断图上叠加解析阴影参考环。
    pub overlay_analytic_shadow: bool,
    /// Beauty 路径对 Novikov-Thorne 盘面走 UCF `disk_shade_novikov` 解释器。
    pub use_ucf_shade: bool,
    /// Beauty 路径像素射线走 UCF `generate_camera_ray` 解释器。
    pub use_ucf_rays: bool,
}

impl RenderConfig {
    pub fn exposure(mut self, exposure: f64) -> Self {
        self.exposure = exposure;
        self
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            black_hole: BlackHole::schwarzschild(1.0),
            camera: Camera::cinematic(BlackHole::schwarzschild(1.0)),
            disk: None,
            trace: TraceConfig::default(),
            integrator: IntegratorKind::Geodesic3d,
            disk_model: DiskModelKind::NovikovThorneThin,
            mode: RenderMode::Beauty,
            exposure: DEFAULT_EXPOSURE,
            overlay_analytic_shadow: false,
            use_ucf_shade: false,
            use_ucf_rays: false,
        }
    }
}

pub struct Renderer {
    config: RenderConfig,
}

/// 单帧渲染结果（含可选诊断直方图）。
#[derive(Debug, Clone)]
pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub termination_histogram: Option<TerminationHistogram>,
}

impl Renderer {
    pub fn new(config: RenderConfig) -> Self {
        Self { config }
    }

    pub fn render(&self) -> Framebuffer {
        self.render_with_histogram().framebuffer
    }

    /// 渲染一帧；所有模式均可附带终止直方图（Beauty 用于验收失败率）。
    pub fn render_with_histogram(&self) -> RenderOutput {
        match self.config.mode {
            RenderMode::Beauty => {
                let (fb, hist) = self.render_beauty();
                RenderOutput {
                    framebuffer: fb,
                    termination_histogram: Some(hist),
                }
            }
            RenderMode::TerminationMap => {
                let (fb, hist) =
                    self.render_diagnostic(DiagnosticColorMode::TerminationMap);
                RenderOutput {
                    framebuffer: fb,
                    termination_histogram: Some(hist),
                }
            }
            RenderMode::CaptureSilhouette => {
                let (mut fb, hist) =
                    self.render_diagnostic(DiagnosticColorMode::CaptureSilhouette);
                if self.config.overlay_analytic_shadow {
                    draw_analytic_shadow_ring(&mut fb, self.config.black_hole, self.config.camera);
                }
                RenderOutput {
                    framebuffer: fb,
                    termination_histogram: Some(hist),
                }
            }
            RenderMode::FailureMap => {
                let (fb, hist) = self.render_diagnostic(DiagnosticColorMode::FailureOnly);
                RenderOutput {
                    framebuffer: fb,
                    termination_histogram: Some(hist),
                }
            }
        }
    }

    fn render_beauty(&self) -> (Framebuffer, TerminationHistogram) {
        let w = self.config.width;
        let h = self.config.height;
        let mut fb = Framebuffer::new(w, h);
        let mut histogram = TerminationHistogram::default();
        let cam = self.config.camera;
        let bh = self.config.black_hole;
        let trace_cfg = self.trace_config_with_disk();
        let exposure = self.config.exposure;
        let mut ucf_shade = self.config.use_ucf_shade.then(DiskShadeUcf::new);
        let mut ucf_rays = self.config.use_ucf_rays.then(CameraRayUcf::new);
        for y in 0..h {
            for x in 0..w {
                let dir = self.pixel_ray_dir(x, y, w, h, cam, ucf_rays.as_mut());
                let outcome = trace_ray_radiance_outcome(
                    self.config.integrator,
                    bh,
                    cam.position,
                    dir,
                    trace_cfg,
                    |hit| self.emit_disk(hit, ucf_shade.as_mut()),
                    sky_color,
                );
                histogram.record(outcome.diagnostic.termination);
                let pixel = if outcome.diagnostic.is_trace_failure() {
                    Rgb::BLACK
                } else {
                    linear_exposure(outcome.radiance.radiance, exposure)
                };
                fb.set(x, y, pixel);
            }
        }
        (fb, histogram)
    }

    fn render_diagnostic(
        &self,
        color_mode: DiagnosticColorMode,
    ) -> (Framebuffer, TerminationHistogram) {
        let w = self.config.width;
        let h = self.config.height;
        let mut fb = Framebuffer::new(w, h);
        let mut histogram = TerminationHistogram::default();
        let cam = self.config.camera;
        let bh = self.config.black_hole;
        let trace_cfg = match color_mode {
            DiagnosticColorMode::CaptureSilhouette | DiagnosticColorMode::FailureOnly => {
                TraceConfig {
                    disk_inner: None,
                    disk_outer: None,
                    ..self.config.trace
                }
            }
            DiagnosticColorMode::TerminationMap => self.trace_config_with_disk(),
        };
        let mut ucf_rays = self.config.use_ucf_rays.then(CameraRayUcf::new);
        for y in 0..h {
            for x in 0..w {
                let dir = self.pixel_ray_dir(x, y, w, h, cam, ucf_rays.as_mut());
                let diag = trace_ray_diagnostic(
                    self.config.integrator,
                    bh,
                    cam.position,
                    dir,
                    trace_cfg,
                );
                histogram.record(diag.termination);
                fb.set(x, y, diagnostic_to_display_rgb(diag, color_mode));
            }
        }
        fb.encoding = FramebufferEncoding::Display;
        (fb, histogram)
    }

    fn trace_config_with_disk(&self) -> TraceConfig {
        let disk_trace = self
            .config
            .disk
            .filter(|_| self.config.disk_model != DiskModelKind::None);
        TraceConfig {
            disk_inner: disk_trace.map(|d| d.config.inner_radius),
            disk_outer: disk_trace.map(|d| d.config.outer_radius),
            ..self.config.trace
        }
    }

    fn pixel_ray_dir(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        cam: Camera,
        ucf_rays: Option<&mut CameraRayUcf>,
    ) -> Vec3 {
        let u = ((x as f64 + 0.5) / w as f64) * 2.0 - 1.0;
        let v = 1.0 - ((y as f64 + 0.5) / h as f64) * 2.0;
        let aspect = w as f64 / h as f64;
        let u_scaled = u * aspect;
        if let Some(rays) = ucf_rays {
            rays.direction(cam, u_scaled, v)
        } else {
            cam.ray_direction(u_scaled, v)
        }
    }

    fn emit_disk(&self, hit: DiskHit, ucf_shade: Option<&mut DiskShadeUcf>) -> DiskEmission {
        let disk = self.config.disk;
        let bh = self.config.black_hole;
        let radius = hit.radius;
        let position = hit.position;
        match self.config.disk_model {
            DiskModelKind::None => DiskEmission::opaque(Rgb::BLACK),
            DiskModelKind::NovikovThorneThin => disk
                .map(|d| {
                    let doppler =
                        disk_frame_doppler(position, hit.incoming_to_observer, radius, bh);
                    let sample = if let Some(shade) = ucf_shade {
                        shade.sample(d, radius, doppler)
                    } else {
                        sample_disk_radiance(d, radius, doppler)
                    };
                    DiskEmission::with_absorption(sample.emissive_radiance, sample.absorption)
                })
                .unwrap_or(DiskEmission::with_absorption(Rgb::new(1.0, 0.6, 0.2), 0.15)),
            DiskModelKind::ProceduralFbm => disk
                .map(|d| {
                    let phi = disk_azimuth(position);
                    let color = procedural_fbm_color(
                        radius,
                        phi,
                        d.config.inner_radius,
                        d.config.outer_radius,
                        0.0,
                    );
                    let peak = color.r.max(color.g).max(color.b);
                    DiskEmission::with_absorption(color, (peak * 0.1).clamp(0.05, 0.4))
                })
                .unwrap_or(DiskEmission::with_absorption(Rgb::new(1.0, 0.45, 0.1), 0.15)),
            DiskModelKind::SolidUniform => disk
                .map(|d| {
                    if radius < d.config.inner_radius || radius > d.config.outer_radius {
                        DiskEmission::with_absorption(Rgb::BLACK, 0.0)
                    } else {
                        DiskEmission::opaque_sheet(Rgb::new(1.0, 0.72, 0.28))
                    }
                })
                .unwrap_or_else(|| DiskEmission::opaque_sheet(Rgb::new(1.0, 0.72, 0.28))),
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

fn linear_exposure(hdr: Rgb, exposure: f64) -> Rgb {
    hdr.scale(exposure.max(0.0))
}

fn diagnostic_to_display_rgb(diag: TraceDiagnostic, mode: DiagnosticColorMode) -> Rgb {
    let (r, g, b) = diag.to_rgb8(mode);
    Rgb::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
}

/// 盘面切向 Doppler：观测方向沿测地线入射，而非相机直线。
fn disk_frame_doppler(
    position: Vec3,
    incoming_to_observer: Vec3,
    radius: f64,
    bh: BlackHole,
) -> f64 {
    let (_, _, n) = DiskFrame::equatorial_xy().tangent_basis();
    let m = bh.mass.max(1e-12);
    let speed = (m / radius.max(2.0 * m)).sqrt();
    let tangent = n.cross(position);
    let vel = if tangent.length() > 1e-12 {
        tangent.normalize().scale(speed)
    } else {
        Vec3::new(speed, 0.0, 0.0)
    };
    let view = incoming_to_observer.normalize();
    let cos_theta = vel.dot(view) / speed.max(1e-12);
    (1.0 / (1.0 - cos_theta * 0.35).max(0.05)).clamp(0.25, 4.0)
}

fn disk_azimuth(position: Vec3) -> f64 {
    let (t1, t2, _) = DiskFrame::equatorial_xy().tangent_basis();
    let u = position.dot(t1);
    let v = position.dot(t2);
    v.atan2(u)
}

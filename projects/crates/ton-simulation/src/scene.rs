use ton_disk::{AccretionDisk, DiskConfig};
use ton_geodesic::TraceConfig;
use ton_renderer::{
    apply_post_process, build_frame_graph_for_resolution, default_camera, default_integrator, Framebuffer, Graph,
    RenderConfig, RenderMode, RenderOutput, Renderer, DEFAULT_EXPOSURE,
};
use ton_types::{BlackHole, Camera, DiskModelKind, IntegratorKind, TonError, TonResult};

use crate::ucf_cpu::{execute_frame_graph_cpu, CpuGraphExecution};
use crate::ucf_physics::{
    schwarzschild_capture_silhouette_ucf_cpu,
    schwarzschild_capture_silhouette_ucf_vulkan,
    schwarzschild_beauty_ucf_vulkan,
};
use crate::SchwarzschildSpectralGpuBuilder;

#[derive(Debug, Clone)]
pub struct Scene {
    pub black_hole: BlackHole,
    pub disk: Option<AccretionDisk>,
    pub camera: Camera,
    pub width: u32,
    pub height: u32,
    pub trace: TraceConfig,
    pub integrator: IntegratorKind,
    pub disk_model: DiskModelKind,
    /// UCF 帧图是否插入 `ton_bloom` 任务。
    pub post_bloom: bool,
    pub render_mode: RenderMode,
    pub overlay_analytic_shadow: bool,
}

#[derive(Debug, Clone)]
pub struct SceneBuilder {
    black_hole: BlackHole,
    disk: Option<AccretionDisk>,
    camera: Option<Camera>,
    width: u32,
    height: u32,
    trace: TraceConfig,
    integrator: IntegratorKind,
    disk_model: DiskModelKind,
    post_bloom: bool,
    render_mode: RenderMode,
    overlay_analytic_shadow: bool,
}

impl SceneBuilder {
    pub fn new(black_hole: BlackHole) -> Self {
        Self {
            black_hole,
            disk: None,
            camera: None,
            width: 640,
            height: 360,
            trace: TraceConfig::default(),
            // cinematic 相机不是赤道二维轨道的适用域。默认使用可处理倾斜盘面的三维近似路径，
            // 二维轨道积分器只通过 CLI 显式选择，用于赤道参考测试。
            integrator: if black_hole.spin.abs() <= f64::EPSILON {
                IntegratorKind::SemiClassicalRayMarch
            } else {
                default_integrator(black_hole)
            },
            disk_model: DiskModelKind::NovikovThorneThin,
            post_bloom: false,
            render_mode: RenderMode::Beauty,
            overlay_analytic_shadow: false,
        }
    }

    pub fn with_overlay_analytic_shadow(mut self, enabled: bool) -> Self {
        self.overlay_analytic_shadow = enabled;
        self
    }

    pub fn with_render_mode(mut self, mode: RenderMode) -> Self {
        self.render_mode = mode;
        self
    }

    pub fn with_post_bloom(mut self, enabled: bool) -> Self {
        self.post_bloom = enabled;
        self
    }

    pub fn with_integrator(mut self, integrator: IntegratorKind) -> Self {
        self.integrator = integrator;
        self
    }

    pub fn with_disk_model(mut self, disk_model: DiskModelKind) -> Self {
        self.disk_model = disk_model;
        self
    }

    pub fn with_default_disk(mut self) -> Self {
        self.disk = Some(self.default_disk_for_model());
        self
    }

    fn default_disk_for_model(&self) -> AccretionDisk {
        let config = if self.disk_model == DiskModelKind::SolidUniform {
            DiskConfig::solid_lensing_test(self.black_hole)
        } else {
            DiskConfig::schwarzschild_default(self.black_hole)
        };
        AccretionDisk::new(self.black_hole, config)
    }

    pub fn with_camera(mut self, camera: Camera) -> Self {
        self.camera = Some(camera);
        self
    }

    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_step_size(mut self, step_size: f64) -> Self {
        let inner = self.trace.disk_inner;
        let outer = self.trace.disk_outer;
        let adaptive = self.trace.adaptive_near_horizon;
        self.trace = TraceConfig::with_step_size_adaptive(step_size, adaptive);
        self.trace.disk_inner = inner;
        self.trace.disk_outer = outer;
        self
    }

    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.trace.max_steps = max_steps;
        self
    }

    pub fn with_trace(mut self, trace: TraceConfig) -> Self {
        self.trace = trace;
        self
    }

    pub fn build(self) -> Scene {
        let disk = if self.disk_model != DiskModelKind::None && self.disk.is_none() {
            Some(self.default_disk_for_model())
        } else {
            self.disk
        };
        Scene {
            black_hole: self.black_hole,
            disk,
            camera: self.camera.unwrap_or_else(|| default_camera(self.black_hole)),
            width: self.width,
            height: self.height,
            trace: self.trace,
            integrator: self.integrator,
            disk_model: self.disk_model,
            post_bloom: self.post_bloom,
            render_mode: self.render_mode,
            overlay_analytic_shadow: self.overlay_analytic_shadow,
        }
    }
}

impl Scene {
    pub fn builder(black_hole: BlackHole) -> SceneBuilder {
        SceneBuilder::new(black_hole)
    }

    pub fn build_ucf_graph(&self) -> Graph {
        build_frame_graph_for_resolution(
            self.integrator,
            self.disk_model,
            self.post_bloom,
            self.width,
            self.height,
        )
    }

    /// UCF 帧图 CPU 参考路径：校验蓝图 `op` 后渲染一帧。
    pub fn render_frame_ucf_cpu(&self) -> TonResult<(Framebuffer, CpuGraphExecution)> {
        let graph = self.build_ucf_graph();
        execute_frame_graph_cpu(self, &graph)
    }

    /// 使用 CPU UCF interpreter 生成场景尺寸的捕获/逃逸剪影。
    pub fn render_capture_silhouette_ucf_cpu(&self) -> TonResult<Framebuffer> {
        schwarzschild_capture_silhouette_ucf_cpu(
            self.black_hole,
            self.camera,
            self.width,
            self.height,
        )
    }

    /// 使用原生 Vulkan UCF backend 生成场景尺寸的捕获/逃逸剪影。
    ///
    /// Vulkan 不可用、程序能力不足或资源绑定失败时直接返回错误，不回退 CPU。
    pub fn render_capture_silhouette_ucf_vulkan(&self) -> TonResult<Framebuffer> {
        schwarzschild_capture_silhouette_ucf_vulkan(
            self.black_hole,
            self.camera,
            self.width,
            self.height,
        )
    }

    /// 使用原生 Vulkan UCF backend 生成已色调映射的 Beauty 帧。
    ///
    /// GPU 不可用、程序能力不足、终止预算耗尽或输出非法时直接返回错误，
    /// 不回退 CPU，也不调用 TON 原生渲染器。
    pub fn render_frame_ucf_vulkan(&self) -> TonResult<Framebuffer> {
        let (inner_radius, outer_radius, accretion_rate) = self
            .disk
            .as_ref()
            .map(|disk| (disk.config.inner_radius, disk.config.outer_radius, disk.config.accretion_rate))
            .ok_or_else(|| TonError::Domain("Vulkan Beauty requires an accretion disk".into()))?;
        schwarzschild_beauty_ucf_vulkan(
            self.black_hole,
            self.camera,
            self.width,
            self.height,
            inner_radius,
            outer_radius,
            accretion_rate,
            self.trace.step_size,
            self.trace.escape_radius,
            self.trace.max_steps,
        )
    }

    /// 使用经过事件包和辐射传输验证的 Schwarzschild 光谱 GPU Beauty 链。
    ///
    /// 该入口只接受无自旋黑洞和标准薄盘模型。其他物理模型必须使用
    /// 对应的独立参考实现，不能在这里静默降级或伪装成 Schwarzschild。
    pub fn render_spectral_frame_ucf_vulkan(&self) -> TonResult<Framebuffer> {
        if self.black_hole.spin.abs() > f64::EPSILON {
            return Err(TonError::Domain("spectral GPU entry currently supports Schwarzschild only".into()));
        }
        if self.disk_model != DiskModelKind::NovikovThorneThin {
            return Err(TonError::Domain("spectral GPU entry requires the thin disk model".into()));
        }
        self.disk.as_ref().ok_or_else(|| TonError::Domain("spectral GPU entry requires an accretion disk".into()))?;
        let frame = SchwarzschildSpectralGpuBuilder::new(self.black_hole.mass)
            .exposure(3.0e-9)
            .execute(self.camera, self.width, self.height)?;
        Ok(frame.framebuffer)
    }

    pub fn render_frame(&self) -> Framebuffer {
        self.render_frame_with_histogram().framebuffer
    }

    /// 渲染一帧；`termination` / `capture` 模式下附带终止直方图。
    pub fn render_frame_with_histogram(&self) -> RenderOutput {
        self.render_frame_with_histogram_opts(false)
    }

    /// 渲染线性 HDR 帧；`use_ucf_kernels` 为真时射线生成与 Novikov 盘面走 UCF 解释器。
    pub fn render_frame_with_histogram_opts(&self, use_ucf_kernels: bool) -> RenderOutput {
        let mut output = Renderer::new(RenderConfig {
            width: self.width,
            height: self.height,
            black_hole: self.black_hole,
            camera: self.camera,
            disk: self.disk,
            trace: self.trace,
            integrator: self.integrator,
            disk_model: self.disk_model,
            mode: self.render_mode,
            exposure: DEFAULT_EXPOSURE,
            overlay_analytic_shadow: self.overlay_analytic_shadow,
            use_ucf_shade: use_ucf_kernels,
            use_ucf_rays: use_ucf_kernels,
        })
        .render_with_histogram();
        if self.post_bloom && self.render_mode == RenderMode::Beauty {
            apply_post_process(&mut output.framebuffer, true);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ton_types::BlackHole;

    #[test]
    fn scene_exposes_cpu_ucf_silhouette() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_resolution(16, 9)
            .build();
        let frame = scene.render_capture_silhouette_ucf_cpu().expect("CPU UCF silhouette");
        assert_eq!((frame.width, frame.height), (16, 9));
        assert_eq!(frame.pixels.len(), 16 * 9);
    }

    #[test]
    fn scene_cpu_and_vulkan_ucf_silhouettes_match_when_available() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_resolution(16, 9)
            .build();
        let cpu = scene.render_capture_silhouette_ucf_cpu().expect("CPU UCF silhouette");
        let gpu = match scene.render_capture_silhouette_ucf_vulkan() {
            Ok(value) => value,
            Err(error) if error.to_string().contains("create Vulkan backend") => {
                eprintln!("skip scene Vulkan silhouette parity test: {error}");
                return;
            }
            Err(error) => panic!("scene Vulkan UCF silhouette: {error}"),
        };
        assert_eq!(cpu.pixels, gpu.pixels);
    }

    #[test]
    fn scene_vulkan_ucf_beauty_smoke_when_available() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_default_disk()
            .with_resolution(1, 1)
            .with_step_size(0.08)
            .with_max_steps(1024)
            .build();
        let frame = match scene.render_frame_ucf_vulkan() {
            Ok(frame) => frame,
            Err(error) if error.to_string().contains("create Vulkan backend") => return,
            Err(error) => panic!("Vulkan UCF Beauty smoke: {error}"),
        };
        assert_eq!((frame.width, frame.height), (1, 1));
        assert_eq!(frame.pixels.len(), 1);
        assert!(frame.pixels[0].r.is_finite());
        assert!(frame.pixels[0].g.is_finite());
        assert!(frame.pixels[0].b.is_finite());
    }

    #[test]
    fn scene_exposes_verified_spectral_gpu_beauty_entry() {
        let scene = Scene::builder(BlackHole::schwarzschild(1.0))
            .with_resolution(8, 4)
            .with_default_disk()
            .build();
        let frame = match scene.render_spectral_frame_ucf_vulkan() {
            Ok(frame) => frame,
            Err(error) if error.to_string().contains("create spectral backend") => return,
            Err(error) => panic!("spectral GPU Beauty: {error}"),
        };
        assert_eq!((frame.width, frame.height), (8, 4));
        assert_eq!(frame.encoding, ton_renderer::FramebufferEncoding::Display);
        assert_eq!(frame.pixels.len(), 32);
        assert!(frame.pixels.iter().all(|pixel| pixel.r.is_finite() && pixel.g.is_finite() && pixel.b.is_finite()));
    }
}

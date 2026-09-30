use ton_disk::{AccretionDisk, DiskConfig};
use ton_geodesic::TraceConfig;
use ton_renderer::{
    build_frame_graph_with_post, default_camera, default_integrator, Framebuffer, Graph,
    RenderConfig, RenderMode, RenderOutput, Renderer, DEFAULT_EXPOSURE,
};
use ton_types::{BlackHole, Camera, DiskModelKind, IntegratorKind, TonResult};

use crate::ucf_cpu::{execute_frame_graph_cpu, CpuGraphExecution};

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
            integrator: default_integrator(black_hole),
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
        self.trace = TraceConfig::with_step_size(step_size);
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
        build_frame_graph_with_post(self.integrator, self.disk_model, self.post_bloom)
    }

    /// UCF 帧图 CPU 参考路径：校验蓝图 `op` 后渲染一帧。
    pub fn render_frame_ucf_cpu(&self) -> TonResult<(Framebuffer, CpuGraphExecution)> {
        let graph = self.build_ucf_graph();
        execute_frame_graph_cpu(self, &graph)
    }

    pub fn render_frame(&self) -> Framebuffer {
        self.render_frame_with_histogram().framebuffer
    }

    /// 渲染一帧；`termination` / `capture` 模式下附带终止直方图。
    pub fn render_frame_with_histogram(&self) -> RenderOutput {
        Renderer::new(RenderConfig {
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
        })
        .render_with_histogram()
    }
}

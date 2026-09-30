use ton_disk::{AccretionDisk, DiskConfig};
use ton_geodesic::TraceConfig;
use ton_render::{default_camera, Framebuffer, RenderConfig, Renderer};
use ton_types::{BlackHole, Camera, DiskModelKind, IntegratorKind, TonResult};
use ton_ucf::{build_frame_graph_with_post, Graph};

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
            integrator: IntegratorKind::default(),
            disk_model: DiskModelKind::default(),
            post_bloom: false,
        }
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
        self.disk = Some(AccretionDisk::new(
            self.black_hole,
            DiskConfig::schwarzschild_default(self.black_hole),
        ));
        self
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

    pub fn build(self) -> Scene {
        Scene {
            black_hole: self.black_hole,
            disk: self.disk,
            camera: self.camera.unwrap_or_else(|| default_camera(self.black_hole)),
            width: self.width,
            height: self.height,
            trace: self.trace,
            integrator: self.integrator,
            disk_model: self.disk_model,
            post_bloom: self.post_bloom,
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
        Renderer::new(RenderConfig {
            width: self.width,
            height: self.height,
            black_hole: self.black_hole,
            camera: self.camera,
            disk: self.disk,
            trace: self.trace,
            integrator: self.integrator,
            disk_model: self.disk_model,
        })
        .render()
    }
}

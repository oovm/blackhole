//! 经验证轨迹、圆轨道频移、温度基线与谱色转换组成的原生 GPU 渲染链。

use std::collections::BTreeMap;
use ton_renderer::{Framebuffer, FramebufferEncoding};
use ton_types::{Camera, Rgb, TonError, TonResult};
use ucf_backend_vulkan::VulkanBackend;
use ucf_ir::{Graph, ResourceGraph, TaskGraph};
use ucf_scheduler::Backend;
use ucf_types::{Access, Domain, ParamValue, ResourceId, ResourceKind, ResourceNode};
use crate::{F64GpuTraceBatch, SchwarzschildF64GpuTraceBuilder};

fn execute_stage(backend: &mut VulkanBackend, program: &ton_ucf_frontend::Program, input: &[f32],
    input_stride: u32, output_stride: u32, base: u64, uniforms: &[f64], table: Option<&[f32]>,
) -> TonResult<Vec<f32>> {
    if input.is_empty() || input.len() % input_stride as usize != 0 { return Err(TonError::Domain("invalid spectral stage input layout".into())); }
    let count = u32::try_from(input.len() / input_stride as usize).map_err(|_| TonError::Domain("spectral batch exceeds u32".into()))?;
    let mut params = uniforms.iter().enumerate().map(|(slot, value)| (format!("uniform_{slot}"), ParamValue::F64(*value))).collect::<BTreeMap<_, _>>();
    params.insert("buffer_2".into(), ParamValue::I64((base + 1) as i64));
    let mut resources = [(base, input_stride, Access::Read), (base + 1, output_stride, Access::Write), (base + 2, 1, Access::Write)]
        .into_iter().map(|(id, stride, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram,
            access, byte_size: Some(u64::from(count) * u64::from(stride) * 4) }).collect::<Vec<_>>();
    if let Some(table) = table {
        params.insert("buffer_3".into(), ParamValue::I64((base + 3) as i64));
        resources.push(ResourceNode { id: ResourceId(base + 3), kind: ResourceKind::Buffer, domain: Domain::Vram,
            access: Access::Read, byte_size: Some(table.len() as u64 * 4) });
    }
    let task = crate::ucf_physics::dispatch_task(base, program, count, ResourceId(base + 2), 2, 1, Some(ResourceId(base)), params)?;
    let graph = Graph { resources: ResourceGraph { nodes: resources }, tasks: TaskGraph { nodes: vec![task], edges: Vec::new() } };
    graph.validate().map_err(|error| TonError::Domain(format!("validate spectral stage: {error}")))?;
    backend.prepare(&graph).map_err(|error| TonError::Domain(format!("prepare spectral stage: {error}")))?;
    backend.write_f32(ResourceId(base), input).map_err(|error| TonError::Domain(format!("upload spectral stage: {error}")))?;
    if let Some(table) = table { backend.write_f32(ResourceId(base + 3), table).map_err(|error| TonError::Domain(format!("upload spectral response: {error}")))?; }
    backend.run_prepared(&graph).map_err(|error| TonError::Domain(format!("execute spectral stage: {error}")))?;
    let output = backend.read_f32(ResourceId(base + 1)).map_err(|error| TonError::Domain(format!("read spectral output: {error}")))?;
    let status = backend.read_f32(ResourceId(base + 2)).map_err(|error| TonError::Domain(format!("read spectral status: {error}")))?;
    if output.len() != count as usize * output_stride as usize || status.len() != count as usize
        || output.iter().any(|value| !value.is_finite()) || status.iter().any(|value| *value != 0.0 && *value != 1.0) {
        return Err(TonError::Domain("spectral stage rejected input or produced invalid output".into()));
    }
    Ok(output)
}

/// 当前无纹理不透明薄盘基线，尚无星空、透射或 Bloom。
#[derive(Debug)]
pub struct SpectralGpuFrame {
    /// 已经 sRGB 编码的显示帧，不可再执行一次色调映射。
    pub framebuffer: Framebuffer,
    /// 曝光后但尚未色调映射的线性 RGB，保留 HDR 和色域外分量。
    pub hdr: Vec<[f32; 3]>,
    /// 真实 F64 原生轨迹，不以仅存在 hash 作为执行证据。
    pub trace: F64GpuTraceBatch,
    /// 依序完成的圆轨道频移、温度、谱色、显示程序哈希。
    pub shading_program_hashes: Vec<u64>,
}

/// 独立的可验证光谱薄盘 GPU 入口，不将温度基线称为 Novikov-Thorne。
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildSpectralGpuBuilder {
    mass: f64,
    peak_kelvin: f64,
    exposure: f64,
    step: f64,
    maximum_steps: u32,
    wavelength_step_nm: u32,
}

impl SchwarzschildSpectralGpuBuilder {
    /// 默认峰值 6500K、曝光 1e-10、最大步长 0.02M。
    pub fn new(mass: f64) -> Self { Self { mass, peak_kelvin: 6500.0, exposure: 1.0e-10, step: 0.02 * mass, maximum_steps: 32768, wavelength_step_nm: 5 } }
    /// 设置物理发射温度峰值。
    pub fn peak_temperature(mut self, value: f64) -> Self { self.peak_kelvin = value; self }
    /// 设置显示曝光，不改变几何或频移。
    pub fn exposure(mut self, value: f64) -> Self { self.exposure = value; self }
    /// 设置 F64 轨迹最大步长。
    pub fn maximum_step(mut self, value: f64) -> Self { self.step = value; self }
    /// 设置 F64 轨迹总预算。
    pub fn maximum_steps(mut self, value: u32) -> Self { self.maximum_steps = value; self }
    /// 控制谱积分采样，范围 1..20nm 且须整除 400nm，不改变几何。
    pub fn wavelength_step_nm(mut self, value: u32) -> Self { self.wavelength_step_nm = value; self }
    /// 执行完整原生程序链，失败直接返回，不 fallback。
    pub fn execute(self, camera: Camera, width: u32, height: u32) -> TonResult<SpectralGpuFrame> {
        if !self.peak_kelvin.is_finite() || self.peak_kelvin <= 0.0 || !self.exposure.is_finite() || self.exposure <= 0.0 {
            return Err(TonError::Domain("invalid spectral disk temperature or exposure".into()));
        }
        let spectrum = ton_ucf_frontend::VisibleSpectrumColorBuilder::new().wavelength_step_nm(self.wavelength_step_nm);
        let table = spectrum.response_table().map_err(TonError::Domain)?;
        let trace = SchwarzschildF64GpuTraceBuilder::new(self.mass).maximum_step(self.step).maximum_steps(self.maximum_steps).execute_camera(camera, width, height)?;
        trace.require_renderable()?;
        let mut backend = VulkanBackend::new().map_err(|error| TonError::Domain(format!("create spectral backend: {error}")))?;
        let records = trace.records.iter().map(|record| record.map(|value| value as f32)).collect::<Vec<_>>();
        if records.iter().flatten().any(|value| !value.is_finite()) { return Err(TonError::Domain("F64 trajectory cannot be represented by F32 radiance inputs".into())); }
        let frequency = crate::adaptive_gpu::execute_cartesian_frequency_with_backend(&mut backend, &records, self.mass as f32, camera.position.length() as f32)?;
        let mut hashes = vec![ton_ucf_frontend::program_hash(&ton_ucf_frontend::SchwarzschildDiskFrequencyBuilder::new().build().map_err(TonError::Domain)?)];
        let temperature = ton_ucf_frontend::ZeroTorqueDiskInputsBuilder::new().build().map_err(TonError::Domain)?;
        let inputs = execute_stage(&mut backend, &temperature, &frequency.iter().flatten().copied().collect::<Vec<_>>(), 4, 3, 101,
            &[6.0 * self.mass, self.peak_kelvin, self.exposure], None)?;
        hashes.push(ton_ucf_frontend::program_hash(&temperature));
        let color = spectrum.build().map_err(TonError::Domain)?;
        let colors = execute_stage(&mut backend, &color, &inputs, 3, 7, 111, &[], Some(&table))?;
        hashes.push(ton_ucf_frontend::program_hash(&color));
        let hdr = colors.chunks_exact(7).map(|record| record[3..6].try_into().unwrap()).collect::<Vec<_>>();
        let display = ton_ucf_frontend::SpectralDisplayBuilder::new().build().map_err(TonError::Domain)?;
        let pixels = execute_stage(&mut backend, &display, &colors, 7, 3, 121, &[], None)?;
        if pixels.iter().any(|value| *value < 0.0 || *value > 1.0) { return Err(TonError::Domain("spectral display exceeds encoded sRGB range".into())); }
        hashes.push(ton_ucf_frontend::program_hash(&display));
        let framebuffer = Framebuffer { width, height, encoding: FramebufferEncoding::Display,
            pixels: pixels.chunks_exact(3).map(|pixel| Rgb::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)).collect() };
        Ok(SpectralGpuFrame { framebuffer, hdr, trace, shading_program_hashes: hashes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_gpu_same_frame_step_and_sampling_converge_without_postprocessing() {
        let inclination = 80.0f64.to_radians();
        let camera = Camera::look_at(ton_types::Vec3::new(50.0 * inclination.sin(), 0.0, 50.0 * inclination.cos()),
            ton_types::Vec3::ZERO, ton_types::Vec3::new(0.0, 0.0, 1.0), 0.58);
        let frames = [0.04, 0.02, 0.01].map(|step| SchwarzschildSpectralGpuBuilder::new(1.0)
            .maximum_step(step).execute(camera, 32, 18).unwrap());
        let finest = &frames[2];
        let mut radius_errors = [0.0f64; 2];
        let mut hdr_errors = [0.0f64; 2];
        let mut hits = 0;
        for pixel in 0..finest.hdr.len() {
            for level in 0..2 {
                assert_eq!(frames[level].trace.terminations[pixel], finest.trace.terminations[pixel], "termination pixel {pixel}");
                assert_eq!(frames[level].trace.records[pixel][13], finest.trace.records[pixel][13], "crossing count pixel {pixel}");
                radius_errors[level] = radius_errors[level].max((frames[level].trace.records[pixel][14] - finest.trace.records[pixel][14]).abs());
                for channel in 0..3 {
                    hdr_errors[level] = hdr_errors[level].max((frames[level].hdr[pixel][channel] - finest.hdr[pixel][channel]).abs() as f64);
                }
            }
            if finest.trace.records[pixel][13] > 0.0 { hits += 1; }
        }
        assert!(hits > 0);
        assert!(radius_errors[0] < 1.0e-5 && radius_errors[1] < 1.0e-6, "radius_errors={radius_errors:?}");
        assert!(radius_errors[1] < radius_errors[0], "refinement must reduce geometric error: {radius_errors:?}");
        assert!(hdr_errors.iter().all(|error| *error < 1.0e-5), "linear HDR errors={hdr_errors:?}");
        let sampled = SchwarzschildSpectralGpuBuilder::new(1.0).maximum_step(0.01).wavelength_step_nm(1).execute(camera, 32, 18).unwrap();
        assert_eq!(sampled.trace.records, finest.trace.records);
        assert_eq!(sampled.shading_program_hashes[0..2], finest.shading_program_hashes[0..2]);
        assert_ne!(sampled.shading_program_hashes[2], finest.shading_program_hashes[2]);
        let mut relative_spectrum_error = 0.0f64;
        let mut display_error = 0.0f64;
        for pixel in 0..finest.hdr.len() {
            let scale = sampled.hdr[pixel].iter().fold(1.0e-10f64, |scale, value| scale.max((*value as f64).abs()));
            for channel in 0..3 {
                relative_spectrum_error = relative_spectrum_error.max((finest.hdr[pixel][channel] - sampled.hdr[pixel][channel]).abs() as f64 / scale);
            }
            let actual = finest.framebuffer.pixels[pixel];
            let expected = sampled.framebuffer.pixels[pixel];
            display_error = display_error.max((actual.r - expected.r).abs()).max((actual.g - expected.g).abs()).max((actual.b - expected.b).abs());
        }
        assert!(relative_spectrum_error < 5.0e-4, "5nm/1nm linear HDR error={relative_spectrum_error}");
        assert!(display_error < 1.0 / 255.0, "encoded display error={display_error}");
        eprintln!("32x18 native convergence: disk_pixels={hits}, radius_errors={radius_errors:?}, hdr_absolute_errors={hdr_errors:?}, 5nm/1nm_hdr_relative={relative_spectrum_error}, display_absolute={display_error}");
        assert!(SchwarzschildSpectralGpuBuilder::new(1.0).wavelength_step_nm(3).execute(camera, 32, 18).is_err());
    }

    #[test]
    fn spectral_gpu_temperature_and_display_match_independent_formulas() {
        let mut backend = VulkanBackend::new().unwrap();
        let program = ton_ucf_frontend::ZeroTorqueDiskInputsBuilder::new().build().unwrap();
        let input = [[6.0f32, 1.0, 1.0, 1.0], [f32::from_bits(6.0f32.to_bits() + 1), 1.0, 1.0, 1.0],
            [8.166667, 0.7, 0.2401, 1.0], [20.0, 1.5, 5.0625, 1.0], [0.0, 0.0, 0.0, 0.0]];
        let actual = execute_stage(&mut backend, &program, &input.iter().flatten().copied().collect::<Vec<_>>(), 4, 3, 101, &[6.0, 6500.0, 3.0e-9], None).unwrap();
        for (input, actual) in input.iter().zip(actual.chunks_exact(3)) {
            let expected = if input[3] == 0.0 { 0.0 } else { ton_geodesic::zero_torque_disk_temperature(input[0] as f64, 6.0, 6500.0).unwrap() };
            assert!((actual[0] as f64 - expected).abs() < 0.002, "input={input:?}, temperature={}, reference={expected}", actual[0]);
            assert_eq!(actual[1], if input[3] == 0.0 { 1.0 } else { input[1] });
        }
        let program = ton_ucf_frontend::SpectralDisplayBuilder::new().build().unwrap();
        let pixels = [[0.0, 0.0, 0.0, -0.1, 0.18, 4.0, 1.0], [0.0; 7]];
        let actual = execute_stage(&mut backend, &program, &pixels[0], 7, 3, 121, &[], None).unwrap();
        for (value, actual) in pixels[0][3..6].iter().zip(actual) {
            let mapped = ton_renderer::aces_filmic(*value as f64);
            let expected = if mapped <= 0.0031308 { 12.92 * mapped } else { 1.055 * mapped.powf(1.0 / 2.4) - 0.055 };
            assert!((actual as f64 - expected).abs() < 1.0e-6);
        }
        let mut invalid_backend = VulkanBackend::new().unwrap();
        assert!(execute_stage(&mut invalid_backend, &program, &pixels[1], 7, 3, 121, &[], None).is_err());
    }

    #[test]
    fn spectral_gpu_full_chain_has_hdr_and_no_diagnostic_colors() {
        let camera = Camera::cinematic(ton_types::BlackHole::schwarzschild(1.0));
        let output = SchwarzschildSpectralGpuBuilder::new(1.0).exposure(3.0e-9).execute(camera, 16, 9).unwrap();
        assert_eq!(output.framebuffer.encoding, FramebufferEncoding::Display);
        assert_eq!(output.shading_program_hashes.len(), 4);
        assert_eq!(output.hdr.len(), 144);
        let mut luminous = 0;
        for ((hdr, pixel), trace) in output.hdr.iter().zip(&output.framebuffer.pixels).zip(&output.trace.records) {
            if trace[13] == 0.0 { assert_eq!(*hdr, [0.0; 3]); assert_eq!(*pixel, Rgb::BLACK); }
            else { luminous += 1; assert!(hdr.iter().any(|value| *value > 0.0)); }
        }
        assert!(output.hdr.iter().flatten().any(|value| *value > 1.0));
        assert_eq!(luminous, 90);
    }

    #[test]
    fn spectral_gpu_color_matches_independent_si_integration_and_cpu() {
        let inputs = [[1500.0f32, 1.0, 3.0e-9], [3000.0, 0.7, 3.0e-9], [6500.0, 1.0, 3.0e-9], [6500.0, 1.5, 3.0e-9], [12000.0, 1.0, 3.0e-9], [0.0, 1.0, 3.0e-9]];
        let builder = ton_ucf_frontend::VisibleSpectrumColorBuilder::new();
        let program = builder.build().unwrap(); let table = builder.response_table().unwrap();
        let input = inputs.iter().flatten().copied().collect::<Vec<_>>();
        let mut backend = VulkanBackend::new().unwrap();
        let output = execute_stage(&mut backend, &program, &input, 3, 7, 111, &[], Some(&table)).unwrap();
        let mut cpu = ton_ucf_frontend::CpuInterpreter::new();
        cpu.with_buffer(1, input.into_iter().map(ton_ucf_frontend::Value::F32).collect());
        cpu.with_buffer(2, vec![ton_ucf_frontend::Value::F32(0.0); inputs.len() * 7]);
        cpu.with_buffer(3, table.into_iter().map(ton_ucf_frontend::Value::F32).collect()); cpu.run_parallel(&program, inputs.len() as u32).unwrap();
        let mut maximum_relative = 0.0f64;
        for (input, actual) in inputs.iter().zip(output.chunks_exact(7)) {
            let xyz = ton_geodesic::observed_blackbody_xyz(input[0] as f64, input[1] as f64).unwrap();
            let rgb = ton_geodesic::xyz_to_linear_srgb(xyz).unwrap().map(|value| value * input[2] as f64);
            assert_eq!(actual[6], 1.0);
            for (actual, expected) in actual[..6].iter().zip(xyz.into_iter().chain(rgb)) {
                let relative = (*actual as f64 - expected).abs() / expected.abs().max(1.0e-10);
                maximum_relative = maximum_relative.max(relative);
                assert!(relative < 5.0e-4, "input={input:?}, actual={actual}, expected={expected}, relative={relative}");
            }
        }
        for (actual, expected) in output.iter().zip(cpu.buffer(2).unwrap()) {
            assert!((*actual as f64 - expected.as_f64().unwrap()).abs() < 1.0e-5 * expected.as_f64().unwrap().abs().max(1.0));
        }
        eprintln!("native visible spectrum maximum relative error to independent 1nm SI integral={maximum_relative}");
    }
}

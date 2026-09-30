//! wasmtime 加载 `ton_wasm.wasm` 并调用导出（与浏览器 C ABI 一致）。

use std::path::PathBuf;

use wasmtime::*;

fn wasm_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("target/wasm32-unknown-unknown/release/ton_wasm.wasm")
}

fn load_module(engine: &Engine) -> Result<Module, String> {
    let path = wasm_path();
    if !path.exists() {
        return Err(format!(
            "missing {} — run: cargo build -p ton-wasm --target wasm32-unknown-unknown --release",
            path.display()
        ));
    }
    Module::from_file(engine, &path).map_err(|e| e.to_string())
}

fn instantiate(engine: &Engine, module: &Module) -> (Store<()>, Instance) {
    let mut linker = Linker::new(engine);
    linker
        .func_wrap(
            "ton_host",
            "ton_host_present_rgba8_webgpu",
            |_: u32, _: u32, _: u32, _: u32| -> i32 { 0 },
        )
        .expect("stub ton_host");
    linker
        .func_wrap(
            "ucf_host",
            "ucf_host_bind_external_buffer",
            |_: u32, _: u32| -> i32 { 0 },
        )
        .expect("stub ucf_host bind");
    linker
        .func_wrap(
            "ucf_host",
            "ucf_host_l2_color_fill_webgpu",
            |_: u32, _: u32, _: i32, _: i32, _: i32, _: i32| -> i32 { 0 },
        )
        .expect("stub ucf_host fill");
    linker
        .func_wrap(
            "ucf_host",
            "ucf_host_l2_raster_clear_webgpu",
            |_: u32, _: u32, _: i32, _: i32, _: i32, _: i32| -> i32 { 0 },
        )
        .expect("stub ucf_host raster clear");
    linker
        .func_wrap(
            "ucf_host",
            "ucf_host_l2_raster_tri_webgpu",
            |_: u32, _: u32| -> i32 { 0 },
        )
        .expect("stub ucf_host tri");
    let mut store = Store::new(engine, ());
    let instance = linker
        .instantiate(&mut store, module)
        .expect("instantiate");
    (store, instance)
}

#[test]
fn ton_version_and_ucf_diag_exports() {
    let engine = Engine::default();
    let module = load_module(&engine).expect("ton_wasm.wasm");
    let (mut store, instance) = instantiate(&engine, &module);

    let version = instance
        .get_typed_func::<(), i32>(&mut store, "ton_version_code")
        .expect("ton_version_code")
        .call(&mut store, ())
        .expect("call");
    assert_eq!(version, 1_000);

    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");
    let out_ptr = 256u32;
    let diag = instance
        .get_typed_func::<u32, i32>(&mut store, "ton_ucf_application_contract_diag_mask")
        .expect("diag export")
        .call(&mut store, out_ptr)
        .expect("diag call");
    assert_eq!(diag, 0);
    let mask = memory.data(&store)[out_ptr as usize..out_ptr as usize + 4]
        .try_into()
        .map(u32::from_le_bytes)
        .expect("mask bytes");
    assert_ne!(mask, 0);
}

#[test]
fn ton_render_schwarzschild_small_frame() {
    let engine = Engine::default();
    let module = load_module(&engine).expect("ton_wasm.wasm");
    let (mut store, instance) = instantiate(&engine, &module);
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");

    let width = 32u32;
    let height = 24u32;
    let out_ptr = 4096u32;
    let status = instance
        .get_typed_func::<(u32, u32, f64, u32), i32>(
            &mut store,
            "ton_render_schwarzschild_rgba8",
        )
        .expect("render export")
        .call(&mut store, (width, height, 1.0_f64, out_ptr))
        .expect("render call");
    assert_eq!(status, 0);

    let need = (width * height * 4) as usize;
    let slice = &memory.data(&store)[out_ptr as usize..out_ptr as usize + need];
    assert!(slice.iter().any(|&b| b > 0), "frame should not be all zero");
}

#[test]
fn ton_render_frame_sem_classical_fbm() {
    let engine = Engine::default();
    let module = load_module(&engine).expect("ton_wasm.wasm");
    let (mut store, instance) = instantiate(&engine, &module);
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");

    let width = 16u32;
    let height = 12u32;
    let out_ptr = 8192u32;
    let render = instance
        .get_typed_func::<(u32, u32, f64, f64, i32, i32, i32, u32), i32>(
            &mut store,
            "ton_render_frame_rgba8",
        )
        .expect("ton_render_frame_rgba8 export");
    let status = render
        .call(&mut store, (width, height, 1.0_f64, 0.0_f64, 1, 2, 0, out_ptr))
        .expect("render call");
    assert_eq!(status, 0);

    let need = (width * height * 4) as usize;
    let slice = &memory.data(&store)[out_ptr as usize..out_ptr as usize + need];
    assert!(slice.iter().any(|&b| b > 0));
}

#[test]
fn ton_render_frame_rejects_invalid_discriminants() {
    let engine = Engine::default();
    let module = load_module(&engine).expect("ton_wasm.wasm");
    let (mut store, instance) = instantiate(&engine, &module);
    let render = instance
        .get_typed_func::<(u32, u32, f64, f64, i32, i32, i32, u32), i32>(
            &mut store,
            "ton_render_frame_rgba8",
        )
        .expect("ton_render_frame_rgba8 export");
    let bad_integrator = render
        .call(&mut store, (8u32, 8u32, 1.0_f64, 0.0_f64, 99, 1, 0, 4096u32))
        .expect("call");
    assert_eq!(bad_integrator, -3);
    let bad_disk = render
        .call(&mut store, (8u32, 8u32, 1.0_f64, 0.0_f64, 0, 99, 0, 4096u32))
        .expect("call");
    assert_eq!(bad_disk, -4);
    let bad_spin = render
        .call(&mut store, (8u32, 8u32, 1.0_f64, 1.0_f64, 3, 1, 0, 4096u32))
        .expect("call");
    assert_eq!(bad_spin, -5);
}

#[test]
fn ton_render_frame_ucf_tone_map_differs_from_raw() {
    let engine = Engine::default();
    let module = load_module(&engine).expect("ton_wasm.wasm");
    let (mut store, instance) = instantiate(&engine, &module);
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");
    let render = instance
        .get_typed_func::<(u32, u32, f64, f64, i32, i32, i32, u32), i32>(
            &mut store,
            "ton_render_frame_rgba8",
        )
        .expect("export");
    let width = 16u32;
    let height = 12u32;
    let out_raw = 4096u32;
    let out_ucf = 16384u32;
    let need = (width * height * 4) as usize;
    assert_eq!(
        render
            .call(&mut store, (width, height, 1.0_f64, 0.0_f64, 0, 1, 0, out_raw))
            .expect("raw"),
        0
    );
    assert_eq!(
        render
            .call(&mut store, (width, height, 1.0_f64, 0.0_f64, 0, 1, 1, out_ucf))
            .expect("ucf"),
        0
    );
    let raw = &memory.data(&store)[out_raw as usize..out_raw as usize + need];
    let ucf = &memory.data(&store)[out_ucf as usize..out_ucf as usize + need];
    assert_ne!(raw, ucf);
}

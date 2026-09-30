# Blackhole / TON

超大质量黑洞数值模拟与可视化（几何单位 G = c = 1）。crate 前缀 **ton-** 取自已知最大黑洞 **TON 618**。

## 快速开始

```bash
cargo check --workspace
cargo run -p ton-view -- blackhole.png --width 800 --height 450 --mass 1.0
```

## Crate 分层

```text
ton-types          ← 共享类型（栈底）
ton-metric         ← 史瓦西 / 克尔度规
ton-geodesic       ← 零测地线积分
ton-disk           ← 吸积盘辐射
ton-renderer       ← UCF 任务图 + CPU 光线追踪
ton-simulation     ← 场景模拟
ton-view           ← CLI 二进制
ton-wasm           ← Wasm cdylib（Web）
ton-napi           ← Node N-API cdylib（native）
```

| crate          | 职责                       |
|----------------|----------------------------|
| `ton-types`    | 质量、自旋、相机、RGB      |
| `ton-metric`   | 线元求值                   |
| `ton-geodesic` | 光线追踪（史瓦西赤道平面） |
| `ton-disk`     | Novikov-Thorne 薄盘近似    |
| `ton-renderer` | UCF 任务图 + 逐像素着色 + 后处理 |
| `ton-simulation` | `Scene` 场景模拟         |
| `ton-view`     | 命令行输出 PNG             |
| `ton-wasm`     | Wasm 导出 + UCF 同模块链入 |
| `ton-napi`     | Node native RGBA8 渲染     |

## 仓库布局

```text
blackhole/
  projects/
    crates/              # ton-* 算法 + ton-wasm / ton-napi
    packages/
      blackhole/                 # @doki-land/blackhole（src/wasm + src/node 宿主）
      blackhole-unknown-wasm32/  # wasm 平台袋
      blackhole-<triple>/        # node 平台袋（win32/linux/darwin × arch）
      homepage/                  # VMZ 演示（仅 wasm）
    examples/
```

## Web 演示

```bash
pnpm install
pnpm run build:wasm
pnpm run build:ts
pnpm run dev:homepage
```

浏览器打开后点击「渲染一帧」。物理在 `ton-simulation` + `ton-renderer` CPU 测地线；UCF 同模块 wasm + `ucf_host` L2 WebGPU 上屏（降级 L0 readback）。

设计文档与路线图见工作区 `规划设计/` 与 `决策和进度表/`（不进本仓 README 外链）。

## 示例

```bash
cargo run -p ton-example-render-schwarzschild
# → schwarzschild.png
```

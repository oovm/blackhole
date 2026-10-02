# ton-view

黑洞模拟器可执行入口：`cargo run -p ton-view` 输出 PNG。

## 原生光谱薄盘

```sh
cargo run -p ton-view -- blackhole.png --ucf-spectral --camera level --width 640 --height 360
```

`--ucf-spectral` 使用 UCF 原生 Vulkan 程序执行 F64 轨迹、F32 盘面频移、可见光谱积分与 sRGB 显示编码。当前为无纹理不透明薄盘，温度采用无内缘力矩牛顿基线，不是 Novikov-Thorne 通量模型。观察者响应采用 CIE1931 高斯近似和 5nm 积分，不是官方标准表。

- `--peak-kelvin` 设置盘面峰值温度，默认 6500K。
- `--exposure` 设置显示曝光，默认 `1e-10`。
- `--step-size` 设置最大仿射步长，默认 `0.02M`。
- `--max-steps` 设置逐射线预算，默认 32768。
- `--camera level` 使用 80 度倾角与 50M 观察距离，保持完整盘面构图。
- `--graybody` 启用有界、随圆轨道平流的程序化灰体发射结构。它只调制发射强度，不是流体模拟或 Novikov-Thorne 通量。

此入口暂不支持 Kerr、Bloom、阴影覆盖、HDR 导出或其他盘/积分器选项。设备、程序或数值失败均返回非零退出码，不切换到 CPU 渲染。

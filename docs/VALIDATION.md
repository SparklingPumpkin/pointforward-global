# PFG-v2 交付验证

验证日期：2026-09-28（Asia/Shanghai）。未运行大型预训练。

## 轻量检查

- 全部 PointForward、入口脚本与 vendored 预处理 Python 文件通过 `py_compile`。
- `pytest -q tests`：包含 PFG-v2 编排合同测试，结果见 release run 证据。
- `scripts/validate_install.py`：Git 交付权重通过文件大小和 SHA256 校验；`pointforward` 从本交付仓库的 `src/` 导入。
- 项目控制面 guard：PASS。

## PFG-v2 fast-g8 推理复现

当前主线使用交付脚本同构的 GR-v1-s 运行，输入为 GR-v1-i 的固定 26.668159 dB payload，并加载 `weights/refiner_scene017_fast_g8.pt`。对全部 156 帧、三前向相机得到：

| 指标 | 结果 |
|---|---:|
| 全局 PSNR | 27.085058566851494 dB |
| 非天空 PSNR | 26.529556188827907 dB |
| 动态 bbox PSNR | 21.998108961643318 dB |
| Gaussian 数量 | 20,677,973 |
| RGB 优化输入到 PLY | 424.772963 s（7.07955 min） |
| 峰值 CUDA allocation | 约 76.41 GiB |

完整 manifest、payload SHA256、逐帧指标、计时与关键帧对照图位于 [scene017-fast-g8](validation/scene017-fast-g8/manifest.json)。其中 payload SHA256 为：

- `final_static_128d.pt`: `afcbef52f124678b5f5989da09cc7e9bdd5e51a1ef949bebde397d8488954693`
- `final_dynamic_128d.pt`: `e975437af16f43abb31b3f74c495136dd0163e1431d7e3afa5f6a5d829e678d5`
- `global_dynamic_meta.pt`: `15287f560041925e3e10bca7fbc756b61e828d5b721d8fbf74a7bd743c31e03e`

7.08 分钟是固定的 RGB 后优化输入到最终 PLY 的计时边界，不包含原始数据预处理、前馈深度、全局建图、动态车辆重建或此前优化。它不能表述为从原始 RGB 开始的完整流水线耗时。

## PFG-v1 legacy 复现

验证使用交付工作树内的：

- `scripts/render_final_global_runtime.py`
- `src/pointforward/`
- `weights/pointforward_waymo_ds_paper_ckpt_020000.pt`
- 本地可选的 `weights/refiner_scene017_27db.pt`（该文件不进入 Git 上传）
- 由 `configs/argoverse_global.template.yaml` 在交付仓库解析的配置

输入是未打包的旧 PFG-v1 最终 scene017 payload；其三个文件 SHA256 已写入完整 [manifest](validation/scene017-27db-metrics/manifest.json)。在全部 156 帧、三前向相机上得到：

| 指标 | 结果 |
|---|---:|
| 全局 PSNR | 27.000626453986534 dB |
| 非天空 PSNR | 26.47900737860264 dB |
| 动态 bbox PSNR | 21.75951957091307 dB |
| Gaussian 数量 | 20,677,973 |
| 指标渲染耗时 | 242.290 s |

共享机器当时没有完全空闲的 48 GiB GPU，因此验证使用 `camera_batch_size=1`；相机、帧、Gaussian、128D+UNet 渲染和指标公式均未改变，数值与原验收 manifest 完全一致。

## PLY 合同

使用交付仓库的 `export_runtime_payload_ply.py` 成功导出 frame 0：

- 20,227,973 静态 + 450,000 动态 = 20,677,973 Gaussian
- 文件大小：1,406,102,582 bytes
- 导出耗时：8.054 s

测试 PLY 写在临时目录，没有进入交付仓库。PLY 是 preview RGB / degree-0 SH，不等同于 128D+UNet 的 27 dB 输出。

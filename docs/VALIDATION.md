# PFG-v1 交付验证

验证日期：2026-09-26（Asia/Shanghai）。未运行大型预训练。

## 轻量检查

- 全部 PointForward、入口脚本与 vendored 预处理 Python 文件通过 `py_compile`。
- `pytest -q tests`：48 passed。
- `scripts/validate_install.py`：5 个权重全部通过文件大小和 SHA256 校验；`pointforward` 从本交付仓库的 `src/` 导入。
- 项目控制面 guard：PASS。

## 27 dB 推理复现

验证使用交付仓库内的：

- `scripts/render_final_global_runtime.py`
- `src/pointforward/`
- `weights/pointforward_waymo_ds_paper_ckpt_020000.pt`
- `weights/refiner_scene017_27db.pt`
- 由 `configs/argoverse_global.template.yaml` 在交付仓库解析的配置

输入是未打包的既有最终 scene017 payload；其三个文件 SHA256 已写入完整 [manifest](validation/scene017-27db-metrics/manifest.json)。在全部 156 帧、三前向相机上得到：

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

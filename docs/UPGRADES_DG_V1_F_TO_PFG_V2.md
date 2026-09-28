# 从 DG-v1-f 到 PFG-v2 的技术升级

本文记录 scene017 从动态车辆基线 DG-v1-f 到当前 GR-v1-s / PFG-v2 交付主线的已采用升级。指标均为 156 帧、三前向相机的同一渲染合同；后优化只改变场景 payload 或 refiner overlay，不代表通用 checkpoint 已获得相同能力。

## 1. 起点：DG-v1-f

DG-v1-f 已经采用真正全局的静态和动态表示。静态背景来自逆序滑窗全局建图；动态车辆按全局 ID 汇总完整轨迹，从所有可见 `(frame, camera)` 中只重建一次。每辆车贪心选择 12 个视图，分数为：

```text
0.65 * min_circular_azimuth_distance / pi
+ 0.35 * sqrt(projected_area / max_projected_area)
```

这同时偏好观察方位多样性和较大的车辆投影。45 辆车各 10,000 点，共 450,000 个动态 Gaussian。基线全局 PSNR 为 19.6822 dB，非天空 19.6184 dB，动态 bbox 16.7232 dB；相比均匀时序选图，动态 bbox 提高 1.2605 dB。

## 2. 已采用升级

| 阶段 | 技术变化 | 全局 PSNR |
|---|---|---:|
| GR-v1-b | 冻结几何、动态、网络，只优化静态 128D feature；groups=4，3 epochs | 23.3375 |
| GR-v1-c | 静态 groups=4 继续到 6 epochs | 23.994672 |
| GR-v1-d | 增加静态 feature 自由度，groups=8 | 24.177398 |
| GR-v1-e | 优化 refiner RGB head | 24.338966 |
| GR-v1-f | 优化最后 up block + head | 24.647833 |
| full refiner e1 | 解冻完整 refiner | 25.081662 |
| full refiner e3 | 延长完整 refiner 优化 | 25.626057 |
| log-MSE | 以 log-MSE 改善低质量帧 | 25.981048 |
| lower-LR log-MSE | 降低学习率继续优化 | 26.105534 |
| frame-log-MSE | loss 对齐逐帧三相机 PSNR 聚合 | 26.231108 |
| GR-v1-g | 优化动态车辆 128D feature | 26.293228 |
| GR-v1-i | 仅在非天空且排除动态 bbox 的区域优化静态 XYZ，最大位移 5 cm | 26.668159 |
| GR-v1-j / PFG-v1 | refiner 与 2 cm、1 cm XYZ 交替优化 | 27.000626 |
| GR-v1-s / PFG-v2 | 全 156 帧、groups=8 的一次静态 feature 尾部，三相机 batching | **27.085059** |

最终非天空 PSNR 为 26.529556 dB，动态 bbox PSNR 为 21.998109 dB；静态 20,227,973 点、动态 450,000 点，总计 20,677,973 个 Gaussian。

## 3. 当前快速尾部为何有效

GR-v1-i 已经把几何和动态表示推进到 26.668159 dB。旧 GR-v1-j 随后交替训练 refiner 和静态 XYZ，四个串行阶段从共同输入到 PLY 需要 1391.46 秒。实验表明这一阶段的主要剩余误差可以由静态 128D feature 吸收，不必继续移动几何。

PFG-v2 因而只运行一次 groups=8、1 epoch、学习率 0.005、anchor weight 0.0001 的静态 feature 优化，并保留全部 156 帧监督。groups=8 比 groups=4 提供更细的每点特征残差；同一帧三个相机一起 rasterize，减少 Python 调度和重复场景构造；`empty_cache_every=0` 避免每帧同步清 cache。结果是质量略升，同时尾部降至 424.77 秒（7.08 分钟）。

这个时间合同从固定的 26.668159 dB RGB 后优化输入开始，到最终 PLY 完成结束，不包含原始 RGB 预处理、InfiniDepth、全局建图、动态重建以及此前的场景优化。PFG-v2 峰值 CUDA allocation 约 76.41 GiB，需要 80 GiB 级 GPU。

## 4. 未采用的实验分支

以下结果用于判断方向，但没有进入默认主线：

| 分支 | 结果 | 未采用原因 |
|---|---:|---|
| 动态 scale multiplier | scale=1.0 最好 | 改尺度不能解决动态误差 |
| 四阶段 frame stride=3 | 26.828986 dB，576.76 s | 未达到 27 dB |
| 减少阶段且 stride=2 | 26.780413 dB | 质量进一步下降 |
| 每 Gaussian RGB residual e2 | 27.067535 dB | 快，但引入额外直接 RGB 容量和不同渲染合同 |
| worst-10 / groups=32 | 26.690885 dB | 局部帧优化污染共享 Gaussian，整体退化 |
| full-frame groups=4 | 27.046574 dB，399.17 s | 已达标，但 groups=8 质量更高 |

## 5. 交付实现

`scripts/run_current_pipeline.py` 默认使用 `--final-recipe fast-g8`，在 `13_static_xyz_5cm` 后生成 `14_static_g8_fast_final/runtime-payload-refined`。旧 PFG-v1 尾部完整保留，可用 `--final-recipe legacy-alternating` 回退。

`scripts/refine_static_features_fullscene_rgb.py` 新增 `--refiner-state`、`--frame-list`、`--frame-stride`、`--camera-batch-size` 和 `--empty-cache-every`，并在 manifest 中记录实际训练帧、batch 和 overlay 身份。

最终高质量结果仍依赖一套真正全局的静态 Gaussian、一套持久动态 Gaussian、128D feature 和 UNet/refiner。导出的 PLY 只含 preview RGB / degree-0 SH，用于几何交付，不等价于 27.085059 dB 的运行时渲染。

## 6. 证据位置

- DG-v1-f：`experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-diagnostics/DG-v1-f/run-20260923T091123Z-scene017-diverse-large-12views/report.md`
- GR-v1-s：`experiments/argoverse-inference/global-dynamic-consistency/gaussian-post-refinement/GR-v1-s/run-20260927T202155Z-full-frame-g8-batch3/`
- 交付验证：`docs/validation/scene017-fast-g8/`
- 旧 PFG-v1 证据：`docs/validation/scene017-27db-metrics/`

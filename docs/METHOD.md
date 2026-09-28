# PFG-v2 当前主线方法

PFG-v2 的最终渲染不是把逐 clip 的 PLY 拼起来，也不是 degree-0 RGB 直接渲染。它始终使用一套最终静态全局 Gaussian、一套带全局车辆 ID 和完整 box 轨迹的动态 Gaussian，并通过原 PointForward 的 128D feature renderer 与 UNet/refiner 解码 RGB。

主流程如下：

1. 逆序处理 8 帧窗口，步长 2（重叠 6 帧），使用 2 cm 体素维护静态全局 Gaussian。
2. 历史点仅在获得更近观测时更新属性；20 m 内更新权重 0.5，20–80 m 平滑衰减，80 m 外冻结；静态中心不在跨 clip 融合中移动。
3. 每个全局车辆 ID 从完整轨迹中按“投影面积 + 观察方位多样性”选择 12 个视图，仅前馈重建一次动态 Gaussian。
4. 对静态 128D feature、refiner、动态 128D feature和 5 cm 约束内的静态 XYZ 依次进行场景级 RGB 后优化。
5. 默认尾部用全部 156 帧监督，对静态 128D feature 做一次 groups=8 优化；同一帧三相机合并为一个 batch，且不在每帧强制清 CUDA cache。
6. 每一帧都以完整全局 Gaussian、对应时刻动态 box、128D feature 和 UNet/refiner 渲染。

scene017 的 PFG-v2 固定验收结果为：156 帧、三前向相机、全局 PSNR 27.085059 dB，非天空 26.529556 dB，动态 bbox 21.998109 dB，共 20,677,973 个 Gaussian。相对 PFG-v1，质量提高 0.084432 dB；固定 26.668159 dB 输入 payload 到最终 PLY 的尾部耗时由 1391.46 秒降至 424.77 秒。

`scripts/run_current_pipeline.py` 固化了完整可恢复阶段。默认 `fast-g8`，旧四阶段尾部可通过 `--final-recipe legacy-alternating` 恢复。这里的 7.08 分钟不是原始 RGB 到 PLY 的完整端到端耗时，只是共同输入后的最终优化合同；完整前序阶段仍会另外耗时。27 dB 也是 scene017 数据、相机、mask、版本和硬件数值环境共同定义的复现合同，不是任意场景保证。

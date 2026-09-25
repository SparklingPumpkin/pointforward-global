# PFG-v1 当前主线方法

PFG-v1 的最终渲染不是把逐 clip 的 PLY 拼起来，也不是 degree-0 RGB 直接渲染。它始终使用一套最终静态全局 Gaussian、一套带全局车辆 ID 和完整 box 轨迹的动态 Gaussian，并通过原 PointForward 的 128D feature renderer 与 UNet/refiner 解码 RGB。

主流程如下：

1. 逆序处理 8 帧窗口，步长 2（重叠 6 帧），使用 2 cm 体素维护静态全局 Gaussian。
2. 历史点仅在获得更近观测时更新属性；20 m 内更新权重 0.5，20–80 m 平滑衰减，80 m 外冻结；静态中心不在跨 clip 融合中移动。
3. 每个全局车辆 ID 从完整轨迹中按“投影面积 + 观察方位多样性”选择 12 个视图，仅前馈重建一次动态 Gaussian。
4. 对静态 128D feature、refiner、动态 128D feature和小范围静态 XYZ 依次进行场景级 RGB 后优化。
5. 每一帧都以完整全局 Gaussian、对应时刻动态 box、128D feature 和 UNet/refiner 渲染。

scene017 的固定验收结果为：156 帧、三前向相机、全局 PSNR 27.000626 dB，非天空 26.479007 dB，动态 bbox 21.759520 dB，共 20,677,973 个 Gaussian。最终静态 XYZ 优化的位移上限依次为 5 cm、2 cm、1 cm，最后一步平均位移只有 0.463 mm。

`scripts/run_current_pipeline.py` 固化了完整可恢复阶段。它刻意保留了达到验收结果所经过的已接受迭代，而没有把这些场景级优化伪装成通用 checkpoint 的单次前馈能力。对新场景可以执行同一流程，但 27 dB 是 scene017 数据、相机、mask、版本和硬件数值环境共同定义的复现合同，不是任意场景保证。

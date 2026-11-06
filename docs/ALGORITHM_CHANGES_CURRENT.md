# 算法与交付变更记录

此文件持续维护当前交付变更。[旧变更记录](ALGORITHM_CHANGES.md) 因历史验收冻结保留原文；晋级事实由 [版本总账](project-control/VERSION_LEDGER.md) 记录。

## 2026-10-22 · 可选车辆前期校准（未重测）

新fresh入口增加 `--vehicle-calibration bbox-expand|adaptive|both`，默认adaptive保持原策略。bbox-expand跳过几何准备/SAM/自适应对齐及LiDAR过滤，使用原始LiDAR，并将车辆前馈局部候选/解码框前后各加0.5m、左右各加0.3m；both先执行原自适应流程，再对该重建框扩张。扩大量参数为 `--bbox-front-back-margin` / `--bbox-left-right-margin`，按单侧米数输入，传给已有解码器时转换为总长+1.0m、总宽+0.6m；高度不变。原始评估框、选视图评分框、静态前馈框不扩大，自适应阶段的LiDAR归属范围也不被扩框改变。

复用已冻结车辆前馈的decode margin参数及原数值实现；当前可维护编排为 `scripts/run_vr_v1_s_generalized.py`，每次输出保存选项及实际扩大量。自适应几何准备与fit-apply显式传场景号，避免旧默认017误用于其他场景。replay维持冻结状态，不接受改变校准选项。仅代码及命令检查，未新增训练、渲染或性能结论；双质量阈值/60min停止策略保持不变，历史验收源码和资产不变。

## 2026-10-22 · 新场景停止条件更新（未重测）

按用户要求，仅更改后续fresh运行停止逻辑：全局PSNR≥30且车辆原始bbox PSNR≥28，两项须在同一次完整场景三相机评估同时满足；或者冷启动起点累计3600s触发停止。优化前及每100步检查，评估成本计入统一计时；空车辆帧按正式指标口径跳过，不将无车场景误判为车辆达标。取消原1700步提前停止，其他训练公式、LR、输入和输出保持原配方。达到停止条件后保存、渲染视频及导出PLY，后续交付开销另记，不声称完整交付一定在3600s内。

当前默认别名转到 `run_vr_v1_s_generalized.py`，复用原编排及数值模块，新停止参数在 `configs/runtime-stop/`，训练器为 `scripts/refine_unified_scene_rgb_stopping.py`。PFG-v4冻结源码、配方、指标、权重和验收证据不覆盖；保留原版本入口供历史复现。此次没有启动实验、重新验收或增加主线版本号。

## 2026-10-22 · VR-v1-s / PFG-v4

用户指定当前最佳版本覆盖默认交付。固化scene017第1670步32D Gaussian、配套PCA/U-Net、修正车辆轨迹、SMPL姿态与模板、通用权重、隔离源码、原始计时、PLY和完整视频。

相对旧交付VR-v1-b，增加车辆几何输入校正、冲突背景LiDAR过滤，取消bbox扩张，加入SMPL全局行人与姿态空缺插值；静态/车辆前馈可并行。128D前馈特征经当场PCA降到32D并适配U-Net，实际32D raster节省联合优化成本。完整三相机训练改为车辆union ROI权重1，LR在600/1200步乘0.5，最多1700步及原始流程1480s停止请求。静态源RGB合成0.75及主要静态前馈规则维持源配方。

全156帧三相机全局31.131654、车辆bbox28.329244、非天空30.685005dB。源实测输入预处理至优化完成1485.371s，至完整视频/指标/PLY交付1609.087s。车辆29dB研究目标未达标，交付接受不改写该结论；本次固定状态独立重放不声称重新冷启动训练。

新默认入口 `scripts/run_pipeline.py`，版本专属 `scripts/run_vr_v1_s_pipeline.py`支持fresh/replay。旧 `run_current_pipeline.py`被历史验收冻结，保留为VR-v1-b复现入口。旧资产与证据保留，新主线覆盖根说明和默认交付选择。

详见 [release](releases/VR-v1-s/README.md)、源码差异及资产SHA256。后续算法变动在此追加日期、基线、实际改动、指标、完整计时边界及验收链接。

## 2026-10-28：接受并固化LIM-v1-r

fresh默认新增独立原始输入校准，详见 `LIDAR_BBOX_PREPROCESSING.md`；数值核心与研究版保持一致，工程新增完整场景深度/框导出及显式消费桥。旧adaptive/both、replay与冻结源码保留。此为用户接受的前处理数据合同变化，不追认原研究质量验收，也不复用旧PSNR/计时作为新默认实测。

## 2026-11-06：三个独立可选预处理模块（PFG-v5-a，非主线晋级）

新增 `scripts/preprocess_scene.py` 与 `pointforward_preprocessing`，可独立选择LIM-r、新深度预测＋天空mask修复、场景范围测量＋有限天空壳，支持任意组合、逐模块manifest与计时及数组API。数值定义保留研究版公式；输入复用显式登记，预处理不执行Gaussian拟合或训练。天空导出Torch/无pickle NumPy；下游需显式接入合成。默认PFG-v5重建、原LIM入口、历史冻结源码和交付资产保持不变。

34项测试通过；scene017帧1/61三相机真实新推理验证通过，明确复用既有LIM-r校准，无全场景冷启动或新PSNR结论。[说明](PREPROCESSING.md)、[验收证据](releases/PFG-v5-a/report.md)。

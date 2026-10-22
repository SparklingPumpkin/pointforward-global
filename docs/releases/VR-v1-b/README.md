# VR-v1-b 固化交付

版本对应研究实验 `VR-v1-b/run-20261020T012848Z-feedforward-unified-all950`，最终选择 950 步。包内 `source/` 保存该实验数值源码，资产位于 `artifacts/scene017_vr_v1_b/`，专用解码网络为 `weights/refiner_scene017_vr_v1_b.pt`。研究原始证据位于 `evidence/`，不依赖原研究仓库执行。

## 算法与检查点

静态前馈逆序、20 帧窗口/步长 20、context stride=5、1 cm 体素；静态 isotropic-trace 尺度上限 0.005 m。静态采用 three-clip-mutable-frozen-position、closest-smoothstep（near=20 m，far=80 m，alpha=0.5），clip 查询原点、static offset gain=0。车辆按全局 track 选择 diverse-large 的 12 个视图一次前馈，角度权重 0.65，scene 查询原点、尺度上限 0.05 m；车辆解码 bbox 总长额外 1 m、总宽额外 0.6 m。

随后只用原始前馈状态与通用 checkpoint 开始一次连续 **950 步 AdamW**。静态、动态 Gaussian 的 XYZ、尺度、旋转、opacity、完整 128D feature、time/sigma，与 U-Net、车辆轨迹平移/yaw 同时优化；检查点不冻结、不重置、不恢复优化器。冻结深度/图像融合网络、点数量与身份、缓存源 RGB、标定、原评估框及 canonical 参考位姿。

loss = 三相机平均 log RGB MSE + 0.25 × 严格可见、按 actor 观测数平衡的 ROI log MSE + 0.02 × 轨迹平滑 + 0.0001 × anchor。ROI 使用 strict visibility 避免画外框变成画内裁剪；正式 bbox 指标沿用原始固定评估框，未扩大评估区域或改动历史指标定义。

| 步数 | 全局 PSNR | 动态 bbox PSNR | 非天空 PSNR |
|---:|---:|---:|---:|
|350|28.106923|23.961671|27.622513|
|550|28.914482|24.734797|28.425473|
|750|29.835084|25.870118|29.348719|
|950|30.294842|26.334712|29.824187|

初始静态学习率 feature=0.03，XYZ/scale/rotation=0.02，opacity/time/sigma=0.01；动态 feature=0.006，其余=0.004；U-Net=0.0002，pose=0.001。所有学习率在零基 step 240、700 乘 0.4。静态 tanh 上限：feature=0.5、XYZ 每轴 0.02 m、log-scale=0.25、rotation=5°、opacity-logit=0.5、time=0.05、log-sigma=0.25；动态对应 1、0.3 m、0.7、20°、2、0.05、0.25。轨迹每 4 帧控制点、平移上限 0.5 m、yaw 上限 15°。完整参数见 `source/scripts/refine_unified_scene_rgb.py` 与 `evidence/joint-all/parameter_contract.json`。

## 执行

从项目根目录执行，输出目录必须为空：

```bash
# 固定场景资产重渲染视频、指标并导出 PLY。
python scripts/run_current_pipeline.py --mode replay \
  --data-root /data/argoverse/processed/training --output-root /work/vr-replay --gpu 0

# 从处理后输入重新前馈并联合优化950步，不加载历史场景状态。
python scripts/run_current_pipeline.py --mode fresh \
  --data-root /data/argoverse/processed/training --output-root /work/vr-fresh --gpu 0
```

`fresh` 默认保存/渲染 350、550、750、950 四个检查点。输出包含配置、实际命令、逐步正值 MSE 与训练 loss、参数变化、四检查点指标/视频、最终 PLY 和 `pipeline_timing.json`。入口创建统一计时器后才读取 RGB；所有子进程结束、PLY 导出、视频复制并 fsync 后结束。所有成本保留，失败也记录已耗时间。`fresh` 不接受续训，避免混入场景缓存。原始传感器数据预处理不包含在入口内，输入是处理后 RGB/LiDAR/标定/实例/masks。

依赖见项目根环境文件；帧数从数据读取，三相机分辨率 576×768，预测拼接视频 2304×576、10 fps。`replay` 仅支持固定 scene017；其他场景用 `fresh --scene ...`，质量未经本次验证。

## 资产及数值移植边界

静态 8,596,883 点，动态 450,000 点，完整 128D。最终 payload 包含修正轨迹 meta 与原评估框；不能混用旧 refiner。原始前馈 payload 同时保留用于审计起点。

唯一已有数值源码文件改动是 `refine_unified_scene_rgb.py` 的编排工具 import，改为包内 `delivery_runtime_helpers.py`：消除研究仓库绝对路径，使用当前 Python 和同一渲染参数，不改优化公式。配置仅替换数据/深度库/权重路径，关闭初始化 ImageNet 下载，随后加载同一通用 checkpoint。原始配置原样保存在 `evidence/`。

快照含研究期间其他脚本以保留 import 闭包；未被交付入口调用的历史编排脚本仍可能带原路径，不应直接执行。受支持入口只有根目录 `run_current_pipeline.py` / `run_vr_v1_b_pipeline.py`。这些入口和调用的数值模块不需原研究仓库、workflow skill 或其总控状态。

视频使用静态 source-RGB blend=0.75、固定最终全局高斯和对应车辆轨迹，非 clip 历史快照拼接。RGB 为重建输入，指标是源帧重建 PSNR，并非新视角泛化质量。PLY 为 frame000 顶点 RGB 预览，不承载完整 128D+U-Net 运行时。

## 耗时审计

缓存后实际增量 2065.830 s，历史前馈及其源视频前缀 442.946 s，历史累计 2508.776 s。四次检查点保存/评估/视频合计 756.254 s；末尾 PLY 导出 9.543 s。这是历史依赖账目，**不是一次新 RGB 冷启动实测**。本次只执行包内独立完整重渲染和 PLY 导出，做配置/代码/命令核验，不重复950步冷启动训练。

资产来源和 SHA256 见 `delivery_manifest.json`；源码指纹见 `source_manifest.json`，入口与模板指纹见 `entrypoint_manifest.json`。交付验证见 [完整报告](../PFG-v2-a/run-20261020T063212Z-unified-delivery-validation/REPORT.md)。独立重渲染三项PSNR与原实验绝对差均为0，PLY的SHA256一致，重渲染及导出共151.719s。该数字仅衡量读取交付场景状态后的重渲染。旧 FFR-v1-l ver2 和原 PFG-v2 入口保留，不改写历史结论。

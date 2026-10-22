# VR-v1-s 固化交付 · PFG-v4

来源 `VR-v1-s/run-20261022T054208Z-cold-compact-union1700`，选择实际保存的第 **1670步**。包内数值实现为 `source/`，场景资产位于 `artifacts/scene017_vr_v1_s/`，匹配U-Net为 `weights/refiner_scene017_vr_v1_s.pt`。运行资产都是包内实际文件，不依赖原研究仓库。

|完整156帧 × 三相机|PSNR|
|---|---:|
|全局源帧|31.13165373679919 dB|
|原始车辆bbox|28.32924439356877 dB|
|非天空|30.68500505349575 dB|

车辆29dB研究目标未达标，原始失败合同/报告保留在 `evidence/`。用户选择将此结果交付，与研究验收通过是两回事。本次独立重渲染验证固定资产、代码和指标一致性，未重新训练冷启动全流程。

## 算法与参数

静态前馈保持逆序20帧窗口/步长20、context stride5、1cm体素、three-clip-mutable-frozen-position；closest-smoothstep near20m/far80m/alpha0.5，clip query、static offset gain0，isotropic-trace最大尺度0.005m。车辆按全局ID选12个diverse-large视图重建一次，角度权重0.65、scene query、最大尺度0.05m。

几何输入先用大且具有方向信息的车辆观测拟合全局共享有效时差，再过滤与车辆可见表面冲突的背景LiDAR提示。只修正几何输入与车辆框，不迁移SH颜色，不拟合每车局部残差；车辆bbox解码额外长宽均为0。原始评估框不变。

行人使用全局canonical SMPL实例；仅在有效bbox支持下填补姿态：内部最多30帧空缺做旋转SLERP/平移插值，边缘最多保持5帧，保留原有效姿态。标注姿态是离散值，运行时按冻结实现变形。

前馈128D颜色特征只用本次场景未优化Gaussian特征拟合PCA，压到32D，对U-Net首层做alpha-aware折叠。实际raster为32D，不是渲染128D后再压缩。`final_*_128d.pt` 为兼容文件名，真实维度由张量及manifest `feature_dim=32`确认。**feature_projection.pt、Gaussian、U-Net、轨迹必须成套使用。**

一个连续AdamW联合优化静态/动态Gaussian的XYZ、尺度、旋转、opacity、feature、time/sigma，U-Net及车辆轨迹。完整三相机，global log RGB MSE + 1×严格可见车辆union ROI log MSE + 0.02×轨迹平滑 + 0.0001×anchor。静态不提前冻结；冻结深度/图像融合网络、点数和身份、缓存源RGB、标定与原评估框。静态source-RGB blend0.75属于构建时缓存颜色合成，非目标帧GT查询。

最多1700步，零基600/1200步LR乘0.5；原实验在原始计时1480s附近请求停止，实际完成1670步及保存。精确数值见 `pipeline_recipe.json`、`source/scripts/refine_unified_scene_rgb.py`、`evidence/joint_options.json`。`parameter_contract.json`保留的actor-balanced标签是旧元数据，实际执行 `vehicle_loss=union`；PLY/render描述残留128D文字也不代表真实维度。

## 使用与依赖

从项目根目录执行，输出目录必须为空：

```bash
python scripts/run_pipeline.py --mode replay \
  --data-root /data/argoverse/processed/training --output-root /work/replay --gpu 0

python scripts/run_pipeline.py --mode fresh \
  --data-root /data/argoverse/processed/training --output-root /work/fresh \
  --scene 017 --gpu 0 --vehicle-gpu 1

python scripts/validate_vr_v1_s_delivery.py --replay /work/replay
```

fresh从处理后RGB/LiDAR/标定/实例/天空mask/SMPL标注开始，重新生成几何校正、SAM mask、前馈Gaussian、SMPL tracks、PCA、优化状态、视频和PLY；不读取已交付场景状态。两GPU时静态/车辆前馈并行，一GPU则串行，统一计时保留真实耗时。1700步上限及1480s停止请求沿用源配方，硬件速度会影响实际步数，读取本次 `checkpoints.json` 和计时器为准；未验证其他场景质量。

replay仅支持固定scene017，第1670步状态，156帧、每相机576×768、10fps；读原RGB/masks作GT对照及指标，计时为cached_incremental。单独PLY为frame000几何/顶点RGB预览，不代表32D+U-Net的PSNR或整个动态序列。

依赖沿用根 `requirements.txt`；SAM2源码在 `source/SAM2`，需要Hydra/OmegaConf及兼容Torch/torchvision。`--sam-python`支持单独SAM解释器。本次主数值环境Python3.9.25、Torch2.8.0+cu128、A10080GB；根历史环境文件的CUDA版本不等于预编译扩展ABI。可选 `gsplat_cuda.so` 仅在显式 `--use-pinned-extension`时加载，其他平台使用正常安装的gsplat扩展。FFmpeg/ffprobe需在PATH；源码LICENSE随包保留，SMPL模板/权重沿用其原许可。

通用离线权重为PointForward、InfiniDepth depthsensor、SAM2.1 Hiera Tiny、pedestrian head和SMPL_NEUTRAL.npz。关闭额外ImageNet初始化下载后加载同一PointForward checkpoint；SAM2及InfiniDepth均从当前包源码加载。

## 时间与验证

|源实际阶段|子进程秒|
|---|---:|
|几何准备|116.464|
|SAM RGB mask|51.242|
|全局偏移拟合/冲突LiDAR过滤|108.557|
|静态前馈（与车辆并行）|74.333|
|车辆前馈|170.793|
|SMPL前馈|22.374|
|本场景PCA|9.481|
|优化/初始化/保存|1006.387|
|完整视频/指标|118.768|
|PLY导出|4.850|

源真实墙钟：输入预处理至优化保存并退出 **1485.371231320314s**；完整视频、指标、PLY **1609.0870128450915s**。两前馈分支重叠，不能直接求和；不包含数据集原始格式转换及SMPL标注生成。所有场景状态均在源冷启动内新生成，未借历史优化前缀；原始证据见 `evidence/pipeline_timing.json`、`optimization_timing.json`、`run_cold.py`、`input_provenance.md`。

本包独立重放见 [验证结果](../PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/validation.json) 及 [报告](../PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/REPORT.md)。重放耗时不属于新冷启动训练结果。

`source_manifest.json`记录全部源码指纹；仅三项便携改动：训练器改包内编排helper import，SAM模型路径改参数，预编译扩展改显式启用。模型、loss、优化器、渲染公式不改。`delivery_manifest.json`记录源/交付资产SHA256；`entrypoint_manifest.json`冻结版本专属入口和模板，默认别名留作后续维护。`source/scripts`为保留导入闭包而附带的历史脚本可能含旧机器路径，受支持入口仅根 `run_pipeline.py` / `run_vr_v1_s_pipeline.py`。

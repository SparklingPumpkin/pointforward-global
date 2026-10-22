# PointForward Global · VR-v1-s

当前交付主线 **PFG-v4 / VR-v1-s**，固化 Argoverse017 第 **1670 步**完整状态：32D Gaussian、PCA 投影基、配套 U-Net、车辆轨迹、SMPL 行人、通用权重、隔离源码、PLY 预览和视频。

完整156帧、三前向相机源帧重建：全局 **31.131654 dB**，车辆 bbox **28.329244 dB**，非天空 **30.685005 dB**。指标属于特征 Gaussian + 配套 U-Net 运行时；单独预览 PLY 不等价。车辆29dB研究目标仍未达到。

- [GT / 预测对照视频](artifacts/scene017_vr_v1_s/final_global_runtime_gt_vs_predicted_front3.mp4)
- [预测视频](artifacts/scene017_vr_v1_s/final_global_runtime_predicted_front3.mp4)
- [车辆局部对比](artifacts/scene017_vr_v1_s/vehicle_crops.png)
- [loss / PSNR 曲线](artifacts/scene017_vr_v1_s/loss_psnr_trends.png)
- [第0帧 PLY 预览](artifacts/scene017_vr_v1_s/final_global_gaussians_frame000_preview.ply)

算法、固定参数及验收见 [VR-v1-s说明](docs/releases/VR-v1-s/README.md)；持续维护 [算法变更记录](docs/ALGORITHM_CHANGES_CURRENT.md)；主线身份以 [CURRENT](docs/project-control/CURRENT.md) 为准。

## 使用

默认入口为 **`scripts/run_pipeline.py`**。`run_current_pipeline.py` 是历史冻结的VR-v1-b入口，保留给历史复现。

```bash
cd /path/to/pointforward-global

# 固定场景资产重放：完整视频、指标、PLY。
python scripts/run_pipeline.py --mode replay \
  --data-root /data/argoverse/processed/training \
  --output-root /work/vr-s-replay --gpu 0

# 原始处理后输入全新构建，不加载历史场景状态。
python scripts/run_pipeline.py --mode fresh \
  --data-root /data/argoverse/processed/training \
  --output-root /work/vr-s-fresh --scene 017 --gpu 0 --vehicle-gpu 1
```

输出目录必须为空。`--dry-run` 打印配方；两GPU并行静态/车辆前馈，省略 `--vehicle-gpu` 则串行。`--sam-python` 可指定SAM2解释器。`--use-pinned-extension` 仅适用于release记录的相同Torch/CUDA ABI；普通环境默认使用gsplat自身扩展。

当前新场景fresh默认停止条件：**全局PSNR≥30dB且车辆bbox PSNR≥28dB，或从冷启动输入开始累计达到3600s（60min）**。优化前及每100步检查完整场景、全部三相机，统计口径同正式渲染器；质量检查成本计入冷启动时间。取消原1700步提前停止，达到条件后保存状态并完成视频/PLY；最终保存与交付耗时仍单独如实记录，60min是优化停止触发点。配置在 `configs/runtime-stop/`；此次仅改停止逻辑，未重测。冻结历史入口 `run_vr_v1_s_pipeline.py` 保留原实验配方，replay固定资产保持原状态。

前期车辆校准通过 `--vehicle-calibration` 选择，只影响fresh：

|选项|处理|
|---|---|
|`adaptive`（默认）|现有自适应bbox/轨迹对齐与冲突背景LiDAR提示过滤，不额外扩大框|
|`bbox-expand`|跳过自适应校准，保留原LiDAR；仅将车辆前馈局部候选/解码框前后各加0.5m、左右各加0.3m|
|`both`|先做现有自适应bbox/LiDAR校准，再扩大车辆前馈局部候选/解码框|

扩大量沿车辆自身长宽轴计算，默认总长增加1.0m、总宽增加0.6m、高度不变。可用 `--bbox-front-back-margin 0.5 --bbox-left-right-margin 0.3` 调整单侧值。扩框不改变原始评估框或静态前馈框，也不改变选视图评分的原框；两者同时使用时，自适应拟合/过滤仍用原框支持，扩框用于之后的车辆重建范围，不重复扩张LiDAR归属范围。配置和实际总扩大量写入每次运行的 `delivery_options.json`。replay使用已固化资产，不重新应用这些选项。

```bash
# 例：同时启用两者，沿用默认扩大量。
python scripts/run_pipeline.py --mode fresh \
  --data-root /data/argoverse/processed/training \
  --output-root /work/vr-s-both --scene 017 --gpu 0 --vehicle-gpu 1 \
  --vehicle-calibration both --bbox-front-back-margin 0.5 --bbox-left-right-margin 0.3
```

输入为已预处理RGB、LiDAR、标定、自车位姿、实例、天空mask与SMPL标注，数据集不随包提供。冷启动起点是这些输入，不包含数据集格式转换或SMPL标注生成。环境见 [requirements.txt](requirements.txt) / [environment.yml](environment.yml) 及release。

## 时间与交付边界

源一次真实冷启动：预处理、前馈、PCA、优化初始化、1670步与保存共 **1485.371s（24.76min）**；继续到完整视频、指标、PLY的总耗时 **1609.087s（26.82min）**。静态/车辆前馈并行，不能累加GPU阶段时间。这是已固化历史状态的原计时；当前fresh使用上面的双质量阈值/60min停止策略，尚未重测。

完整复制须包含 `artifacts/scene017_vr_v1_s/`、`weights/` 及 `docs/releases/VR-v1-s/source/`，保留许可证。大资产沿用Git ignore策略，单独clone不包含完整交付包。资产SHA256、源码差异及独立验证见release。旧版本冻结证据保留，当前默认交付已由新主线替代。

# FFR-v1-l ver2 交付版本

本目录对应 `FFR-v1-l` 的 ver2 固化交付包，覆盖旧交付包中的 PFG-v2 主线说明。代码、通用权重和 scene017 的最终运行资产均已放入 `/mnt/fjn/projects/fjn/pointforward-global`。

## 固定结果

- 冷启动边界：从原始 RGB/raw scene 首次读取开始，到 PLY 和完整 156 帧三相机视频写盘完成。
- 总耗时：`1153.131 s`（`19.219 min`）。
- 全局 PSNR：`28.088604 dB`。
- 非天空 PSNR：`27.598707 dB`。
- 动态 bbox PSNR：`23.446180 dB`。
- 20 分钟时间约束通过；29 dB 质量目标未通过，不能将本版本标记为 29 dB 验收版本。

原始证据位于：
`/mnt/data/fjn/projects/PointForward/experiments/argoverse-inference/cold-scene-delivery/joint-speed29/FFR-v1-l/run-20261016T052523Z-fresh-parallel-350-gpu1/`。

## 算法固定项

1. 静态前馈和动态车辆前馈并行执行。
2. 联合优化使用 fused AdamW。
3. 移除 128D identity feature gather。
4. 冷启动计时范围内预加载训练帧。
5. 全量 Gaussian 参数与 U-Net 联合更新 350 步，240 步后学习率乘 `0.4`。

## 代码与资产

- `src/pointforward/`：FFR-v1-l 运行时 PointForward 模型代码。
- `scripts/`：预训练、RGB 冷启动建图、动态车辆处理、联合优化、PLY 导出、指标和视频渲染脚本。
- `weights/`：PointForward、InfiniDepth、SegFormer 以及 FFR-v1-l scene017 refiner 权重。
- `artifacts/scene017_ffr_v1_l/runtime-payload-refined/`：联合优化后的 scene017 128D 静态/动态 payload 和动态元数据。
- `artifacts/scene017_ffr_v1_l/runtime-payload/`：未联合优化的原始运行 payload。

文件大小和 SHA256 见同目录的 `delivery_manifest.json` 以及 `weights/manifest.json`。

## 复现入口

通用新场景流程：

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/run_current_pipeline.py \
  --config /work/scene017/config.yaml \
  --output-root /work/scene017/ffr-v1-l \
  --scene 017
```

对已生成的 payload 渲染 156 帧视频或导出预览 PLY 时，使用 `scripts/render_final_global_runtime.py` 和 `scripts/export_runtime_payload_ply.py`；refiner 使用 `weights/refiner_scene017_ffr_v1_l.pt`。该 refiner 与 scene017 payload 配套，不应直接用于其他场景。

## 边界

交付包包含预训练和推理源码以及本次固定场景的完整运行 payload，但不包含原始数据集。scene017 payload 是场景级资产，不是可泛化的模型 checkpoint；新场景仍需从 RGB 重新执行完整流程。

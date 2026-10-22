# 权重说明

当前 **VR-v1-s** 的 `refiner_scene017_vr_v1_s.pt` 必须与 `artifacts/scene017_vr_v1_s/runtime-payload-refined/` 内32D Gaussian、`feature_projection.pt`及轨迹成套使用。新增通用权重 `sam2.1_hiera_tiny.pt`、`pedestrian_head_ckpt_002000.pt`、`SMPL_NEUTRAL.npz` 均实际保存在本目录，SHA256见 `manifest.json`。新默认入口 `scripts/run_pipeline.py --mode fresh`只加载通用权重，`--mode replay`才读取固定scene017状态。

以下为历史说明，`run_current_pipeline.py`对应旧VR-v1-b。完整复制须包含Git忽略的大资产；单独Git clone不构成完整包。

当前 VR-v1-b 配套网络为 `refiner_scene017_vr_v1_b.pt`，只能与 `artifacts/scene017_vr_v1_b/runtime-payload-refined/` 成套使用。通用 PointForward / InfiniDepth 权重不变，清单见 `manifest.json`。`scripts/run_current_pipeline.py --mode fresh` 只加载通用权重，生成自己的场景网络；`--mode replay` 才加载上述 scene017 专属状态。旧版权重继续保留，以下段落为前序交付说明。

Git 只跟踪 `pointforward_waymo_ds_paper_ckpt_020000.pt`，通过 Git LFS 下载。FFR-v1-l ver2 交付目录同时提供 InfiniDepth、SegFormer、EfficientNetV2-S 初始化和 `refiner_scene017_ffr_v1_l.pt`；这些大文件按仓库原有策略不进入 Git 历史，但复制或归档交付目录时必须一并保留。

克隆后执行 `git lfs pull`，再运行 `python scripts/validate_install.py`。验证脚本只检查 Git 交付的 PointForward checkpoint，不会把外部权重误判为缺失。

所有随包权重的 SHA256、大小和用途均在 `manifest.json` 中，便于交付后校验。

`refiner_scene017_ffr_v1_l.pt` 是本交付固定结果的 scene017 联合 refiner；`refiner_scene017_fast_g8.pt` 和 `refiner_scene017_27db.pt` 仅作为历史配方保留。scene017 refiner 不是通用预训练权重，对新场景应由 `run_current_pipeline.py` 的 refiner 阶段重新生成。最终渲染还需要 `artifacts/scene017_ffr_v1_l/` 中相同配方生成的 runtime payload。

# 权重说明

Git 只跟踪 `pointforward_waymo_ds_paper_ckpt_020000.pt`，通过 Git LFS 下载。FFR-v1-l ver2 交付目录同时提供 InfiniDepth、SegFormer、EfficientNetV2-S 初始化和 `refiner_scene017_ffr_v1_l.pt`；这些大文件按仓库原有策略不进入 Git 历史，但复制或归档交付目录时必须一并保留。

克隆后执行 `git lfs pull`，再运行 `python scripts/validate_install.py`。验证脚本只检查 Git 交付的 PointForward checkpoint，不会把外部权重误判为缺失。

所有随包权重的 SHA256、大小和用途均在 `manifest.json` 中，便于交付后校验。

`refiner_scene017_ffr_v1_l.pt` 是本交付固定结果的 scene017 联合 refiner；`refiner_scene017_fast_g8.pt` 和 `refiner_scene017_27db.pt` 仅作为历史配方保留。scene017 refiner 不是通用预训练权重，对新场景应由 `run_current_pipeline.py` 的 refiner 阶段重新生成。最终渲染还需要 `artifacts/scene017_ffr_v1_l/` 中相同配方生成的 runtime payload。

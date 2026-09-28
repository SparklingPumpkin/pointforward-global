# 权重说明

Git 只跟踪 `pointforward_waymo_ds_paper_ckpt_020000.pt`，通过 Git LFS 下载。其余大权重列在 `manifest.json` 的 `optional_local_files` 中，仅保留在已有工作目录，不进入 Git 上传；新机器需按 README 中的说明自行下载或通过内部制品分发。

克隆后执行 `git lfs pull`，再运行 `python scripts/validate_install.py`。验证脚本只检查 Git 交付的 PointForward checkpoint，不会把外部权重误判为缺失。

可选权重的 SHA256 仍保留在清单中，便于下载后校验。

`refiner_scene017_fast_g8.pt` 是 PFG-v2 默认配方使用的 scene017 overlay；`refiner_scene017_27db.pt` 是 PFG-v1 旧配方 overlay。两者都不是通用预训练权重，也不进入 Git。对新场景应由 `run_current_pipeline.py` 的 refiner 阶段重新生成。最终 27 dB 渲染还需要相同配方生成的 scene017 runtime payload；payload 属于场景数据，按交付要求不进入本仓库。

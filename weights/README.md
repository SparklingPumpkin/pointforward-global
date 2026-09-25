# 权重说明

权重均由 Git LFS 跟踪，完整校验信息在 `manifest.json`。克隆后先执行 `git lfs pull`，再运行 `python scripts/validate_install.py`。

`refiner_scene017_27db.pt` 是 scene017 场景级 RGB 后优化得到的 overlay，不是通用预训练权重。对新场景应由 `run_current_pipeline.py` 的 refiner 阶段重新生成。最终 27 dB 渲染还需要相同版本生成的 scene017 runtime payload；payload 属于场景数据，按交付要求不进入本仓库。

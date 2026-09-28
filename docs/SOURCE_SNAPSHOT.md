# 源码快照身份

本交付不是从一个干净上游 tag 机械发布，而是对当前研究工作树中已验收文件的可审计快照。源研究仓库当时存在其他用户实验改动，因此没有对源仓库执行 reset、checkout 或清理。

| 组件 | 基础 commit | 交付方式 |
|---|---|---|
| PointForward | `e3ce44c52273cf0b9423e1f8a4c420cc94938e3a` | 复制当前 `pointforward/`、`scripts/`、`configs/`、`tests/` 工作树；源仓库当时有 65 个 porcelain 条目 |
| 51w-DriveStudio preprocessing | `69404678417b06027934ba2fb0397e462b3c3099` | 复制当前 `datasets/`、Argoverse/Waymo 文档和许可证 |
| InfiniDepth | `7c78b2253105a89d57c0a7bab9fc431f9def2123` | 去除 checkpoint、example assets 和 Git 元数据后的源码快照 |
| NVlabs SegFormer | `65fa8cfa9b52b6ee7e8897a98705abf8570f9e32` | shallow clone 后去除嵌套 Git 元数据 |

PFG-v2 另外同步了研究仓库中 GR-v1-s 已验收的最终静态 feature 优化能力：`refine_static_features_fullscene_rgb.py` 支持已有 refiner overlay、全帧选择、同帧多相机 batching 和可配置的 CUDA cache 清理频率；`run_current_pipeline.py` 将其编排为默认 `fast-g8` 尾部。其来源实验为 `GR-v1-s/run-20260927T202155Z-full-frame-g8-batch3`。

交付仓库自己的 Git commit 是最终可恢复身份；第三方许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。

# 源码快照身份

## 当前 VR-v1-s 隔离快照（2026-10-22）

当前数值源码 `docs/releases/VR-v1-s/source/` 来自 `VR-v1-s/run-20261022T054208Z-cold-compact-union1700/source-snapshot/`，指纹与便携改动见该版本 `source_manifest.json`。Gaussian、PCA、U-Net、轨迹、SMPL成套固化；仅修改训练器编排helper import、SAM路径参数及预编译扩展启用条件，不改数值公式。

默认入口为 `scripts/run_pipeline.py`，旧 `run_current_pipeline.py` 因历史验收冻结保留为VR-v1-b。当前以 [VR-v1-s release](releases/VR-v1-s/README.md) 为准，下面记录为历史版本。

## 当前 VR-v1-b 独立快照（2026-10-20）

当前入口使用 `docs/releases/VR-v1-b/source/`，来自已验收 `VR-v1-b/run-20261020T012848Z-feedforward-unified-all950/source-snapshot/`，不覆盖旧 `src/pointforward`。完整原始/交付源码 SHA256 见 `docs/releases/VR-v1-b/source_manifest.json`。唯一已有数值文件差异为联合训练器的编排helper import，新增portable helper解析包内路径；优化公式、模型和renderer未改动。新增根统一入口支持fresh/replay，原入口完整保存为 `scripts/run_legacy_pipeline.py`。配置移植替换机器路径并禁用初始化下载，随后加载同一通用checkpoint。

当前默认数值实现及其验证身份以上述快照和 [release](releases/VR-v1-b/README.md) 为准；以下内容属于更早交付历史。

本交付不是从一个干净上游 tag 机械发布，而是对当前研究工作树中已验收文件的可审计快照。源研究仓库当时存在其他用户实验改动，因此没有对源仓库执行 reset、checkout 或清理。

| 组件 | 基础 commit | 交付方式 |
|---|---|---|
| PointForward | `e3ce44c52273cf0b9423e1f8a4c420cc94938e3a` | 复制当前 `pointforward/`、`scripts/`、`configs/`、`tests/` 工作树；源仓库当时有 65 个 porcelain 条目 |
| 51w-DriveStudio preprocessing | `69404678417b06027934ba2fb0397e462b3c3099` | 复制当前 `datasets/`、Argoverse/Waymo 文档和许可证 |
| InfiniDepth | `7c78b2253105a89d57c0a7bab9fc431f9def2123` | 去除 checkpoint、example assets 和 Git 元数据后的源码快照 |
| NVlabs SegFormer | `65fa8cfa9b52b6ee7e8897a98705abf8570f9e32` | shallow clone 后去除嵌套 Git 元数据 |

PFG-v2 另外同步了研究仓库中 GR-v1-s 已验收的最终静态 feature 优化能力：`refine_static_features_fullscene_rgb.py` 支持已有 refiner overlay、全帧选择、同帧多相机 batching 和可配置的 CUDA cache 清理频率；`run_current_pipeline.py` 将其编排为默认 `fast-g8` 尾部。其来源实验为 `GR-v1-s/run-20260927T202155Z-full-frame-g8-batch3`。

交付仓库自己的 Git commit 是最终可恢复身份；第三方许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。

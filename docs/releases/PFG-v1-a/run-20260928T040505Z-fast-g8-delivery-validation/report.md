# PFG-v2 fast-g8 交付验证

结论：通过。交付包已同步 GR-v1-s 的 fast-g8 最终优化能力，并保留 PFG-v1 `legacy-alternating` 回退路径。

## 验证结果

- Python 语法检查通过。
- 测试：50 passed。
- 安装与 Git checkpoint 完整性检查通过。
- 交付入口公开 `fast-g8` 默认配方和 `legacy-alternating` 旧配方。
- 使用交付包自身渲染代码独立复算 156 帧、三相机，得到 27.085058566851494 dB；非天空 26.529556188827907 dB；动态 bbox 21.998108961643318 dB。
- 每帧使用完整 20,677,973 个 Gaussian。

独立复算 manifest 位于 `independent-metrics/manifest.json`。当前配方的源实验指标、计时和关键帧见 `docs/validation/scene017-fast-g8/`。

## 性能合同

fast-g8 从固定 GR-v1-i 26.668159 dB payload 开始，经最终静态 128D feature 优化到 PLY 导出完成，共 424.772963 秒（7.07955 分钟）；旧四阶段尾部为 1391.46 秒。该时间不含原始数据预处理及所有前序建图和优化阶段。

## 打包边界

仓库只保存代码、文档、指标 manifest、计时与关键帧证据。约 12 GB 的最终 payload、约 1.4 GB 的 PLY 和视频不进入 Git；scene017 refiner 作为本地可选制品登记 SHA256。

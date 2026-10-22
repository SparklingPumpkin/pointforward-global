# 算法与交付变更记录

## 2026-10-20 · VR-v1-b

当前交付切换到 VR-v1-b 的前馈后连续950步联合优化；保留 FFR-v1-l ver2 资产与旧 PFG-v2 编排 `scripts/run_legacy_pipeline.py`，不覆盖旧核心实现。VR-v1-b 使用独立源码、可移植配置、新配套 U-Net 及原始/最终 payload，附完整950步视频、PLY、loss曲线和车辆对比。

相对 FFR-v1-l 全场景350步流程，本版本在同一优化器中训练到950步，动态Gaussian、静态Gaussian、U-Net与车辆轨迹始终共同优化；增加正确处理画外可见性的 actor-balanced ROI 目标、轨迹平滑与锚定。没有scene350后冻结静态/U-Net再单训车辆的边界。零基240/700步学习率衰减；350/550/750/950仅作状态保存和评估。完整参数与来源见 [release](releases/VR-v1-b/README.md)。

全局30.294842dB、动态bbox26.334712dB属于固定156帧三相机128D+U-Net源帧运行时。保留legacy bbox指标定义；strict visibility只改变训练ROI。PLY只作frame000预览。增量2065.830s、历史前缀442.946s、历史累计2508.776s，未认证一次新的端到端冷启动时间。

独立交付新增统一fresh/replay入口，仅替换训练器编排helper import；数值模型、loss与优化器不变。包内全156帧重渲染及资产校验另存验证报告。fresh入口拒绝非空输出，统一计时覆盖前馈、联合优化、四检查点渲染、最终视频及PLY落盘。

## 前序交付

- [FFR-v1-l ver2 固定记录](releases/FFR-v1-l-ver2/README.md)：1153.131s / 28.088604dB，维持原记录边界，非VR-v1-b耗时。
- [原根说明归档](releases/FFR-v1-l-ver2/ROOT_README_ARCHIVE.md)：含预处理、预训练及旧PFG入口说明；其中run_current入口已替换，旧配方应使用run_legacy_pipeline。
- 更早PFG-v1/PFG-v2沿革见项目控制面版本总账；不同版本质量与耗时不能跨链拼接。

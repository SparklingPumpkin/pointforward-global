# 当前项目控制面

> 由机器状态生成，只展示当前版本与仍活跃的候选；完整历史见 `VERSION_LEDGER.md`。

项目：PointForward Global（`pointforward-global`）  
更新时间：2026-10-22T07:18:06+00:00

## 项目主线

当前版本：`PFG-v4`（compact32-union-smpl-delivery）

当前交付VR-v1-s第1670步：车辆几何校正、32D Gaussian/PCA/U-Net、SMPL插值、全三相机union联合训练；fresh/replay便携入口；独立156帧全局31.131654/车辆28.329244dB，源完整RGB处理后输入至视频PLY1609.087s，研究29dB目标仍未达标

当前证据：
- `docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/validation.json`：完整独立重放及资产核验通过
- `docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/validation.json`：用户指定当前最佳交付，独立完整验证通过

活跃候选：
- 无。

## 交付验证

当前版本：`DEL-v1`（validation-baseline）

PFG-v1交付包轻量验证与scene017 27dB复现

当前证据：
- `docs/VALIDATION.md`：交付包通过48项测试并复现27.000626453986534 dB

活跃候选：
- `DEL-v1-a` [passed]：交付仓库内源码、权重和配置可以在不复制场景payload的前提下复现scene017的27.000626 dB指标并导出PLY。

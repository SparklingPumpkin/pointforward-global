# PFG-v4：compact32-union-smpl-delivery

> 该文件是版本建立时的不可变快照。状态变化和后续候选见 `../VERSION_LEDGER.md`。

作用域：项目主线（`project`）  
建立时间：2026-10-22T07:07:04+00:00  
父版本：`PFG-v3`  
来源候选：`PFG-v3-a`  
旧标识：`无`

## 当前真相摘要

当前交付VR-v1-s第1670步：车辆几何校正、32D Gaussian/PCA/U-Net、SMPL插值、全三相机union联合训练；fresh/replay便携入口；独立156帧全局31.131654/车辆28.329244dB，源完整RGB处理后输入至视频PLY1609.087s，研究29dB目标仍未达标

## 改变类型

- `code_architecture`

## 证据

- `docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/validation.json`：完整独立重放及资产核验通过
- `docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/validation.json`：用户指定当前最佳交付，独立完整验证通过

## 晋级依据

- 候选：`PFG-v3-a`
- 假设：用户选定VR-v1-s可独立固化并精确重渲染；交付验收独立于未通过的研究29dB目标
- 结论：VR-v1-s第1670步32D运行时独立交付：全局31.131654、车辆28.329244dB精确复现；研究车辆29dB目标仍失败

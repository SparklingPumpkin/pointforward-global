# PFG-v3：unified-scene-joint950-delivery

> 该文件是版本建立时的不可变快照。状态变化和后续候选见 `../VERSION_LEDGER.md`。

作用域：项目主线（`project`）  
建立时间：2026-10-20T06:47:34+00:00  
父版本：`PFG-v2`  
来源候选：`PFG-v2-a`  
旧标识：`无`

## 当前真相摘要

当前交付算法VR-v1-b：原始前馈后全参数、U-Net与车辆轨迹连续950步联合优化；portable fresh/replay成为默认；旧入口保留；独立完整156帧复现30.294842dB/动态26.334712dB，历史累计非新冷启动实测

## 改变类型

- `code_architecture`

## 证据

- `docs/releases/PFG-v2-a/run-20261020T063212Z-unified-delivery-validation/validation.json`：独立完整重渲染与校验通过
- `docs/releases/PFG-v2-a/run-20261020T063212Z-unified-delivery-validation/validation.json`：支持将已独立验收的交付架构设为当前主线

## 晋级依据

- 候选：`PFG-v2-a`
- 假设：VR-v1-b联合优化可独立固化交付并在交付环境精确复现30.294842dB
- 结论：VR-v1-b固定950步完整状态独立交付，156帧全局30.294842/动态26.334712dB精确重现；保留旧入口与资产，无新冷启动耗时认证

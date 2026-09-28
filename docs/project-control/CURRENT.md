# 当前项目控制面

> 由机器状态生成，只展示当前版本与仍活跃的候选；完整历史见 `VERSION_LEDGER.md`。

项目：PointForward Global（`pointforward-global`）  
更新时间：2026-09-28T04:27:48+00:00

## 项目主线

当前版本：`PFG-v2`（fast-g8-27db-mainline）

PFG-v2 makes the validated full-frame groups=8 batched feature tail the default, retains the PFG-v1 legacy recipe, and documents the DG-v1-f to current upgrade path.

当前证据：
- `docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/validation-summary.json`：Validates the new default recipe and its fallback contract.
- `docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/validation-summary.json`：Supports promotion to the PFG-v2 mainline.

活跃候选：
- 无。

## 交付验证

当前版本：`DEL-v1`（validation-baseline）

PFG-v1交付包轻量验证与scene017 27dB复现

当前证据：
- `docs/VALIDATION.md`：交付包通过48项测试并复现27.000626453986534 dB

活跃候选：
- `DEL-v1-a` [passed]：交付仓库内源码、权重和配置可以在不复制场景payload的前提下复现scene017的27.000626 dB指标并导出PLY。

# PFG-v2：fast-g8-27db-mainline

> 该文件是版本建立时的不可变快照。状态变化和后续候选见 `../VERSION_LEDGER.md`。

作用域：项目主线（`project`）  
建立时间：2026-09-28T04:27:48+00:00  
父版本：`PFG-v1`  
来源候选：`PFG-v1-a`  
旧标识：`无`

## 当前真相摘要

PFG-v2 makes the validated full-frame groups=8 batched feature tail the default, retains the PFG-v1 legacy recipe, and documents the DG-v1-f to current upgrade path.

## 改变类型

- `validated_default_change`

## 证据

- `docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/validation-summary.json`：Validates the new default recipe and its fallback contract.
- `docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/validation-summary.json`：Supports promotion to the PFG-v2 mainline.

## 晋级依据

- 候选：`PFG-v1-a`
- 假设：以DG-v1-f到GR-v1-i的既有链路为基础，用全156帧groups8静态feature优化和三相机批处理替换GR-v1-j末端四阶段，可在10分钟RGB后优化合同内达到至少27dB并保留旧配方回退
- 结论：Adopt fast-g8 as the PFG-v2 default: 27.085059 dB, 7.08-minute accepted tail, with legacy-alternating preserved.

# 项目版本总账

> 本文件由 `fjn-project-workflow` 根据不可追加覆盖的事件账本生成；请勿手改。

更新时间：2026-09-28T04:27:48+00:00

## 项目主线（project）

作用域类型：`project`  
当前版本：`PFG-v2`

### 主线版本

| 版本 | 日期 | 状态 | 父版本 | 来源候选 | 变更类型 | 摘要 | 证据 |
|---|---|---|---|---|---|---|---|
| PFG-v1 | 2026-09-25T15:51:53+00:00 | superseded | — | — | — | scene017最终全局Gaussian 27.000626dB方法的可移植交付主线 | — |
| PFG-v2 | 2026-09-28T04:27:48+00:00 | current | PFG-v1 | PFG-v1-a | validated_default_change | PFG-v2 makes the validated full-frame groups=8 batched feature tail the default, retains the PFG-v1 legacy recipe, and documents the DG-v1-f to current upgrade path. | docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/validation-summary.json：Validates the new default recipe and its fallback contract.; docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/validation-summary.json：Supports promotion to the PFG-v2 mainline. |

### 子实验候选

| 候选 | 基线 | 状态 | 假设 | 结论 | 运行数 | 证据 |
|---|---|---|---|---|---:|---|
| PFG-v1-a | PFG-v1 | promoted | 以DG-v1-f到GR-v1-i的既有链路为基础，用全156帧groups8静态feature优化和三相机批处理替换GR-v1-j末端四阶段，可在10分钟RGB后优化合同内达到至少27dB并保留旧配方回退 | Adopt fast-g8 as the PFG-v2 default: 27.085059 dB, 7.08-minute accepted tail, with legacy-alternating preserved. | 1 | docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/validation-summary.json：Validates the new default recipe and its fallback contract. |

## 交付验证（delivery-validation）

作用域类型：`workstream`  
当前版本：`DEL-v1`

### 主线版本

| 版本 | 日期 | 状态 | 父版本 | 来源候选 | 变更类型 | 摘要 | 证据 |
|---|---|---|---|---|---|---|---|
| DEL-v1 | 2026-09-25T16:30:57+00:00 | current | — | — | — | PFG-v1交付包轻量验证与scene017 27dB复现 | docs/VALIDATION.md：交付包通过48项测试并复现27.000626453986534 dB |

### 子实验候选

| 候选 | 基线 | 状态 | 假设 | 结论 | 运行数 | 证据 |
|---|---|---|---|---|---:|---|
| DEL-v1-a | DEL-v1 | passed | 交付仓库内源码、权重和配置可以在不复制场景payload的前提下复现scene017的27.000626 dB指标并导出PLY。 | 交付代码、权重和配置通过48项测试，并用外部scene017最终payload复现27.000626453986534 dB全局PSNR及PLY导出。 | 1 | docs/VALIDATION.md：记录完整轻量检查、156帧27dB复现和PLY合同证据 |

## 血缘解释

仅当事件账本中同时保存父版本、晋级候选和证据时，才生成承接关系。没有证据的旧关系应登记为未知，不得推断连线。

# 当前项目控制面

> 由机器状态生成，只展示当前版本与仍活跃的候选；完整历史见 `VERSION_LEDGER.md`。

项目：PointForward Global（`pointforward-global`）  
更新时间：2026-11-06T06:23:28+00:00

## 项目主线

当前版本：`PFG-v5`（lim-r-default-preprocessing）

VR-v1-s骨干加用户接受的LIM-v1-r固有前处理；fresh默认lim-r，保留legacy与冻结replay；独立原始输入及消费链验证完成，未新增整场PSNR/完整计时结论

当前证据：
- `docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/report.md`：实质前处理数据合同与工程验证

活跃候选：
- `PFG-v5-a` [passed]：Independent portable preprocessing stages can preserve research numerics and delivery default while serving other projects

## 交付验证

当前版本：`DEL-v1`（validation-baseline）

PFG-v1交付包轻量验证与scene017 27dB复现

当前证据：
- `docs/VALIDATION.md`：交付包通过48项测试并复现27.000626453986534 dB

活跃候选：
- `DEL-v1-a` [passed]：交付仓库内源码、权重和配置可以在不复制场景payload的前提下复现scene017的27.000626 dB指标并导出PLY。

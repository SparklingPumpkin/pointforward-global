# 文件夹职责总表

> 每个受管文件夹必须有唯一职责、层级和父级。新增实验目录前先记录布局决策。

活跃平级目录预算：6  
最大受管深度：8

| ID | 路径 | 层级 | 父级 | 作用域 | 状态 | 管理直接子目录 | 职责 |
|---|---|---|---|---|---|---|---|
| dir-0001 | . | project | — | project | active | 否 | 项目边界；其子目录必须按业务职责继续登记。 |
| dir-0022 | artifacts | area | dir-0001 | project | active | 否 | 版本化场景载荷、PLY和视频 |
| dir-0004 | configs | area | dir-0001 | project | active | 否 | 可移植训练和推理配置模板 |
| dir-0007 | docs | area | dir-0001 | project | active | 否 | 使用说明、方法说明、验证证据与项目控制面 |
| dir-0003 | scripts | area | dir-0001 | project | active | 否 | 训练、推理、优化、PLY导出和视频渲染入口 |
| dir-0002 | src | area | dir-0001 | project | active | 否 | 核心模型、外部预处理与深度估计源码快照 |
| dir-0006 | tests | area | dir-0001 | project | active | 否 | 轻量语法、导入与交付合同测试 |
| dir-0009 | validation-delivery | workstream | dir-0001 | delivery-validation | active | 否 | 交付包独立可执行性验证工作流 |
| dir-0005 | weights | area | dir-0001 | project | active | 否 | 交付所需模型权重及校验清单 |
| dir-0023 | artifacts/scene017_vr_v1_b | archive | dir-0022 | project | active | 否 | VR-v1-b原始前馈和950步资产 |
| dir-0033 | artifacts/scene017_vr_v1_s | archive | dir-0022 | project | active | 否 | 当前VR-v1-s运行时及预览资产 |
| dir-0034 | configs/runtime-stop | area | dir-0004 | project | active | 否 | 当前fresh运行停止条件，保留已冻结VR-v1-s历史配方 |
| dir-0013 | docs/releases | experiment_family | dir-0007 | project | active | 否 | 主线交付升级候选与验证记录 |
| dir-0008 | docs/validation | workstream | dir-0007 | project | active | 否 | PFG-v1独立交付的轻量与27dB复现证据 |
| dir-0035 | src/pointforward_calibration | area | dir-0002 | project | active | 否 | LIM-v1-r独立前处理模块及冻结数值实现 |
| dir-0042 | src/pointforward_preprocessing | area | dir-0002 | project | active | 否 | Reusable depth mask repair and measured scene sky alongside independent LIM-r |
| dir-0010 | validation-delivery/scene017-27db | experiment_family | dir-0009 | delivery-validation | active | 否 | scene017 27dB交付复现验证实验族 |
| dir-0014 | docs/releases/PFG-v1-a | candidate | dir-0013 | project | active | 否 | DG-v1-f到GR-v1-s快速27dB主线升级 |
| dir-0018 | docs/releases/PFG-v2-a | candidate | dir-0013 | project | active | 否 | VR-v1-b独立交付包升级验证 |
| dir-0025 | docs/releases/PFG-v3-a | candidate | dir-0013 | project | active | 否 | VR-v1-s独立交付验收 |
| dir-0036 | docs/releases/PFG-v4-a | candidate | dir-0013 | project | active | 否 | 用户接受LIM-v1-r固化为默认前处理 |
| dir-0043 | docs/releases/PFG-v5-a | candidate | dir-0013 | project | active | 否 | Optional modular preprocessing engineering integration; PFG-v5 default preserved |
| dir-0020 | docs/releases/VR-v1-b | archive | dir-0013 | project | active | 否 | VR-v1-b版本说明、源码与固定证据 |
| dir-0028 | docs/releases/VR-v1-s | archive | dir-0013 | project | active | 否 | VR-v1-s不可变版本源码与证据 |
| dir-0017 | docs/validation/scene017-fast-g8 | experiment_family | dir-0008 | project | active | 否 | scene017 fast-g8指标计时与关键帧证据 |
| dir-0011 | validation-delivery/scene017-27db/DEL-v1-a | candidate | dir-0010 | delivery-validation | active | 否 | 交付包27dB复现候选 |
| dir-0015 | docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation | run | dir-0014 | project | active | 否 | PFG-v1-a源码、配方、指标与文档交付验证 |
| dir-0019 | docs/releases/PFG-v2-a/run-20261020T063212Z-unified-delivery-validation | run | dir-0018 | project | active | 否 | 独立交付复制核验与156帧重渲染 |
| dir-0026 | docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation | run | dir-0025 | project | active | 否 | 完整独立重渲染与交付校验 |
| dir-0037 | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration | run | dir-0036 | project | active | 否 | 独立代码与前处理消费链验证 |
| dir-0044 | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration | run | dir-0043 | project | active | 否 | Module contracts and real-input limited-frame integration evidence |
| dir-0021 | docs/releases/VR-v1-b/source | archive | dir-0020 | project | active | 否 | VR-v1-b精确数值源码快照 |
| dir-0032 | docs/releases/VR-v1-s/evidence | archive | dir-0028 | project | active | 否 | 原始计时指标与研究失败验收记录 |
| dir-0029 | docs/releases/VR-v1-s/source | archive | dir-0028 | project | active | 否 | VR-v1-s数值源码及便携依赖 |
| dir-0012 | validation-delivery/scene017-27db/DEL-v1-a/run-20260925T162548Z-scene017-27db-delivery-validation | run | dir-0011 | delivery-validation | active | 否 | scene017交付复现执行记录 |
| dir-0016 | docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/independent-metrics | artifact | dir-0015 | project | active | 否 | 交付包独立156帧复算manifest |
| dir-0024 | docs/releases/PFG-v2-a/run-20261020T063212Z-unified-delivery-validation/replay | artifact | dir-0019 | project | active | 否 | 包内完整156帧独立重渲染 |
| dir-0027 | docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/replay | artifact | dir-0026 | project | active | 否 | 包内重渲染输出 |
| dir-0040 | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/argo-fresh | artifact | dir-0037 | project | active | 否 | Correct isolated-source fresh integration sample |
| dir-0038 | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/argo-probe | artifact | dir-0037 | project | active | 否 | Fresh raw preprocessing integration sample |
| dir-0041 | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/ncore-fresh | artifact | dir-0037 | project | active | 否 | Correct isolated-source fresh integration sample |
| dir-0039 | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/ncore-probe | artifact | dir-0037 | project | active | 否 | Fresh ncore raw preprocessing sample |
| dir-0047 | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/independent-sky | artifact | dir-0044 | project | active | 否 | Scoped portable preprocessing validation independent-sky |
| dir-0049 | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/real-combined-retry | artifact | dir-0044 | project | active | 否 | Retry real integration after explicit missing-ego adapter handling; first failure retained |
| dir-0045 | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/real-depth-mask | artifact | dir-0044 | project | active | 否 | Scoped portable preprocessing validation real-depth-mask |
| dir-0046 | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/real-scene-sky | artifact | dir-0044 | project | active | 否 | Scoped portable preprocessing validation real-scene-sky |
| dir-0048 | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/source-snapshot | artifact | dir-0044 | project | active | 否 | Scoped portable preprocessing validation source-snapshot |
| dir-0031 | docs/releases/VR-v1-s/source/InfiniDepth | archive | dir-0029 | project | active | 否 | 本次原始深度模型源码 |
| dir-0030 | docs/releases/VR-v1-s/source/SAM2 | archive | dir-0029 | project | active | 否 | SAM2.1 Tiny源码及许可 |

## 布局决策历史

| 决策 | 状态 | 动作 | 目标路径 | 理由 | 备选方案 | 豁免理由 |
|---|---|---|---|---|---|---|
| layout-0001 | applied | new_child | src | 用src统一容纳本项目代码和固定版本外部源码，保持根目录职责数量可控 | 将三个源码树散列在根目录会增加交付结构复杂度 | — |
| layout-0002 | applied | new_child | scripts | 入口脚本与可导入源码分离，便于新机器按阶段执行 | 把编排逻辑嵌入模型包会混淆职责 | — |
| layout-0003 | applied | new_child | configs | 集中管理数据根、权重和输出参数，消除机器路径硬编码 | 直接复用研究配置会携带本机绝对路径 | — |
| layout-0004 | applied | new_child | weights | 必要权重随包交付并通过SHA256审计 | 依赖隐式缓存不满足离线新机器使用目标 | — |
| layout-0005 | applied | new_child | tests | 验证代码和交付资产，不执行大型预训练 | 只在原研究仓库验证无法证明交付包独立性 | — |
| layout-0006 | applied | new_child | docs | 登记初始化时已创建的文档目录并保留控制面 | 另建说明目录会分裂文档事实源 | — |
| layout-0007 | applied | new_child | docs/validation | 验证记录属于文档证据，不包含场景payload或逐帧图像 | 放入outputs会被忽略且无法随主线审计 | — |
| layout-0008 | applied | new_child | validation-delivery | 交付验证使用独立workstream承载后续实验记录 | 不把交付验证混入PFG主线目录 | — |
| layout-0009 | applied | new_child | validation-delivery/scene017-27db | 按场景和合同隔离交付验证证据 | 直接在workstream下登记candidate会丢失实验族语义 | — |
| layout-0010 | applied | new_child | validation-delivery/scene017-27db/DEL-v1-a | 候选验证交付包与固定scene017合同 | 不把验证作为PFG方法候选，避免误升主线版本 | — |
| layout-0011 | applied | new_child | validation-delivery/scene017-27db/DEL-v1-a/run-20260925T162548Z-scene017-27db-delivery-validation | 记录独立交付代码的实际验证命令与证据 | 不把run事实混入candidate摘要 | — |
| layout-0012 | applied | new_group | docs/releases | 将方法升级候选和运行证据集中于文档区域，不混入源码或交付验证工作流 | 直接修改主线而不保留候选和run审计 | — |
| layout-0013 | applied | new_child | docs/releases/PFG-v1-a | 当前方法、性能合同和默认流水线均发生实质变化，应作为PFG-v1独立候选验证后晋级 | 覆盖PFG-v1会丢失旧27dB四阶段配方身份 | — |
| layout-0014 | applied | new_child | docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation | 候选晋级前需保存独立运行与持久证据 | 仅依赖源研究仓库结果不能证明交付包已经更新 | — |
| layout-0015 | applied | new_child | docs/releases/PFG-v1-a/run-20260928T040505Z-fast-g8-delivery-validation/independent-metrics | 将独立渲染产生的可审计指标证据登记在对应验证run下 | — | — |
| layout-0016 | applied | new_group | docs/validation/scene017-fast-g8 | 保存PFG-v2验收所需的小型可审计证据，不纳入大payload和PLY | — | — |
| layout-0017 | applied | new_child | docs/releases/PFG-v2-a | 独立固化VR-v1-b并保留旧交付版本 | 覆盖旧源码和资产会损害旧版本复现 | — |
| layout-0018 | applied | new_child | docs/releases/PFG-v2-a/run-20261020T063212Z-unified-delivery-validation | 独立固化VR-v1-b并保留旧交付版本 | 覆盖旧源码和资产会损害旧版本复现 | — |
| layout-0019 | applied | new_child | docs/releases/VR-v1-b | 独立固化VR-v1-b并保留旧交付版本 | 覆盖旧源码和资产会损害旧版本复现 | — |
| layout-0020 | applied | new_child | docs/releases/VR-v1-b/source | 独立固化VR-v1-b并保留旧交付版本 | 覆盖旧源码和资产会损害旧版本复现 | — |
| layout-0021 | planned | new_child | artifacts | 独立固化VR-v1-b并保留旧交付版本 | 覆盖旧源码和资产会损害旧版本复现 | — |
| layout-0022 | applied | new_child | artifacts | 独立固化VR-v1-b并保留旧交付版本 | 覆盖旧源码和资产会损害旧版本复现 | — |
| layout-0023 | applied | new_child | artifacts/scene017_vr_v1_b | 独立固化VR-v1-b并保留旧交付版本 | 覆盖旧源码和资产会损害旧版本复现 | — |
| layout-0024 | applied | new_child | docs/releases/PFG-v2-a/run-20261020T063212Z-unified-delivery-validation/replay | 验证独立交付的数值一致性 | 仅复制旧视频不足以验证执行能力 | — |
| layout-0025 | applied | new_child | docs/releases/PFG-v3-a | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0026 | applied | new_child | docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0027 | applied | new_child | docs/releases/PFG-v3-a/run-20261022T064659Z-compact-delivery-validation/replay | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0028 | applied | new_child | docs/releases/VR-v1-s | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0029 | applied | new_child | docs/releases/VR-v1-s/source | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0030 | applied | new_child | docs/releases/VR-v1-s/source/SAM2 | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0031 | applied | new_child | docs/releases/VR-v1-s/source/InfiniDepth | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0032 | applied | new_child | docs/releases/VR-v1-s/evidence | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0033 | applied | new_child | artifacts/scene017_vr_v1_s | 用户指定将当前VR-v1-s固化为当前交付主线 | 保留历史冻结证据，仅覆盖当前入口和根说明 | 用户授权在既有包内固化新主线，历史文件不删除 |
| layout-0034 | applied | new_child | configs/runtime-stop | 用户要求只修改新场景停止条件，不重测；运行时覆盖与历史证据分离 | 直接修改已验收配方会破坏历史指纹 | — |
| layout-0035 | applied | new_child | src/pointforward_calibration | 与模型快照和历史实验隔离 | 直接引用研究仓库不满足独立固化 | — |
| layout-0036 | applied | new_child | docs/releases/PFG-v4-a | 明确固化过程与原研究质量验收区别 | 覆盖历史版本 | — |
| layout-0037 | applied | new_child | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration | 保存工程验收证据，不重写研究验收 | 混入旧研究run | — |
| layout-0038 | applied | new_child | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/argo-probe | 实际检查独立运行和消费链 | 仅静态代码检查 | — |
| layout-0039 | applied | new_child | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/ncore-probe | 实际检查独立运行和消费链 | 仅静态代码检查 | — |
| layout-0040 | applied | new_child | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/argo-fresh | 修正可导入模块优先级后重试；初次日志保留 | 覆盖初次证据 | — |
| layout-0041 | applied | new_child | docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration/ncore-fresh | 修正可导入模块优先级后重试；初次日志保留 | 覆盖初次证据 | — |
| layout-0042 | applied | new_child | src/pointforward_preprocessing | User requested reusable independently selectable preprocessing without changing delivery mainline | Keep research-only wrappers coupled to experiments | — |
| layout-0043 | applied | new_child | docs/releases/PFG-v5-a | User requested reusable independently selectable preprocessing without changing delivery mainline | Keep research-only wrappers coupled to experiments | — |
| layout-0044 | applied | new_child | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration | User requested reusable independently selectable preprocessing without changing delivery mainline | Keep research-only wrappers coupled to experiments | — |
| layout-0045 | applied | new_child | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/real-depth-mask | User requested reusable independently selectable preprocessing without changing delivery mainline | Keep research-only wrappers coupled to experiments | — |
| layout-0046 | applied | new_child | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/real-scene-sky | User requested reusable independently selectable preprocessing without changing delivery mainline | Keep research-only wrappers coupled to experiments | — |
| layout-0047 | applied | new_child | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/independent-sky | User requested reusable independently selectable preprocessing without changing delivery mainline | Keep research-only wrappers coupled to experiments | — |
| layout-0048 | applied | new_child | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/source-snapshot | User requested reusable independently selectable preprocessing without changing delivery mainline | Keep research-only wrappers coupled to experiments | — |
| layout-0049 | applied | new_child | docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration/real-combined-retry | Keep failed first output immutable and generate fresh depth/mask/sky in a separate empty location | Overwrite failed evidence | — |

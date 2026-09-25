# 文件夹职责总表

> 每个受管文件夹必须有唯一职责、层级和父级。新增实验目录前先记录布局决策。

活跃平级目录预算：6  
最大受管深度：8

| ID | 路径 | 层级 | 父级 | 作用域 | 状态 | 管理直接子目录 | 职责 |
|---|---|---|---|---|---|---|---|
| dir-0001 | . | project | — | project | active | 否 | 项目边界；其子目录必须按业务职责继续登记。 |
| dir-0004 | configs | area | dir-0001 | project | active | 否 | 可移植训练和推理配置模板 |
| dir-0007 | docs | area | dir-0001 | project | active | 否 | 使用说明、方法说明、验证证据与项目控制面 |
| dir-0003 | scripts | area | dir-0001 | project | active | 否 | 训练、推理、优化、PLY导出和视频渲染入口 |
| dir-0002 | src | area | dir-0001 | project | active | 否 | 核心模型、外部预处理与深度估计源码快照 |
| dir-0006 | tests | area | dir-0001 | project | active | 否 | 轻量语法、导入与交付合同测试 |
| dir-0009 | validation-delivery | workstream | dir-0001 | delivery-validation | active | 否 | 交付包独立可执行性验证工作流 |
| dir-0005 | weights | area | dir-0001 | project | active | 否 | 交付所需模型权重及校验清单 |
| dir-0008 | docs/validation | workstream | dir-0007 | project | active | 否 | PFG-v1独立交付的轻量与27dB复现证据 |
| dir-0010 | validation-delivery/scene017-27db | experiment_family | dir-0009 | delivery-validation | active | 否 | scene017 27dB交付复现验证实验族 |
| dir-0011 | validation-delivery/scene017-27db/DEL-v1-a | candidate | dir-0010 | delivery-validation | active | 否 | 交付包27dB复现候选 |
| dir-0012 | validation-delivery/scene017-27db/DEL-v1-a/run-20260925T162548Z-scene017-27db-delivery-validation | run | dir-0011 | delivery-validation | active | 否 | scene017交付复现执行记录 |

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

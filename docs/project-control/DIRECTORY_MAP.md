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
| dir-0005 | weights | area | dir-0001 | project | active | 否 | 交付所需模型权重及校验清单 |
| dir-0008 | docs/validation | workstream | dir-0007 | project | active | 否 | PFG-v1独立交付的轻量与27dB复现证据 |

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

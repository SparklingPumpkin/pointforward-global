# LIM-v1-r 默认前处理固化

用户接受当前研究版本，固化到 pointforward-global 的 fresh 前处理。新的默认为 `--vehicle-calibration lim-r`；`adaptive`/`both`可重现旧前处理，replay自动使用旧冻结资产。

数值实现从接受版本及其依赖提取，73个函数/方法与原源码AST一致（仅SAM导入路径及绘图缓存装饰器作便携处理），所有数值配置逐字保留。移除了历史场景缓存加载路径；Q前置参考在当前原始输入上重新计算，供R原有的可信点保护使用。工程层新增完整场景深度/框应用及显式dataset消费包装，冻版本源码不修改。

本次原始输入验证：Argo017、ncore各两车三观测，重新执行YOLO/SAM2、真实邻帧选点、前置参考及R拟合。各应用1帧×3相机；最终导出实现复用本run拟合结果再次应用，独立记录cached计时。修正框、曝光相机姿态、深度逐值由既有dataset读取通过。默认几何选车也分别实际执行；5项专用合同测试通过，包含连续时间、独立时钟、首尾偏移边界和z-buffer。

亲自查看两张五列诊断图。原有外溢与框轮廓差异并未消失；仅两车子集的时钟不等同于原九批合并时钟。这不改变用户接受决定，也不改写原研究失败结论。未启动新的全场景Gaussian训练、未重测PSNR或端到端冷启动性能。旧VR-v1-s指标/计时仍只属于其历史资产。

证据：`validation.json`、`contract-tests.log`、`consumer_verification.json`、`default_selection_verification.json`、`personal_review.json`、`source_research_report.md`、`integration_code_snapshot.zip`。复现入口 `reproduce.sh`。初次导入优先级错误及取消尝试在 `attempts.json` 保留。

控制面 guard PASS；4条接管时已有警告：DEL-v1及DEL-v1-a的docs/VALIDATION.md指纹变化；旧run的docs/validation/summary.json变化；layout-0021未应用。均非本轮新增，不重写旧账消警告。

产物评估：本run是用户指定固化版本的工程证据，全部keep，无删除候选；源研究实验及其依赖未删除。详见独立artifact-review.json。

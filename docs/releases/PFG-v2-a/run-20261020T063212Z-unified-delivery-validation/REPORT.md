# VR-v1-b 独立交付验证

研究算法版本：VR-v1-b，选择950步。交付升级候选：PFG-v2-a；二者是算法身份与交付仓库身份，不是两次不同数值实验。

## 验证结果

在交付仓库内用隔离源码、复制的最终payload、配套U-Net及通用权重独立渲染完整156帧三相机，并重新导出frame000预览PLY。执行的具体命令、参数和统一计时见 `replay/delivery_options.json` 与 `replay/pipeline_timing.json`。

| 指标 | 原实验 | 交付包重渲染 | 绝对差 |
|---|---:|---:|---:|
|全局PSNR|30.2948417297|30.2948417297|0|
|非天空PSNR|29.8241869853|29.8241869853|0|
|动态bbox PSNR|26.3347121752|26.3347121752|0|

预测视频2304×576、GT/预测视频2304×1152，均156帧、10fps。复渲染及PLY导出共151.719s，仅是本次缓存资产的重渲染，不是RGB冷启动。PLY与原导出SHA256一致。复制资产、原始证据与专用refiner共33文件SHA256核验通过；原始和交付源码逐项记录，共110个Python文件。通用PointForward与InfiniDepth权重哈希一致，旧FFR-v1-l ver2权重和场景资产继续核验。

亲自查看 `replay/runtime-render-source/frames/comparison/frame_0080.jpg` 以及交付车辆局部对比：场景及车辆与原950步结果一致，近处车辆仍有细节模糊和轮廓误差，固化不代表这些原有缺陷已解决。

## 代码与入口

原数值源码只替换联合训练器中一行编排helper import，新增portable helper；完整差异与原始/交付SHA256见release的 `source_manifest.json`。全流程入口自动使用隔离源码，源码目录下的static/track/joint950/PLY命令均已通过真实import和`--help`检查，exact命令保存于 `pipeline_review.json`；根默认入口支持同一配方。路径替换集中在配置/编排，参数和数值公式不变。

原默认入口逐字节保留为 `scripts/run_legacy_pipeline.py`，旧根README逐字节归档。新的fresh入口拒绝非空输出、只从原始处理后输入和通用权重开始，统计到最终PLY和完整视频落盘。本次未重新执行950步cold fresh，故不把入口审查当作冷启动性能认证。

## 历史耗时与限制

原VR-v1-b缓存后增量2065.830s；原始前馈及其视频历史前缀442.946s，历史累计2508.776s。未作为一次新RGB冷启动实测。原timing、input manifest、检查点指标和逐步loss均原样保存。正式质量属于128D+U-Net运行时且静态source-RGB blend=0.75；单独PLY只是frame000顶点RGB预览。

原始传感器数据预处理仍是独立步骤，本包fresh入口输入为处理后RGB/LiDAR/标定/实例/masks。本次验证不覆盖新场景泛化或完整冷启动训练重现，不宣称10/20/30分钟预算通过。

## 控制面提示

guard无错误。接管时已存在三条旧证据变化警告：DEL-v1与DEL-v1-a引用的 `docs/VALIDATION.md`、旧交付验证run引用的 `docs/validation/summary.json`。本次没有改写这些文件。另有一次登记既有artifacts目录时产生的重复布局计划 `layout-0021` 未应用；实际目录已按 `layout-0022` 登记。控制器无撤销布局计划接口，因此保留该提醒，不手改追加式账本。以上均不影响本次资产与数值验证。

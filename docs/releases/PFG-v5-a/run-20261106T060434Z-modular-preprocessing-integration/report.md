# PFG-v5-a：三个可选预处理模块

完成可复用工程集成，PFG-v5交付主线保持不变，未晋级。原LIM-r已具有独立脚本/包；本次新增统一入口与两个可选模块，三项可单独或组合使用。

| 模块 | 实际职责 | 复用出口 |
|---|---|---|
| lim-r | LiDAR/bbox时差、归属与小空间校准 | 既有独立CLI、pointforward_calibration |
| depth-mask | 新InfiniDepth预测＋深度引导天空mask修复 | 独立CLI、repair数组API、逐视图米制深度和mask |
| scene-sky | 场景范围测量＋有限球面天空壳构建 | 独立CLI、measure_scene/build_sky数组API、Torch/无pickle NumPy资产 |

模块顺序固定为lim-r→depth-mask→scene-sky，但只执行显式选中的项。未选中模块不隐式启动；已有校准、深度和mask作为显式输入登记。每项保存manifest与独立计时，总计时用统一墙钟。没有Gaussian拟合或后优化。默认VR-v1-y等价重建脚本/配置、既有LIM-r入口与冻结资产没有变动。

代码局部迁移保留原数值公式，9个原数值定义有AST身份校验。其他项目可以调用数组API，不需导入PointForward研究仓库；本包的数据集CLI使用交付包自带adapter和源码，数据格式差异集中在adapter。天空资产需消费方在前景后显式alpha合成，不会因生成文件而自动改变默认渲染器。远场径向压缩函数作为可选API导出，实际重建Gaussian压缩不计入预处理。

## 验证与证据

`tests.log`：34项通过，覆盖7种非空模块组合、原数值身份、深度修复不增加天空、有限射线几何、范围公式、mask帧/相机顺序及原交付合同。修正一项旧测试对历史入口文件的陈旧引用，转为检查其实际保留的run_legacy_pipeline.py；未改旧运行逻辑。

真实数据：scene017帧1与61、相机0/1/2；明确复用ASR-v1-w的LIM-r校准，深度网络与权重使用交付包本地副本，重新推理后修复mask，不读旧depth-cache。`validation.json`记录：6视图与历史对应新深度结果有限像素最大差0，修复mask差异0像素；范围公式误差≤1e-10；修复mask消费保持RGB与内参不变；Torch/NumPy导出逐数组一致。

组合与独立天空调用的means/scales/quats/opacities/rgb_logits/observed渲染参数逐位一致。未观测补全球元数据keys有3584项浮点floor边界差异（最多一个纬度行），已如实记录；观测keys一致，保留原公式，不宣称全部元数据位级复现。天空共157523个Gaussian，其中28322个来自观测，129201个为明确标记的方向补全；不把补全称为真实观测。

| 本次有限帧调用 | 实测墙钟 / s |
|---|---:|
| depth-mask模块（含模型/数据加载、推理、修复与写盘） | 37.742 |
| 同次scene-sky模块（测量、构建与导出） | 18.265 |
| 组合整体（含编排与配置开销） | 57.273 |
| 独立scene-sky（用修复mask，整体） | 20.020 |
| 独立scene-sky（用原mask，整体） | 18.874 |

以上仅是两帧三相机工程测试，不代表全场景或RGB冷启动耗时，没有新PSNR/视频质量结论，完整LIM fit未重跑。精确边界与原始start/end见各preprocess_timing.json。

第一次组合运行depth-mask成功，scene-sky因冻结数据adapter在无ego mask时省略字段而失败。已在新预处理adapter中显式加载可用ego mask，否则用全空mask；原始失败目录与日志保留，新的空目录完整重试成功，没有覆盖历史结果。

本人查看了两张全图RGB/原mask/修复mask诊断图，确认帧1细杆/线缆与远吊车、帧61吊车立柱/横臂从误天空区域释放，建筑与道路保持非天空。图见 `real-combined-retry/depth-mask/diagnostics/`，记录见 `root_visual_review.json`。

复现命令、环境与输入见 `REPRODUCE.md`；源码快照、哈希见 `source-snapshot/` 和 `code_identity.json`。使用说明见 [PREPROCESSING.md](../../../PREPROCESSING.md)。

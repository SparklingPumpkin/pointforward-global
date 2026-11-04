# 固有前处理：LIM-v1-r

用户于2026-10-28接受研究版 LIM-v1-r，要求固化为此仓库的前处理。新建场景的默认入口 `scripts/run_pipeline.py --mode fresh` 现在先执行该流程，然后才执行静态/车辆前馈、联合优化和导出。历史 replay、VR-v1-s 冻结源码及已交付场景资产保持各自的旧配方身份。

```bash
python -m pip install -r requirements-calibration.txt
python scripts/run_pipeline.py --mode fresh \
  --data-root /data/argoverse/processed/training --scene 017 \
  --output-root /work/new-scene --gpu 0 --vehicle-gpu 1
```

`--vehicle-calibration lim-r` 为 fresh 默认。`lim-r-both` 在新校准后增加车辆重建框边距；`adaptive` / `both` 明确选择旧校准 / 旧校准加扩框；`bbox-expand` 跳过校准。replay未指定选项时使用原来的adaptive场景资产，不能把旧场景标成LIM-v1-r。

仅运行前处理（不启动Gaussian构建/训练）：

```bash
python scripts/preprocess_lidar_bbox.py --config /work/config_build.yaml \
  --scene 017 --out /work/geometry-input
```

该config采用已有训练配置格式，`data.geometry_directory` 在校准输入阶段会被移除，防止读入已校准场景。输出目录必须为空；需更换SAM解释器时，主入口使用 `--sam-python /path/to/python`。独立入口可指定 `--detector`、`--sam-checkpoint`、`--sam-root`。权重均随交付目录提供，不下载新模型。预处理模块为 `src/pointforward_calibration/`，默认命令不依赖研究仓库或历史P/Q缓存。

## 固定算法

1. 按原框几何、原图面积、可见性和轨迹位移选车；优先较大且运动明显的侧向观测。默认最多8辆，每车最多9次观测；可用 `--selection` 提供几何预选列表以复现指定车辆。所有保留车辆与观测都拟合，包括仅一个观测；没有可靠mask时仍作几何拟合，明确标记图像对应关系未核验。
2. 原框扩大1.5倍（单轴至少增加原算法规定边距），局部RANSAC去地面，再以原始种子为锚补充邻近聚类。保存原始点ID、扩充点池与保留/排除标记。
3. YOLO11n-seg独立从RGB产生车/公交/卡车mask；图像翻转一致性和SAM2.1 Tiny细化后匹配身份。3D框不生成mask。原算法的身份恢复和“大车误配小轿车mask”处理保留。
4. LiDAR与bbox分别测试真实源帧偏移 `-1/0/+1`；按既有fit/gate规则选整数偏移，之后所有观测一起拟合。这里的gate只核验粗帧选择，并非后续留出车辆。源帧偏移统一；需要下一帧而末帧缺失、需要上一帧而首帧缺失时均剔除，绝不退回当前帧。各车轨迹端点缺少选定bbox标注也显式排除该车该帧。
5. 用同一个时间假设同时满足RGB mask与3D时间管的点归属；保留多块有效车表面及既有可信点保护。共享时间加受约束逐车时间残差，LiDAR、bbox两路独立拟合。逐车残差加权均值为零；共享±0.2s，残差±0.06s；5ms粗剖面、0.5ms局部剖面、PCHIP与连续SLSQP，轨迹线性位置插值/SLERP，显式有界端点外推不超过0.27s。
6. 仅当原算法增益与回退条件允许，增加很小的逐车空间偏置。静态点世界坐标不施加动态时间变换；相机不拟合额外偏移，存在曝光时间时逐相机使用各自曝光时刻。

数值定义、配置及来源清单在 `provenance.json` / `numerical_contract.json`。73个函数/方法已逐项与研究源码AST核对；工程改动集中在原始输入、可移植导入、源帧边界和完整场景消费适配。未继续研究或改变已接受的拟合公式、阈值。

## 完整场景如何使用

研究版针对选车生成诊断；固化版新增完整场景应用桥。选中车辆使用共享时间、逐车残差与被接受的微小空间项；未参与拟合的车辆使用共享时间、零逐车残差/空间偏置，采用几何归属，不能声称这些车已经单独校准或人工核验。

每个相机保留RGB帧原始scan中的非归属动态点坐标，移除当前归属动态点，再加入所选真实源scan经该相机曝光时刻补偿后的归属动态点，以最近z生成稀疏深度。重叠候选按归一化框中心距离独占源点ID；拒绝点保留原始位置。静态点没有整体平移/时间变换，但动态点变化后其可见性及z-buffer结果可能变化。相邻帧和多相机共享相同的几何候选/无图像归属结果，避免重复计算。

`corrected_instances.pt` 提供按组时间采样的公共车辆轨迹，沿用现有模型的共同时间槽；诊断及稀疏深度在各相机曝光时刻计算。现有Gaussian模型仍共享每帧的actor时间槽，本次没有改成每相机一条车辆轨迹，也没有改变网络时间结构。原曝光信息缺失时使用原有10Hz名义时钟；曝光越界时原有自车姿态保持策略被明确记录。未校正滚动快门。

`run_calibrated_stage.py` 仅在新LIM-r分支显式启用消费适配：过滤不具备统一源帧的边界帧、加载修正框、读取每相机稀疏深度及原曝光相机姿态。直接运行冻结脚本不能自动获得这些新增行为；单独使用dataset时调用 `pointforward_calibration.consumer.install()` 后再构建dataset。旧adaptive/replay不会启用该包装。

主要输出位于 `geometry-input/`：

|文件|用途|
|---|---|
|`calibration.json`, `reference_calibration.json`|新时间/空间参数与R可信参考的前置拟合|
|`frame_selection.json`, `source_consistency.json`, `unsupported_tracks.json`|两路粗帧选择、统一源帧及不支持观测的明确记录|
|`*_result.npz`, `*_ownership.npz`|原始点池、ID、保留/拒绝标记、旧/时间/最终结果|
|`overview.jpg`, `ownership_overview.jpg`, `worst_overview.jpg`, `index.html`|与研究版一致的五列诊断与归属图|
|`corrected_instances.pt`, `camera_poses.pt`, `sparse/*.npz`|下游实际消费的修正框、逐相机曝光姿态与深度|
|`application_manifest.json`|应用帧/相机、原始源帧、逐车应用状态和静态不动审计|
|`input_manifest.json`, `raw_input_identity.json`, `*_config.json`, `preprocess_timing.json`|输入、权重/源码身份及前处理实测计时|

`--apply-frames` 仅用于独立校准入口的小范围应用验证，主入口永远导出全部受支持帧；部分导出会在result_summary中标记。不要将诊断用部分导出作为完整重建结果。

## 验收口径

本次为用户接受方案的独立工程固化，不重新宣布它满足原研究的所有质量目标。研究版原九批结果仍保留其原始失败记录：相同保留点上仍有17个观测比旧参考退化超过0.3px，mask和归属属于自动代理，时间偏移也不是已验证的硬件物理延迟。两个场景的小范围原始输入验证只证明封装和消费链可执行，不替代全场景重建质量、PSNR或端到端性能评测。旧VR-v1-s的31.13dB与26.82min不能套用于新默认前处理。

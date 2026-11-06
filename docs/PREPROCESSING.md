# 可独立选用的三个预处理模块

现有LIM-r已经具备独立入口 `scripts/preprocess_lidar_bbox.py` 和 `src/pointforward_calibration/`。现在增加统一入口 `scripts/preprocess_scene.py`，可分别选择下表三项，不执行前馈Gaussian构建、PCA、U-Net或后优化。

| 模块名 | 功能 | 主要输入 | 主要输出 |
|---|---|---|---|
| `lim-r` | 原LIM-v1-r车辆时差、点归属及小空间修正 | processed RGB/LiDAR/相机/实例，既有YOLO/SAM权重 | `corrected_instances.pt`、`camera_poses.pt`、逐相机`sparse/*.npz`、诊断与计时 |
| `depth-mask` | 新InfiniDepth预测＋深度引导天空mask修复 | RGB、内外参、原始sky mask、稀疏深度和通用深度权重；可显式使用LIM-r产物 | 逐视图米制`depth-cache/*.npy`、`masks/*.png`、全图修复前后图、`mask_manifest.json` |
| `scene-sky` | 测量场景范围＋实际有限球面交点投影＋共享天空壳 | raw LiDAR/相机位姿、RGB、sky/ego mask；可显式使用修复mask与校准位姿 | `measured_range.json`、`shared_sky.pt`、通用`sky_arrays.npz`、`sky_manifest.json` |

**交付主线仍为PFG-v5（VR-v1-y对应的LIM-r＋VR-v1-s重建组合）。** 默认 `run_pipeline.py` 的算法与配置没有切换。新增模块为可选工具；独立预处理可以服务其他项目，不代表已将ASR-v1-k重建或优化算法晋级为交付主线。

## 独立运行与组合

工作目录为本交付包根目录，使用已准备好的训练格式配置（含实际数据根、深度权重及源码路径），输出目录必须为空。以下均为processed数据输入，不包含原始TFRecord/AV2转换、SMPL标注生成。

```bash
# 三项全选；按lim-r→depth-mask→scene-sky顺序运行，均计入预处理。
python scripts/preprocess_scene.py --config /work/config_build.yaml \
  --scene 017 --out /work/preprocessing-all --gpu 0 \
  --modules lim-r depth-mask scene-sky

# 各项可单独选择。未选择的模块不会隐式运行。
python scripts/preprocess_scene.py --config /work/config_build.yaml \
  --out /work/geometry-only --modules lim-r
python scripts/preprocess_scene.py --config /work/config_build.yaml \
  --out /work/masks-only --modules depth-mask --gpu 0
python scripts/preprocess_scene.py --config /work/config_build.yaml \
  --out /work/sky-only --modules scene-sky

# 只选后两项，显式复用已生成的LIM-r校准。
python scripts/preprocess_scene.py --config /work/config_build.yaml \
  --out /work/mask-and-sky --modules depth-mask scene-sky \
  --geometry-dir /work/preprocessing-all/lim-r --gpu 0

# 天空单独用修复mask；省略mask-dir则使用原数据集mask。
python scripts/preprocess_scene.py --config /work/config_build.yaml \
  --out /work/new-sky --modules scene-sky \
  --mask-dir /work/mask-and-sky/depth-mask/masks

# 不加载模型/数据、不写输出，查看模块选择与依赖。
python scripts/preprocess_scene.py --config /work/config_build.yaml \
  --out /work/plan --modules depth-mask scene-sky --dry-run
```

若不执行LIM-r且配置本身带`data.geometry_directory`，会使用并明确登记该配置中的已有校准；要完全使用原始几何，请从输入配置移除此字段。选中LIM-r时会移除旧字段，防止新校准读取旧校准结果。`--geometry-dir`为显式已有校准；与新LIM-r互斥。

`--frames 1 61`仅作局部诊断，所有输出标记`partial_export`，不能当作全场景预处理结果。`--depth-dir /path/depth-cache`可供depth-mask显式使用已有深度（文件名`001_0.npy`等）；默认始终加载通用深度权重重新预测。复用校准、深度或mask均进入输入清单，不把阶段复用声称为独立RGB冷启动。

## 数值约定

深度修复沿用ASR-v1-k／w：正disparity直接取倒数保留未被旧近远裁剪压平的深度，无效值置NaN。31像素邻域第75百分位背景深度、前景/背景比例<0.6、逆深度差>0.0005 m⁻¹、8连通区域至少4像素、有效深度≤10000m。仅从原sky mask中移除被误判为天空的有限深度前景；不增添sky像素，不使用手工吊车ROI。原数值函数和来源身份保存在`src/pointforward_preprocessing/provenance.json`。

场景测量沿用同一公式：所有输入帧的raw LiDAR与相机中心统一到首个支持帧ego米制坐标，联合AABB中点为中心，精确最大距离为R；默认场景边界1.5R、天空半径2R。`--scene-multiplier` / `--sky-multiplier`可调整，要求1≤前者<后者。点云列布局由相应dataset adapter读取，不把Waymo/ncore误当作Argoverse Nx4。

天空RGB按相机射线与实际有限球面交点落入共享方向网格，原2048×1024观察网格、边界腐蚀4像素、默认像素步长2、两轮Huber外观累积，512×256网格补全未观察方向。补全颜色来自最近观察方向，`observed`明确区分；不声称补全方向有真实观测。输出Gaussian几何/透明度不训练。离线消费方可自行训练RGB，但那是后续优化，不属于此预处理。

原远场径向压缩公式也作为`range.compress(points, center, lower, upper, maximum)`提供，参数来自`measured_range.json`；**此处仅导出范围与天空，不读写任何重建Gaussian**。实际对前馈Gaussian执行压缩应归入重建阶段，不混入预处理耗时。

## 供其他项目使用

安装本包或将`src`加入`PYTHONPATH`，数值API不导入研究仓库，不要求启动重建网络：

```python
from pointforward_preprocessing.depth_mask import repair
from pointforward_preprocessing.range import measure_scene
from pointforward_preprocessing.sky import SkyView, build_sky

fixed_mask, removed = repair(depth_m, sky_mask_bool)   # H×W; True表示天空
sizing = measure_scene(world_point_chunks, camera_centers_m)
state, records = build_sky(repeatable_view_factory, sizing)
```

`SkyView`包含帧/相机ID、H×W×3 RGB（0..1 NumPy）、H×W sky/ego布尔mask、Torch三维相机中心与H×W×3世界方向。所有几何必须在同一米制坐标中。工厂每次迭代返回相同来源、相同顺序，支持两轮外观累积。仅修复已有深度时不需要InfiniDepth权重；独立scene-sky也不加载深度网络、YOLO或SAM。LIM-r使用既有独立模块，详见[LIM-r文档](LIDAR_BBOX_PREPROCESSING.md)。

`sky_arrays.npz`不含pickle，包含means/scales/quats/opacities/rgb_logits/keys/observed，与Torch文件的对应数组一致；元数据见`sky_manifest.json`。四元数为wxyz，RGB=sigmoid(rgb_logits)，坐标与`measured_range.json`同系。

## 输出消费与计时

统一入口输出`config_preprocessed.yaml`、`input_manifest.json`、`preprocessing_manifest.json`及`preprocess_timing.json`。各模块有独立计时；组合计时用同一墙钟，不把子模块开销重复叠加。

修复mask保持dataset裁剪/缩放后的像素坐标，不能直接当作传感器原分辨率mask覆盖数据源。PointForward下游若需使用它，显式调用：

```bash
python scripts/run_preprocessed_stage.py /path/to/stage.py \
  --config /work/preprocessing-all/config_preprocessed.yaml [其他stage参数]
```

包装器仅启用LIM-r消费桥与`data.sky_mask_directory`的逐帧mask消费；不修改冻结源码或全局数据集类，也不使普通PFG-v5渲染器自动获得天空壳。`preprocessing.scene_sky_state`是明确的下游资产引用：其他渲染器需将该层在前景后独立alpha合成。**生成天空文件不等于原默认渲染已启用它**，本次没有更换默认渲染/优化算法。

验证范围与实际运行证据见[模块化工程验收](releases/PFG-v5-a/report.md)。此次不宣称新全场景PSNR或冷启动速度。

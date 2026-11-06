# 模块化预处理复现

工作目录：交付包根目录。环境：`/root/miniconda3/envs/drivestudio/bin/python`，使用交付包现有依赖与通用权重。配置、数据、权重与校准来源见 `config_input.yaml`、`real_input_identity.json`、各模块 `input_manifest.json` / `mask_manifest.json`；源代码快照与身份见 `source-snapshot/`、`code_identity.json`。

这是scene017帧1/61、相机0/1/2的工程验证，明确复用ASR-v1-w的LIM-r校准；不是全流程冷启动，不进行Gaussian拟合、训练或视频质量验收。原始数据不被覆盖。输出路径须换成新的空目录，勿覆盖本次证据。

```bash
PFG=/mnt/data/fjn/projects/fjn/pointforward-global
RUN="$PFG/docs/releases/PFG-v5-a/run-20261106T060434Z-modular-preprocessing-integration"
PY=/root/miniconda3/envs/drivestudio/bin/python
GEOMETRY=/mnt/data/fjn/projects/PointForward/experiments/argoverse-inference/cold-scene-delivery/vehicle-roi-visibility/feedforward-parameter-search/ASR-v1-w/run-20261106T013721Z-asrk-independent-rgb-cold/geometry-input
cd "$PFG"

# 新深度、修复mask和天空组合；depth-mask内部seed=0。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" scripts/preprocess_scene.py \
 --config "$RUN/config_input.yaml" --geometry-dir "$GEOMETRY" \
 --frames 1 61 --gpu 1 --out /work/new-combined \
 --modules depth-mask scene-sky

# 同一输入独立天空；不加载深度网络。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" scripts/preprocess_scene.py \
 --config "$RUN/config_input.yaml" --geometry-dir "$GEOMETRY" \
 --frames 1 61 --out /work/new-independent-sky --modules scene-sky \
 --mask-dir /work/new-combined/depth-mask/masks

# 原始mask也能单独构建天空，省略mask-dir即可。
# 完整LIM-r独立入口及参数见 docs/LIDAR_BBOX_PREPROCESSING.md；此次未重跑LIM fit。
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src "$PY" -m pytest \
 tests/test_modular_preprocessing.py tests/test_lidar_bbox_preprocessing.py \
 tests/test_pipeline_contract.py tests/test_data.py -q

# 只读复核本次证据；不会覆盖validation.json或修改默认流程。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" "$RUN/source-snapshot/validate_integration.py" \
 --project-root "$PFG" --run "$RUN"
```

所有7种非空模块子集由单元测试检查。独立深度mask数值函数、通用范围与天空数组接口及全三模块CLI用法见 `docs/PREPROCESSING.md`。真实组合运行命令与原始日志保留于 `real-input.log`；首次失败和恢复说明见 `report.md`。

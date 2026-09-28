# PointForward Global（PFG-v2 当前主线）

这是当前 27 dB 全局 Gaussian 版本的独立交付仓库。它包含原始数据预处理、PointForward 预训练、全局静态/动态 Gaussian 推理、场景级 RGB 后优化、PLY 导出和 128D+UNet 视频渲染所需源码，以及 Git 中唯一保留的 PointForward checkpoint；不包含任何数据集、历史实验、payload、逐帧图片或成品视频。

当前版本由 [项目控制面](docs/project-control/CURRENT.md) 标记。PFG-v2 将最终优化默认切换为 `fast-g8`，在 scene017 上达到 27.085059 dB，并把固定输入 payload 到最终 PLY 的阶段压缩到 7.08 分钟；旧版四阶段尾部仍可选。方法见 [METHOD.md](docs/METHOD.md)，从 DG-v1-f 开始的完整技术演进见 [UPGRADES_DG_V1_F_TO_PFG_V2.md](docs/UPGRADES_DG_V1_F_TO_PFG_V2.md)。

## 1. 硬件与环境

完整 scene017 流程建议使用 Linux、CUDA 12.x、FFmpeg 6 和 A100 80 GiB 级别显存；`fast-g8` 验证峰值 CUDA allocation 约 76.41 GiB。预训练按论文配置使用 8 GPU、全局 batch 8；单 GPU 可以运行但会显著变慢。

```bash
cd /path/to/pointforward-global
git lfs pull
conda env create -f environment.yml
conda activate pointforward-global
export PYTHONPATH="$PWD/src:$PWD"
export HF_HOME="$PWD/weights/huggingface"
python scripts/validate_install.py
```

如果已有 CUDA 版 PyTorch，亦可执行 `pip install -r requirements.txt && pip install -e .`。`weights/manifest.json` 给出 Git 保留 checkpoint 以及外部可选权重的大小、SHA256 和用途。为控制 Git 体积，InfiniDepth、SegFormer、EfficientNetV2-S 初始化权重和 scene017 refiner 不再上传到 Git。

## 2. 原始数据预处理

### 外部可选模型权重

Git 只提供 `weights/pointforward_waymo_ds_paper_ckpt_020000.pt`。运行完整推理前，还需将以下可选权重放到对应路径（或修改生成配置指向其他路径）：

- `weights/infinidepth_depthsensor.ckpt`：从 InfiniDepth 发布渠道获取；
- `weights/segformer.b5.1024x1024.city.160k.pth`：按下方 Argoverse 文档的 NVlabs SegFormer 下载说明获取；
- `weights/huggingface/.../model.safetensors`：运行大规模预训练时由 timm/Hugging Face 下载，或从内部缓存复制；
- `weights/refiner_scene017_fast_g8.pt`：用于复现 PFG-v2 scene017 的 refiner overlay；
- `weights/refiner_scene017_27db.pt`：旧 PFG-v1 / `legacy-alternating` overlay。

下载后可用 `weights/manifest.json` 中 `optional_local_files` 的 SHA256 校验。`scripts/validate_install.py` 默认只检查 Git 中的 PointForward checkpoint。

### Argoverse 2

先按 `src/drivestudio_preprocess/docs/ArgoVerse.md` 下载原始 sensor 数据，然后运行：

```bash
export PYTHONPATH="$PWD/src/drivestudio_preprocess"
python src/drivestudio_preprocess/datasets/preprocess.py \
  --data_root /data/argoverse/raw \
  --target_dir /data/argoverse/processed/training \
  --dataset argoverse \
  --split_file /path/to/scene-list.txt \
  --workers 32 \
  --process_keys images lidar calib pose dynamic_masks objects
```

当前流程使用 `images/`、相机内外参、ego pose、objects/instances、LiDAR 稀疏深度和 sky mask。场景目录应符合 vendored 文档末尾的数据结构。

### Sky mask / fine dynamic mask

SegFormer 依赖 PyTorch 1.8.1 和 mmcv-full 1.2.7，应创建独立环境；不要把它安装进主渲染环境。源码和 Cityscapes 权重已经随包提供：

```bash
conda create -n pfg-segformer python=3.8 -y
conda activate pfg-segformer
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2 mmcv-full==1.2.7 --no-build-isolation
pip install -e src/SegFormer imageio opencv-python tqdm
export PYTHONPATH="$PWD/src/drivestudio_preprocess:$PWD/src/SegFormer"
python src/drivestudio_preprocess/datasets/tools/extract_masks.py \
  --data_root /data/argoverse/processed/training \
  --segformer_path "$PWD/src/SegFormer" \
  --checkpoint "$PWD/weights/segformer.b5.1024x1024.city.160k.pth" \
  --split_file /path/to/scene-list.txt \
  --process_dynamic_mask
```

Waymo 的下载和转换方法见 `src/drivestudio_preprocess/docs/Waymo.md`，依赖放在 `requirements-preprocess.txt`。数据 SDK 受数据集许可约束，因此本包只提供转换代码，不提供原始或处理后数据。

## 3. 生成机器本地配置

模板不含交付机器的绝对路径：

```bash
conda activate pointforward-global
export PYTHONPATH="$PWD/src:$PWD"
python scripts/prepare_config.py \
  --data-root /data/argoverse/processed/training \
  --output-root /work/pfg/scene017 \
  --out /work/pfg/scene017/config.yaml
```

如处理其他场景，编辑生成的 YAML 中 `train_scenes` / `val_scenes`。三前向相机、分辨率和 12 视图输入合同保持不变。

## 4. 模型预训练

论文式 8 GPU 预训练：

```bash
export HF_HOME="$PWD/weights/huggingface"
torchrun --nproc_per_node=8 -m scripts.train \
  --config /work/pfg/scene017/config.yaml \
  --out /work/pfg/pretrain
```

单机 smoke（只检查训练链，不代表模型质量）：

```bash
python -m scripts.train --config /work/pfg/scene017/config.yaml \
  --depth constant --steps 4 --out /work/pfg/pretrain-smoke
```

本次交付未重跑大型预训练。默认推理权重为 `weights/pointforward_waymo_ds_paper_ckpt_020000.pt`，InfiniDepth 为冻结深度先验。

## 5. 当前主线完整推理

下列命令从处理后 RGB/标定/box/LiDAR 开始，分阶段生成真正全局的 128D Gaussian payload、指标、标准预览 PLY 和 60 fps 视频：

```bash
python scripts/run_current_pipeline.py \
  --config /work/pfg/scene017/config.yaml \
  --output-root /work/pfg/scene017/current \
  --scene 017
```

默认等价于 `--final-recipe fast-g8`。如需严格复现旧 PFG-v1 四阶段尾部，增加 `--final-recipe legacy-alternating`。

每个阶段以 manifest 或目标文件为完成证据，重复执行会自动跳过已完成阶段。可用 `--from-stage NAME`、`--through-stage NAME` 选择区间，用 `--force` 重做，用 `--skip-video` 只做到指标和 PLY。列出阶段名可运行 `python scripts/run_current_pipeline.py --help`，或查看脚本中的 `Stage` 定义。

主要输出：

- `14_static_g8_fast_final/runtime-payload-refined/`：PFG-v2 默认最终 128D 全局 payload；
- `17_static_xyz_1cm/runtime-payload-refined/`：仅旧 `legacy-alternating` 最终 payload；
- `18_metrics/manifest.json`：全局/非天空/动态 bbox PSNR；
- `final_global_gaussians_frame000_preview.ply`：frame 0 动态姿态下的标准 PLY；
- `19_video/final_global_runtime_predicted_front3_60fps.mp4`：相机与车辆均插值的 60 fps 视频。

PLY 只有缓存 preview RGB / degree-0 SH，适合几何查看，不等价于 27 dB 的 128D+UNet 输出。高质量视频必须保留 payload、base checkpoint 和最终 refiner overlay。

## 6. 对已有 payload 渲染、评估和导出

```bash
PAYLOAD=/work/pfg/scene017/current/14_static_g8_fast_final/runtime-payload-refined
CONFIG=/work/pfg/scene017/config.yaml
REFINER="$PWD/weights/refiner_scene017_fast_g8.pt"
CKPT="$PWD/weights/pointforward_waymo_ds_paper_ckpt_020000.pt"

# 156 个源帧指标
python scripts/render_final_global_runtime.py --config "$CONFIG" --ckpt "$CKPT" \
  --refiner-state "$REFINER" --payload-dir "$PAYLOAD" --out /work/pfg/metrics \
  --scene 017 --metrics-only --dynamic-bbox-metrics --camera-batch-size 3

# 60 fps：相机和车辆都插值
python scripts/render_final_global_runtime.py --config "$CONFIG" --ckpt "$CKPT" \
  --refiner-state "$REFINER" --payload-dir "$PAYLOAD" --out /work/pfg/video60 \
  --scene 017 --fps 60 --camera-interpolation-factor 6 --interpolate-dynamics \
  --camera-batch-size 3

# 标准预览 PLY
python scripts/export_runtime_payload_ply.py --payload-dir "$PAYLOAD" \
  --out /work/pfg/final_frame000_preview.ply --frame 0
```

两个 scene017 refiner 都只能和对应配方的 payload 配套使用，不能直接套到新场景。

## 7. 新视角

```bash
# 左移 2 m / 4 m
python scripts/render_final_global_runtime.py --config "$CONFIG" --ckpt "$CKPT" \
  --refiner-state "$REFINER" --payload-dir "$PAYLOAD" --out /work/pfg/left2 \
  --scene 017 --camera-lateral-left-m 2
python scripts/render_final_global_runtime.py --config "$CONFIG" --ckpt "$CKPT" \
  --refiner-state "$REFINER" --payload-dir "$PAYLOAD" --out /work/pfg/left4 \
  --scene 017 --camera-lateral-left-m 4

# 抬升 3 m、俯视 20 度
python scripts/render_final_global_runtime.py --config "$CONFIG" --ckpt "$CKPT" \
  --refiner-state "$REFINER" --payload-dir "$PAYLOAD" --out /work/pfg/high3 \
  --scene 017 --camera-vertical-m 3 --camera-pitch-down-deg 20

# 360 度环绕；中心和半径按场景调整
python scripts/render_final_global_runtime.py --config "$CONFIG" --ckpt "$CKPT" \
  --refiner-state "$REFINER" --payload-dir "$PAYLOAD" --out /work/pfg/orbit \
  --scene 017 --camera-trajectory orbit --orbit-frame-count 360 \
  --orbit-radius-m 35 --orbit-height-m 5 --fps 60
```

## 8. 验收边界

交付仓库的轻量检查包括语法、导入、权重 SHA256、编排合同检查，以及目标仓库代码对既有最终 payload 的 156 帧复算。不会把最终 payload 打包进 Git，因为它是约 20.7M Gaussian 的场景数据产物。7.08 分钟仅统计固定 26.668159 dB 输入 payload 的最后 RGB 优化到 PLY，不包含原始数据预处理、深度预测、全局建图及此前优化阶段。复现记录见 `docs/VALIDATION.md`。

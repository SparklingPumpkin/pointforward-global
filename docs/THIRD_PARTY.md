# 第三方代码与模型

- `src/InfiniDepth`：InfiniDepth 源码，遵循其目录中的 `LICENSE`。
- `src/drivestudio_preprocess`：51w-DriveStudio 数据预处理快照，遵循其目录中的 `LICENSE`。
- `src/SegFormer`：NVlabs SegFormer 源码，遵循其目录中的许可证。
- `weights/infinidepth_depthsensor.ckpt`：PointForward 深度先验。
- `weights/segformer.b5.1024x1024.city.160k.pth`：仅用于生成 sky/fine dynamic masks。
- `weights/huggingface`：timm EfficientNetV2-S 的离线 Hugging Face 缓存，用于从头预训练初始化；加载已训练 PointForward checkpoint 时不需要再次下载。

数据集本身不在本仓库中。使用 Argoverse 2、Waymo 等数据前必须接受并遵守各自许可和使用条款。

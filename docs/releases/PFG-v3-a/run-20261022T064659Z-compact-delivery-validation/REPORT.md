# VR-v1-s 独立交付验收

用户指定将当前最佳版本固化为交付主线。研究目标车辆29dB仍失败，本报告只验收打包完整性、可移植入口与固定状态重放，不以交付晋级改写研究结论。

独立执行 `scripts/run_pipeline.py --mode replay`，仅使用包内源码、通用权重、scene017专属状态，以及指定原始处理后数据。全156帧三相机：全局31.13165373679919、车辆bbox28.32924439356877、非天空30.68500505349575dB，三项与源实验误差均为0。行人bbox也一致，为25.01632857322693dB。

预测视频2304×576、GT对照2304×1152，均156帧、10fps；独立导出的frame000 PLY与源交付预览SHA256相同。实际重放视频、评估与PLY导出211.9892128231004s，为cached_incremental，未重新冷启动训练。

验证3922项资产与443项源码指纹，所有关键场景文件均实际复制，无指向研究仓库的资产软链接。源/交付Gaussian、PCA、U-Net、轨迹、SMPL、通用权重逐文件SHA256一致。源码只有训练器编排helper import、SAM模型路径参数、显式启用预编译扩展三项便携差异，未改优化和渲染公式。完整结果见 `validation.json`，fresh入口配置/命令/import核对见 `pipeline_review.json`。

亲自查看重渲染第20/80/140帧三相机GT对比（`replay/review.png`）及车辆局部图：街道/建筑/路面结构完整，白色轿车及远处车辆外观与原结果一致；第140帧近处SUV仍有车头/车窗模糊，细纹理仍软，未宣称车辆质量达29dB或新视角无缺陷。

源一次真实冷启动，从处理后RGB/LiDAR/标定/实例/天空mask/SMPL标注开始，到预处理+前馈+PCA+优化保存退出1485.371231320314s；至完整视频/指标/PLY1609.0870128450915s。两GPU静态/车辆前馈并行，阶段秒不直接相加；不包含原始数据集转换与SMPL标注生成。PLY仅预览，正式PSNR属于32D特征Gaussian+PCA/U-Net+轨迹+source-RGB合成运行时。

新默认入口为 `scripts/run_pipeline.py`，版本专属实现 `scripts/run_vr_v1_s_pipeline.py`；旧 `run_current_pipeline.py` 和 `docs/ALGORITHM_CHANGES.md` 被历史验收冻结，保留给历史复现，持续变更改在 `docs/ALGORITHM_CHANGES_CURRENT.md` 维护。旧资产未删除。新验收不冻结可持续维护的默认别名及变更文档，版本专属入口、模板、说明、清单与验证证据冻结。

控制面guard保留4项旧警告：DEL-v1及DEL-v1-a的旧VALIDATION证据发生过变化、旧run的summary证据变化、layout-0021未应用；均为本次接管前已有警告，本次未改写这些历史证据。

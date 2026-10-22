# Raw cold replay provenance

This run starts before scene preprocessing. Inputs are the original processed
Argoverse training/017 RGB, LiDAR, annotations, calibration and pedestrian pose
annotations under /mnt/data/dataset/argoverse/processed/training. These are
the dataset input boundary, not a previously fitted Gaussian scene.

The inherited configuration and prior pipeline JSON are recipes only. The runner
rewrites all scene-output paths to this run. Geometry preparation, SAM masks,
geometry offset fitting/filtering, static FF, vehicle FF, SMPL FF and PCA are
all regenerated inside the measured timer. No historical scene Gaussian payload,
scene PCA basis, optimizer state, Gaussian delta or fitted U-Net is loaded.

Generic offline prerequisites: PointForward ckpt_020000_waymo_ds_paper_original.pt,
InfiniDepth depthsensor checkpoint, SAM2.1 tiny checkpoint, pedestrian
ckpt_002000.pt and neutral SMPL model. The pinned gsplat binary is generic
software, with SHA256
ea65eb53c4b4e62681ab9afec2a31065b781f0a2ee124a3bf5fd6f7933db55c1.

The raw-to-optimization endpoint follows successful trainer exit and checkpoint
serialization. Independent static/vehicle producers run on GPUs 0 and 5 and
are joined before SMPL/PCA; their elapsed GPU times must not be summed as wall
time. A 1480-second in-training guard requests final save, but only the actual
recorded endpoint decides the 1500-second criterion.

optimization_timing.json measures preprocessing+FF+optimization.
pipeline_timing.json continues through evaluation/video/PLY delivery. Exported
PLY is a preview; formal RGB quality belongs to the rank32 features, U-Net,
trajectory, source-compositing assets and renderer together.

Metadata caveat: parameter_contract.json retains an inherited actor-loss label.
The actual loss is selected by options.json vehicle_loss=union and the union
branch of refine_unified_scene_rgb.py. A documentation-only label edit during
startup was reverted; the source snapshot retains the executed numerical code.

---
name: lidar-bbox-overlay-diagnostics
description: Generate unchanged RGB, 3-D vehicle bounding box, and raw LiDAR overlay diagnostics for checking calibration and point ownership.
---

# LiDAR/BBox overlay diagnostics

Use this skill when a scene needs visual evidence of the *unmodified* geometry before changing a LiDAR/BBox calibration strategy. The output is diagnostic only: it does not fit a transform, correct timestamps, deskew points, move boxes, or run network inference.

Run the bundled script with an audit scene, the parent run containing its saved labels and manifest, and a new output directory:

```bash
python /mnt/fjn/projects/fjn/pointforward-global/skills/lidar-bbox-overlay-diagnostics/scripts/export_overlays.py \
  --project-root /mnt/fjn/projects/PointForward \
  --audit-dir /path/to/audit/scene \
  --parent-run /path/to/parent/run \
  --output-dir /path/to/output/raw-geometry-overlays \
  --frames 95 99 101 104 108 112 \
  --track-id 49 --camera 1
```

The default contract expects `data_contract.json` and `poses.pt` under `--audit-dir`, `manifest.json` and files named `{scene}_t{track:03d}_f{frame:03d}_c{camera}_labels.npz` under `--parent-run`, raw points under `raw_root/lidar/{frame:03d}.bin`, and rectified images under `audit/images/{frame:03d}_{camera}.png`. Override these with `--contract`, `--poses`, `--raw-root`, `--label-dir`, `--label-pattern`, `--image-dir`, `--fallback-image-dir`, or `--image-pattern` when a run uses another layout. Use `--scene` when the scene name cannot be read from `poses.pt`.

By default, the script verifies the RGB and LiDAR hashes, camera intrinsics, saved point membership, and saved source UVs from the parent run. `--skip-replay-checks` is available only for a new source without those records and is recorded in the output manifest.

Outputs include one full-frame JPG, one aspect-ratio-preserving detail JPG, one PNG overlay per frame, `overview_six_frames.jpg`, and `manifest.json`. Colors mean:

- yellow: projected original 3-D box;
- orange: raw LiDAR points inside the original box;
- cyan: points in the 1.5× box ring but outside the box;
- gray: other displayed raw LiDAR points.

These colors express geometric containment only; they do not prove physical vehicle ownership or visibility/occlusion correctness. Inspect the detail images and manifest before proposing a calibration change. Keep the generated evidence until the project’s later large-file cleanup pass.

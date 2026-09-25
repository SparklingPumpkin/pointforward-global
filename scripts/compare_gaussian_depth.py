"""Compare an existing posed Gaussian PLY with LiDAR-conditioned InfiniDepth.

Uses camera-z expected depth from the repository's gsplat renderer, not a
Gaussian-centre z-buffer. No query resampling, RGB decoding or hole filling.
Run from the repository root with python -m scripts.compare_gaussian_depth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from plyfile import PlyData

from pointforward.models.renderer import render_features
from scripts.train import build_dataset, build_depth


def error_stats(residual: np.ndarray, mask: np.ndarray) -> dict:
    values = residual[mask].astype(np.float64)
    result = {"pixels": int(mask.sum()), "coverage": float(mask.mean())}
    if not values.size:
        return result
    absolute = np.abs(values)
    result.update(
        mean_signed_m=float(values.mean()),
        mae_m=float(absolute.mean()),
        median_absolute_m=float(np.median(absolute)),
        rmse_m=float(np.sqrt(np.mean(values ** 2))),
        p90_absolute_m=float(np.percentile(absolute, 90)),
        p98_absolute_m=float(np.percentile(absolute, 98)),
        max_absolute_m=float(absolute.max()),
    )
    return result


def draw_comparison(rgb, gaussian, reference, valid, stats, title, prefix):
    residual = gaussian - reference
    depth_valid = np.isfinite(reference) & (reference > 0)
    high = max(float(np.percentile(np.concatenate(
        [reference[depth_valid], gaussian[valid]]), 98)), 1.0)
    error_high = max(stats["p98_absolute_m"], 0.1)
    depth_cmap = plt.get_cmap("turbo").copy()
    error_cmap = plt.get_cmap("inferno").copy()
    depth_cmap.set_bad("#777777")
    error_cmap.set_bad("#777777")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(title + "\nNative resolution; gray = invalid/low-alpha; no hole filling", fontsize=15)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input RGB (same frame and camera)")
    depth_artist = axes[0, 1].imshow(np.ma.array(gaussian, mask=~valid),
                                   cmap=depth_cmap, vmin=0, vmax=high)
    axes[0, 1].set_title("Gaussian rendered expected depth (camera Z)")
    reference_artist = axes[1, 0].imshow(np.ma.array(reference, mask=~depth_valid),
                                       cmap=depth_cmap, vmin=0, vmax=high)
    axes[1, 0].set_title("InfiniDepth + LiDAR (camera Z)")
    error_artist = axes[1, 1].imshow(np.ma.array(np.abs(residual), mask=~valid),
                                   cmap=error_cmap, vmin=0, vmax=error_high)
    axes[1, 1].set_title(
        f"Absolute difference |Gaussian - InfiniDepth|\n"
        f"Valid {stats['coverage']:.2%}; MAE {stats['mae_m']:.3f} m; "
        f"median {stats['median_absolute_m']:.3f} m")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.colorbar(depth_artist, ax=axes[0, 1], label="Depth (m); shared scale", shrink=0.85, extend="max")
    fig.colorbar(reference_artist, ax=axes[1, 0], label="Depth (m); shared scale", shrink=0.85, extend="max")
    fig.colorbar(error_artist, ax=axes[1, 1], label="Absolute difference (m); clipped at P98", shrink=0.85, extend="max")
    fig.savefig(str(prefix) + "_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    artist = ax.imshow(np.ma.array(np.abs(residual), mask=~valid),
                       cmap=error_cmap, vmin=0, vmax=error_high)
    ax.set_title(title + f"\n|Gaussian - InfiniDepth|; valid {stats['coverage']:.2%}; gray = invalid")
    ax.set_axis_off()
    fig.colorbar(artist, ax=ax, label="Absolute depth difference (m); clipped at P98", shrink=0.85, extend="max")
    fig.savefig(str(prefix) + "_heatmap.png", dpi=150)
    plt.close(fig)
    return {"depth_range_m": [0, high], "absolute_difference_range_m": [0, error_high],
            "clipping": "display only; statistics use unclipped values"}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--out", default="visualization")
    args = parser.parse_args()
    if not 0 < args.alpha_threshold < 1:
        raise ValueError("alpha threshold must be in (0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat rendering")
    device = torch.device("cuda")
    ply_path = Path(args.ply)
    snapshot = json.loads(ply_path.with_suffix(".json").read_text())
    config = yaml.safe_load(Path(args.config).read_text())
    if config["depth"]["kind"] != "infinidepth":
        raise ValueError("This diagnostic requires real InfiniDepth, not ConstantDepth")
    dataset = build_dataset(config, args.split)
    scene_index, start = dataset.index[snapshot["clip_index"]]
    metadata = dataset.scenes[scene_index]
    window = metadata["frames"][start:start + dataset.clip_length]
    if metadata["name"] != snapshot["scene"] or list(window) != snapshot["window_frames"]:
        raise ValueError("Dataset clip does not match PLY metadata")
    if snapshot["coordinate_frame"] != "world recentered at the first ego pose of this clip":
        raise ValueError("Unrecognized PLY coordinate frame")
    if args.camera not in dataset.cameras:
        raise ValueError("Requested camera is not in the dataset config")
    frame = snapshot["snapshot_frame"]
    if window[snapshot["snapshot_frame_offset"]] != frame:
        raise ValueError("Inconsistent snapshot frame")
    world_from = np.linalg.inv(dataset._ego_pose(metadata, window[0]))
    batch = dataset._collect(metadata, [frame], {f: i for i, f in enumerate(window)},
                             dataset._normalised_times(window), dataset.resolution,
                             world_from, lidar=True)
    view = dataset.cameras.index(args.camera)
    if "sparse_depth" not in batch or not batch["sparse_mask"][view].any():
        raise RuntimeError("No LiDAR support for the selected view")
    image = batch["images"][view:view + 1].to(device)
    lidar = batch["sparse_depth"][view:view + 1].to(device)
    lidar_mask = batch["sparse_mask"][view:view + 1].to(device)
    torch.manual_seed(config.get("seed", 0))
    depth_model = build_depth(config).to(device).eval()
    reference = depth_model(image, sparse_depth=lidar, sparse_mask=lidar_mask)[0].float().cpu().numpy()
    del depth_model

    vertices = PlyData.read(str(ply_path))["vertex"].data
    if len(vertices) != snapshot["gaussian_count"]:
        raise ValueError("PLY vertex count differs from metadata")

    def fields(names):
        array = np.stack([vertices[name] for name in names], axis=-1).astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite PLY fields: {names}")
        return torch.from_numpy(array).to(device)

    means = fields(["x", "y", "z"])
    quats = fields([f"rot_{i}" for i in range(4)])
    scales = fields([f"scale_{i}" for i in range(3)]).exp()
    opacity = fields(["opacity"]).squeeze(-1).sigmoid()
    height, width = image.shape[-2:]
    # A dummy feature channel suffices: RGB/features do not influence depth or alpha.
    _, alpha_tensor, depth_tensor = render_features(
        means, quats, scales, opacity, torch.zeros((len(vertices), 1), device=device),
        batch["intrinsics"][view:view + 1].to(device),
        torch.linalg.inv(batch["cam_to_world"][view:view + 1].to(device)),
        height, width, render_depth=True)
    assert depth_tensor is not None
    gaussian = depth_tensor[0, 0].cpu().numpy()
    alpha = alpha_tensor[0, 0].cpu().numpy()
    finite = np.isfinite(gaussian) & (gaussian > 0) & np.isfinite(reference) & (reference > 0) & np.isfinite(alpha)
    valid = finite & (alpha >= args.alpha_threshold)
    if not valid.any():
        raise RuntimeError("No valid rendered pixels at the selected alpha threshold")
    residual = gaussian - reference
    stats = error_stats(residual, valid)
    sky = (batch["sky_mask"][view, 0].numpy() > 0.5) if "sky_mask" in batch else None
    prefix = Path(args.out) / f"{metadata['name']}_clip{snapshot['clip_index']:04d}_frame{frame:03d}_cam{args.camera}_gaussian_vs_infinidepth"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb = image[0].cpu().numpy().transpose(1, 2, 0)
    display = draw_comparison(rgb, gaussian, reference, valid, stats,
                              f"Scene {metadata['name']} | frame {frame} | camera {args.camera} | {width}x{height} | alpha >= {args.alpha_threshold}", prefix)
    arrays = dict(gaussian_depth_m=gaussian, infinidepth_m=reference, alpha=alpha,
                  valid_mask=valid, signed_difference_m=np.where(valid, residual, np.nan),
                  absolute_difference_m=np.where(valid, np.abs(residual), np.nan),
                  lidar_depth_m=lidar[0, 0].cpu().numpy(), lidar_mask=lidar_mask[0, 0].cpu().numpy(),
                  intrinsics=batch["intrinsics"][view].numpy(), cam_to_world=batch["cam_to_world"][view].numpy())
    if sky is not None:
        arrays["sky_mask"] = sky
    np.savez_compressed(str(prefix) + "_depths.npz", **arrays)
    metrics = dict(scene=metadata["name"], frame=frame, camera=args.camera,
                   resolution_hw=[height, width], gaussian_count=len(vertices),
                   source_ply=str(ply_path.resolve()), source_ply_sha256=hashlib.sha256(ply_path.read_bytes()).hexdigest(),
                   source_snapshot=snapshot, depth_config=config["depth"],
                   depth_source="fresh InfiniDepth inference conditioned on same-frame LiDAR",
                   gaussian_source="existing posed PLY; no resampling; invert log scales/logit opacity",
                   depth_semantics="camera-Z metres; gsplat RGB+ED expected depth, alpha-normalized, near=0.1 far=500",
                   alpha_threshold=args.alpha_threshold, valid_pixel_stats=stats,
                   alpha_threshold_sensitivity={str(t): error_stats(residual, finite & (alpha >= t)) for t in [0.01, 0.1, 0.5, 0.9, 0.99]},
                   alpha_percentiles={str(p): float(np.percentile(alpha, p)) for p in [0, 1, 5, 50, 95, 99, 100]},
                   lidar_pixels=int((lidar_mask > 0.5).sum().item()), display=display,
                   interpretation="Gaussian-vs-InfiniDepth consistency, not independent ground-truth accuracy; no hole filling or downsampling")
    if sky is not None:
        metrics["sky_stats"] = error_stats(residual, valid & sky)
        metrics["non_sky_stats"] = error_stats(residual, valid & ~sky)
    Path(str(prefix) + "_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved comparison, heatmap, depths and metrics: {prefix}")


if __name__ == "__main__":
    main()
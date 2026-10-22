"""Re-run InfiniDepth with rendered Gaussian depth as the sparse prompt."""

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

from scripts.train import build_dataset, build_depth


def error_stats(residual: np.ndarray, mask: np.ndarray) -> dict:
    values = residual[mask].astype(np.float64)
    absolute = np.abs(values)
    result = {"pixels": int(mask.sum()), "coverage": float(mask.mean())}
    if not values.size:
        return result
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


def masked_depth(array: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(array, mask=~mask)


def add_colorbar(fig, artist, axis, label: str) -> None:
    fig.colorbar(artist, ax=axis, label=label, shrink=0.82, extend="max")


def draw_seven_panel(
    rgb: np.ndarray,
    prompt: np.ndarray,
    prompt_mask: np.ndarray,
    old_depth: np.ndarray,
    new_depth: np.ndarray,
    old_prompt_stats: dict,
    new_old_stats: dict,
    new_prompt_stats: dict,
    title: str,
    output: Path,
) -> dict:
    depth_values = np.concatenate(
        (prompt[prompt_mask], old_depth[np.isfinite(old_depth)], new_depth[np.isfinite(new_depth)])
    )
    depth_high = max(float(np.percentile(depth_values, 98)), 1.0)
    old_prompt_error = np.abs(prompt - old_depth)
    new_old_error = np.abs(new_depth - old_depth)
    new_prompt_error = np.abs(new_depth - prompt)
    old_prompt_high = max(old_prompt_stats["p98_absolute_m"], 0.1)
    new_old_high = max(new_old_stats["p98_absolute_m"], 0.1)
    new_prompt_high = max(new_prompt_stats["p98_absolute_m"], 0.1)

    depth_cmap = plt.get_cmap("turbo").copy()
    error_cmap = plt.get_cmap("inferno").copy()
    depth_cmap.set_bad("#777777")
    error_cmap.set_bad("#777777")

    fig = plt.figure(figsize=(24, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 12)
    axes = [
        fig.add_subplot(grid[0, 0:3]),
        fig.add_subplot(grid[0, 3:6]),
        fig.add_subplot(grid[0, 6:9]),
        fig.add_subplot(grid[0, 9:12]),
        fig.add_subplot(grid[1, 0:4]),
        fig.add_subplot(grid[1, 4:8]),
        fig.add_subplot(grid[1, 8:12]),
    ]
    fig.suptitle(title + "\nGaussian alpha mask is used unchanged; gray = invalid", fontsize=17)

    axes[0].imshow(rgb)
    axes[0].set_title("1. Input RGB")

    artist = axes[1].imshow(masked_depth(prompt, prompt_mask), cmap=depth_cmap, vmin=0, vmax=depth_high)
    axes[1].set_title(f"2. Gaussian sparse depth prompt\ncoverage {prompt_mask.mean():.2%}")
    add_colorbar(fig, artist, axes[1], "Camera-Z depth (m)")

    artist = axes[2].imshow(old_depth, cmap=depth_cmap, vmin=0, vmax=depth_high)
    axes[2].set_title("3. Old InfiniDepth + LiDAR")
    add_colorbar(fig, artist, axes[2], "Camera-Z depth (m)")

    artist = axes[3].imshow(
        masked_depth(old_prompt_error, prompt_mask), cmap=error_cmap, vmin=0, vmax=old_prompt_high
    )
    axes[3].set_title(
        "4. |Old prediction - Gaussian prompt|\n"
        f"MAE {old_prompt_stats['mae_m']:.3f} m; median {old_prompt_stats['median_absolute_m']:.3f} m"
    )
    add_colorbar(fig, artist, axes[3], "Absolute difference (m), clipped at P98")

    artist = axes[4].imshow(new_depth, cmap=depth_cmap, vmin=0, vmax=depth_high)
    axes[4].set_title("5. New InfiniDepth + Gaussian sparse prompt")
    add_colorbar(fig, artist, axes[4], "Camera-Z depth (m)")

    new_old_mask = np.isfinite(new_depth) & np.isfinite(old_depth) & (new_depth > 0) & (old_depth > 0)
    artist = axes[5].imshow(
        masked_depth(new_old_error, new_old_mask), cmap=error_cmap, vmin=0, vmax=new_old_high
    )
    axes[5].set_title(
        "6. |New prediction - Old prediction|\n"
        f"MAE {new_old_stats['mae_m']:.3f} m; median {new_old_stats['median_absolute_m']:.3f} m"
    )
    add_colorbar(fig, artist, axes[5], "Absolute difference (m), clipped at P98")

    artist = axes[6].imshow(
        masked_depth(new_prompt_error, prompt_mask), cmap=error_cmap, vmin=0, vmax=new_prompt_high
    )
    axes[6].set_title(
        "7. |New prediction - Gaussian prompt|\n"
        f"MAE {new_prompt_stats['mae_m']:.3f} m; median {new_prompt_stats['median_absolute_m']:.3f} m"
    )
    add_colorbar(fig, artist, axes[6], "Absolute difference (m), clipped at P98")

    for axis in axes:
        axis.set_axis_off()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return {
        "depth_range_m": [0.0, depth_high],
        "old_prompt_absolute_difference_range_m": [0.0, old_prompt_high],
        "new_old_absolute_difference_range_m": [0.0, new_old_high],
        "new_prompt_absolute_difference_range_m": [0.0, new_prompt_high],
        "clipping": "display only; statistics use unclipped values",
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-depths", required=True)
    parser.add_argument("--source-metrics", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--alpha-threshold", type=float, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InfiniDepth inference")
    config = yaml.safe_load(Path(args.config).read_text())
    source_metrics = json.loads(Path(args.source_metrics).read_text())
    source_snapshot = source_metrics["source_snapshot"]
    arrays = np.load(args.input_depths)
    prompt = arrays["gaussian_depth_m"].astype(np.float32)
    old_depth = arrays["infinidepth_m"].astype(np.float32)
    alpha = arrays["alpha"].astype(np.float32)
    alpha_threshold = (
        float(source_metrics["alpha_threshold"])
        if args.alpha_threshold is None
        else args.alpha_threshold
    )
    if not 0 < alpha_threshold < 1:
        raise ValueError("alpha threshold must be in (0, 1)")
    prompt_mask = np.isfinite(prompt) & (prompt > 0) & np.isfinite(alpha) & (alpha >= alpha_threshold)

    dataset = build_dataset(config, "val")
    scene_index, start = dataset.index[source_snapshot["clip_index"]]
    metadata = dataset.scenes[scene_index]
    window = metadata["frames"][start:start + dataset.clip_length]
    frame = source_snapshot["snapshot_frame"]
    if metadata["name"] != source_metrics["scene"] or list(window) != source_snapshot["window_frames"]:
        raise ValueError("Dataset identity does not match the source diagnostic")
    batch = dataset._collect(
        metadata,
        [frame],
        {value: index for index, value in enumerate(window)},
        dataset._normalised_times(window),
        dataset.resolution,
        np.linalg.inv(dataset._ego_pose(metadata, window[0])),
        lidar=False,
    )
    view = dataset.cameras.index(source_metrics["camera"])
    device = torch.device("cuda")
    image = batch["images"][view:view + 1].to(device)
    sparse_depth = torch.from_numpy(np.where(prompt_mask, prompt, 0.0))[None, None].to(device)
    sparse_mask = torch.from_numpy(prompt_mask.astype(np.float32))[None, None].to(device)
    torch.manual_seed(config.get("seed", 0))
    depth_model = build_depth(config).to(device).eval()
    new_depth = depth_model(
        image,
        sparse_depth=sparse_depth,
        sparse_mask=sparse_mask,
    )[0].float().cpu().numpy()

    old_valid = np.isfinite(old_depth) & (old_depth > 0)
    new_valid = np.isfinite(new_depth) & (new_depth > 0)
    new_old_mask = new_valid & old_valid
    old_prompt_stats = error_stats(old_depth - prompt, prompt_mask & old_valid)
    new_old_stats = error_stats(new_depth - old_depth, new_old_mask)
    new_prompt_stats = error_stats(new_depth - prompt, prompt_mask & new_valid)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_image = Path(str(output_prefix) + "_7panel.png")
    rgb = image[0].cpu().numpy().transpose(1, 2, 0)
    display = draw_seven_panel(
        rgb,
        prompt,
        prompt_mask,
        old_depth,
        new_depth,
        old_prompt_stats,
        new_old_stats,
        new_prompt_stats,
        (
            f"Scene {metadata['name']} | frame {frame} | camera {source_metrics['camera']} | "
            f"{prompt.shape[1]}x{prompt.shape[0]} | Gaussian prompt alpha >= {alpha_threshold}"
        ),
        output_image,
    )

    output_arrays = Path(str(output_prefix) + "_depths.npz")
    np.savez_compressed(
        output_arrays,
        rgb=rgb,
        gaussian_sparse_depth_m=np.where(prompt_mask, prompt, np.nan),
        gaussian_sparse_mask=prompt_mask,
        old_infinidepth_lidar_m=old_depth,
        new_infinidepth_gaussian_prompt_m=new_depth,
        old_minus_prompt_m=np.where(prompt_mask & old_valid, old_depth - prompt, np.nan),
        new_minus_old_m=np.where(new_old_mask, new_depth - old_depth, np.nan),
        new_minus_prompt_m=np.where(prompt_mask & new_valid, new_depth - prompt, np.nan),
        intrinsics=arrays["intrinsics"],
        cam_to_world=arrays["cam_to_world"],
    )
    metrics = {
        "scene": metadata["name"],
        "frame": frame,
        "camera": source_metrics["camera"],
        "resolution_hw": list(prompt.shape),
        "experiment": "InfiniDepth reconditioned on Gaussian rendered sparse depth",
        "source_depths": str(Path(args.input_depths).resolve()),
        "source_depths_sha256": hashlib.sha256(Path(args.input_depths).read_bytes()).hexdigest(),
        "prompt": {
            "source": "gaussian_depth_m",
            "mask": f"finite positive depth and alpha >= {alpha_threshold}",
            "pixels": int(prompt_mask.sum()),
            "coverage": float(prompt_mask.mean()),
            "depth_range_m": [float(prompt[prompt_mask].min()), float(prompt[prompt_mask].max())],
        },
        "old_prediction": "InfiniDepth_DepthSensor conditioned on same-frame LiDAR",
        "new_prediction": "InfiniDepth_DepthSensor conditioned on Gaussian rendered depth prompt",
        "old_prediction_vs_prompt": old_prompt_stats,
        "new_prediction_vs_old_prediction": new_old_stats,
        "new_prediction_vs_prompt": new_prompt_stats,
        "display": display,
        "outputs": {
            "seven_panel": str(output_image.resolve()),
            "arrays": str(output_arrays.resolve()),
        },
        "interpretation": (
            "This measures how replacing the LiDAR prompt with rendered Gaussian depth changes InfiniDepth; "
            "the Gaussian prompt is not independent ground truth."
        ),
    }
    output_metrics = Path(str(output_prefix) + "_metrics.json")
    output_metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
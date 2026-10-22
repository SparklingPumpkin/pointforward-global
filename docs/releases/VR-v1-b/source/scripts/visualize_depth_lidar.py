"""Export a 2x2 dense-depth and LiDAR diagnostic for one PointForward clip.

The sparse LiDAR depth is projected by the same DriveStudio loader that feeds
the frozen InfiniDepth depth-sensor variant. The error panel only colours pixels
with an actual LiDAR return; black pixels are unobserved, not zero-depth ground
truth.

Usage:
    python -m scripts.visualize_depth_lidar \
        --config runs/waymo_ds_paper/config.yaml \
        --out visualization
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from pointforward.data import to_scene_input
from scripts.train import build_dataset, build_depth


def image_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a CHW RGB tensor in [0, 1] to an HWC uint8 image."""
    array = image.detach().cpu().clamp(0, 1).numpy()
    return (array.transpose(1, 2, 0) * 255).astype(np.uint8)


def depth_colormap(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map depth from red (near) to blue (far) without matplotlib."""
    normalized = np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)
    hue6 = normalized * 0.66 * 6.0
    x = 1.0 - np.abs(np.mod(hue6, 2.0) - 1.0)
    zeros = np.zeros_like(hue6)
    red = np.select([hue6 < 1, hue6 < 2, hue6 < 3, hue6 < 4, hue6 < 5], [1.0, x, zeros, zeros, x], default=1.0)
    green = np.select([hue6 < 1, hue6 < 2, hue6 < 3, hue6 < 4, hue6 < 5], [x, 1.0, 1.0, x, zeros], default=zeros)
    blue = np.select([hue6 < 1, hue6 < 2, hue6 < 3, hue6 < 4, hue6 < 5], [zeros, zeros, x, 1.0, 1.0], default=x)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def error_colormap(values: np.ndarray, high: float) -> np.ndarray:
    """Map absolute error from blue (low) through yellow to red (high)."""
    normalized = np.clip(values / max(high, 1e-6), 0.0, 1.0)
    red = np.minimum(1.0, normalized * 2.0)
    green = np.minimum(1.0, 2.0 - normalized * 2.0)
    blue = np.maximum(0.0, 1.0 - normalized * 2.0)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def annotate(panel: np.ndarray, title: str, detail: str) -> Image.Image:
    """Add a compact ASCII label band; image content remains at native scale."""
    output = Image.fromarray(panel)
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 34), fill=(0, 0, 0))
    draw.text((8, 4), title, fill=(255, 255, 255))
    draw.text((8, 19), detail, fill=(210, 210, 210))
    return output


def make_grid(panels: list[Image.Image]) -> Image.Image:
    """Compose equally sized panels into a 2x2 image."""
    width, height = panels[0].size
    grid = Image.new("RGB", (width * 2, height * 2), (0, 0, 0))
    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        grid.paste(panel, (column * width, row * height))
    return grid


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="visualization")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--view-index", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    dataset = build_dataset(config, args.split)
    if not 0 <= args.clip_index < len(dataset):
        raise ValueError(f"clip index {args.clip_index} is outside [0, {len(dataset) - 1}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = dataset[args.clip_index]
    scene, _, _ = to_scene_input(clip, device)
    if scene.sparse_depth is None or scene.sparse_mask is None:
        raise RuntimeError("selected clip has no projected LiDAR depth")
    if not 0 <= args.view_index < scene.num_views:
        raise ValueError(f"view index {args.view_index} is outside [0, {scene.num_views - 1}]")

    depth_model = build_depth(config).to(device)
    depth_model.eval()
    predicted = depth_model(
        scene.images,
        sparse_depth=scene.sparse_depth,
        sparse_mask=scene.sparse_mask,
    )

    view = args.view_index
    rgb = image_uint8(scene.images[view])
    dense_depth = predicted[view].detach().cpu().numpy().astype(np.float32)
    lidar_depth = scene.sparse_depth[view, 0].detach().cpu().numpy().astype(np.float32)
    lidar_mask = scene.sparse_mask[view, 0].detach().cpu().numpy() > 0.5
    if not lidar_mask.any():
        raise RuntimeError("selected view has no LiDAR pixels after projection")

    valid_depth = np.concatenate((dense_depth.reshape(-1), lidar_depth[lidar_mask]))
    depth_low, depth_high = np.percentile(valid_depth, (2, 98))
    depth_panel = depth_colormap(dense_depth, float(depth_low), float(depth_high))

    absolute_error = np.abs(dense_depth - lidar_depth)
    error_values = absolute_error[lidar_mask]
    error_high = max(float(np.percentile(error_values, 98)), 0.1)
    error_panel = np.zeros_like(rgb)
    error_panel[lidar_mask] = error_colormap(error_values, error_high)

    coverage = float(lidar_mask.mean())
    metrics = {
        "scene": clip["scene"],
        "clip_index": args.clip_index,
        "view_index": view,
        "frame_index": int(scene.frame_index[view].item()),
        "image_height": int(rgb.shape[0]),
        "image_width": int(rgb.shape[1]),
        "lidar_pixels": int(lidar_mask.sum()),
        "lidar_coverage": coverage,
        "dense_depth_range_m": [float(dense_depth.min()), float(dense_depth.max())],
        "lidar_depth_range_m": [float(lidar_depth[lidar_mask].min()), float(lidar_depth[lidar_mask].max())],
        "absolute_error_mean_m": float(error_values.mean()),
        "absolute_error_median_m": float(np.median(error_values)),
        "absolute_error_rmse_m": float(np.sqrt(np.mean(error_values**2))),
        "error_colour_max_m": error_high,
    }

    panels = [
        annotate(rgb, "1 Input RGB", f"scene {clip['scene']}; context view {view}"),
        annotate(depth_panel, "2 Dense depth prediction", f"colour range {depth_low:.1f}-{depth_high:.1f} m"),
        annotate(rgb.copy(), "3 GT RGB aligned with LiDAR", "same camera frame as depth comparison"),
        annotate(error_panel, "4 |dense depth - LiDAR|", f"black=unobserved; 98pct={error_high:.2f} m"),
    ]

    os.makedirs(args.out, exist_ok=True)
    prefix = f"{clip['scene']}_clip{args.clip_index:04d}_view{view}"
    grid_path = os.path.join(args.out, f"{prefix}_depth_lidar_2x2.png")
    metrics_path = os.path.join(args.out, f"{prefix}_depth_lidar_metrics.json")
    make_grid(panels).save(grid_path)
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"saved {grid_path}")
    print(f"saved {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
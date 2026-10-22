"""Create 12-view contact sheets for track-once dynamic reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from scripts.train import build_dataset


SIGNS = np.asarray([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
], dtype=np.float64)


def panel_for_view(dataset, metadata, track_id, frame, camera, world_from):
    tensor, intrinsic = dataset._load_calibrated_image(
        metadata,
        f"images/{frame:03d}_{camera}.jpg",
        dataset.resolution,
        camera,
    )
    array = (tensor.permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    raw_pose, size = metadata["instances"][track_id][frame]
    object_to_world = world_from @ raw_pose
    corners = SIGNS * (size * 0.5)
    corners = corners @ object_to_world[:3, :3].T + object_to_world[:3, 3]
    world_to_camera = np.linalg.inv(
        dataset._cam_to_world(metadata, frame, camera, world_from)
    )
    camera_points = corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    front = camera_points[:, 2] > 1e-3
    visible = False
    if front.any():
        points = camera_points[front]
        uv = (points / points[:, 2:3]) @ intrinsic.T
        x0, y0 = uv[:, :2].min(axis=0)
        x1, y1 = uv[:, :2].max(axis=0)
        width, height = image.size
        visible = x1 >= 0 and x0 < width and y1 >= 0 and y0 < height
        if visible:
            x0, x1 = max(0, x0), min(width - 1, x1)
            y0, y1 = max(0, y0), min(height - 1, y1)
            draw.rectangle((x0, y0, x1, y1), outline=(255, 40, 40), width=5)
    draw.rectangle((0, 0, image.width, 26), fill=(0, 0, 0))
    label = f"track {track_id} | frame {frame} | cam {camera}"
    if not visible:
        label += " | outside this camera"
    draw.text((7, 7), label, fill=(255, 255, 255))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--tracks", nargs="+", type=int, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    # Contact sheets need only RGB/calibration/boxes; avoid rejecting a scene
    # when the optional LiDAR probe is unavailable in the visualization env.
    config["data"]["load_lidar"] = False
    config["data"]["require_lidar"] = False
    config["data"]["require_instances"] = False
    config["data"]["rigid_classes"] = [
        "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
        "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
        "MOTORCYCLIST",
    ]
    dataset = build_dataset(config, "val")
    metadata = next(meta for meta in dataset.scenes if meta["name"] == args.scene)
    manifest = json.loads(Path(args.manifest).read_text())
    records = {record["track_id"]: record for record in manifest["tracks"]}
    world_from = np.linalg.inv(dataset._ego_pose(metadata, metadata["frames"][0]))
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    for track_id in args.tracks:
        record = records[track_id]
        if "selected_views" in record:
            views = [(view["frame"], view["camera"]) for view in record["selected_views"]]
        else:
            views = [
                (frame, camera)
                for frame in record["sampled_timestamps"]
                for camera in dataset.cameras
            ]
        panels = [
            panel_for_view(dataset, metadata, track_id, frame, camera, world_from)
            for frame, camera in views
        ]
        tile_width = 480
        ratio = tile_width / panels[0].width
        tile_height = round(panels[0].height * ratio)
        panels = [panel.resize((tile_width, tile_height), Image.Resampling.LANCZOS) for panel in panels]
        sheet = Image.new("RGB", (tile_width * 3, tile_height * 4), (0, 0, 0))
        for index, panel in enumerate(panels):
            sheet.paste(panel, ((index % 3) * tile_width, (index // 3) * tile_height))
        sheet.save(output_dir / f"track_{track_id:03d}_12views.jpg", quality=94)


if __name__ == "__main__":
    main()

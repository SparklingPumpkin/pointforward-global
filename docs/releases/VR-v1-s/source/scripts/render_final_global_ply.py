"""Render every frame from one fixed preview-color Gaussian PLY.

This is a diagnostic renderer for the final global-map snapshot.  It bypasses
PointForward's 128-D feature refiner because a standard preview PLY stores only
degree-0 SH/RGB values.  The same Gaussian set is passed to gsplat for every
frame; only the calibrated camera pose changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from plyfile import PlyData

from pointforward.models.renderer import render_features
from scripts.train import build_dataset


SH_C0 = 0.28209479177387814


def to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return (array * 255).round().astype(np.uint8)


def label(image: np.ndarray, text: str) -> np.ndarray:
    output = Image.fromarray(image)
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 24), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(output)


def horizontal_strip(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def vertical_strip(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=0)


def encode_video(frame_pattern: Path, output_path: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_pattern),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(output_path),
            "-loglevel", "error",
        ],
        check=True,
    )


def ply_fields(vertices, names: list[str]) -> np.ndarray:
    values = np.stack([vertices[name] for name in names], axis=-1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite PLY fields: {names}")
    return values


def load_preview_ply(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
    vertices = PlyData.read(str(path))["vertex"].data
    means = ply_fields(vertices, ["x", "y", "z"])
    quats = ply_fields(vertices, [f"rot_{i}" for i in range(4)])
    scales = np.exp(ply_fields(vertices, [f"scale_{i}" for i in range(3)]))
    opacity = 1.0 / (1.0 + np.exp(-np.clip(ply_fields(vertices, ["opacity"]), -80.0, 80.0)))
    # export_gaussians_ply.py encoded query RGB as (rgb - 0.5) / SH_C0.
    colors = ply_fields(vertices, [f"f_dc_{i}" for i in range(3)]) * SH_C0 + 0.5
    arrays = {
        "means": torch.from_numpy(means).to(device=device),
        "quats": torch.from_numpy(quats).to(device=device),
        "scales": torch.from_numpy(scales).to(device=device),
        "opacities": torch.from_numpy(opacity[:, 0]).to(device=device),
        "colors": torch.from_numpy(colors).to(device=device),
    }
    return arrays, len(vertices)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-frames", type=int, default=None,
                        help="diagnostic limit; omit to render all 156 frames")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for rendering this Gaussian set")
    device = torch.device(args.device)
    config_path = Path(args.config).resolve()
    ply_path = Path(args.ply).resolve()
    output_dir = Path(args.out).resolve()
    predicted_dir = output_dir / "frames" / "predicted"
    comparison_dir = output_dir / "frames" / "comparison"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text())
    dataset = build_dataset(config, args.split)
    matches = [metadata for metadata in dataset.scenes if metadata["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    frames = list(metadata["frames"])
    if len(frames) != 156:
        raise ValueError(f"Expected 156 scene frames, found {len(frames)}")
    if args.limit_frames is not None:
        if not 1 <= args.limit_frames <= len(frames):
            raise ValueError("limit-frames must be in [1, 156]")
        frames = frames[:args.limit_frames]
    if list(dataset.cameras) != [0, 1, 2]:
        raise ValueError(f"Expected cameras [0, 1, 2], found {dataset.cameras}")

    gaussians, gaussian_count = load_preview_ply(ply_path, device)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    normalized_times = dataset._normalised_times(frames)
    slot_of = {frame: slot for slot, frame in enumerate(frames)}
    height, width = dataset.resolution
    source_hash = hashlib.sha256(ply_path.read_bytes()).hexdigest()
    started = time.perf_counter()
    frame_records = []

    print(f"loaded {gaussian_count:,} Gaussians from {ply_path}", flush=True)
    for output_index, frame in enumerate(frames):
        target = dataset._collect(
            metadata, [frame], slot_of, normalized_times,
            dataset.resolution, world_from, lidar=False,
        )
        intrinsics = target["intrinsics"].to(device)
        world_to_cam = torch.linalg.inv(target["cam_to_world"].to(device))
        rendered, alpha, _ = render_features(
            means=gaussians["means"],
            quats=gaussians["quats"],
            scales=gaussians["scales"],
            opacities=gaussians["opacities"],
            colors=gaussians["colors"],
            intrinsics=intrinsics,
            world_to_cam=world_to_cam,
            height=height,
            width=width,
        )
        # render_features returns three RGB views in dataset.cameras order.
        prediction_panels = [
            label(to_uint8(rendered[index]), f"Fixed final Gaussian | frame {frame:03d} | camera {camera}")
            for index, camera in enumerate(dataset.cameras)
        ]
        ground_truth_panels = [
            label((target["images"][index].clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8),
                  f"GT RGB | frame {frame:03d} | camera {camera}")
            for index, camera in enumerate(dataset.cameras)
        ]
        prediction_strip = horizontal_strip(prediction_panels)
        comparison = vertical_strip((horizontal_strip(ground_truth_panels), prediction_strip))
        Image.fromarray(prediction_strip).save(predicted_dir / f"frame_{output_index:04d}.jpg", quality=92)
        Image.fromarray(comparison).save(comparison_dir / f"frame_{output_index:04d}.jpg", quality=92)
        frame_records.append({
            "output_index": output_index,
            "frame": int(frame),
            "alpha_mean": float(alpha.mean().cpu()),
            "alpha_covered_fraction": float((alpha > 0.01).float().mean().cpu()),
        })
        if output_index % 10 == 0 or output_index == len(frames) - 1:
            print(json.dumps({"frame": int(frame), "index": output_index, "gaussian_count": gaussian_count}), flush=True)
        del target, intrinsics, world_to_cam, rendered, alpha

    predicted_video = output_dir / "final_global_ply_predicted_front3.mp4"
    comparison_video = output_dir / "final_global_ply_gt_vs_predicted_front3.mp4"
    encode_video(predicted_dir / "frame_%04d.jpg", predicted_video, args.fps)
    encode_video(comparison_dir / "frame_%04d.jpg", comparison_video, args.fps)
    manifest = {
        "status": "completed",
        "scene": args.scene,
        "frame_count": len(frames),
        "frame_range": [int(frames[0]), int(frames[-1])],
        "fps": args.fps,
        "cameras": list(dataset.cameras),
        "camera_semantics": {"0": "front", "1": "front_left", "2": "front_right"},
        "resolution_hw_per_camera": [height, width],
        "config": str(config_path),
        "source_ply": str(ply_path),
        "source_ply_sha256": source_hash,
        "gaussian_count": gaussian_count,
        "rendering": "fixed final PLY, direct degree-0 SH/RGB gsplat rendering; no PointForward 128-D feature UNet",
        "coordinate_frame": "shared world recentered at the first ego pose of scene017",
        "dynamic_semantics": "dynamic Gaussians are frozen at the pose encoded in the final PLY",
        "predicted_video": str(predicted_video),
        "comparison_video": str(comparison_video),
        "frame_records": frame_records,
        "wall_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

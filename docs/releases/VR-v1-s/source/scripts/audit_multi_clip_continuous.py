"""Measure per-frame quality across real non-overlapping clip boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.losses import psnr, ssim
from pointforward.models import InstanceBoxes, SceneInput, TargetViews
from scripts.eval import build_model
from scripts.train import build_dataset


def boxes_to(boxes: InstanceBoxes | None, device: torch.device) -> InstanceBoxes | None:
    if boxes is None:
        return None
    return InstanceBoxes(
        centers=boxes.centers.to(device), dims=boxes.dims.to(device),
        rots=boxes.rots.to(device), times=boxes.times.to(device), valid=boxes.valid.to(device),
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = yaml.safe_load(Path(args.config).read_text())
    dataset = build_dataset(config, "val")
    metadata = next((item for item in dataset.scenes if item["name"] == args.scene), None)
    if metadata is None:
        raise ValueError(f"scene {args.scene} is not in validation split")
    frames = metadata["frames"][args.start:args.start + args.num_frames]
    if len(frames) != args.num_frames or args.num_frames % dataset.clip_length:
        raise ValueError("requested range must contain a whole number of full clips")
    device = torch.device("cuda")
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    model = build_model(config, device, args.ckpt)
    camera_offset = dataset.cameras.index(args.camera)
    rows = []

    for window_index, start in enumerate(range(0, len(frames), dataset.clip_length)):
        window = frames[start:start + dataset.clip_length]
        slot_of = {frame: index for index, frame in enumerate(window)}
        times = dataset._normalised_times(window)
        context_frames = sorted({
            window[min(index * dataset.frame_stride, len(window) - 1)]
            for index in range(dataset.num_context)
        })
        context = dataset._collect(
            metadata, context_frames, slot_of, times, dataset.resolution,
            world_from, lidar=dataset.load_lidar,
        )
        target = dataset._collect(
            metadata, window, slot_of, times, dataset.resolution,
            world_from, lidar=True,
        )
        selected = torch.arange(
            camera_offset, len(window) * len(dataset.cameras), len(dataset.cameras)
        )
        boxes = boxes_to(dataset._build_boxes(metadata, window, world_from), device)
        scene = SceneInput(
            images=context["images"].to(device), intrinsics=context["intrinsics"].to(device),
            cam_to_world=context["cam_to_world"].to(device), times=context["times"].to(device),
            frame_index=context["frame_index"].to(device), boxes=boxes,
            sparse_depth=context["sparse_depth"].to(device),
            sparse_mask=context["sparse_mask"].to(device),
        )
        targets = TargetViews(
            intrinsics=target["intrinsics"][selected].to(device),
            cam_to_world=target["cam_to_world"][selected].to(device),
            times=target["times"][selected].to(device),
            frame_index=target["frame_index"][selected].to(device),
        )
        seed_all(args.seed + window_index)
        depth = model.predict_depth(scene)
        generator = torch.Generator(device=device).manual_seed(args.seed + window_index)
        sampled = model.build_queries(scene, depth, generator=generator)
        gaussians = model.decode(scene, depth, sampled)
        output = model.render(
            gaussians, boxes, targets,
            height=dataset.resolution[0], width=dataset.resolution[1], render_depth=True,
        )
        gt = target["images"][selected].to(device)
        dynamic = target["dynamic_mask"][selected].to(device) > 0.5
        lidar_depth = target["sparse_depth"][selected].to(device)
        lidar_mask = target["sparse_mask"][selected].to(device) > 0.5
        for local_index, frame in enumerate(window):
            pred = output["rgb"][local_index:local_index + 1]
            truth = gt[local_index:local_index + 1]
            dyn = dynamic[local_index:local_index + 1]
            sta = ~dyn
            depth_valid = lidar_mask[local_index] & (output["alpha"][local_index] >= 0.01)
            depth_error = output["render_depth"][local_index][depth_valid] - lidar_depth[local_index][depth_valid]
            rows.append({
                "scene": args.scene,
                "camera": args.camera,
                "frame": int(frame),
                "window_index": window_index,
                "local_index": local_index,
                "local_time": float(times[local_index]),
                "is_context": int(frame in context_frames),
                "distance_to_nearest_context": min(abs(frame - value) for value in context_frames),
                "is_window_start": int(local_index == 0),
                "is_window_end": int(local_index == len(window) - 1),
                "psnr": float(psnr(pred, truth)),
                "ssim": float(ssim(pred, truth)),
                "dynamic_psnr": float(psnr(pred, truth, dyn)),
                "static_psnr": float(psnr(pred, truth, sta)),
                "l1": float((pred - truth).abs().mean()),
                "alpha_mean": float(output["alpha"][local_index].mean()),
                "lidar_pixels": int(depth_valid.sum()),
                "lidar_depth_rmse_m": None if not depth_error.numel() else float(torch.sqrt((depth_error ** 2).mean())),
            })
        del scene, targets, boxes, depth, sampled, gaussians, output
        torch.cuda.empty_cache()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"scene{args.scene}_continuous_per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    boundary_pairs = []
    for index in range(1, len(rows)):
        if rows[index]["window_index"] != rows[index - 1]["window_index"]:
            boundary_pairs.append({
                "from_frame": rows[index - 1]["frame"],
                "to_frame": rows[index]["frame"],
                "psnr_change_db": rows[index]["psnr"] - rows[index - 1]["psnr"],
                "ssim_change": rows[index]["ssim"] - rows[index - 1]["ssim"],
                "l1_change": rows[index]["l1"] - rows[index - 1]["l1"],
                "depth_rmse_change_m": (
                    None if rows[index]["lidar_depth_rmse_m"] is None or rows[index - 1]["lidar_depth_rmse_m"] is None
                    else rows[index]["lidar_depth_rmse_m"] - rows[index - 1]["lidar_depth_rmse_m"]
                ),
            })
    summary = {
        "interpretation": "per-frame reconstruction quality; adjacent raw-frame differences are not temporal consistency metrics",
        "scene": args.scene,
        "camera": args.camera,
        "frames": list(map(int, frames)),
        "context_offsets": [0, 5, 10, 15],
        "boundary_pairs": boundary_pairs,
        "mean_psnr_context": float(np.mean([row["psnr"] for row in rows if row["is_context"]])),
        "mean_psnr_noncontext": float(np.mean([row["psnr"] for row in rows if not row["is_context"]])),
        "mean_psnr_window_start": float(np.mean([row["psnr"] for row in rows if row["is_window_start"]])),
        "mean_psnr_window_end": float(np.mean([row["psnr"] for row in rows if row["is_window_end"]])),
        "mean_psnr_by_context_distance": {
            str(distance): float(np.mean([row["psnr"] for row in rows if row["distance_to_nearest_context"] == distance]))
            for distance in sorted({row["distance_to_nearest_context"] for row in rows})
        },
    }
    json_path = output_dir / f"scene{args.scene}_continuous_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
"""Jointly refine dynamic Gaussian canonical geometry and grouped features.

Static Gaussians, dynamic boxes, opacity/scale/rotation/time and the network
remain frozen. The dynamic bbox RGB loss updates a bounded canonical XYZ delta
and a compact grouped 128-D feature delta in one pass over the source frames.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.models import TargetViews, box_dynamic_masks
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--refiner-state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--max-xyz-delta", type=float, default=0.15)
    parser.add_argument("--max-feature-delta", type=float, default=0.25)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if 128 % args.groups:
        parser.error("--groups must divide 128")
    if args.frame_stride < 1 or args.camera_batch_size < 1:
        parser.error("frame-stride and camera-batch-size must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source_dir = Path(args.payload_dir).resolve()
    output_dir = Path(args.out).resolve()
    refined_dir = output_dir / "runtime-payload-refined"
    refined_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, "val")
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    all_frames = list(metadata["frames"])
    frames = all_frames[::args.frame_stride]
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    if not frames:
        parser.error("no training frames")
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    normalized_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    base_means = dynamic.means.detach()
    base_colors = dynamic.colors.detach()
    raw_xyz = torch.nn.Parameter(torch.zeros_like(base_means))
    raw_feature = torch.nn.Parameter(torch.zeros((dynamic.count, args.groups), device=device))
    group_index = torch.arange(128, device=device) // (128 // args.groups)
    optimizer = torch.optim.Adam((raw_xyz, raw_feature), lr=args.learning_rate)

    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    overlay = torch.load(Path(args.refiner_state).resolve(), map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(overlay.get("refiner", overlay), strict=True)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()

    for epoch in range(args.epochs):
        order = torch.randperm(len(frames), generator=generator).tolist()
        for order_index in order:
            frame = int(frames[order_index])
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            optimizer.zero_grad(set_to_none=True)
            xyz_delta = torch.tanh(raw_xyz) * args.max_xyz_delta
            feature_delta = torch.tanh(raw_feature)[:, group_index] * args.max_feature_delta
            current_dynamic = replace(
                dynamic, means=base_means + xyz_delta,
                colors=base_colors + feature_delta,
            )
            gaussians = combine(static, current_dynamic)
            camera_count = len(dataset.cameras)
            losses: list[float] = []
            valid_views = 0
            for camera_start in range(0, camera_count, args.camera_batch_size):
                camera_end = min(camera_start + args.camera_batch_size, camera_count)
                intrinsics = batch["intrinsics"][camera_start:camera_end].to(device)
                cam_to_world = batch["cam_to_world"][camera_start:camera_end].to(device)
                view_count = camera_end - camera_start
                targets = TargetViews(
                    intrinsics=intrinsics,
                    cam_to_world=cam_to_world,
                    times=normalized_times[slot:slot + 1].to(device).expand(view_count),
                    frame_index=torch.full((view_count,), slot, dtype=torch.long, device=device),
                )
                rendered = model.render(
                    gaussians, boxes, targets, height=height, width=width,
                    render_depth=False,
                )["rgb"]
                mask = box_dynamic_masks(
                    evaluation_boxes,
                    torch.full((view_count,), slot, dtype=torch.long, device=device),
                    intrinsics, torch.linalg.inv(cam_to_world), height, width,
                ).unsqueeze(1)
                if bool(mask.any()):
                    truth = batch["images"][camera_start:camera_end].to(device)
                    squared = (rendered - truth).square()
                    mse = (squared * mask).sum() / (mask.sum() * 3.0).clamp_min(1.0)
                    (mse / camera_count).backward()
                    losses.append(float(mse.detach()))
                    valid_views += view_count
                del targets, rendered, mask
            xyz_delta = torch.tanh(raw_xyz) * args.max_xyz_delta
            feature_delta = torch.tanh(raw_feature)[:, group_index] * args.max_feature_delta
            anchor = (
                (xyz_delta / args.max_xyz_delta).square().mean()
                + (feature_delta / args.max_feature_delta).square().mean()
            )
            (args.anchor_weight * anchor).backward()
            for parameter in (raw_xyz, raw_feature):
                torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw_xyz, raw_feature), 1.0)
            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(raw_xyz, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nan_to_num_(raw_feature, nan=0.0, posinf=0.0, neginf=0.0)
            step = epoch * len(frames) + order_index + 1
            if step % 10 == 0 or step == 1:
                with torch.no_grad():
                    xyz_norm = xyz_delta.norm(dim=-1)
                    feature_norm = feature_delta.norm(dim=-1)
                    records.append({
                        "step": step, "epoch": epoch + 1, "frame": frame,
                        "valid_views": valid_views,
                        "dynamic_bbox_rgb_mse": float(np.mean(losses)) if losses else None,
                        "anchor": float(anchor.detach()), "grad_norm": float(grad_norm),
                        "xyz_rms_m": float(xyz_delta.square().mean().sqrt()),
                        "xyz_p95_m": float(torch.quantile(xyz_norm, 0.95)),
                        "feature_rms": float(feature_delta.square().mean().sqrt()),
                        "feature_p95": float(torch.quantile(feature_norm, 0.95)),
                        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    })
                    print(json.dumps(records[-1]), flush=True)
            del batch, current_dynamic, gaussians, xyz_delta, feature_delta

    with torch.no_grad():
        xyz_delta = torch.tanh(raw_xyz) * args.max_xyz_delta
        feature_delta = torch.tanh(raw_feature)[:, group_index] * args.max_feature_delta
        refined_dynamic = replace(
            dynamic, means=base_means + xyz_delta,
            colors=base_colors + feature_delta,
        )
    static_target = refined_dir / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source_dir / "final_static_128d.pt").resolve(), static_target)
    save_gaussian_payload(refined_dir / "final_dynamic_128d.pt", refined_dynamic)
    meta_target = refined_dir / "global_dynamic_meta.pt"
    if not meta_target.exists():
        os.symlink((source_dir / "global_dynamic_meta.pt").resolve(), meta_target)
    torch.save({"dynamic_xyz_delta_m": xyz_delta.cpu(), "dynamic_feature_delta": feature_delta.cpu(), "groups": args.groups}, output_dir / "dynamic_joint_delta.pt")
    xyz_norm = xyz_delta.norm(dim=-1)
    feature_norm = feature_delta.norm(dim=-1)
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": str(Path(args.refiner_state).resolve()),
        "optimized": "dynamic canonical XYZ plus grouped 128-D feature deltas",
        "groups": args.groups,
        "frozen": "static Gaussian map, dynamic boxes, opacity/scale/rotation/time, cameras, network and UNet",
        "supervision": "dynamic evaluation bbox RGB MSE over all source frames and cameras",
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "max_xyz_delta_m": args.max_xyz_delta, "max_feature_delta": args.max_feature_delta,
        "anchor_weight": args.anchor_weight, "training_frame_stride": args.frame_stride,
        "training_frame_count": len(frames), "training_frames": [int(frame) for frame in frames],
        "camera_batch_size": args.camera_batch_size,
        "xyz_rms_m": float(xyz_delta.square().mean().sqrt()),
        "xyz_p95_m": float(torch.quantile(xyz_norm, 0.95)),
        "feature_rms": float(feature_delta.square().mean().sqrt()),
        "feature_p95": float(torch.quantile(feature_norm, 0.95)),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started, "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

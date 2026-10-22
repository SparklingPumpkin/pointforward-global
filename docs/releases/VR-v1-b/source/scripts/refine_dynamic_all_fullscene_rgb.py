"""Jointly refine all bounded dynamic Gaussian attributes and per-frame poses.

The static map, camera calibration, network, and RGB refiner stay frozen.  A
single pass over all source frames updates dynamic canonical XYZ, full 128-D
features, scale, opacity, orientation, and non-canonical box translation/yaw.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pointforward.models import InstanceBoxes, TargetViews, box_dynamic_masks
from pointforward.models.rotation import quaternion_multiply
from scripts.deliver_continuous_scene import boxes_payload, save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def rotation_z_batch(angle: torch.Tensor) -> torch.Tensor:
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((
        c, -s, zero,
        s, c, zero,
        zero, zero, one,
    ), dim=-1).reshape(*angle.shape, 3, 3)


def rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1e-7,
        torch.sin(half) / angle.clamp_min(1e-7),
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((torch.cos(half), rotvec * scale), dim=-1)


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
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-batch-size", type=int, default=3)
    parser.add_argument("--feature-lr", type=float, default=0.01)
    parser.add_argument("--geometry-lr", type=float, default=0.003)
    parser.add_argument("--opacity-lr", type=float, default=0.01)
    parser.add_argument("--box-lr", type=float, default=0.02)
    parser.add_argument("--max-feature-delta", type=float, default=0.50)
    parser.add_argument("--max-xyz-delta", type=float, default=0.20)
    parser.add_argument("--max-log-scale-delta", type=float, default=0.70)
    parser.add_argument("--max-opacity-logit-delta", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=25.0)
    parser.add_argument("--max-box-translation-m", type=float, default=0.35)
    parser.add_argument("--max-box-yaw-deg", type=float, default=12.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--box-smoothness-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
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
    base_scales = dynamic.scales.detach()
    base_opacity_logit = torch.logit(dynamic.opacities.detach().clamp(1e-5, 1.0 - 1e-5))
    base_quats = F.normalize(dynamic.quats.detach(), dim=-1)

    raw_xyz = torch.nn.Parameter(torch.zeros_like(base_means))
    raw_feature = torch.nn.Parameter(torch.zeros_like(base_colors))
    raw_log_scale = torch.nn.Parameter(torch.zeros_like(base_scales))
    raw_opacity = torch.nn.Parameter(torch.zeros_like(dynamic.opacities))
    raw_rotation = torch.nn.Parameter(torch.zeros_like(base_means))

    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None
    raw_box_translation = torch.nn.Parameter(torch.zeros_like(boxes.centers))
    raw_box_yaw = torch.nn.Parameter(torch.zeros_like(boxes.times))
    valid_float = boxes.valid.unsqueeze(-1).float()
    movable = boxes.valid.clone()
    rows = torch.arange(boxes.num_instances, device=device)
    movable[rows, boxes.canonical_index()] = False
    movable_float = movable.unsqueeze(-1).float()
    adjacent_valid = movable[:, 1:] & movable[:, :-1]

    optimizer = torch.optim.Adam([
        {"params": [raw_feature], "lr": args.feature_lr},
        {"params": [raw_xyz, raw_log_scale, raw_rotation], "lr": args.geometry_lr},
        {"params": [raw_opacity], "lr": args.opacity_lr},
        {"params": [raw_box_translation, raw_box_yaw], "lr": args.box_lr},
    ])
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    overlay = torch.load(Path(args.refiner_state).resolve(), map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(overlay.get("refiner", overlay), strict=True)
    max_rotation = math.radians(args.max_rotation_deg)
    max_box_yaw = math.radians(args.max_box_yaw_deg)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()

    def current_state():
        xyz_delta = torch.tanh(raw_xyz) * args.max_xyz_delta
        feature_delta = torch.tanh(raw_feature) * args.max_feature_delta
        log_scale_delta = torch.tanh(raw_log_scale) * args.max_log_scale_delta
        opacity_delta = torch.tanh(raw_opacity) * args.max_opacity_logit_delta
        rotvec = torch.tanh(raw_rotation) * max_rotation
        quats = F.normalize(
            quaternion_multiply(rotvec_to_quaternion(rotvec), base_quats), dim=-1
        )
        current_dynamic = replace(
            dynamic,
            means=base_means + xyz_delta,
            colors=base_colors + feature_delta,
            scales=base_scales * torch.exp(log_scale_delta),
            opacities=torch.sigmoid(base_opacity_logit + opacity_delta),
            quats=quats,
        )
        box_translation = (
            torch.tanh(raw_box_translation) * args.max_box_translation_m * movable_float
        )
        box_yaw = torch.tanh(raw_box_yaw) * max_box_yaw * movable.float()
        current_boxes = InstanceBoxes(
            centers=boxes.centers + box_translation,
            dims=boxes.dims,
            rots=rotation_z_batch(box_yaw) @ boxes.rots,
            times=boxes.times,
            valid=boxes.valid,
        )
        return (
            current_dynamic, current_boxes, xyz_delta, feature_delta,
            log_scale_delta, opacity_delta, rotvec, box_translation, box_yaw,
        )

    for epoch in range(args.epochs):
        order = torch.randperm(len(frames), generator=generator).tolist()
        for within_epoch, order_index in enumerate(order):
            frame = int(frames[order_index])
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            optimizer.zero_grad(set_to_none=True)
            state = current_state()
            current_dynamic, current_boxes = state[:2]
            gaussians = combine(static, current_dynamic)
            camera_count = len(dataset.cameras)
            losses: list[float] = []
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
                    gaussians, current_boxes, targets,
                    height=height, width=width, render_depth=False,
                )["rgb"]
                mask = box_dynamic_masks(
                    evaluation_boxes,
                    torch.full((view_count,), slot, dtype=torch.long, device=device),
                    intrinsics, torch.linalg.inv(cam_to_world), height, width,
                ).unsqueeze(1)
                if bool(mask.any()):
                    truth = batch["images"][camera_start:camera_end].to(device)
                    mse = ((rendered - truth).square() * mask).sum() / (
                        mask.sum() * 3.0
                    ).clamp_min(1.0)
                    (mse / camera_count).backward()
                    losses.append(float(mse.detach()))
                del targets, rendered, mask

            # The RGB backward above consumes the render graph.  Rebuild the
            # lightweight parameter transforms for a separate regularisation
            # graph instead of retaining the 20M-Gaussian render graph.
            del current_dynamic, current_boxes, gaussians, state
            state = current_state()
            current_dynamic, current_boxes = state[:2]
            (
                _, _, xyz_delta, feature_delta, log_scale_delta,
                opacity_delta, rotvec, box_translation, box_yaw,
            ) = state
            anchor = (
                (xyz_delta / args.max_xyz_delta).square().mean()
                + (feature_delta / args.max_feature_delta).square().mean()
                + (log_scale_delta / args.max_log_scale_delta).square().mean()
                + (opacity_delta / args.max_opacity_logit_delta).square().mean()
                + (rotvec / max_rotation).square().mean()
                + (box_translation / args.max_box_translation_m).square().sum()
                  / valid_float.sum().clamp_min(1.0)
                + (box_yaw / max_box_yaw).square().sum()
                  / boxes.valid.float().sum().clamp_min(1.0)
            )
            box_diff = box_translation[:, 1:] - box_translation[:, :-1]
            smoothness = (
                box_diff[adjacent_valid].square().mean()
                if bool(adjacent_valid.any()) else anchor * 0.0
            )
            (
                args.anchor_weight * anchor
                + args.box_smoothness_weight * smoothness / len(frames)
            ).backward()
            params = (
                raw_xyz, raw_feature, raw_log_scale, raw_opacity,
                raw_rotation, raw_box_translation, raw_box_yaw,
            )
            for parameter in params:
                torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            with torch.no_grad():
                for parameter in params:
                    torch.nan_to_num_(parameter, nan=0.0, posinf=0.0, neginf=0.0)
            step = epoch * len(frames) + within_epoch + 1
            if step % 10 == 0 or step == 1:
                row = {
                    "step": step, "epoch": epoch + 1, "frame": frame,
                    "dynamic_bbox_rgb_mse": float(np.mean(losses)) if losses else None,
                    "anchor": float(anchor.detach()), "grad_norm": float(grad_norm),
                    "xyz_rms_m": float(xyz_delta.square().mean().sqrt()),
                    "feature_rms": float(feature_delta.square().mean().sqrt()),
                    "scale_multiplier_mean": float(torch.exp(log_scale_delta).mean()),
                    "opacity_mean": float(current_dynamic.opacities.mean()),
                    "rotation_rms_deg": float(torch.rad2deg(rotvec.square().mean().sqrt())),
                    "box_translation_mean_m": float(
                        box_translation.norm(dim=-1)[movable].mean()
                    ) if bool(movable.any()) else 0.0,
                    "box_yaw_abs_mean_deg": float(
                        torch.rad2deg(box_yaw[movable].abs().mean())
                    ) if bool(movable.any()) else 0.0,
                    "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                records.append(row)
                print(json.dumps(row), flush=True)
            del batch, current_dynamic, current_boxes, state

    with torch.no_grad():
        state = current_state()
        refined_dynamic, refined_boxes = state[:2]
        deltas = state[2:]
    static_target = refined_dir / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source_dir / "final_static_128d.pt").resolve(), static_target)
    save_gaussian_payload(refined_dir / "final_dynamic_128d.pt", refined_dynamic)
    refined_meta = dict(meta)
    refined_meta["boxes"] = boxes_payload(refined_boxes)
    refined_meta["evaluation_boxes"] = meta.get("evaluation_boxes", meta["boxes"])
    torch.save(refined_meta, refined_dir / "global_dynamic_meta.pt")
    torch.save({
        "xyz_delta_m": deltas[0].cpu(),
        "feature_delta": deltas[1].cpu(),
        "log_scale_delta": deltas[2].cpu(),
        "opacity_logit_delta": deltas[3].cpu(),
        "rotation_vector_rad": deltas[4].cpu(),
        "box_translation_world_m": deltas[5].cpu(),
        "box_yaw_rad": deltas[6].cpu(),
    }, output_dir / "dynamic_all_deltas.pt")

    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": str(Path(args.refiner_state).resolve()),
        "optimized": "dynamic XYZ/full128D feature/scale/opacity/orientation and per-frame noncanonical box translation/yaw",
        "frozen": "static Gaussian map, canonical boxes, camera calibration, network and UNet",
        "supervision": "dynamic evaluation bbox RGB MSE over all source frames and cameras",
        "epochs": args.epochs, "training_frame_stride": args.frame_stride,
        "training_frame_count": len(frames), "camera_batch_size": args.camera_batch_size,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

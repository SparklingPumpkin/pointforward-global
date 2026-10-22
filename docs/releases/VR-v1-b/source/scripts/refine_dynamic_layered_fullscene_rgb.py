"""Refine dynamic Gaussians with cached static-layer feature compositing.

For each source frame the frozen 20M-point static layer is rasterised once.
Several inexpensive dynamic-only steps then composite the posed dynamic feature
layer over that cached background before the frozen RGB refiner.  Original box
poses remain fixed, avoiding the drift found in GR-v1-u.
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

from pointforward.models import TargetViews, box_dynamic_masks
from pointforward.models.renderer import render_features
from pointforward.models.rotation import quaternion_multiply
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import load_boxes, load_payload
from scripts.train import build_dataset


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
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--feature-lr", type=float, default=0.005)
    parser.add_argument("--geometry-lr", type=float, default=0.0015)
    parser.add_argument("--opacity-lr", type=float, default=0.005)
    parser.add_argument("--max-feature-delta", type=float, default=0.50)
    parser.add_argument("--max-xyz-delta", type=float, default=0.20)
    parser.add_argument("--max-log-scale-delta", type=float, default=0.70)
    parser.add_argument("--max-opacity-logit-delta", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=25.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("mse", "log_mse"), default="log_mse")
    parser.add_argument(
        "--composite", choices=("dynamic_front", "depth_order"),
        default="depth_order",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.frame_stride < 1 or args.inner_steps < 1:
        parser.error("frame-stride and inner-steps must be positive")

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
    parameters = (raw_xyz, raw_feature, raw_log_scale, raw_opacity, raw_rotation)
    optimizer = torch.optim.Adam([
        {"params": [raw_feature], "lr": args.feature_lr},
        {"params": [raw_xyz, raw_log_scale, raw_rotation], "lr": args.geometry_lr},
        {"params": [raw_opacity], "lr": args.opacity_lr},
    ])

    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    overlay = torch.load(Path(args.refiner_state).resolve(), map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(overlay.get("refiner", overlay), strict=True)
    max_rotation = math.radians(args.max_rotation_deg)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()

    def current_dynamic_state():
        xyz_delta = torch.tanh(raw_xyz) * args.max_xyz_delta
        feature_delta = torch.tanh(raw_feature) * args.max_feature_delta
        log_scale_delta = torch.tanh(raw_log_scale) * args.max_log_scale_delta
        opacity_delta = torch.tanh(raw_opacity) * args.max_opacity_logit_delta
        rotvec = torch.tanh(raw_rotation) * max_rotation
        quats = F.normalize(
            quaternion_multiply(rotvec_to_quaternion(rotvec), base_quats), dim=-1
        )
        current = replace(
            dynamic,
            means=base_means + xyz_delta,
            colors=base_colors + feature_delta,
            scales=base_scales * torch.exp(log_scale_delta),
            opacities=torch.sigmoid(base_opacity_logit + opacity_delta),
            quats=quats,
        )
        return current, xyz_delta, feature_delta, log_scale_delta, opacity_delta, rotvec

    global_step = 0
    for epoch in range(args.epochs):
        for order_index in torch.randperm(len(frames), generator=generator).tolist():
            frame = int(frames[order_index])
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            intrinsics = batch["intrinsics"].to(device)
            cam_to_world = batch["cam_to_world"].to(device)
            camera_count = len(dataset.cameras)
            targets = TargetViews(
                intrinsics=intrinsics, cam_to_world=cam_to_world,
                times=normalized_times[slot:slot + 1].to(device).expand(camera_count),
                frame_index=torch.full((camera_count,), slot, dtype=torch.long, device=device),
            )
            mask = box_dynamic_masks(
                evaluation_boxes,
                torch.full((camera_count,), slot, dtype=torch.long, device=device),
                intrinsics, torch.linalg.inv(cam_to_world), height, width,
            ).unsqueeze(1)
            if not bool(mask.any()):
                continue
            truth = batch["images"].to(device)
            with torch.no_grad():
                posed_static = model.pose_gaussians(
                    static, None, slot, normalized_times[slot].to(device)
                )
                static_features, static_alpha, static_depth = render_features(
                    means=posed_static["means"], quats=posed_static["quats"],
                    scales=posed_static["scales"], opacities=posed_static["opacities"],
                    colors=posed_static["colors"], intrinsics=intrinsics,
                    world_to_cam=torch.linalg.inv(cam_to_world),
                    height=height, width=width, render_depth=True,
                )

            for inner_step in range(args.inner_steps):
                optimizer.zero_grad(set_to_none=True)
                state = current_dynamic_state()
                current_dynamic = state[0]
                posed_dynamic = model.pose_gaussians(
                    current_dynamic, boxes, slot, normalized_times[slot].to(device)
                )
                dynamic_features, dynamic_alpha, dynamic_depth = render_features(
                    means=posed_dynamic["means"], quats=posed_dynamic["quats"],
                    scales=posed_dynamic["scales"], opacities=posed_dynamic["opacities"],
                    colors=posed_dynamic["colors"], intrinsics=intrinsics,
                    world_to_cam=torch.linalg.inv(cam_to_world),
                    height=height, width=width, render_depth=True,
                )
                if args.composite == "depth_order":
                    assert static_depth is not None and dynamic_depth is not None
                    dynamic_valid = dynamic_alpha > 1e-4
                    static_valid = static_alpha > 1e-4
                    dynamic_front = dynamic_valid & (
                        ~static_valid | (dynamic_depth < static_depth)
                    )
                    dyn_over = dynamic_features + (1.0 - dynamic_alpha) * static_features
                    static_over = static_features + (1.0 - static_alpha) * dynamic_features
                    features = torch.where(dynamic_front, dyn_over, static_over)
                else:
                    features = dynamic_features + (1.0 - dynamic_alpha) * static_features
                alpha = dynamic_alpha + (1.0 - dynamic_alpha) * static_alpha
                rendered = model.refiner(features, alpha)
                mse = ((rendered - truth).square() * mask).sum() / (
                    mask.sum() * 3.0
                ).clamp_min(1.0)
                data_loss = torch.log(mse.clamp_min(1e-8)) if args.loss == "log_mse" else mse
                _, xyz_delta, feature_delta, log_scale_delta, opacity_delta, rotvec = state
                anchor = (
                    (xyz_delta / args.max_xyz_delta).square().mean()
                    + (feature_delta / args.max_feature_delta).square().mean()
                    + (log_scale_delta / args.max_log_scale_delta).square().mean()
                    + (opacity_delta / args.max_opacity_logit_delta).square().mean()
                    + (rotvec / max_rotation).square().mean()
                )
                loss = data_loss + args.anchor_weight * anchor
                loss.backward()
                for parameter in parameters:
                    torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                with torch.no_grad():
                    for parameter in parameters:
                        torch.nan_to_num_(parameter, nan=0.0, posinf=0.0, neginf=0.0)
                global_step += 1
                if global_step % 20 == 0 or global_step == 1:
                    row = {
                        "step": global_step, "epoch": epoch + 1, "frame": frame,
                        "inner_step": inner_step + 1, "dynamic_bbox_rgb_mse": float(mse.detach()),
                        "anchor": float(anchor.detach()), "grad_norm": float(grad_norm),
                        "xyz_rms_m": float(xyz_delta.detach().square().mean().sqrt()),
                        "feature_rms": float(feature_delta.detach().square().mean().sqrt()),
                        "scale_multiplier_mean": float(torch.exp(log_scale_delta.detach()).mean()),
                        "opacity_mean": float(current_dynamic.opacities.detach().mean()),
                        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                    records.append(row)
                    print(json.dumps(row), flush=True)
                del state, current_dynamic, posed_dynamic, dynamic_features
                del dynamic_alpha, dynamic_depth, features, alpha, rendered, loss
            del batch, targets, mask, truth, static_features, static_alpha, static_depth

    with torch.no_grad():
        state = current_dynamic_state()
        refined_dynamic = state[0]
    static_target = refined_dir / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source_dir / "final_static_128d.pt").resolve(), static_target)
    save_gaussian_payload(refined_dir / "final_dynamic_128d.pt", refined_dynamic)
    meta_target = refined_dir / "global_dynamic_meta.pt"
    if not meta_target.exists():
        os.symlink((source_dir / "global_dynamic_meta.pt").resolve(), meta_target)
    torch.save({
        "xyz_delta_m": state[1].cpu(), "feature_delta": state[2].cpu(),
        "log_scale_delta": state[3].cpu(), "opacity_logit_delta": state[4].cpu(),
        "rotation_vector_rad": state[5].cpu(),
    }, output_dir / "dynamic_layered_deltas.pt")
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": str(Path(args.refiner_state).resolve()),
        "optimized": "dynamic XYZ/full128D feature/scale/opacity/orientation",
        "frozen": "static Gaussian map, original box graph, camera calibration, network and UNet",
        "training_render": f"cached static and dynamic feature/alpha layers composed with {args.composite}",
        "loss": args.loss, "epochs": args.epochs, "inner_steps": args.inner_steps,
        "training_frame_count": len(frames), "optimizer_steps": global_step,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started, "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

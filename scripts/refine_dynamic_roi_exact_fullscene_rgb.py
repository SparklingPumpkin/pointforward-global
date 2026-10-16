"""Refine dynamic Gaussian attributes with exact joint sorting in RGB ROIs.

Each camera is cropped to the union of projected dynamic boxes plus context.
Frozen static Gaussians whose projected centres can affect that crop are joined
with all dynamic Gaussians in one gsplat call, preserving exact cross-layer
depth sorting while avoiding a full-frame 20M-point render for every update.
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

from pointforward.models import (
    Gaussians,
    InstanceBoxes,
    TargetViews,
    box_dynamic_masks,
    render_features,
)
from pointforward.models.rotation import quaternion_multiply
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


_FIELDS = (
    "means", "quats", "scales", "opacities", "colors", "sigma", "times",
    "instance_id",
)


def subset_gaussians(gaussians: Gaussians, indices: torch.Tensor) -> Gaussians:
    return Gaussians(
        **{name: getattr(gaussians, name)[indices] for name in _FIELDS},
        extras={name: value[indices] for name, value in gaussians.extras.items()},
    )


def subset_boxes(boxes: InstanceBoxes, indices: torch.Tensor) -> InstanceBoxes:
    return InstanceBoxes(
        centers=boxes.centers[indices], dims=boxes.dims[indices],
        rots=boxes.rots[indices], times=boxes.times[indices],
        valid=boxes.valid[indices],
    )


def rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1e-7,
        torch.sin(half) / angle.clamp_min(1e-7),
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((torch.cos(half), rotvec * scale), dim=-1)


def crop_bounds(mask: torch.Tensor, height: int, width: int, margin: int):
    pixels = torch.nonzero(mask, as_tuple=False)
    if pixels.numel() == 0:
        return None
    y0 = max(0, int(pixels[:, 0].min()) - margin)
    y1 = min(height, int(pixels[:, 0].max()) + 1 + margin)
    x0 = max(0, int(pixels[:, 1].min()) - margin)
    x1 = min(width, int(pixels[:, 1].max()) + 1 + margin)
    y0 = max(0, (y0 // 4) * 4)
    x0 = max(0, (x0 // 4) * 4)
    y1 = min(height, ((y1 + 3) // 4) * 4)
    x1 = min(width, ((x1 + 3) // 4) * 4)
    return y0, y1, x0, x1


def projected_static_indices(
    means: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_cam: torch.Tensor,
    bounds,
    image_height: int,
    image_width: int,
    extra_margin: int,
    chunk: int,
) -> torch.Tensor:
    y0, y1, x0, x1 = bounds
    lo_x = max(0, x0 - extra_margin)
    hi_x = min(image_width, x1 + extra_margin)
    lo_y = max(0, y0 - extra_margin)
    hi_y = min(image_height, y1 + extra_margin)
    selected = []
    rotation = world_to_cam[:3, :3]
    translation = world_to_cam[:3, 3]
    for start in range(0, means.shape[0], chunk):
        end = min(start + chunk, means.shape[0])
        camera = means[start:end] @ rotation.T + translation
        z = camera[:, 2]
        safe_z = z.clamp_min(1e-3)
        u = intrinsics[0, 0] * camera[:, 0] / safe_z + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / safe_z + intrinsics[1, 2]
        keep = (
            (z > 0.05) & (u >= lo_x) & (u < hi_x)
            & (v >= lo_y) & (v < hi_y)
        )
        selected.append(torch.nonzero(keep, as_tuple=False).flatten() + start)
    return torch.cat(selected, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--refiner-state", default=None)
    parser.add_argument(
        "--refiner-states", default=None,
        help=(
            "Optional comma-separated camera-specific refiner checkpoints. "
            "When supplied, camera i loads the i-th state; --refiner-state "
            "remains the single-state compatibility fallback."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--train-frame-mod", type=int, default=1)
    parser.add_argument("--train-frame-offset", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--crop-margin-px", type=int, default=64)
    parser.add_argument("--static-center-margin-px", type=int, default=96)
    parser.add_argument(
        "--full-context",
        action="store_true",
        help=(
            "Render the frozen static layer at full resolution, replace only "
            "the dynamic ROI with an exact joint static+dynamic raster, and run "
            "the frozen UNet on the full feature canvas.  The default keeps the "
            "original crop-only UNet path."
        ),
    )
    parser.add_argument(
        "--step-per-camera",
        action="store_true",
        help=(
            "Apply one optimizer step after every camera instead of accumulating "
            "all cameras of a frame.  The default preserves one step per frame."
        ),
    )
    parser.add_argument("--projection-chunk", type=int, default=2_000_000)
    parser.add_argument("--feature-lr", type=float, default=0.01)
    parser.add_argument("--geometry-lr", type=float, default=0.003)
    parser.add_argument("--opacity-lr", type=float, default=0.01)
    parser.add_argument("--max-feature-delta", type=float, default=0.50)
    parser.add_argument("--max-xyz-delta", type=float, default=0.20)
    parser.add_argument("--max-log-scale-delta", type=float, default=0.70)
    parser.add_argument("--max-opacity-logit-delta", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=25.0)
    parser.add_argument(
        "--feature-only", action="store_true",
        help=(
            "Freeze dynamic XYZ, scale, opacity and rotation deltas; optimize "
            "only the selected points' 128D feature vectors."
        ),
    )
    parser.add_argument(
        "--track-shared-feature", action="store_true",
        help=(
            "Use one trainable 128D feature delta per selected instance_id, "
            "broadcast to all points in that track. Implies feature-only."
        ),
    )
    parser.add_argument(
        "--opacity-only", action="store_true",
        help=(
            "Freeze dynamic XYZ, feature, scale and rotation deltas; optimize "
            "only dynamic opacity logits."
        ),
    )
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--track-slots",
        default=None,
        help=(
            "Optional comma-separated scene-global box slots. Only Gaussian "
            "points assigned to these tracks are trainable, and the RGB loss "
            "mask contains only these boxes. The default preserves all tracks."
        ),
    )
    parser.add_argument(
        "--focus-track-slots",
        default=None,
        help=(
            "Optional comma-separated box slots whose pixels receive extra "
            "loss weight while all selected --track-slots remain trainable."
        ),
    )
    parser.add_argument(
        "--focus-weight", type=float, default=1.0,
        help="Extra positive pixel weight inside --focus-track-slots boxes.",
    )
    parser.add_argument(
        "--loss-mask", choices=("box", "fine-vehicle"), default="box",
        help="Use projected boxes or calibrated fine vehicle masks for RGB loss.",
    )
    parser.add_argument(
        "--ownership-loss", action="store_true",
        help="Intersect RGB loss with frozen-baseline visible dynamic ownership support.",
    )
    args = parser.parse_args()
    if not args.refiner_states and not args.refiner_state:
        parser.error("provide --refiner-state or --refiner-states")
    if args.focus_weight <= 0:
        parser.error("--focus-weight must be positive")
    if args.opacity_only and (args.feature_only or args.track_shared_feature):
        parser.error("--opacity-only cannot be combined with feature-only modes")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source = Path(args.payload_dir).resolve()
    output = Path(args.out).resolve()
    refined_dir = output / "runtime-payload-refined"
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
    frames = [
        frame for frame in all_frames[::args.frame_stride]
        if frame % args.train_frame_mod == args.train_frame_offset
    ]
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    normalized_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source / "final_static_128d.pt", device)
    dynamic = load_payload(source / "final_dynamic_128d.pt", device)
    selected_track_slots = None
    if args.track_slots:
        selected_track_slots = sorted({
            int(value) for value in args.track_slots.split(",") if value.strip()
        })
        if not selected_track_slots:
            parser.error("--track-slots must contain at least one integer")
        if min(selected_track_slots) < 0:
            parser.error("--track-slots cannot contain negative values")
        slot_tensor = torch.tensor(selected_track_slots, dtype=torch.long, device=device)
        selected_point_mask = torch.isin(dynamic.instance_id.long(), slot_tensor)
        optimized_indices = torch.nonzero(
            selected_point_mask, as_tuple=False
        ).flatten()
        if optimized_indices.numel() == 0:
            parser.error("--track-slots selected no dynamic Gaussian points")
        optimize_all_tracks = False
    else:
        optimized_indices = torch.arange(dynamic.count, device=device)
        optimize_all_tracks = True
    base_means = dynamic.means[optimized_indices].detach()
    base_colors = dynamic.colors[optimized_indices].detach()
    base_scales = dynamic.scales[optimized_indices].detach()
    base_opacity_logit = torch.logit(
        dynamic.opacities[optimized_indices].detach().clamp(1e-5, 1 - 1e-5)
    )
    base_quats = F.normalize(dynamic.quats[optimized_indices].detach(), dim=-1)
    raw_xyz = torch.nn.Parameter(torch.zeros_like(base_means))
    shared_track_slots = None
    point_track_indices = None
    if args.track_shared_feature:
        if selected_track_slots is None:
            shared_track_slots = sorted({
                int(value) for value in dynamic.instance_id.detach().cpu().tolist()
            })
        else:
            shared_track_slots = selected_track_slots
        shared_track_tensor = torch.tensor(
            shared_track_slots, dtype=torch.long, device=device
        )
        point_track_indices = torch.empty(
            (optimized_indices.numel(),), dtype=torch.long, device=device
        )
        point_instance_ids = dynamic.instance_id[optimized_indices].long()
        for track_index, track_slot in enumerate(shared_track_slots):
            point_track_indices[point_instance_ids == track_slot] = track_index
        raw_feature = torch.nn.Parameter(torch.zeros(
            (len(shared_track_slots), base_colors.shape[-1]),
            device=device, dtype=base_colors.dtype,
        ))
    else:
        raw_feature = torch.nn.Parameter(torch.zeros_like(base_colors))
    raw_log_scale = torch.nn.Parameter(torch.zeros_like(base_scales))
    raw_opacity = torch.nn.Parameter(torch.zeros_like(base_opacity_logit))
    raw_rotation = torch.nn.Parameter(torch.zeros_like(base_means))
    if args.opacity_only:
        raw_xyz.requires_grad_(False)
        raw_feature.requires_grad_(False)
        raw_log_scale.requires_grad_(False)
        raw_rotation.requires_grad_(False)
        parameters = (raw_opacity,)
        optimizer = torch.optim.Adam([
            {"params": [raw_opacity], "lr": args.opacity_lr},
        ])
    elif args.feature_only or args.track_shared_feature:
        # Keep the tensors present for a byte-identical payload/delta manifest,
        # but exclude every non-feature tensor from autograd and the optimizer.
        raw_xyz.requires_grad_(False)
        raw_log_scale.requires_grad_(False)
        raw_opacity.requires_grad_(False)
        raw_rotation.requires_grad_(False)
        parameters = (raw_feature,)
        optimizer = torch.optim.Adam([
            {"params": [raw_feature], "lr": args.feature_lr},
        ])
    else:
        parameters = (raw_xyz, raw_feature, raw_log_scale, raw_opacity, raw_rotation)
        optimizer = torch.optim.Adam([
            {"params": [raw_feature], "lr": args.feature_lr},
            {"params": [raw_xyz, raw_log_scale, raw_rotation], "lr": args.geometry_lr},
            {"params": [raw_opacity], "lr": args.opacity_lr},
        ])
    meta = torch.load(source / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None
    if selected_track_slots is not None:
        if max(selected_track_slots) >= boxes.num_instances:
            parser.error("--track-slots contains an out-of-range box slot")
        selected_box_indices = torch.tensor(
            selected_track_slots, dtype=torch.long, device=device
        )
        loss_boxes = subset_boxes(evaluation_boxes, selected_box_indices)
    else:
        loss_boxes = evaluation_boxes
    focus_track_slots = None
    focus_boxes = None
    if args.focus_track_slots:
        focus_track_slots = sorted({
            int(value) for value in args.focus_track_slots.split(",") if value.strip()
        })
        if not focus_track_slots:
            parser.error("--focus-track-slots must contain at least one integer")
        if min(focus_track_slots) < 0 or max(focus_track_slots) >= boxes.num_instances:
            parser.error("--focus-track-slots contains an out-of-range box slot")
        focus_box_indices = torch.tensor(
            focus_track_slots, dtype=torch.long, device=device
        )
        focus_boxes = subset_boxes(evaluation_boxes, focus_box_indices)
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    if args.refiner_states:
        refiner_paths = [
            Path(value.strip()).resolve()
            for value in args.refiner_states.split(",") if value.strip()
        ]
        if len(refiner_paths) != len(dataset.cameras):
            parser.error(
                f"--refiner-states must contain {len(dataset.cameras)} paths"
            )
    else:
        refiner_paths = [Path(args.refiner_state).resolve()] * len(dataset.cameras)
    refiner_states = []
    for refiner_path in refiner_paths:
        overlay = torch.load(refiner_path, map_location="cpu", weights_only=True)
        refiner_states.append(overlay.get("refiner", overlay))
    model.refiner.load_state_dict(refiner_states[0], strict=True)
    max_rotation = math.radians(args.max_rotation_deg)
    generator = torch.Generator().manual_seed(args.seed)
    records = []
    activity = {
        "steps": 0,
        "finite_nonzero_grad_steps": 0,
        "opacity_rms_samples": [],
        "gradient_norm_samples": [],
        "loss_samples": [],
    }
    started = time.perf_counter()

    def current_dynamic_state():
        xyz_delta = torch.tanh(raw_xyz) * args.max_xyz_delta
        if args.track_shared_feature:
            shared_feature_delta = torch.tanh(raw_feature) * args.max_feature_delta
            feature_delta = shared_feature_delta[point_track_indices]
        else:
            feature_delta = torch.tanh(raw_feature) * args.max_feature_delta
        log_scale_delta = torch.tanh(raw_log_scale) * args.max_log_scale_delta
        opacity_delta = torch.tanh(raw_opacity) * args.max_opacity_logit_delta
        rotvec = torch.tanh(raw_rotation) * max_rotation
        updated_means = base_means + xyz_delta
        updated_colors = base_colors + feature_delta
        updated_scales = base_scales * torch.exp(log_scale_delta)
        updated_opacities = torch.sigmoid(base_opacity_logit + opacity_delta)
        updated_quats = F.normalize(
            quaternion_multiply(rotvec_to_quaternion(rotvec), base_quats), dim=-1
        )
        if optimize_all_tracks:
            current = replace(
                dynamic, means=updated_means, colors=updated_colors,
                scales=updated_scales, opacities=updated_opacities,
                quats=updated_quats,
            )
        else:
            means = dynamic.means.clone()
            colors = dynamic.colors.clone()
            scales = dynamic.scales.clone()
            opacities = dynamic.opacities.clone()
            quats = dynamic.quats.clone()
            means[optimized_indices] = updated_means
            colors[optimized_indices] = updated_colors
            scales[optimized_indices] = updated_scales
            opacities[optimized_indices] = updated_opacities
            quats[optimized_indices] = updated_quats
            current = replace(
                dynamic, means=means, colors=colors, scales=scales,
                opacities=opacities, quats=quats,
            )
        return current, xyz_delta, feature_delta, log_scale_delta, opacity_delta, rotvec

    global_step = 0

    def apply_optimizer_step(
        epoch: int,
        frame: int,
        losses: list[float],
        crop_area_fraction: list[float],
        selected_static_counts: list[int],
    ) -> None:
        nonlocal global_step
        state = current_dynamic_state()
        _, xyz_delta, feature_delta, log_scale_delta, opacity_delta, rotvec = state
        anchor = (
            (xyz_delta / args.max_xyz_delta).square().mean()
            + (feature_delta / args.max_feature_delta).square().mean()
            + (log_scale_delta / args.max_log_scale_delta).square().mean()
            + (opacity_delta / args.max_opacity_logit_delta).square().mean()
            + (rotvec / max_rotation).square().mean()
        )
        (args.anchor_weight * anchor).backward()
        raw_grad_sq = torch.zeros((), device=device)
        finite_grad = True
        for parameter in parameters:
            if parameter.grad is None:
                finite_grad = False
                continue
            finite_grad = finite_grad and bool(torch.isfinite(parameter.grad).all())
            raw_grad_sq = raw_grad_sq + parameter.grad.detach().square().sum()
        raw_grad_norm = raw_grad_sq.sqrt()
        activity["steps"] += 1
        if finite_grad and bool(raw_grad_norm > 0):
            activity["finite_nonzero_grad_steps"] += 1
        activity["gradient_norm_samples"].append(float(raw_grad_norm.detach()))
        activity["opacity_rms_samples"].append(
            float(opacity_delta.detach().square().mean().sqrt())
        )
        activity["loss_samples"].append(float(np.mean(losses)))
        for parameter in parameters:
            torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        with torch.no_grad():
            for parameter in parameters:
                torch.nan_to_num_(parameter, nan=0.0, posinf=0.0, neginf=0.0)
        global_step += 1
        if global_step % 10 == 0 or global_step == 1:
            row = {
                "step": global_step, "epoch": epoch + 1, "frame": frame,
                "dynamic_bbox_rgb_mse": float(np.mean(losses)),
                "crop_area_fraction_mean": float(np.mean(crop_area_fraction)),
                "selected_static_count_mean": float(np.mean(selected_static_counts)),
                "anchor": float(anchor.detach()), "grad_norm": float(grad_norm),
                "xyz_rms_m": float(xyz_delta.detach().square().mean().sqrt()),
                "feature_rms": float(feature_delta.detach().square().mean().sqrt()),
                "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(row)
            print(json.dumps(row), flush=True)

    for epoch in range(args.epochs):
        for order_index in torch.randperm(len(frames), generator=generator).tolist():
            frame = int(frames[order_index])
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            intrinsics_all = batch["intrinsics"].to(device)
            cam_to_world_all = batch["cam_to_world"].to(device)
            frame_indices = torch.full(
                (len(dataset.cameras),), slot, dtype=torch.long, device=device
            )
            masks = box_dynamic_masks(
                loss_boxes, frame_indices, intrinsics_all,
                torch.linalg.inv(cam_to_world_all), height, width,
            )
            if args.loss_mask == "fine-vehicle":
                fine_masks = []
                for camera in dataset.cameras:
                    relative = f"fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png"
                    fine, _ = dataset._load_calibrated_image(
                        metadata, relative, dataset.resolution, camera, gray=True
                    )
                    fine_masks.append(fine.squeeze(0).to(device) > 0.5)
                masks = torch.stack(fine_masks, dim=0)
            focus_masks = (
                box_dynamic_masks(
                    focus_boxes, frame_indices, intrinsics_all,
                    torch.linalg.inv(cam_to_world_all), height, width,
                )
                if focus_boxes is not None else None
            )
            camera_specs = []
            for camera_index in range(len(dataset.cameras)):
                bounds = crop_bounds(
                    masks[camera_index], height, width, args.crop_margin_px
                )
                if bounds is not None:
                    camera_specs.append((camera_index, bounds))
            if not camera_specs:
                continue
            if not args.step_per_camera:
                optimizer.zero_grad(set_to_none=True)
            losses = []
            crop_area_fraction = []
            selected_static_counts = []
            for camera_index, bounds in camera_specs:
                # RF-v1-c uses one independently adapted UNet per camera.  The
                # dynamic optimization must preserve this routing exactly.
                model.refiner.load_state_dict(
                    refiner_states[camera_index], strict=True
                )
                if args.step_per_camera:
                    optimizer.zero_grad(set_to_none=True)
                y0, y1, x0, x1 = bounds
                intrinsics = intrinsics_all[camera_index:camera_index + 1]
                cam_to_world = cam_to_world_all[camera_index:camera_index + 1]
                static_indices = projected_static_indices(
                    static.means, intrinsics[0], torch.linalg.inv(cam_to_world)[0],
                    bounds, height, width, args.static_center_margin_px,
                    args.projection_chunk,
                )
                static_crop = subset_gaussians(static, static_indices)
                state = current_dynamic_state()
                gaussians = combine(static_crop, state[0])
                crop_intrinsics = intrinsics.clone()
                crop_intrinsics[:, 0, 2] -= x0
                crop_intrinsics[:, 1, 2] -= y0
                targets = TargetViews(
                    intrinsics=crop_intrinsics,
                    cam_to_world=cam_to_world,
                    times=normalized_times[slot:slot + 1].to(device),
                    frame_index=torch.full(
                        (1,), slot, dtype=torch.long, device=device
                    ),
                )
                if args.full_context:
                    target_time = targets.times[0]
                    with torch.no_grad():
                        posed_static = model.pose_gaussians(
                            static, None, slot, target_time
                        )
                        static_features, static_alpha, _ = render_features(
                            means=posed_static["means"],
                            quats=posed_static["quats"],
                            scales=posed_static["scales"],
                            opacities=posed_static["opacities"],
                            colors=posed_static["colors"],
                            intrinsics=intrinsics,
                            world_to_cam=torch.linalg.inv(cam_to_world),
                            height=height,
                            width=width,
                            render_depth=False,
                        )
                    posed_joint = model.pose_gaussians(
                        gaussians, boxes, slot, target_time
                    )
                    joint_features, joint_alpha, _ = render_features(
                        means=posed_joint["means"],
                        quats=posed_joint["quats"],
                        scales=posed_joint["scales"],
                        opacities=posed_joint["opacities"],
                        colors=posed_joint["colors"],
                        intrinsics=crop_intrinsics,
                        world_to_cam=torch.linalg.inv(cam_to_world),
                        height=y1 - y0,
                        width=x1 - x0,
                        render_depth=False,
                    )
                    ownership_support = None
                    if args.ownership_loss:
                        # Frozen-baseline dynamic ownership channel: static
                        # Gaussians carry 0 and dynamic Gaussians carry 1.
                        static_owner = replace(
                            static_crop,
                            colors=torch.cat(
                                [static_crop.colors,
                                 torch.zeros((static_crop.count, 1), device=device)],
                                dim=-1,
                            ),
                        )
                        dynamic_owner = replace(
                            dynamic,
                            colors=torch.cat(
                                [dynamic.colors,
                                 torch.ones((dynamic.count, 1), device=device)],
                                dim=-1,
                            ),
                        )
                        owner_joint = combine(static_owner, dynamic_owner)
                        with torch.no_grad():
                            owner_posed = model.pose_gaussians(
                                owner_joint, boxes, slot, target_time
                            )
                            owner_features, _, _ = render_features(
                                means=owner_posed["means"],
                                quats=owner_posed["quats"],
                                scales=owner_posed["scales"],
                                opacities=owner_posed["opacities"],
                                colors=owner_posed["colors"],
                                intrinsics=crop_intrinsics,
                                world_to_cam=torch.linalg.inv(cam_to_world),
                                height=y1 - y0,
                                width=x1 - x0,
                                render_depth=False,
                            )
                        ownership_support = owner_features[:, -1:] >= 0.05
                    full_features = static_features.clone()
                    full_alpha = static_alpha.clone()
                    full_features[:, :, y0:y1, x0:x1] = joint_features
                    full_alpha[:, :, y0:y1, x0:x1] = joint_alpha
                    rendered = model.refiner(full_features, full_alpha)
                    truth = batch["images"][camera_index:camera_index + 1].to(device)
                    mask = masks[camera_index:camera_index + 1].unsqueeze(1)
                    if ownership_support is not None:
                        support_full = torch.zeros_like(mask, dtype=torch.bool)
                        support_full[:, :, y0:y1, x0:x1] = ownership_support
                        mask = mask & support_full
                else:
                    rendered = model.render(
                        gaussians, boxes, targets,
                        height=y1 - y0, width=x1 - x0, render_depth=False,
                    )["rgb"]
                    truth = batch["images"][camera_index:camera_index + 1].to(device)[
                        :, :, y0:y1, x0:x1
                    ]
                    mask = masks[
                        camera_index:camera_index + 1, y0:y1, x0:x1
                    ].unsqueeze(1)
                weights = mask.float()
                if focus_masks is not None:
                    focus_mask = focus_masks[camera_index:camera_index + 1].unsqueeze(1)
                    if not args.full_context:
                        focus_mask = focus_mask[:, :, y0:y1, x0:x1]
                    weights = weights + args.focus_weight * focus_mask.float()
                mse = ((rendered - truth).square() * weights).sum() / (
                    weights.sum() * 3.0
                ).clamp_min(1.0)
                (torch.log(mse.clamp_min(1e-8)) / len(camera_specs)).backward()
                losses.append(float(mse.detach()))
                crop_area_fraction.append((y1 - y0) * (x1 - x0) / (height * width))
                selected_static_counts.append(int(static_indices.numel()))
                if args.step_per_camera:
                    apply_optimizer_step(
                        epoch,
                        frame,
                        [losses[-1]],
                        [crop_area_fraction[-1]],
                        [selected_static_counts[-1]],
                    )
                del static_indices, static_crop, state, gaussians, targets
                del rendered, truth, mask, weights
                if focus_masks is not None:
                    del focus_mask
                if args.full_context:
                    del posed_static, static_features, static_alpha
                    del posed_joint, joint_features, joint_alpha
                    del full_features, full_alpha
                    if ownership_support is not None:
                        del static_owner, dynamic_owner, owner_joint, owner_posed
                        del owner_features, ownership_support, support_full

            if not args.step_per_camera:
                apply_optimizer_step(
                    epoch, frame, losses, crop_area_fraction,
                    selected_static_counts,
                )
            del batch, masks
            if focus_masks is not None:
                del focus_masks

    with torch.no_grad():
        state = current_dynamic_state()
        refined_dynamic = state[0]
    os.symlink((source / "final_static_128d.pt").resolve(), refined_dir / "final_static_128d.pt")
    save_gaussian_payload(refined_dir / "final_dynamic_128d.pt", refined_dynamic)
    os.symlink((source / "global_dynamic_meta.pt").resolve(), refined_dir / "global_dynamic_meta.pt")
    torch.save({
        "optimized_indices": optimized_indices.cpu(),
        "selected_track_slots": selected_track_slots,
        "focus_track_slots": focus_track_slots,
        "focus_weight": args.focus_weight if focus_track_slots is not None else None,
        "xyz_delta_m": state[1].cpu(), "feature_delta": state[2].cpu(),
        "log_scale_delta": state[3].cpu(), "opacity_logit_delta": state[4].cpu(),
        "rotation_vector_rad": state[5].cpu(),
    }, output / "dynamic_roi_exact_deltas.pt")
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source), "refined_payload": str(refined_dir),
        "optimized": (
            "dynamic opacity only with ordinary projected-box RGB loss"
            if args.opacity_only else
            "selected dynamic per-track shared 128D feature only"
            if args.track_shared_feature else
            "selected dynamic 128D feature only"
            if args.feature_only else
            "dynamic XYZ/full128D feature/scale/opacity/orientation"
        ),
        "selected_track_slots": selected_track_slots,
        "refiner_states": [str(path) for path in refiner_paths],
        "optimized_dynamic_count": int(optimized_indices.numel()),
        "frozen": "static attributes, original box graph, cameras, network and UNet",
        "training_render": (
            "full-resolution frozen static feature canvas with exact joint ROI "
            "replacement and full-frame UNet"
            if args.full_context else
            "per-camera bbox ROI; projected static centre cull; exact joint gsplat sorting"
        ),
        "full_context": args.full_context,
        "ownership_loss": args.ownership_loss,
        "feature_only": args.feature_only,
        "track_shared_feature": args.track_shared_feature,
        "opacity_only": args.opacity_only,
        "shared_track_slots": shared_track_slots,
        "train_frame_mod": args.train_frame_mod,
        "train_frame_offset": args.train_frame_offset,
        "optimizer_step_contract": (
            "one step per camera" if args.step_per_camera else "one step per frame"
        ),
        "step_per_camera": args.step_per_camera,
        "epochs": args.epochs, "training_frame_count": len(frames),
        "optimizer_steps": global_step, "crop_margin_px": args.crop_margin_px,
        "static_center_margin_px": args.static_center_margin_px,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
        "activity": {
            "steps": activity["steps"],
            "finite_nonzero_grad_steps": activity["finite_nonzero_grad_steps"],
            "median_raw_gradient_norm": float(np.median(activity["gradient_norm_samples"])) if activity["gradient_norm_samples"] else 0.0,
            "opacity_logit_delta_rms": float(np.sqrt(np.mean(np.square(activity["opacity_rms_samples"])))) if activity["opacity_rms_samples"] else 0.0,
            "training_loss_first_window": float(np.mean(activity["loss_samples"][:10])) if activity["loss_samples"] else None,
            "training_loss_last_window": float(np.mean(activity["loss_samples"][-10:])) if activity["loss_samples"] else None,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

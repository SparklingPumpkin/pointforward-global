"""Refine scene-global dynamic box poses against vehicle RGB/masks.

The PointForward Gaussian payload and all network weights stay frozen.  A
small, track-local translation and yaw correction is learned for each global
vehicle from cropped differentiable renders.  The corrected boxes are written
as render boxes while the original boxes are retained as evaluation boxes.
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

from pointforward.models import Gaussians, InstanceBoxes, TargetViews
from pointforward.models.rotation import matrix_to_quaternion, quaternion_multiply
from pointforward.models.scene_graph import from_box_local, to_box_local
from scripts.build_track_once_dynamic import (
    annotate_fine_mask_visibility,
    calibrated_intrinsics,
    select_diverse_visible_views,
    track_view_candidates,
)
from scripts.deliver_continuous_scene import boxes_payload, save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import load_boxes, load_payload
from scripts.train import build_dataset


def rotation_z(angle: torch.Tensor) -> torch.Tensor:
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((
        torch.stack((c, -s, zero)),
        torch.stack((s, c, zero)),
        torch.stack((zero, zero, one)),
    ))


def corrected_pose(
    center: torch.Tensor,
    rotation: torch.Tensor,
    local_translation: torch.Tensor,
    yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_translation = torch.einsum("j,ji->i", local_translation, rotation)
    return center + world_translation, rotation_z(yaw) @ rotation


def crop_bounds(
    xyxy: list[float], height: int, width: int, padding: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = xyxy
    pad_x = max((x1 - x0) * padding, 8.0)
    pad_y = max((y1 - y0) * padding, 8.0)
    left = max(0, int(math.floor(x0 - pad_x)))
    top = max(0, int(math.floor(y0 - pad_y)))
    right = min(width, int(math.ceil(x1 + pad_x)) + 1)
    bottom = min(height, int(math.ceil(y1 + pad_y)) + 1)
    right = min(width, max(right, left + 8))
    bottom = min(height, max(bottom, top + 8))
    # The two-level UNet is most stable when both dimensions are divisible by 4.
    right -= (right - left) % 4
    bottom -= (bottom - top) % 4
    if right <= left or bottom <= top:
        raise ValueError(f"Empty crop from {xyxy}")
    return left, top, right, bottom


def track_gaussians(dynamic: Gaussians, track_index: int) -> Gaussians:
    keep = dynamic.instance_id == track_index
    if not bool(keep.any()):
        raise ValueError(f"Track {track_index} has no Gaussian")
    return Gaussians(
        means=dynamic.means[keep],
        quats=dynamic.quats[keep],
        scales=dynamic.scales[keep],
        opacities=dynamic.opacities[keep],
        colors=dynamic.colors[keep],
        sigma=dynamic.sigma[keep],
        times=dynamic.times[keep],
        instance_id=torch.zeros(int(keep.sum()), dtype=torch.long, device=dynamic.means.device),
        extras={key: value[keep] for key, value in dynamic.extras.items()},
    )


def two_slot_boxes(
    boxes: InstanceBoxes,
    track_index: int,
    frame_slot: int,
    local_translation: torch.Tensor,
    yaw: torch.Tensor,
) -> InstanceBoxes:
    canonical_slot = int(boxes.canonical_index()[track_index])
    target_center, target_rotation = corrected_pose(
        boxes.centers[track_index, frame_slot],
        boxes.rots[track_index, frame_slot],
        local_translation,
        yaw,
    )
    return InstanceBoxes(
        centers=torch.stack((
            boxes.centers[track_index, canonical_slot], target_center,
        )).unsqueeze(0),
        dims=torch.stack((
            boxes.dims[track_index, canonical_slot], boxes.dims[track_index, frame_slot],
        )).unsqueeze(0),
        rots=torch.stack((
            boxes.rots[track_index, canonical_slot], target_rotation,
        )).unsqueeze(0),
        times=torch.stack((
            boxes.times[track_index, canonical_slot], boxes.times[track_index, frame_slot],
        )).unsqueeze(0),
        valid=torch.ones((1, 2), dtype=torch.bool, device=boxes.device),
    )


def centroid_loss(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    height, width = alpha.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=alpha.device),
        torch.linspace(-1.0, 1.0, width, device=alpha.device),
        indexing="ij",
    )
    pred_weight = alpha[0, 0].clamp_min(0.0)
    target_weight = mask[0, 0]
    pred_mass = pred_weight.sum().clamp_min(1e-6)
    target_mass = target_weight.sum().clamp_min(1e-6)
    pred_center = torch.stack(((pred_weight * xx).sum(), (pred_weight * yy).sum())) / pred_mass
    target_center = torch.stack(((target_weight * xx).sum(), (target_weight * yy).sum())) / target_mass
    return ((pred_center - target_center) ** 2).sum()


def rebase_payload(
    dynamic: Gaussians,
    original: InstanceBoxes,
    refined: InstanceBoxes,
) -> Gaussians:
    means = dynamic.means.clone()
    quats = dynamic.quats.clone()
    old_center, old_rotation = original.canonical()
    new_center, new_rotation = refined.canonical()
    for track_index in range(original.num_instances):
        keep = dynamic.instance_id == track_index
        if not bool(keep.any()):
            continue
        local = to_box_local(
            dynamic.means[keep], old_center[track_index], old_rotation[track_index]
        )
        means[keep] = from_box_local(
            local, new_center[track_index], new_rotation[track_index]
        )
        relative = new_rotation[track_index].transpose(0, 1) @ old_rotation[track_index]
        quats[keep] = quaternion_multiply(
            matrix_to_quaternion(relative.expand(int(keep.sum()), -1, -1)),
            dynamic.quats[keep],
        )
    return replace(dynamic, means=means, quats=quats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--views-per-track", type=int, default=12)
    parser.add_argument("--view-angle-weight", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-translation-m", type=float, default=0.75)
    parser.add_argument("--max-yaw-deg", type=float, default=12.0)
    parser.add_argument("--crop-padding", type=float, default=0.35)
    parser.add_argument("--min-mask-pixels", type=int, default=24)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--centroid-weight", type=float, default=0.5)
    parser.add_argument("--rgb-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
    dataset = build_dataset(config, args.split)
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    frames = list(metadata["frames"])
    frame_to_slot = {int(frame): slot for slot, frame in enumerate(frames)}
    selected_set = set(frames)
    normalized_times = dataset._normalised_times(frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    intrinsics = calibrated_intrinsics(dataset, metadata, frames[0])
    height, width = dataset.resolution

    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes_cpu = load_boxes(meta["boxes"], torch.device("cpu"))
    assert boxes_cpu is not None
    boxes = InstanceBoxes(
        centers=boxes_cpu.centers.to(device), dims=boxes_cpu.dims.to(device),
        rots=boxes_cpu.rots.to(device), times=boxes_cpu.times.to(device),
        valid=boxes_cpu.valid.to(device),
    )
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)

    candidates_by_track: dict[int, list[dict]] = {}
    for track_index, track_id in enumerate(meta["track_ids"]):
        if not bool((dynamic.instance_id == track_index).any()):
            continue
        candidates_by_track[int(track_id)] = track_view_candidates(
            dataset, metadata, int(track_id), selected_set, world_from, intrinsics
        )
    annotate_fine_mask_visibility(dataset, metadata, candidates_by_track)

    samples: list[dict] = []
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(meta["track_ids"])}
    for track_id, candidates in candidates_by_track.items():
        visible = [item for item in candidates if item.get("fine_dynamic_pixels", 0) >= args.min_mask_pixels]
        if not visible:
            continue
        chosen = select_diverse_visible_views(
            visible, min(args.views_per_track, len(visible)), args.view_angle_weight
        )
        for item in chosen:
            samples.append({"track_index": track_id_to_index[track_id], **item})
    if not samples:
        raise RuntimeError("No visible vehicle crops were found")

    raw_translation = torch.nn.Parameter(torch.zeros((boxes.num_instances, 3), device=device))
    raw_yaw = torch.nn.Parameter(torch.zeros(boxes.num_instances, device=device))
    optimizer = torch.optim.Adam((raw_translation, raw_yaw), lr=args.learning_rate)
    max_yaw = math.radians(args.max_yaw_deg)
    generator = torch.Generator().manual_seed(args.seed)
    track_cache: dict[int, Gaussians] = {}
    image_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    records: list[dict] = []
    started = time.perf_counter()

    for epoch in range(args.epochs):
        order = torch.randperm(len(samples), generator=generator).tolist()
        for within_epoch, sample_index in enumerate(order):
            sample = samples[sample_index]
            track_index = int(sample["track_index"])
            frame = int(sample["frame"])
            camera = int(sample["camera"])
            slot = frame_to_slot[frame]
            key = (frame, camera)
            if key not in image_cache:
                image, intrinsic = dataset._load_calibrated_image(
                    metadata, f"images/{frame:03d}_{camera}.jpg",
                    dataset.resolution, camera,
                )
                fine_path = Path(metadata["path"]) / f"fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png"
                mask_relative = (
                    f"fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png"
                    if fine_path.exists()
                    else f"dynamic_masks/all/{frame:03d}_{camera}.png"
                )
                mask, _ = dataset._load_calibrated_image(
                    metadata, mask_relative, dataset.resolution, camera, gray=True
                )
                camera_to_world = torch.from_numpy(
                    dataset._cam_to_world(metadata, frame, camera, world_from)
                ).float()
                image_cache[key] = (
                    image, mask, torch.from_numpy(intrinsic).float(), camera_to_world
                )
            image, mask, intrinsic, camera_to_world = image_cache[key]
            left, top, right, bottom = crop_bounds(
                sample["bbox_xyxy"], height, width, args.crop_padding
            )
            target = image[:, top:bottom, left:right].unsqueeze(0).to(device)
            target_mask = mask[:, top:bottom, left:right].unsqueeze(0).to(device)
            if int((target_mask > 0.5).sum()) < args.min_mask_pixels:
                continue
            crop_intrinsic = intrinsic.clone()
            crop_intrinsic[0, 2] -= left
            crop_intrinsic[1, 2] -= top

            if track_index not in track_cache:
                track_cache[track_index] = track_gaussians(dynamic, track_index)
            local_translation = torch.tanh(raw_translation[track_index]) * args.max_translation_m
            yaw = torch.tanh(raw_yaw[track_index]) * max_yaw
            render_boxes = two_slot_boxes(
                boxes, track_index, slot, local_translation, yaw
            )
            targets = TargetViews(
                intrinsics=crop_intrinsic.unsqueeze(0).to(device),
                cam_to_world=camera_to_world.unsqueeze(0).to(device),
                times=normalized_times[slot:slot + 1].to(device),
                frame_index=torch.ones(1, dtype=torch.long, device=device),
            )
            rendered = model.render(
                track_cache[track_index], render_boxes, targets,
                height=bottom - top, width=right - left, render_depth=False,
            )
            alpha = rendered["alpha"].clamp(1e-5, 1.0 - 1e-5)
            target_mask = (target_mask > 0.5).float()
            intersection = (alpha * target_mask).sum()
            dice = 1.0 - (2.0 * intersection + 1.0) / (
                alpha.sum() + target_mask.sum() + 1.0
            )
            centre = centroid_loss(alpha, target_mask)
            rgb_region = torch.maximum(target_mask, (alpha.detach() > 0.05).float())
            rgb = (((rendered["rgb"] - target) ** 2) * rgb_region).sum() / (
                rgb_region.sum() * 3.0
            ).clamp_min(1.0)
            anchor = (
                (local_translation / args.max_translation_m).square().mean()
                + (yaw / max_yaw).square()
            )
            loss = (
                args.dice_weight * dice
                + args.centroid_weight * centre
                + args.rgb_weight * rgb
                + args.anchor_weight * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # gsplat projection derivatives can be undefined for a Gaussian
            # exactly crossing the near plane.  Such isolated entries must not
            # poison Adam's persistent state for the whole track.
            for parameter in (raw_translation, raw_yaw):
                if parameter.grad is not None:
                    torch.nan_to_num_(
                        parameter.grad, nan=0.0, posinf=0.0, neginf=0.0
                    )
            torch.nn.utils.clip_grad_norm_((raw_translation, raw_yaw), 5.0)
            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(raw_translation, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nan_to_num_(raw_yaw, nan=0.0, posinf=0.0, neginf=0.0)

            global_step = epoch * len(samples) + within_epoch + 1
            if global_step % 25 == 0 or global_step == 1:
                row = {
                    "step": global_step, "epoch": epoch + 1,
                    "track_index": track_index, "frame": frame, "camera": camera,
                    "loss": float(loss.detach()), "dice_loss": float(dice.detach()),
                    "centroid_loss": float(centre.detach()), "rgb_mse": float(rgb.detach()),
                    "translation_m": float(local_translation.detach().norm()),
                    "yaw_deg": float(torch.rad2deg(yaw.detach())),
                }
                records.append(row)
                print(json.dumps(row), flush=True)

    with torch.no_grad():
        translations = torch.tanh(raw_translation) * args.max_translation_m
        yaws = torch.tanh(raw_yaw) * max_yaw
        refined_boxes = InstanceBoxes(
            centers=boxes.centers.clone(), dims=boxes.dims.clone(),
            rots=boxes.rots.clone(), times=boxes.times.clone(), valid=boxes.valid.clone(),
        )
        for track_index in range(boxes.num_instances):
            for slot in torch.nonzero(boxes.valid[track_index], as_tuple=False).flatten().tolist():
                center, rotation = corrected_pose(
                    boxes.centers[track_index, slot], boxes.rots[track_index, slot],
                    translations[track_index], yaws[track_index],
                )
                refined_boxes.centers[track_index, slot] = center
                refined_boxes.rots[track_index, slot] = rotation
        refined_dynamic = rebase_payload(dynamic, boxes, refined_boxes)

    static_target = refined_dir / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source_dir / "final_static_128d.pt").resolve(), static_target)
    save_gaussian_payload(refined_dir / "final_dynamic_128d.pt", refined_dynamic)
    refined_meta = dict(meta)
    refined_meta["boxes"] = boxes_payload(refined_boxes)
    refined_meta["evaluation_boxes"] = meta.get("evaluation_boxes", meta["boxes"])
    torch.save(refined_meta, refined_dir / "global_dynamic_meta.pt")

    corrections = []
    for index, track_id in enumerate(meta["track_ids"]):
        corrections.append({
            "track_index": index, "track_id": int(track_id),
            "translation_local_m": translations[index].detach().cpu().tolist(),
            "translation_norm_m": float(translations[index].norm().cpu()),
            "yaw_deg": float(torch.rad2deg(yaws[index]).cpu()),
            "training_views": sum(int(sample["track_index"] == index) for sample in samples),
        })
    (output_dir / "track_pose_corrections.json").write_text(
        json.dumps(corrections, indent=2) + "\n"
    )
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "optimized": "one local translation and yaw correction per global track",
        "frozen": "all Gaussian attributes, static map, cameras, network and UNet weights",
        "supervision": "fine vehicle mask silhouette plus cropped RGB",
        "evaluation_boxes": "original unmodified boxes",
        "epochs": args.epochs, "sample_count": len(samples),
        "views_per_track": args.views_per_track,
        "max_translation_m": args.max_translation_m, "max_yaw_deg": args.max_yaw_deg,
        "translation_norm_m": {
            "mean": float(translations.norm(dim=-1).mean().cpu()),
            "p95": float(torch.quantile(translations.norm(dim=-1), 0.95).cpu()),
            "max": float(translations.norm(dim=-1).max().cpu()),
        },
        "yaw_abs_deg": {
            "mean": float(torch.rad2deg(yaws.abs()).mean().cpu()),
            "p95": float(torch.quantile(torch.rad2deg(yaws.abs()), 0.95).cpu()),
            "max": float(torch.rad2deg(yaws.abs()).max().cpu()),
        },
        "wall_seconds": time.perf_counter() - started,
        "loss_records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

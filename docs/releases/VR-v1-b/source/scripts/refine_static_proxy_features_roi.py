"""Refine static vehicle-proxy features inside swept dynamic boxes.

This optional branch keeps the scene-global dynamic map and every Gaussian
geometry field frozen.  Only static points that fall inside the union of
annotated dynamic boxes (excluding the road-adjacent bottom band) receive a
learnable 128-D feature delta.  Training uses exact joint static/dynamic
sorting in each dynamic ROI and the original full-resolution refiner context.
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

from pointforward.models import TargetViews, box_dynamic_masks, render_features
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.refine_dynamic_roi_exact_fullscene_rgb import (
    crop_bounds,
    projected_static_indices,
    subset_gaussians,
)
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def swept_proxy_indices(
    means: torch.Tensor,
    box_payload: dict,
    bottom_keep_fraction: float,
    box_batch_size: int,
) -> torch.Tensor:
    centers = box_payload["centers"].to(means.device)
    dims = box_payload["dims"].to(means.device)
    rots = box_payload["rots"].to(means.device)
    valid = box_payload["valid"].to(means.device)
    selected_global = torch.zeros(
        means.shape[0], dtype=torch.bool, device=means.device
    )
    for track in range(centers.shape[0]):
        track_valid = valid[track]
        if not bool(track_valid.any()):
            continue
        track_centers = centers[track, track_valid]
        track_dims = dims[track, track_valid]
        track_rots = rots[track, track_valid]
        radii = 0.5 * torch.linalg.vector_norm(track_dims, dim=-1)
        lower = (track_centers - radii[:, None]).amin(dim=0)
        upper = (track_centers + radii[:, None]).amax(dim=0)
        candidates = torch.nonzero(
            ((means >= lower) & (means <= upper)).all(dim=-1),
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            continue
        points = means[candidates]
        hit = torch.zeros(points.shape[0], dtype=torch.bool, device=means.device)
        for start in range(0, track_centers.shape[0], box_batch_size):
            end = min(start + box_batch_size, track_centers.shape[0])
            batch_centers = track_centers[start:end]
            batch_dims = track_dims[start:end]
            batch_rots = track_rots[start:end]
            delta = points[:, None, :] - batch_centers[None, :, :]
            local = torch.einsum("pbj,bij->pbi", delta, batch_rots)
            inside = (local.abs() <= 0.5 * batch_dims[None, :, :]).all(dim=-1)
            if bottom_keep_fraction > 0.0:
                bottom_cut = (-0.5 + bottom_keep_fraction) * batch_dims[None, :, 2]
                inside &= local[:, :, 2] > bottom_cut
            hit |= inside.any(dim=1)
        selected_global[candidates[hit]] = True
    return torch.nonzero(selected_global, as_tuple=False).flatten()


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
    parser.add_argument("--crop-margin-px", type=int, default=64)
    parser.add_argument("--static-center-margin-px", type=int, default=96)
    parser.add_argument("--projection-chunk", type=int, default=2_000_000)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--max-feature-delta", type=float, default=0.50)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--bottom-keep-fraction", type=float, default=0.20)
    parser.add_argument("--box-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.bottom_keep_fraction < 1.0:
        parser.error("--bottom-keep-fraction must be in [0, 1)")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source = Path(args.payload_dir).resolve()
    output = Path(args.out).resolve()
    refined_dir = output / "runtime-payload-refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

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
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    normalized_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source / "final_static_128d.pt", device)
    dynamic = load_payload(source / "final_dynamic_128d.pt", device)
    meta = torch.load(
        source / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    boxes = load_boxes(meta["boxes"], device)
    evaluation_payload = meta.get("evaluation_boxes", meta["boxes"])
    evaluation_boxes = load_boxes(evaluation_payload, device)
    assert boxes is not None and evaluation_boxes is not None

    proxy_indices = swept_proxy_indices(
        static.means, evaluation_payload, args.bottom_keep_fraction,
        args.box_batch_size,
    )
    if proxy_indices.numel() == 0:
        raise RuntimeError("No static proxy Gaussians were selected")
    base_proxy_colors = static.colors[proxy_indices].detach()
    raw_feature = torch.nn.Parameter(torch.zeros_like(base_proxy_colors))
    optimizer = torch.optim.Adam((raw_feature,), lr=args.learning_rate)

    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    overlay = torch.load(
        Path(args.refiner_state).resolve(), map_location="cpu", weights_only=True
    )
    model.refiner.load_state_dict(overlay.get("refiner", overlay), strict=True)
    generator = torch.Generator().manual_seed(args.seed)
    records = []
    global_step = 0

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
                evaluation_boxes, frame_indices, intrinsics_all,
                torch.linalg.inv(cam_to_world_all), height, width,
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

            optimizer.zero_grad(set_to_none=True)
            losses = []
            selected_counts = []
            for camera_index, bounds in camera_specs:
                y0, y1, x0, x1 = bounds
                intrinsics = intrinsics_all[camera_index:camera_index + 1]
                cam_to_world = cam_to_world_all[camera_index:camera_index + 1]
                static_indices = projected_static_indices(
                    static.means, intrinsics[0], torch.linalg.inv(cam_to_world)[0],
                    bounds, height, width, args.static_center_margin_px,
                    args.projection_chunk,
                )
                static_crop = subset_gaussians(static, static_indices)
                positions = torch.searchsorted(proxy_indices, static_indices)
                safe_positions = positions.clamp_max(proxy_indices.numel() - 1)
                matches = (
                    (positions < proxy_indices.numel())
                    & (proxy_indices[safe_positions] == static_indices)
                )
                local_rows = torch.nonzero(matches, as_tuple=False).flatten()
                proxy_rows = positions[matches]
                crop_colors = static_crop.colors.clone()
                if local_rows.numel() > 0:
                    delta = torch.tanh(raw_feature[proxy_rows]) * args.max_feature_delta
                    crop_colors[local_rows] = base_proxy_colors[proxy_rows] + delta
                static_crop = replace(static_crop, colors=crop_colors)
                gaussians = combine(static_crop, dynamic)

                crop_intrinsics = intrinsics.clone()
                crop_intrinsics[:, 0, 2] -= x0
                crop_intrinsics[:, 1, 2] -= y0
                target_time = normalized_times[slot].to(device)
                with torch.no_grad():
                    posed_static = model.pose_gaussians(static, None, slot, target_time)
                    static_features, static_alpha, _ = render_features(
                        means=posed_static["means"], quats=posed_static["quats"],
                        scales=posed_static["scales"],
                        opacities=posed_static["opacities"],
                        colors=posed_static["colors"], intrinsics=intrinsics,
                        world_to_cam=torch.linalg.inv(cam_to_world),
                        height=height, width=width, render_depth=False,
                    )
                targets = TargetViews(
                    intrinsics=crop_intrinsics,
                    cam_to_world=cam_to_world,
                    times=normalized_times[slot:slot + 1].to(device),
                    frame_index=torch.full((1,), slot, dtype=torch.long, device=device),
                )
                posed_joint = model.pose_gaussians(gaussians, boxes, slot, target_time)
                joint_features, joint_alpha, _ = render_features(
                    means=posed_joint["means"], quats=posed_joint["quats"],
                    scales=posed_joint["scales"],
                    opacities=posed_joint["opacities"],
                    colors=posed_joint["colors"], intrinsics=crop_intrinsics,
                    world_to_cam=torch.linalg.inv(cam_to_world),
                    height=y1 - y0, width=x1 - x0, render_depth=False,
                )
                full_features = static_features.clone()
                full_alpha = static_alpha.clone()
                full_features[:, :, y0:y1, x0:x1] = joint_features
                full_alpha[:, :, y0:y1, x0:x1] = joint_alpha
                rendered = model.refiner(full_features, full_alpha)
                truth = batch["images"][camera_index:camera_index + 1].to(device)
                mask = masks[camera_index:camera_index + 1].unsqueeze(1)
                mse = ((rendered - truth).square() * mask).sum() / (
                    mask.sum() * 3.0
                ).clamp_min(1.0)
                (torch.log(mse.clamp_min(1e-8)) / len(camera_specs)).backward()
                losses.append(float(mse.detach()))
                selected_counts.append(int(local_rows.numel()))
                del static_indices, static_crop, positions, safe_positions, matches
                del local_rows, proxy_rows, crop_colors, gaussians, targets
                del posed_static, static_features, static_alpha, posed_joint
                del joint_features, joint_alpha, full_features, full_alpha
                del rendered, truth, mask

            feature_delta = torch.tanh(raw_feature) * args.max_feature_delta
            anchor = (feature_delta / args.max_feature_delta).square().mean()
            (args.anchor_weight * anchor).backward()
            torch.nan_to_num_(raw_feature.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw_feature,), 1.0)
            optimizer.step()
            global_step += 1
            if global_step % 10 == 0 or global_step == 1:
                row = {
                    "step": global_step, "epoch": epoch + 1, "frame": frame,
                    "dynamic_bbox_rgb_mse": float(np.mean(losses)),
                    "selected_proxy_count_mean": float(np.mean(selected_counts)),
                    "anchor": float(anchor.detach()),
                    "grad_norm": float(grad_norm),
                    "feature_rms": float(feature_delta.detach().square().mean().sqrt()),
                    "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                records.append(row)
                print(json.dumps(row), flush=True)
            del batch, masks

    with torch.no_grad():
        final_delta = torch.tanh(raw_feature) * args.max_feature_delta
        refined_colors = static.colors.clone()
        refined_colors[proxy_indices] = base_proxy_colors + final_delta
        refined_static = replace(static, colors=refined_colors)
    save_gaussian_payload(refined_dir / "final_static_128d.pt", refined_static)
    for name in ("final_dynamic_128d.pt", "global_dynamic_meta.pt"):
        os.symlink((source / name).resolve(), refined_dir / name)
    torch.save(
        {"static_indices": proxy_indices.cpu(), "feature_delta": final_delta.cpu()},
        output / "static_proxy_feature_delta.pt",
    )
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source), "refined_payload": str(refined_dir),
        "optimized": "full 128-D feature of swept-box static proxy Gaussians",
        "frozen": "all geometry, scene-global dynamic map, non-proxy static features, boxes, cameras, network and UNet",
        "training_render": "exact joint ROI sorting pasted into full static feature canvas before full-resolution UNet",
        "proxy_count": int(proxy_indices.numel()),
        "bottom_keep_fraction": args.bottom_keep_fraction,
        "epochs": args.epochs, "optimizer_steps": global_step,
        "training_frame_count": len(frames),
        "learning_rate": args.learning_rate,
        "max_feature_delta": args.max_feature_delta,
        "feature_rms": float(final_delta.square().mean().sqrt()),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()


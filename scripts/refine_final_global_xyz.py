"""Refine only static Gaussian centres of a final PointForward global map.

All decoded attributes, dynamic Gaussians, scene boxes, cameras, and network
weights stay frozen.  Sparse LiDAR depth supplies the geometry loss; each
centre is projected back into a distance-dependent displacement ball.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pointforward.models import Gaussians, TargetViews
from pointforward.models.renderer import render_features
from scripts.deliver_continuous_scene import boxes_to, save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload, psnr
from scripts.train import build_dataset


def displacement_bound(distance: torch.Tensor) -> torch.Tensor:
    """2 cm through 10 m, then +2 mm/m, capped at 15 cm."""
    return (0.02 + 0.002 * (distance - 10.0).clamp_min(0.0)).clamp(0.02, 0.15)


def project_to_ball_(delta: torch.Tensor, radius: torch.Tensor) -> None:
    with torch.no_grad():
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta.mul_(torch.minimum(torch.ones_like(norm), radius / norm))


def sampled_quantile(values: torch.Tensor, q: float, max_samples: int = 1_000_000) -> torch.Tensor:
    """Avoid full-tensor CUDA quantile workspace for 20M-point payloads."""
    flat = values.reshape(-1)
    stride = max(1, (flat.numel() + max_samples - 1) // max_samples)
    return torch.quantile(flat[::stride], q)


def collect_frame(dataset, metadata, frames, all_slots, all_times, world_from, frame):
    return dataset._collect(
        metadata, [frame], all_slots, all_times, dataset.resolution, world_from,
        lidar=True,
    )


def static_depth(
    means: torch.Tensor,
    static: Gaussians,
    intrinsic: torch.Tensor,
    cam_to_world: torch.Tensor,
    height: int,
    width: int,
    scalar_colors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, alpha, depth = render_features(
        means=means,
        quats=static.quats,
        scales=static.scales,
        opacities=static.opacities,
        colors=scalar_colors,
        intrinsics=intrinsic[None],
        world_to_cam=torch.linalg.inv(cam_to_world)[None],
        height=height,
        width=width,
        render_depth=True,
    )
    assert depth is not None
    return depth, alpha


@torch.no_grad()
def audit(
    model,
    dataset,
    metadata,
    frames,
    all_slots,
    all_times,
    world_from,
    static: Gaussians,
    dynamic: Gaussians,
    boxes,
    means: torch.Tensor,
    scalar_colors: torch.Tensor,
    audit_frames: list[int],
) -> dict:
    height, width = dataset.resolution
    refined_static = replace(static, means=means)
    scene = combine(refined_static, dynamic)
    rgb_scores, non_sky_scores, depth_errors = [], [], []
    depth_hits = 0
    for frame in audit_frames:
        batch = collect_frame(
            dataset, metadata, frames, all_slots, all_times, world_from, frame
        )
        slot = all_slots[frame]
        for camera_index in range(len(dataset.cameras)):
            targets = TargetViews(
                intrinsics=batch["intrinsics"][camera_index:camera_index + 1].to(means.device),
                cam_to_world=batch["cam_to_world"][camera_index:camera_index + 1].to(means.device),
                times=batch["times"][camera_index:camera_index + 1].to(means.device),
                frame_index=torch.tensor([slot], device=means.device),
            )
            prediction = model.render(
                scene, boxes, targets, height=height, width=width, render_depth=False
            )["rgb"]
            truth = batch["images"][camera_index:camera_index + 1].to(means.device)
            non_sky = batch["sky_mask"][camera_index:camera_index + 1].to(means.device) < 0.5
            rgb_scores.append(psnr(prediction, truth))
            non_sky_scores.append(psnr(prediction, truth, non_sky))

            rendered_depth, alpha = static_depth(
                means, static,
                batch["intrinsics"][camera_index].to(means.device),
                batch["cam_to_world"][camera_index].to(means.device),
                height, width, scalar_colors,
            )
            valid = batch["sparse_mask"][camera_index:camera_index + 1].to(means.device).bool()
            valid &= ~batch["dynamic_mask"][camera_index:camera_index + 1].to(means.device).bool()
            valid &= non_sky
            valid &= alpha > 0.05
            if valid.any():
                truth_depth = batch["sparse_depth"][camera_index:camera_index + 1].to(means.device)
                depth_errors.append(((rendered_depth[valid] - truth_depth[valid]) ** 2).sum())
                depth_hits += int(valid.sum())
    return {
        "psnr": float(np.mean(rgb_scores)),
        "non_sky_psnr": float(np.mean(non_sky_scores)),
        "depth_rmse_m": float(torch.sqrt(torch.stack(depth_errors).sum() / max(depth_hits, 1)).cpu()),
        "depth_hits": depth_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--audit-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.out).resolve()
    refined_dir = output_dir / "runtime-payload-refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.payload_dir).resolve()
    config = yaml.safe_load(Path(args.config).read_text())
    dataset = build_dataset(config, "val")
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    frames = list(metadata["frames"])
    all_slots = {frame: index for index, frame in enumerate(frames)}
    all_times = dataset._normalised_times(frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    dynamic_meta = torch.load(
        source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    boxes = boxes_to(load_boxes(dynamic_meta["boxes"], torch.device("cpu")), device)
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)

    camera_centres = []
    for frame in frames:
        for camera in dataset.cameras:
            camera_centres.append(dataset._cam_to_world(metadata, frame, camera, world_from)[:3, 3])
    camera_centres = torch.from_numpy(np.stack(camera_centres)).float().to(device)
    nearest = torch.empty(static.count, device=device)
    chunk = 100_000
    for start in range(0, static.count, chunk):
        points = static.means[start:start + chunk]
        nearest[start:start + chunk] = torch.cdist(points, camera_centres).amin(dim=1)
    max_displacement = displacement_bound(nearest).unsqueeze(-1)
    del nearest, camera_centres

    base_means = static.means.detach()
    delta = torch.nn.Parameter(torch.zeros_like(base_means))
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    scalar_colors = torch.ones((static.count, 1), device=device)
    audit_frames = [frames[0], frames[len(frames) // 2], frames[-1]]
    train_frames = [frame for frame in frames if frame not in audit_frames]
    generator = torch.Generator().manual_seed(args.seed)
    view_order = torch.randperm(len(train_frames) * len(dataset.cameras), generator=generator).tolist()

    records = []
    started = time.perf_counter()
    initial = audit(
        model, dataset, metadata, frames, all_slots, all_times, world_from,
        static, dynamic, boxes, base_means, scalar_colors, audit_frames,
    )
    records.append({"step": 0, "kind": "audit", **initial})
    print(json.dumps(records[-1]), flush=True)

    for step in range(1, args.steps + 1):
        view_index = view_order[(step - 1) % len(view_order)]
        frame = train_frames[view_index // len(dataset.cameras)]
        camera_index = view_index % len(dataset.cameras)
        batch = collect_frame(
            dataset, metadata, frames, all_slots, all_times, world_from, frame
        )
        means = base_means + delta
        rendered_depth, alpha = static_depth(
            means, static,
            batch["intrinsics"][camera_index].to(device),
            batch["cam_to_world"][camera_index].to(device),
            height, width, scalar_colors,
        )
        truth_depth = batch["sparse_depth"][camera_index:camera_index + 1].to(device)
        valid = batch["sparse_mask"][camera_index:camera_index + 1].to(device).bool()
        valid &= ~batch["dynamic_mask"][camera_index:camera_index + 1].to(device).bool()
        valid &= batch["sky_mask"][camera_index:camera_index + 1].to(device) < 0.5
        valid &= alpha.detach() > 0.05
        if not valid.any():
            continue
        depth_loss = F.smooth_l1_loss(rendered_depth[valid], truth_depth[valid], beta=0.1)
        anchor_loss = ((delta / max_displacement) ** 2).mean()
        loss = depth_loss + args.anchor_weight * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta], 1.0)
        optimizer.step()
        project_to_ball_(delta, max_displacement)

        with torch.no_grad():
            displacement = delta.norm(dim=-1)
            row = {
                "step": step,
                "kind": "train",
                "frame": int(frame),
                "camera": int(dataset.cameras[camera_index]),
                "valid_depth_pixels": int(valid.sum()),
                "loss": float(loss),
                "depth_loss": float(depth_loss),
                "anchor_loss": float(anchor_loss),
                "mean_displacement_m": float(displacement.mean()),
                "p95_displacement_m": float(sampled_quantile(displacement, 0.95)),
                "max_displacement_m": float(displacement.max()),
            }
        records.append(row)
        print(json.dumps(row), flush=True)
        if step % args.audit_every == 0 or step == args.steps:
            score = audit(
                model, dataset, metadata, frames, all_slots, all_times, world_from,
                static, dynamic, boxes, base_means + delta.detach(), scalar_colors,
                audit_frames,
            )
            audit_row = {"step": step, "kind": "audit", **score}
            records.append(audit_row)
            print(json.dumps(audit_row), flush=True)

    refined_static = replace(static, means=base_means + delta.detach())
    save_gaussian_payload(refined_dir / "final_static_128d.pt", refined_static)
    shutil.copy2(source_dir / "final_dynamic_128d.pt", refined_dir / "final_dynamic_128d.pt")
    shutil.copy2(source_dir / "global_dynamic_meta.pt", refined_dir / "global_dynamic_meta.pt")

    with (output_dir / "optimization_metrics.csv").open("w", newline="") as handle:
        fields = sorted({key for row in records for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "optimization_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    final_audit = next(row for row in reversed(records) if row["kind"] == "audit")
    displacement = delta.detach().norm(dim=-1)
    manifest = {
        "status": "completed",
        "scene": args.scene,
        "source_payload": str(source_dir),
        "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "steps": args.steps,
        "learning_rate": args.lr,
        "anchor_weight": args.anchor_weight,
        "optimized": "static means only",
        "frozen": "dynamic means, colors/features, opacity, scale, rotation, boxes, cameras, and all network weights",
        "supervision": "sparse LiDAR depth excluding dynamic and sky pixels",
        "max_displacement_policy": "2cm through 10m, +2mm/m thereafter, capped at 15cm using nearest source-camera distance",
        "audit_frames": audit_frames,
        "initial_audit": initial,
        "final_audit": final_audit,
        "audit_change": {
            key: final_audit[key] - initial[key]
            for key in ("psnr", "non_sky_psnr", "depth_rmse_m")
        },
        "displacement_m": {
            "mean": float(displacement.mean()),
            "p95": float(sampled_quantile(displacement, 0.95)),
            "max": float(displacement.max()),
        },
        "static_count": static.count,
        "dynamic_count": dynamic.count,
        "wall_seconds": time.perf_counter() - started,
        "metrics_csv": str(output_dir / "optimization_metrics.csv"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

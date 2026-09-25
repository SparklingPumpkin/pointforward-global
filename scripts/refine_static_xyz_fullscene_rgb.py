"""Bounded RGB refinement of final global static Gaussian centres."""

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


def bounded_delta(raw: torch.Tensor, radius: float) -> torch.Tensor:
    """Smooth component proposal followed by an exact L2-ball projection."""
    proposed = torch.tanh(raw) * radius
    norm = proposed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return proposed * (radius / norm).clamp_max(1.0)


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
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-displacement-m", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--max-frames", type=int, default=None)
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
    dataset = build_dataset(config, "val")
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    all_frames = list(metadata["frames"])
    frames = all_frames if args.max_frames is None else all_frames[:args.max_frames]
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    normalized_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    base_means = static.means.detach()
    raw_delta = torch.nn.Parameter(torch.zeros_like(base_means))
    optimizer = torch.optim.Adam((raw_delta,), lr=args.learning_rate)
    meta = torch.load(
        source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None

    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    overlay = torch.load(
        Path(args.refiner_state).resolve(), map_location="cpu", weights_only=True
    )
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
            losses = []
            for camera_index in range(len(dataset.cameras)):
                delta = bounded_delta(raw_delta, args.max_displacement_m)
                current_static = replace(static, means=base_means + delta)
                gaussians = combine(current_static, dynamic)
                intrinsics = batch["intrinsics"][camera_index:camera_index + 1].to(device)
                cam_to_world = batch["cam_to_world"][camera_index:camera_index + 1].to(device)
                targets = TargetViews(
                    intrinsics=intrinsics,
                    cam_to_world=cam_to_world,
                    times=normalized_times[slot:slot + 1].to(device),
                    frame_index=torch.full((1,), slot, dtype=torch.long, device=device),
                )
                rendered = model.render(
                    gaussians, boxes, targets,
                    height=height, width=width, render_depth=False,
                )["rgb"]
                dynamic_mask = box_dynamic_masks(
                    evaluation_boxes,
                    torch.tensor([slot], dtype=torch.long, device=device),
                    intrinsics, torch.linalg.inv(cam_to_world), height, width,
                ).unsqueeze(1)
                valid = (batch["sky_mask"][camera_index:camera_index + 1].to(device) < 0.5)
                valid &= ~dynamic_mask
                squared = (rendered - batch["images"][camera_index:camera_index + 1].to(device)).square()
                mse = (squared * valid).sum() / (valid.sum() * 3.0).clamp_min(1.0)
                (torch.log(mse.clamp_min(1e-8)) / len(dataset.cameras)).backward()
                losses.append(float(mse.detach()))
                del delta, current_static, gaussians, rendered, targets

            delta = bounded_delta(raw_delta, args.max_displacement_m)
            anchor = (delta / args.max_displacement_m).square().mean()
            (args.anchor_weight * anchor).backward()
            torch.nan_to_num_(raw_delta.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw_delta,), 1.0)
            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(raw_delta, nan=0.0, posinf=0.0, neginf=0.0)
            step = epoch * len(frames) + order_index + 1
            if step % 10 == 0 or step == 1:
                with torch.no_grad():
                    norms = delta.norm(dim=-1)
                    sample = norms[::max(1, norms.numel() // 1_000_000)]
                    row = {
                        "step": step, "epoch": epoch + 1, "frame": frame,
                        "static_rgb_mse": float(np.mean(losses)),
                        "anchor": float(anchor), "grad_norm": float(grad_norm),
                        "mean_displacement_m": float(norms.mean()),
                        "p95_displacement_m_sampled": float(torch.quantile(sample, 0.95)),
                        "max_displacement_m": float(norms.max()),
                        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                records.append(row)
                print(json.dumps(row), flush=True)
            del batch, delta

    with torch.no_grad():
        delta = bounded_delta(raw_delta, args.max_displacement_m)
        refined_static = replace(static, means=base_means + delta)
    save_gaussian_payload(refined_dir / "final_static_128d.pt", refined_static)
    for name in ("final_dynamic_128d.pt", "global_dynamic_meta.pt"):
        target = refined_dir / name
        if not target.exists():
            os.symlink((source_dir / name).resolve(), target)
    torch.save({"static_xyz_delta_m": delta.cpu()}, output_dir / "static_xyz_delta.pt")
    norms = delta.norm(dim=-1)
    sample = norms[::max(1, norms.numel() // 1_000_000)]
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": str(Path(args.refiner_state).resolve()),
        "optimized": "static Gaussian XYZ only, tanh bounded",
        "frozen": "all features, dynamic map, opacity/scale/rotation/time, boxes, cameras and network",
        "supervision": "log RGB MSE outside sky and dynamic evaluation boxes",
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "max_displacement_m": args.max_displacement_m,
        "anchor_weight": args.anchor_weight,
        "mean_displacement_m": float(norms.mean()),
        "p95_displacement_m_sampled": float(torch.quantile(sample, 0.95)),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

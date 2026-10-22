"""Refine only dynamic Gaussian 128-D features using final full-scene RGB."""

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
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--anchor-weight", type=float, default=0.0001)
    parser.add_argument("--max-feature-delta", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-delta", help="Optional dynamic_feature_delta.pt to resume.")
    parser.add_argument(
        "--refiner-state",
        help="Optional trained refiner overlay used while optimizing dynamic features.",
    )
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
    all_frames = list(metadata["frames"])
    frames = all_frames if args.max_frames is None else all_frames[:args.max_frames]
    if not frames:
        parser.error("--max-frames must be positive")
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    normalized_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    base_colors = dynamic.colors.detach()
    raw_delta = torch.nn.Parameter(torch.zeros_like(base_colors))
    if args.initial_delta:
        initial = torch.load(args.initial_delta, map_location="cpu", weights_only=True)["feature_delta"].to(device)
        if initial.shape != raw_delta.shape:
            raise ValueError(f"Initial feature delta shape {tuple(initial.shape)} != {tuple(raw_delta.shape)}")
        raw_delta.data.copy_(torch.atanh((initial / args.max_feature_delta).clamp(-0.999, 0.999)))
    optimizer = torch.optim.Adam((raw_delta,), lr=args.learning_rate)
    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    if args.refiner_state:
        overlay = torch.load(
            Path(args.refiner_state).resolve(), map_location="cpu", weights_only=True
        )
        model.refiner.load_state_dict(overlay.get("refiner", overlay), strict=True)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()

    for epoch in range(args.epochs):
        for order_index, source_index in enumerate(
            torch.randperm(len(frames), generator=generator).tolist()
        ):
            frame = int(frames[source_index])
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = []
            valid_views = 0
            for camera_index in range(len(dataset.cameras)):
                delta = torch.tanh(raw_delta) * args.max_feature_delta
                current_dynamic = replace(dynamic, colors=base_colors + delta)
                gaussians = combine(static, current_dynamic)
                targets = TargetViews(
                    intrinsics=batch["intrinsics"][camera_index:camera_index + 1].to(device),
                    cam_to_world=batch["cam_to_world"][camera_index:camera_index + 1].to(device),
                    times=normalized_times[slot:slot + 1].to(device),
                    frame_index=torch.full((1,), slot, dtype=torch.long, device=device),
                )
                rendered = model.render(
                    gaussians, boxes, targets,
                    height=height, width=width, render_depth=False,
                )["rgb"]
                mask = box_dynamic_masks(
                    evaluation_boxes,
                    torch.tensor([slot], dtype=torch.long, device=device),
                    batch["intrinsics"][camera_index:camera_index + 1].to(device),
                    torch.linalg.inv(
                        batch["cam_to_world"][camera_index:camera_index + 1].to(device)
                    ),
                    height, width,
                ).unsqueeze(1)
                if not bool(mask.any()):
                    continue
                squared = (rendered - batch["images"][camera_index:camera_index + 1].to(device)).square()
                mse = (squared * mask).sum() / (mask.sum() * 3.0).clamp_min(1.0)
                (mse / len(dataset.cameras)).backward()
                losses.append(float(mse.detach()))
                valid_views += 1
                del gaussians, current_dynamic, rendered

            delta = torch.tanh(raw_delta) * args.max_feature_delta
            anchor = delta.square().mean()
            (args.anchor_weight * anchor).backward()
            torch.nan_to_num_(raw_delta.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw_delta,), 1.0)
            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(raw_delta, nan=0.0, posinf=0.0, neginf=0.0)

            step = epoch * len(frames) + order_index + 1
            if step % 10 == 0 or step == 1:
                sample = delta.detach().reshape(-1)[::100]
                row = {
                    "step": step, "epoch": epoch + 1, "frame": frame,
                    "valid_views": valid_views,
                    "bbox_rgb_mse": float(np.mean(losses)) if losses else None,
                    "anchor": float(anchor.detach()), "grad_norm": float(grad_norm),
                    "feature_delta_rms": float(delta.detach().square().mean().sqrt()),
                    "feature_delta_abs_p95_sampled": float(torch.quantile(sample.abs(), 0.95)),
                    "feature_delta_abs_max": float(delta.detach().abs().max()),
                    "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                records.append(row)
                print(json.dumps(row), flush=True)

    with torch.no_grad():
        delta = torch.tanh(raw_delta) * args.max_feature_delta
        refined_dynamic = replace(dynamic, colors=base_colors + delta)
    static_target = refined_dir / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source_dir / "final_static_128d.pt").resolve(), static_target)
    save_gaussian_payload(refined_dir / "final_dynamic_128d.pt", refined_dynamic)
    meta_target = refined_dir / "global_dynamic_meta.pt"
    if not meta_target.exists():
        os.symlink((source_dir / "global_dynamic_meta.pt").resolve(), meta_target)
    torch.save({"feature_delta": delta.cpu()}, output_dir / "dynamic_feature_delta.pt")

    sample = delta.reshape(-1)[::100]
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": (
            None if args.refiner_state is None else str(Path(args.refiner_state).resolve())
        ),
        "optimized": "dynamic Gaussian 128-D colors/features only",
        "frozen": "static map, all Gaussian geometry/opacity/sigma/time, boxes, cameras, network and UNet",
        "supervision": "full final scene RGB MSE inside original evaluation bbox union",
        "evaluation_boxes": "original unmodified boxes",
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "anchor_weight": args.anchor_weight,
        "max_feature_delta": args.max_feature_delta,
        "initial_delta": None if args.initial_delta is None else str(Path(args.initial_delta).resolve()),
        "feature_delta": {
            "rms": float(delta.square().mean().sqrt()),
            "abs_p95_sampled": float(torch.quantile(sample.abs(), 0.95)),
            "abs_max": float(delta.abs().max()),
        },
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

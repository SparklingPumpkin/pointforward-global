"""Fast target-domain refinement of static Gaussian feature groups.

The payload geometry, dynamic Gaussian set, boxes, camera poses, PointForward
network and UNet stay frozen.  A small per-static-point feature-group delta is
optimized with the final full-scene RGB loss, then saved as a normal runtime
payload.  ``--groups`` must divide the 128-D feature width; each group delta
is broadcast over its corresponding channels to keep memory bounded.
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

from pointforward.models import TargetViews
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def load_refiner_state(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(payload.get("refiner", payload), strict=True)


def sampled_quantile(values: torch.Tensor, q: float, max_samples: int = 1_000_000) -> torch.Tensor:
    flat = values.reshape(-1)
    stride = max(1, (flat.numel() + max_samples - 1) // max_samples)
    return torch.quantile(flat[::stride], q)


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
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--anchor-weight", type=float, default=0.0001)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--frame-list", default=None,
        help="Comma-separated source frame IDs to supervise, overriding stride/max-frames.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--camera-batch-size", type=int, default=1)
    parser.add_argument("--refiner-state", default=None)
    parser.add_argument(
        "--empty-cache-every", type=int, default=1,
        help="Call CUDA empty_cache every N frames; 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if 128 % args.groups:
        parser.error("--groups must divide 128")

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
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.camera_batch_size < 1:
        raise ValueError("--camera-batch-size must be >= 1")
    if args.frame_list:
        requested = [int(value) for value in args.frame_list.split(",") if value.strip()]
        missing = sorted(set(requested) - set(all_frames))
        if missing:
            raise ValueError(f"--frame-list contains frames absent from scene: {missing}")
        frames = requested
    else:
        frames = all_frames[::args.frame_stride]
        if args.max_frames is not None:
            frames = frames[:args.max_frames]
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    normalized_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    base_colors = static.colors.detach()
    n, c = base_colors.shape
    channels_per_group = c // args.groups
    raw_delta = torch.nn.Parameter(torch.zeros((n, args.groups), device=device))
    optimizer = torch.optim.Adam((raw_delta,), lr=args.learning_rate)
    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    assert boxes is not None
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    if args.refiner_state:
        load_refiner_state(model, Path(args.refiner_state).resolve())
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()
    group_index = torch.arange(c, device=device) // channels_per_group

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
            camera_count = len(dataset.cameras)
            camera_batch_size = min(args.camera_batch_size, camera_count)
            for camera_start in range(0, camera_count, camera_batch_size):
                camera_end = min(camera_start + camera_batch_size, camera_count)
                view_count = camera_end - camera_start
                delta = raw_delta[:, group_index]
                current_static = replace(static, colors=base_colors + delta)
                scene = combine(current_static, dynamic)
                targets = TargetViews(
                    intrinsics=batch["intrinsics"][camera_start:camera_end].to(device),
                    cam_to_world=batch["cam_to_world"][camera_start:camera_end].to(device),
                    times=normalized_times[slot:slot + 1].to(device).expand(view_count),
                    frame_index=torch.full((view_count,), slot, dtype=torch.long, device=device),
                )
                rendered = model.render(
                    scene, boxes, targets, height=height, width=width,
                    render_depth=False,
                )["rgb"]
                truth = batch["images"][camera_start:camera_end].to(device)
                per_view_mse = (rendered - truth).square().flatten(1).mean(1)
                (per_view_mse.sum() / camera_count).backward()
                losses.extend(float(value) for value in per_view_mse.detach())
                del targets, rendered, scene, current_static, delta
            anchor = raw_delta.square().mean()
            (args.anchor_weight * anchor).backward()
            torch.nan_to_num_(raw_delta.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw_delta,), 1.0)
            optimizer.step()
            step = epoch * len(frames) + order_index + 1
            if step % 10 == 0 or step == 1:
                with torch.no_grad():
                    magnitude = raw_delta.detach()
                    records.append({
                        "step": step, "epoch": epoch + 1, "frame": frame,
                        "rgb_mse": float(np.mean(losses)),
                        "anchor": float(anchor.detach()),
                        "grad_norm": float(grad_norm),
                        "delta_rms": float(magnitude.square().mean().sqrt()),
                        "delta_abs_p95_sampled": float(sampled_quantile(magnitude.abs(), 0.95)),
                        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    })
                    print(json.dumps(records[-1]), flush=True)
            del batch
            if args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    with torch.no_grad():
        delta = raw_delta[:, group_index]
        refined_static = replace(static, colors=base_colors + delta)
    save_gaussian_payload(refined_dir / "final_static_128d.pt", refined_static)
    for name in ("final_dynamic_128d.pt", "global_dynamic_meta.pt"):
        target = refined_dir / name
        if not target.exists():
            os.symlink((source_dir / name).resolve(), target)
    torch.save({"feature_delta_groups": raw_delta.detach().cpu(), "groups": args.groups}, output_dir / "static_feature_delta.pt")
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": (
            str(Path(args.refiner_state).resolve()) if args.refiner_state else None
        ),
        "optimized": "static Gaussian 128-D colors via per-point grouped deltas",
        "groups": args.groups, "epochs": args.epochs,
        "frozen": "dynamic map, Gaussian geometry/opacity/sigma/time, boxes, cameras, network and UNet",
        "supervision": "full final scene RGB MSE over all pixels and cameras",
        "learning_rate": args.learning_rate, "anchor_weight": args.anchor_weight,
        "training_frame_stride": args.frame_stride,
        "training_frame_count": len(frames),
        "training_frames": [int(frame) for frame in frames],
        "camera_batch_size": args.camera_batch_size,
        "empty_cache_every": args.empty_cache_every,
        "delta_rms": float(delta.square().mean().sqrt()),
        "delta_abs_p95_sampled": float(sampled_quantile(delta.abs(), 0.95)),
        "delta_abs_max": float(delta.abs().max()),
        "static_count": int(static.count), "dynamic_count": int(dynamic.count),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started, "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Fast full-scene optimization of an additive 3-D Gaussian RGB residual.

The original 128-D features and PointForward refiner remain unchanged. Each
Gaussian receives a view-independent RGB residual which is rasterised with
the same geometry, opacity, temporal state and dynamic pose, then added to the
normal 128-D+UNet image.  The optional per-frame mode gives every source frame
an independent dynamic residual slice while keeping geometry and the decoder
frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.models import TargetViews, render_features
from scripts.eval import build_model
from scripts.render_final_global_runtime import (
    box_dynamic_masks,
    combine,
    load_boxes,
    load_payload,
)
from scripts.train import build_dataset


def load_refiner_state(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(payload.get("refiner", payload), strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--refiner-state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--residual-state-in", default=None)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--cache-base-rgb", action="store_true",
        help="Optional in-run CPU cache of frozen base RGB; never loads an external scene cache.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--max-residual", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--frames", default=None,
        help="Explicit comma-separated source frame IDs; overrides stride/max-frames.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--camera-index", type=int, default=-1,
        help="Train against one camera only; -1 keeps the legacy all-camera mode.",
    )
    parser.add_argument(
        "--per-frame", action="store_true",
        help="Learn an independent dynamic RGB residual for each source frame.",
    )
    parser.add_argument(
        "--optimize",
        choices=("all", "dynamic"),
        default="all",
        help="Restrict trainable residuals to dynamic Gaussians when requested.",
    )
    parser.add_argument(
        "--loss-region",
        choices=("full", "dynamic-bbox", "dynamic-support"),
        default="full",
        help="RGB supervision region; default preserves the original behavior.",
    )
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.max_residual <= 0:
        parser.error("--max-residual must be positive")
    if args.per_frame and args.residual_state_in:
        parser.error("--per-frame cannot resume a shared residual state")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source_dir = Path(args.payload_dir).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, "val")
    if args.camera_index >= len(dataset.cameras):
        parser.error("--camera-index must address a configured camera")
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    all_frames = list(metadata["frames"])
    if args.frames:
        frames = [int(value) for value in args.frames.split(",") if value.strip()]
        unknown = sorted(set(frames) - set(all_frames))
        if unknown:
            parser.error(f"--frames contains unknown source frames: {unknown}")
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
    gaussians = combine(static, dynamic)
    meta = torch.load(
        source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    boxes = load_boxes(meta["boxes"], device)
    assert boxes is not None
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    load_refiner_state(model, Path(args.refiner_state).resolve())

    if args.residual_state_in:
        initial_state = torch.load(
            Path(args.residual_state_in).resolve(),
            map_location="cpu", weights_only=True,
        )
        initial_residual = torch.cat((
            initial_state["static_rgb_residual"],
            initial_state["dynamic_rgb_residual"],
        ), dim=0).to(device)
        if initial_residual.shape != (gaussians.count, 3):
            raise ValueError("Input RGB residual shape does not match payload")
        if args.optimize == "dynamic":
            initial_residual = initial_residual[static.count:]
        normalized = (initial_residual / args.max_residual).clamp(-0.999999, 0.999999)
        raw_initial = torch.atanh(normalized)
    else:
        trainable_count = dynamic.count if args.optimize == "dynamic" else gaussians.count
        shape = (
            (len(all_frames), trainable_count, 3)
            if args.per_frame else (trainable_count, 3)
        )
        raw_initial = torch.zeros(
            shape, dtype=torch.float32, device=device
        )
    if args.per_frame:
        raw_residual = torch.nn.ParameterList([
            torch.nn.Parameter(frame_initial)
            for frame_initial in raw_initial.unbind(0)
        ])
        optimizer = torch.optim.Adam(raw_residual.parameters(), lr=args.learning_rate)
    else:
        raw_residual = torch.nn.Parameter(raw_initial)
        optimizer = torch.optim.Adam((raw_residual,), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()
    global_step = 0
    base_rgb_cache = {}

    for epoch in range(args.epochs):
        for order_index in torch.randperm(len(frames), generator=generator).tolist():
            global_step += 1
            frame = int(frames[order_index])
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            camera_indices = (
                list(range(len(dataset.cameras)))
                if args.camera_index < 0 else [args.camera_index]
            )
            view_count = len(camera_indices)
            targets = TargetViews(
                intrinsics=batch["intrinsics"][camera_indices].to(device),
                cam_to_world=batch["cam_to_world"][camera_indices].to(device),
                times=normalized_times[slot:slot + 1].to(device).expand(view_count),
                frame_index=torch.full(
                    (view_count,), slot, dtype=torch.long, device=device
                ),
            )
            with torch.no_grad():
                if args.cache_base_rgb and frame in base_rgb_cache:
                    base_rgb = base_rgb_cache[frame].to(device)
                else:
                    base_rgb = model.render(
                        gaussians, boxes, targets, height=height, width=width,
                        render_depth=False,
                    )["rgb"]
                    if args.cache_base_rgb:
                        base_rgb_cache[frame] = base_rgb.cpu()
                baseline_support = None
                if args.loss_region == "dynamic-support":
                    posed_base = model.pose_gaussians(
                        gaussians, boxes, slot, normalized_times[slot].to(device)
                    )
                    ownership = (gaussians.instance_id >= 0).to(
                        posed_base["colors"].dtype
                    ).unsqueeze(1)
                    base_colors129 = torch.cat((posed_base["colors"], ownership), dim=1)
                    base_feature129, _, _ = render_features(
                        means=posed_base["means"], quats=posed_base["quats"],
                        scales=posed_base["scales"], opacities=posed_base["opacities"],
                        colors=base_colors129, intrinsics=targets.intrinsics,
                        world_to_cam=torch.linalg.inv(targets.cam_to_world),
                        height=height, width=width, render_depth=False,
                    )
                    baseline_support = base_feature129[:, 128:129] >= 0.05
                    del posed_base, ownership, base_colors129, base_feature129

            optimizer.zero_grad(set_to_none=True)
            active_raw_residual = raw_residual[slot] if args.per_frame else raw_residual
            trainable_residual = torch.tanh(active_raw_residual) * args.max_residual
            if args.optimize == "dynamic":
                residual = torch.cat((
                    torch.zeros((static.count, 3), device=device),
                    trainable_residual,
                ), dim=0)
            else:
                residual = trainable_residual
            residual_scene = replace(gaussians, colors=residual)
            posed = model.pose_gaussians(
                residual_scene, boxes, slot, normalized_times[slot].to(device)
            )
            rendered_residual, _, _ = render_features(
                means=posed["means"], quats=posed["quats"],
                scales=posed["scales"], opacities=posed["opacities"],
                colors=posed["colors"], intrinsics=targets.intrinsics,
                world_to_cam=torch.linalg.inv(targets.cam_to_world),
                height=height, width=width, render_depth=False,
            )
            truth = batch["images"][camera_indices].to(device)
            prediction = base_rgb + rendered_residual
            squared_error = (prediction - truth).square()
            if args.loss_region == "dynamic-bbox":
                region = box_dynamic_masks(
                    boxes,
                    torch.full(
                        (view_count,), slot, dtype=torch.long, device=device
                    ),
                    targets.intrinsics,
                    torch.linalg.inv(targets.cam_to_world),
                    height,
                    width,
                ).unsqueeze(1)
                denominator = region.sum().clamp_min(1) * squared_error.shape[1]
                mse = (squared_error * region).sum() / denominator
            elif args.loss_region == "dynamic-support":
                region = baseline_support
                denominator = region.sum().clamp_min(1) * squared_error.shape[1]
                mse = (squared_error * region).sum() / denominator
            else:
                mse = squared_error.mean()
            anchor = trainable_residual.square().mean()
            (mse + args.anchor_weight * anchor).backward()
            torch.nan_to_num_(
                active_raw_residual.grad, nan=0.0, posinf=0.0, neginf=0.0
            )
            grad_norm = torch.nn.utils.clip_grad_norm_((active_raw_residual,), 1.0)
            optimizer.step()
            if global_step % 10 == 0 or global_step == 1:
                with torch.no_grad():
                    magnitude = trainable_residual.norm(dim=-1)
                    sample = magnitude[::max(1, magnitude.numel() // 1_000_000)]
                    row = {
                        "step": global_step, "epoch": epoch + 1, "frame": frame,
                        "rgb_mse": float(mse), "anchor": float(anchor),
                        "grad_norm": float(grad_norm),
                        "mean_residual_norm": float(magnitude.mean()),
                        "p95_residual_norm_sampled": float(torch.quantile(sample, 0.95)),
                        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                records.append(row)
                print(json.dumps(row), flush=True)
            del batch, targets, base_rgb, residual, trainable_residual
            del residual_scene, posed, rendered_residual, truth, prediction
            if baseline_support is not None:
                del baseline_support, region

    with torch.no_grad():
        trainable_residual = (
            torch.stack([
                torch.tanh(frame_residual) * args.max_residual
                for frame_residual in raw_residual
            ])
            if args.per_frame
            else torch.tanh(raw_residual) * args.max_residual
        )
        if args.optimize == "dynamic":
            static_residual = torch.zeros((static.count, 3), dtype=torch.float32)
            dynamic_residual = trainable_residual.cpu()
        else:
            static_residual = trainable_residual[:static.count].cpu()
            dynamic_residual = trainable_residual[static.count:].cpu()
    state_path = output_dir / "rgb_residual_state.pt"
    torch.save(
        {
            "static_rgb_residual": static_residual,
            "dynamic_rgb_residual": dynamic_residual,
            "source_payload": str(source_dir),
            "residual_mode": "per-frame" if args.per_frame else "shared",
            "source_frames": [int(frame) for frame in all_frames],
        },
        state_path,
    )
    magnitude = trainable_residual.norm(dim=-1)
    flat_magnitude = magnitude.flatten()
    sample = flat_magnitude[::max(1, flat_magnitude.numel() // 1_000_000)]
    manifest = {
        "status": "completed", "scene": args.scene,
        "cache_base_rgb": args.cache_base_rgb,
        "source_payload": str(source_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": str(Path(args.refiner_state).resolve()),
        "rgb_residual_state": str(state_path),
        "residual_state_in": (
            str(Path(args.residual_state_in).resolve())
            if args.residual_state_in else None
        ),
        "optimized": (
            "view-independent additive RGB residual on dynamic Gaussians only"
            if args.optimize == "dynamic"
            else "view-independent additive RGB residual on every static and dynamic Gaussian"
        ),
        "residual_mode": "per-frame" if args.per_frame else "shared",
        "optimize_subset": args.optimize,
        "loss_region": args.loss_region,
        "frozen": "128-D features, refiner, geometry, opacity, scale, rotation, time, boxes and cameras",
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "anchor_weight": args.anchor_weight, "max_residual": args.max_residual,
        "training_frame_stride": args.frame_stride,
        "camera_index": args.camera_index,
        "training_frame_count": len(frames),
        "training_frames": [int(frame) for frame in frames],
        "static_count": int(static.count), "dynamic_count": int(dynamic.count),
        "mean_residual_norm": float(magnitude.mean()),
        "p95_residual_norm_sampled": float(torch.quantile(sample, 0.95)),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

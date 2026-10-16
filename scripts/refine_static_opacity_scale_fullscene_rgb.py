"""Refine bounded static Gaussian opacity and optional isotropic scale.

The final global XYZ, rotation, 128-D features, dynamic map, boxes and network
stay frozen.  This is an optional post-refinement path: by default only a
bounded per-point opacity-logit delta is learned; ``--optimize-scale`` also
learns one isotropic multiplier per static Gaussian while preserving the
configured maximum physical radius.
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
import yaml

from pointforward.models import TargetViews
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def load_refiner_state(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(payload.get("refiner", payload), strict=True)


def sampled_quantile(
    values: torch.Tensor, q: float, max_samples: int = 1_000_000
) -> torch.Tensor:
    flat = values.reshape(-1)
    stride = max(1, (flat.numel() + max_samples - 1) // max_samples)
    return torch.quantile(flat[::stride], q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--refiner-state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--opacity-learning-rate", type=float, default=0.02)
    parser.add_argument("--scale-learning-rate", type=float, default=0.01)
    parser.add_argument("--max-opacity-logit-delta", type=float, default=2.0)
    parser.add_argument("--optimize-scale", action="store_true")
    parser.add_argument("--max-scale-factor", type=float, default=2.0)
    parser.add_argument("--max-scale-m", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.camera_batch_size < 1:
        parser.error("--camera-batch-size must be >= 1")
    if args.max_scale_factor < 1.0:
        parser.error("--max-scale-factor must be >= 1")
    if args.max_scale_m <= 0.0:
        parser.error("--max-scale-m must be positive")

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
    frames = all_frames[::args.frame_stride]
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    normalized_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    base_opacities = static.opacities.detach().float().clamp(1e-5, 1.0 - 1e-5)
    base_opacity_logits = torch.logit(base_opacities)
    base_scales = static.scales.detach().float()
    raw_opacity = torch.nn.Parameter(torch.zeros_like(base_opacities))
    parameter_groups = [{"params": [raw_opacity], "lr": args.opacity_learning_rate}]
    raw_scale = None
    if args.optimize_scale:
        raw_scale = torch.nn.Parameter(
            torch.zeros(static.count, dtype=base_scales.dtype, device=device)
        )
        parameter_groups.append(
            {"params": [raw_scale], "lr": args.scale_learning_rate}
        )
    optimizer = torch.optim.Adam(parameter_groups)

    meta = torch.load(
        source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    boxes = load_boxes(meta["boxes"], device)
    assert boxes is not None
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    load_refiner_state(model, Path(args.refiner_state).resolve())
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()
    max_log_scale_delta = math.log(args.max_scale_factor)

    def current_static() -> tuple:
        opacity_delta = torch.tanh(raw_opacity) * args.max_opacity_logit_delta
        opacities = torch.sigmoid(base_opacity_logits + opacity_delta)
        scale_delta = None
        scales = base_scales
        if raw_scale is not None:
            scale_delta = torch.tanh(raw_scale) * max_log_scale_delta
            scales = (base_scales * scale_delta.exp().unsqueeze(-1)).clamp(
                min=1e-4, max=args.max_scale_m
            )
        return replace(static, opacities=opacities, scales=scales), opacity_delta, scale_delta

    global_step = 0
    for epoch in range(args.epochs):
        order = torch.randperm(len(frames), generator=generator).tolist()
        for order_index in order:
            global_step += 1
            frame = int(frames[order_index])
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            optimizer.zero_grad(set_to_none=True)
            camera_count = len(dataset.cameras)
            camera_batch_size = min(args.camera_batch_size, camera_count)
            losses = []
            all_cameras_in_one_graph = camera_batch_size == camera_count
            frame_mse_sum = None
            if not all_cameras_in_one_graph:
                measured = []
                with torch.no_grad():
                    frozen_static, _, _ = current_static()
                    frozen_scene = combine(frozen_static, dynamic)
                    for camera_start in range(0, camera_count, camera_batch_size):
                        camera_end = min(camera_start + camera_batch_size, camera_count)
                        view_count = camera_end - camera_start
                        targets = TargetViews(
                            intrinsics=batch["intrinsics"][camera_start:camera_end].to(device),
                            cam_to_world=batch["cam_to_world"][camera_start:camera_end].to(device),
                            times=normalized_times[slot:slot + 1].to(device).expand(view_count),
                            frame_index=torch.full(
                                (view_count,), slot, dtype=torch.long, device=device
                            ),
                        )
                        rgb = model.render(
                            frozen_scene, boxes, targets, height=height, width=width,
                            render_depth=False,
                        )["rgb"]
                        truth = batch["images"][camera_start:camera_end].to(device)
                        measured.extend((rgb - truth).square().flatten(1).mean(1))
                    del frozen_static, frozen_scene
                frame_mse_sum = torch.stack(measured).sum().clamp_min(1e-8)

            for camera_start in range(0, camera_count, camera_batch_size):
                camera_end = min(camera_start + camera_batch_size, camera_count)
                view_count = camera_end - camera_start
                refined_static, opacity_delta, scale_delta = current_static()
                scene = combine(refined_static, dynamic)
                targets = TargetViews(
                    intrinsics=batch["intrinsics"][camera_start:camera_end].to(device),
                    cam_to_world=batch["cam_to_world"][camera_start:camera_end].to(device),
                    times=normalized_times[slot:slot + 1].to(device).expand(view_count),
                    frame_index=torch.full(
                        (view_count,), slot, dtype=torch.long, device=device
                    ),
                )
                rgb = model.render(
                    scene, boxes, targets, height=height, width=width,
                    render_depth=False,
                )["rgb"]
                truth = batch["images"][camera_start:camera_end].to(device)
                per_view_mse = (rgb - truth).square().flatten(1).mean(1)
                objective = (
                    torch.log(per_view_mse.sum().clamp_min(1e-8))
                    if all_cameras_in_one_graph
                    else per_view_mse.sum() / frame_mse_sum
                )
                objective.backward()
                losses.extend(float(value) for value in per_view_mse.detach())
                del refined_static, scene, targets, rgb, opacity_delta, scale_delta

            anchor = torch.tanh(raw_opacity).square().mean()
            if raw_scale is not None:
                anchor = 0.5 * (
                    anchor + torch.tanh(raw_scale).square().mean()
                )
            (args.anchor_weight * anchor).backward()
            parameters = [raw_opacity] + ([] if raw_scale is None else [raw_scale])
            for parameter in parameters:
                torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            if global_step % 10 == 0 or global_step == 1:
                with torch.no_grad():
                    opacity_delta = (
                        torch.tanh(raw_opacity) * args.max_opacity_logit_delta
                    )
                    row = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "frame": frame,
                        "rgb_mse": float(np.mean(losses)),
                        "anchor": float(anchor),
                        "grad_norm": float(grad_norm),
                        "opacity_logit_delta_rms": float(
                            opacity_delta.square().mean().sqrt()
                        ),
                        "opacity_logit_delta_p95": float(
                            sampled_quantile(opacity_delta.abs(), 0.95)
                        ),
                        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                    if raw_scale is not None:
                        scale_delta = torch.tanh(raw_scale) * max_log_scale_delta
                        row.update({
                            "scale_log_delta_rms": float(
                                scale_delta.square().mean().sqrt()
                            ),
                            "scale_multiplier_p05": float(
                                sampled_quantile(scale_delta.exp(), 0.05)
                            ),
                            "scale_multiplier_p95": float(
                                sampled_quantile(scale_delta.exp(), 0.95)
                            ),
                        })
                    records.append(row)
                    print(json.dumps(row), flush=True)
            del batch

    with torch.no_grad():
        refined_static, opacity_delta, scale_delta = current_static()
    save_gaussian_payload(refined_dir / "final_static_128d.pt", refined_static)
    for name in ("final_dynamic_128d.pt", "global_dynamic_meta.pt"):
        target = refined_dir / name
        if not target.exists():
            os.symlink((source_dir / name).resolve(), target)
    torch.save(
        {
            "opacity_logit_delta": opacity_delta.detach().cpu(),
            "scale_log_delta": (
                scale_delta.detach().cpu() if scale_delta is not None else None
            ),
        },
        output_dir / "static_opacity_scale_delta.pt",
    )
    manifest = {
        "status": "completed",
        "scene": args.scene,
        "source_payload": str(source_dir),
        "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": str(Path(args.refiner_state).resolve()),
        "optimized": (
            "static opacity logits and bounded isotropic scale"
            if args.optimize_scale else "static opacity logits"
        ),
        "frozen": "static XYZ/rotation/128-D feature, dynamic map, boxes, cameras and network",
        "epochs": args.epochs,
        "opacity_learning_rate": args.opacity_learning_rate,
        "scale_learning_rate": args.scale_learning_rate,
        "max_opacity_logit_delta": args.max_opacity_logit_delta,
        "optimize_scale": args.optimize_scale,
        "max_scale_factor": args.max_scale_factor,
        "max_scale_m": args.max_scale_m,
        "anchor_weight": args.anchor_weight,
        "training_frame_stride": args.frame_stride,
        "training_frame_count": len(frames),
        "camera_batch_size": args.camera_batch_size,
        "static_count": int(static.count),
        "dynamic_count": int(dynamic.count),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

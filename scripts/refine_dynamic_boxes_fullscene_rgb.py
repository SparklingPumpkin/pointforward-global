"""Refine per-frame dynamic box translations using the final full-scene RGB loss."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.models import InstanceBoxes, TargetViews, box_dynamic_masks
from scripts.deliver_continuous_scene import boxes_payload, save_gaussian_payload
from scripts.eval import build_model
from scripts.refine_dynamic_boxes_rgb import rebase_payload
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def compact_render_boxes(
    boxes: InstanceBoxes,
    frame_slot: int,
    translations_world: torch.Tensor,
) -> InstanceBoxes:
    """Keep original canonical boxes and place corrected target poses in slot 1."""
    rows = torch.arange(boxes.num_instances, device=boxes.device)
    canonical_slot = boxes.canonical_index()
    return InstanceBoxes(
        centers=torch.stack((
            boxes.centers[rows, canonical_slot],
            boxes.centers[:, frame_slot] + translations_world[:, frame_slot],
        ), dim=1),
        dims=torch.stack((
            boxes.dims[rows, canonical_slot], boxes.dims[:, frame_slot],
        ), dim=1),
        rots=torch.stack((
            boxes.rots[rows, canonical_slot], boxes.rots[:, frame_slot],
        ), dim=1),
        times=torch.stack((
            boxes.times[rows, canonical_slot], boxes.times[:, frame_slot],
        ), dim=1),
        valid=torch.stack((
            torch.ones(boxes.num_instances, dtype=torch.bool, device=boxes.device),
            boxes.valid[:, frame_slot],
        ), dim=1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Optional deterministic frame cap for smoke testing.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-translation-m", type=float, default=0.30)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-corrections",
        help="Optional frame_translation_corrections.pt from an earlier run.",
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
    frames = (
        all_frames if args.max_frames is None else all_frames[:args.max_frames]
    )
    if not frames:
        parser.error("--max-frames must be positive")
    normalized_times = dataset._normalised_times(all_frames)
    all_slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    slots = all_slots
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    gaussians = combine(static, dynamic)
    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes_cpu = load_boxes(meta["boxes"], torch.device("cpu"))
    evaluation_cpu = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), torch.device("cpu"))
    assert boxes_cpu is not None and evaluation_cpu is not None
    boxes = InstanceBoxes(
        centers=boxes_cpu.centers.to(device), dims=boxes_cpu.dims.to(device),
        rots=boxes_cpu.rots.to(device), times=boxes_cpu.times.to(device),
        valid=boxes_cpu.valid.to(device),
    )
    evaluation_boxes = InstanceBoxes(
        centers=evaluation_cpu.centers.to(device), dims=evaluation_cpu.dims.to(device),
        rots=evaluation_cpu.rots.to(device), times=evaluation_cpu.times.to(device),
        valid=evaluation_cpu.valid.to(device),
    )
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)

    raw_translation = torch.nn.Parameter(torch.zeros_like(boxes.centers))
    if args.initial_corrections:
        initial = torch.load(
            args.initial_corrections, map_location="cpu", weights_only=True
        )["translations_world_m"].to(device)
        if initial.shape != raw_translation.shape:
            raise ValueError(
                f"Initial corrections have {tuple(initial.shape)}, expected "
                f"{tuple(raw_translation.shape)}"
            )
        ratio = (initial / args.max_translation_m).clamp(-0.999, 0.999)
        raw_translation.data.copy_(torch.atanh(ratio))
    optimizer = torch.optim.Adam((raw_translation,), lr=args.learning_rate)
    valid_float = boxes.valid.unsqueeze(-1).float()
    adjacent_valid = boxes.valid[:, 1:] & boxes.valid[:, :-1]
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
            view_losses = []
            valid_views = 0
            for camera_index in range(len(dataset.cameras)):
                translations = (
                    torch.tanh(raw_translation) * args.max_translation_m * valid_float
                )
                render_boxes = compact_render_boxes(boxes, slot, translations)
                targets = TargetViews(
                    intrinsics=batch["intrinsics"][camera_index:camera_index + 1].to(device),
                    cam_to_world=batch["cam_to_world"][camera_index:camera_index + 1].to(device),
                    times=normalized_times[slot:slot + 1].to(device),
                    frame_index=torch.ones(1, dtype=torch.long, device=device),
                )
                rendered = model.render(
                    gaussians, render_boxes, targets,
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
                view_losses.append(float(mse.detach()))
                valid_views += 1

            translations = torch.tanh(raw_translation) * args.max_translation_m * valid_float
            current_valid = boxes.valid[:, slot]
            anchor = (
                translations[current_valid, slot].square().mean()
                / (args.max_translation_m ** 2)
                if bool(current_valid.any()) else translations.sum() * 0.0
            )
            differences = translations[:, 1:] - translations[:, :-1]
            smoothness = (
                differences[adjacent_valid].square().mean()
                / (args.max_translation_m ** 2)
                if bool(adjacent_valid.any()) else translations.sum() * 0.0
            )
            regularization = (
                args.anchor_weight * anchor
                + args.smoothness_weight * smoothness / len(frames)
            )
            regularization.backward()
            torch.nan_to_num_(raw_translation.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw_translation,), 1.0)
            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(raw_translation, nan=0.0, posinf=0.0, neginf=0.0)

            step = epoch * len(frames) + order_index + 1
            if step % 10 == 0 or step == 1:
                norms = translations.norm(dim=-1)[boxes.valid]
                row = {
                    "step": step, "epoch": epoch + 1, "frame": frame,
                    "valid_views": valid_views,
                    "bbox_rgb_mse": float(np.mean(view_losses)) if view_losses else None,
                    "anchor": float(anchor.detach()),
                    "smoothness": float(smoothness.detach()),
                    "grad_norm": float(grad_norm),
                    "mean_translation_m": float(norms.mean()),
                    "p95_translation_m": float(torch.quantile(norms, 0.95)),
                    "max_translation_m": float(norms.max()),
                    "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                records.append(row)
                print(json.dumps(row), flush=True)

    with torch.no_grad():
        translations = torch.tanh(raw_translation) * args.max_translation_m * valid_float
        refined_boxes = InstanceBoxes(
            centers=boxes.centers + translations,
            dims=boxes.dims.clone(), rots=boxes.rots.clone(),
            times=boxes.times.clone(), valid=boxes.valid.clone(),
        )
        refined_dynamic = rebase_payload(dynamic, boxes, refined_boxes)

    static_target = refined_dir / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source_dir / "final_static_128d.pt").resolve(), static_target)
    save_gaussian_payload(refined_dir / "final_dynamic_128d.pt", refined_dynamic)
    refined_meta = dict(meta)
    refined_meta["boxes"] = boxes_payload(refined_boxes)
    refined_meta["evaluation_boxes"] = meta.get("evaluation_boxes", meta["boxes"])
    torch.save(refined_meta, refined_dir / "global_dynamic_meta.pt")
    torch.save(
        {"translations_world_m": translations.cpu(), "valid": boxes.valid.cpu()},
        output_dir / "frame_translation_corrections.pt",
    )

    norms = translations.norm(dim=-1)[boxes.valid]
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "optimized": "per-track per-frame world translation only",
        "frozen": "all Gaussian attributes, box rotation/dimensions, cameras, network and UNet weights",
        "supervision": "full final scene 128D+UNet RGB MSE inside original evaluation bbox union",
        "evaluation_boxes": "original unmodified boxes",
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "max_translation_component_m": args.max_translation_m,
        "anchor_weight": args.anchor_weight,
        "smoothness_weight": args.smoothness_weight,
        "initial_corrections": (
            None if args.initial_corrections is None
            else str(Path(args.initial_corrections).resolve())
        ),
        "translation_norm_m": {
            "mean": float(norms.mean()),
            "p95": float(torch.quantile(norms, 0.95)),
            "max": float(norms.max()),
        },
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

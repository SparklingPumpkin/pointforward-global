"""Target-domain RGB refinement for PointForward's final feature refiner.

The global Gaussian payload, dynamic boxes and the rest of PointForward stay
frozen.  Only a deliberately small refiner scope (the RGB head, or the final
up block plus head) is optimized with full-scene RGB supervision.  The result
is saved as an optional refiner-state overlay, leaving the original checkpoint
unchanged and fully recoverable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.models import TargetViews, box_dynamic_masks, render_features
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def load_refiner_state(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("refiner", payload)
    model.refiner.load_state_dict(state, strict=True)


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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument(
        "--full-frame-weight", type=float, default=1.0,
        help="Weight for the original full-frame objective; 1 preserves legacy behavior.",
    )
    parser.add_argument(
        "--dynamic-bbox-weight", type=float, default=0.0,
        help=(
            "Add this weight times log-MSE inside the evaluation dynamic "
            "boxes. Zero preserves the original full-frame-only objective."
        ),
    )
    parser.add_argument(
        "--dynamic-ownership-weight", type=float, default=0.0,
        help=(
            "Add this weight times normalized RGB MSE weighted by the visible "
            "dynamic ownership contribution from the same gsplat rasterization. "
            "Unlike dynamic-bbox-weight, this does not use geometry boxes as the "
            "training mask."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss-mode",
        choices=("mse", "log-mse", "frame-log-mse"),
        default="mse",
        help=(
            "log-mse targets mean per-view PSNR; frame-log-mse targets the "
            "reported per-frame PSNR after combining all cameras."
        ),
    )
    parser.add_argument(
        "--scope", choices=("head", "up-head", "full"), default="head"
    )
    parser.add_argument("--refiner-state-in", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--camera-batch-size", type=int, default=1,
        help="Render this many same-frame cameras together; 1 preserves legacy execution.",
    )
    parser.add_argument(
        "--camera-ids", type=int, nargs="+", default=None,
        help=(
            "Optional dataset camera IDs to train on. By default all configured "
            "cameras are used, preserving legacy behavior."
        ),
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="Uniformly subsample RGB-supervision frames; 1 keeps every frame.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source_dir = Path(args.payload_dir).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "refiner_state.pt"

    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, args.split)
    if args.camera_ids is None:
        camera_indices = list(range(len(dataset.cameras)))
        selected_camera_ids = list(dataset.cameras)
    else:
        unknown = [camera for camera in args.camera_ids if camera not in dataset.cameras]
        if unknown:
            raise ValueError(
                f"Unknown --camera-ids {unknown}; configured cameras are {dataset.cameras}"
            )
        if len(set(args.camera_ids)) != len(args.camera_ids):
            raise ValueError("--camera-ids must not contain duplicates")
        selected_camera_ids = list(args.camera_ids)
        camera_indices = [dataset.cameras.index(camera) for camera in selected_camera_ids]
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    all_frames = list(metadata["frames"])
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.camera_batch_size < 1:
        raise ValueError("--camera-batch-size must be >= 1")
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
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None

    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    if args.refiner_state_in:
        load_refiner_state(model, Path(args.refiner_state_in).resolve())
    modules = [model.refiner.head]
    if args.scope == "up-head":
        modules.insert(0, model.refiner.ups[-1])
    elif args.scope == "full":
        modules = [model.refiner]
    trainable = []
    for module in modules:
        module.requires_grad_(True)
        trainable.extend(module.parameters())
    initial = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()
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
            if args.camera_ids is not None:
                for key in ("intrinsics", "cam_to_world", "images"):
                    batch[key] = batch[key][camera_indices]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            camera_count = len(selected_camera_ids)
            camera_batch_size = min(args.camera_batch_size, camera_count)
            frame_mse_sum = None
            dynamic_mse_value = None
            # With all same-frame cameras in one graph, log(sum MSE) is the
            # exact legacy frame-log objective and needs no extra measurement
            # renders. Smaller chunks retain the detached-denominator path.
            all_cameras_in_one_graph = camera_batch_size == camera_count
            if args.loss_mode == "frame-log-mse" and not all_cameras_in_one_graph:
                measured = []
                with torch.no_grad():
                    for camera_start in range(0, camera_count, camera_batch_size):
                        camera_end = min(camera_start + camera_batch_size, camera_count)
                        view_count = camera_end - camera_start
                        targets = TargetViews(
                            intrinsics=batch["intrinsics"][camera_start:camera_end].to(device),
                            cam_to_world=batch["cam_to_world"][camera_start:camera_end].to(device),
                            times=normalized_times[slot:slot + 1].to(device).expand(view_count),
                            frame_index=torch.full((view_count,), slot, dtype=torch.long, device=device),
                        )
                        rendered = model.render(
                            gaussians, boxes, targets, height=height, width=width,
                            render_depth=False,
                        )
                        truth = batch["images"][camera_start:camera_end].to(device)
                        measured.extend((rendered["rgb"] - truth).square().flatten(1).mean(1))
                        del targets, rendered
                frame_mse_sum = torch.stack(measured).sum().clamp_min(1e-8)
            for camera_start in range(0, camera_count, camera_batch_size):
                camera_end = min(camera_start + camera_batch_size, camera_count)
                view_count = camera_end - camera_start
                targets = TargetViews(
                    intrinsics=batch["intrinsics"][camera_start:camera_end].to(device),
                    cam_to_world=batch["cam_to_world"][camera_start:camera_end].to(device),
                    times=normalized_times[slot:slot + 1].to(device).expand(view_count),
                    frame_index=torch.full((view_count,), slot, dtype=torch.long, device=device),
                )
                if args.dynamic_ownership_weight > 0.0:
                    # Render the ownership indicator as an extra feature in the
                    # same gsplat call.  This preserves identical depth sorting
                    # and makes the loss weight differentiable with respect to
                    # neither the frozen scene nor the refiner parameters.
                    posed = model.pose_gaussians(
                        gaussians, boxes, frame, targets.times[0]
                    )
                    ownership = (gaussians.instance_id >= 0).to(
                        posed["colors"].dtype
                    ).unsqueeze(1)
                    colors129 = torch.cat((posed["colors"], ownership), dim=1)
                    feature129, alpha129, _ = render_features(
                        posed["means"], posed["quats"], posed["scales"],
                        posed["opacities"], colors129, targets.intrinsics,
                        torch.linalg.inv(targets.cam_to_world), height, width,
                        render_depth=False,
                    )
                    rendered = {
                        "rgb": model.refiner(feature129[:, :128], alpha129),
                        "dynamic_contribution": feature129[:, 128:129],
                    }
                else:
                    rendered = model.render(
                        gaussians, boxes, targets, height=height, width=width,
                        render_depth=False,
                    )
                truth = batch["images"][camera_start:camera_end].to(device)
                per_view_mse = (rendered["rgb"] - truth).square().flatten(1).mean(1)
                if args.loss_mode == "frame-log-mse":
                    if all_cameras_in_one_graph:
                        objective = torch.log(per_view_mse.sum().clamp_min(1e-8))
                    else:
                        # d log(sum_i mse_i) / d theta = sum_i d mse_i /
                        # sum_i mse_i; the detached denominator is exact.
                        objective = per_view_mse.sum() / frame_mse_sum
                elif args.loss_mode == "log-mse":
                    objective = torch.log(per_view_mse.clamp_min(1e-8)).sum() / camera_count
                else:
                    objective = per_view_mse.sum() / camera_count
                objective = args.full_frame_weight * objective
                if args.dynamic_bbox_weight > 0.0:
                    dynamic_mask = box_dynamic_masks(
                        evaluation_boxes,
                        torch.full(
                            (view_count,), slot, dtype=torch.long, device=device
                        ),
                        targets.intrinsics,
                        torch.linalg.inv(targets.cam_to_world),
                        height,
                        width,
                    ).unsqueeze(1)
                    if bool(dynamic_mask.any()):
                        dynamic_mse = (
                            (rendered["rgb"] - truth).square() * dynamic_mask
                        ).sum() / (dynamic_mask.sum() * 3.0).clamp_min(1.0)
                        objective = objective + args.dynamic_bbox_weight * torch.log(
                            dynamic_mse.clamp_min(1e-8)
                        )
                        dynamic_mse_value = float(dynamic_mse.detach())
                if args.dynamic_ownership_weight > 0.0:
                    dynamic_contribution = rendered["dynamic_contribution"].clamp(
                        min=0.0, max=1.0
                    ).detach()
                    ownership_mse = (
                        (rendered["rgb"] - truth).square() * dynamic_contribution
                    ).sum() / (dynamic_contribution.sum() * 3.0).clamp_min(1e-8)
                    objective = objective + args.dynamic_ownership_weight * ownership_mse
                objective.backward()
                losses.extend(float(value) for value in per_view_mse.detach())
                del targets, rendered
            anchor = torch.stack([
                (parameter - start).square().mean()
                for parameter, start in zip(trainable, initial)
            ]).mean()
            (args.anchor_weight * anchor).backward()
            for parameter in trainable:
                torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if global_step % 10 == 0 or global_step == 1:
                with torch.no_grad():
                    delta_rms = torch.stack([
                        (parameter - start).square().mean()
                        for parameter, start in zip(trainable, initial)
                    ]).mean().sqrt()
                    record = {
                        "step": global_step, "epoch": epoch + 1, "frame": frame,
                        "rgb_mse": float(np.mean(losses)),
                        "dynamic_bbox_rgb_mse": dynamic_mse_value,
                        "anchor": float(anchor.detach()),
                        "grad_norm": float(grad_norm),
                        "delta_rms": float(delta_rms),
                        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
            del batch

    torch.save(
        {
            "refiner": {k: value.detach().cpu() for k, value in model.refiner.state_dict().items()},
            "scope": args.scope,
            "source_checkpoint": str(Path(args.ckpt).resolve()),
            "source_refiner_state": (
                str(Path(args.refiner_state_in).resolve()) if args.refiner_state_in else None
            ),
        },
        state_path,
    )
    manifest = {
        "status": "completed", "scene": args.scene,
        "payload_dir": str(source_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state_in": args.refiner_state_in,
        "refiner_state": str(state_path),
        "scope": args.scope,
        "optimized": "PointForward feature refiner target-domain overlay",
        "frozen": "all Gaussian fields, dynamic boxes, cameras and model outside selected refiner scope",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "loss_mode": args.loss_mode,
        "anchor_weight": args.anchor_weight,
        "full_frame_weight": args.full_frame_weight,
        "dynamic_bbox_weight": args.dynamic_bbox_weight,
        "weight_decay": args.weight_decay,
        "training_frame_stride": args.frame_stride,
        "training_frame_count": len(frames),
        "training_frames": [int(frame) for frame in frames],
        "camera_ids": selected_camera_ids,
        "camera_batch_size": args.camera_batch_size,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "static_count": int(static.count), "dynamic_count": int(dynamic.count),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

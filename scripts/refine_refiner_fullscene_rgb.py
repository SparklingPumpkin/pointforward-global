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

from pointforward.models import TargetViews
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
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    all_frames = list(metadata["frames"])
    frames = all_frames if args.max_frames is None else all_frames[:args.max_frames]
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
            optimizer.zero_grad(set_to_none=True)
            losses = []
            frame_mse_sum = None
            if args.loss_mode == "frame-log-mse":
                measured = []
                with torch.no_grad():
                    for camera_index in range(len(dataset.cameras)):
                        targets = TargetViews(
                            intrinsics=batch["intrinsics"][camera_index:camera_index + 1].to(device),
                            cam_to_world=batch["cam_to_world"][camera_index:camera_index + 1].to(device),
                            times=normalized_times[slot:slot + 1].to(device),
                            frame_index=torch.full((1,), slot, dtype=torch.long, device=device),
                        )
                        rendered = model.render(
                            gaussians, boxes, targets, height=height, width=width,
                            render_depth=False,
                        )
                        truth = batch["images"][camera_index:camera_index + 1].to(device)
                        measured.append((rendered["rgb"] - truth).square().mean())
                        del targets, rendered
                frame_mse_sum = torch.stack(measured).sum().clamp_min(1e-8)
            for camera_index in range(len(dataset.cameras)):
                targets = TargetViews(
                    intrinsics=batch["intrinsics"][camera_index:camera_index + 1].to(device),
                    cam_to_world=batch["cam_to_world"][camera_index:camera_index + 1].to(device),
                    times=normalized_times[slot:slot + 1].to(device),
                    frame_index=torch.full((1,), slot, dtype=torch.long, device=device),
                )
                rendered = model.render(
                    gaussians, boxes, targets, height=height, width=width,
                    render_depth=False,
                )
                truth = batch["images"][camera_index:camera_index + 1].to(device)
                mse = (rendered["rgb"] - truth).square().mean()
                if args.loss_mode == "frame-log-mse":
                    # d log(sum_i mse_i) / d theta =
                    # sum_i d mse_i / d theta / sum_i mse_i.  The detached
                    # denominator comes from an immediately preceding no-grad
                    # pass, allowing exact frame-level weighting without
                    # retaining three renderer graphs simultaneously.
                    objective = mse / frame_mse_sum
                elif args.loss_mode == "log-mse":
                    objective = torch.log(mse.clamp_min(1e-8))
                else:
                    objective = mse
                if args.loss_mode == "frame-log-mse":
                    objective.backward()
                else:
                    (objective / len(dataset.cameras)).backward()
                losses.append(float(mse.detach()))
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
        "weight_decay": args.weight_decay,
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

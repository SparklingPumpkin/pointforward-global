"""Learn bounded per-frame/per-track opacity corrections for dynamic Gaussians.

Geometry, features, boxes, cameras and the pretrained refiner remain frozen.
Only the temporal visibility of each tracked vehicle is adapted from RGB.  The
state is small (frame x track scalars) and is applied after PointForward's
normal temporal opacity attenuation.
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


def load_refiner(model, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(payload.get("refiner", payload), strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--camera-refiner-states", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-log-scale", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--dynamic-bbox-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.payload_dir).resolve()
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, "val")
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    frames = [int(value) for value in metadata["frames"]]
    slots = {frame: index for index, frame in enumerate(frames)}
    normalized_times = dataset._normalised_times(frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    height, width = dataset.resolution

    static = load_payload(source / "final_static_128d.pt", device)
    dynamic = load_payload(source / "final_dynamic_128d.pt", device)
    gaussians = combine(static, dynamic)
    meta = torch.load(source / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None
    dynamic_mask = gaussians.instance_id >= 0
    track_count = boxes.num_instances

    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    refiner_states = [Path(value).resolve() for value in args.camera_refiner_states]
    if len(refiner_states) != len(dataset.cameras):
        raise ValueError("one refiner state is required per configured camera")

    raw = torch.nn.Parameter(torch.zeros((len(frames), track_count), device=device))
    optimizer = torch.optim.Adam((raw,), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    records = []
    started = time.perf_counter()

    for epoch in range(args.epochs):
        order = torch.randperm(len(frames), generator=generator).tolist()
        for order_index in order:
            frame = frames[order_index]
            slot = slots[frame]
            batch = dataset._collect(
                metadata, [frame], slots, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for camera_index, state_path in enumerate(refiner_states):
                track_scale = torch.exp(torch.tanh(raw[slot]) * args.max_log_scale)
                load_refiner(model, state_path)
                targets = TargetViews(
                    intrinsics=batch["intrinsics"][camera_index:camera_index + 1].to(device),
                    cam_to_world=batch["cam_to_world"][camera_index:camera_index + 1].to(device),
                    times=normalized_times[slot:slot + 1].to(device),
                    frame_index=torch.tensor([slot], dtype=torch.long, device=device),
                )
                posed = model.pose_gaussians(
                    gaussians, boxes, slot, normalized_times[slot].to(device)
                )
                opacity = posed["opacities"].clone()
                opacity[dynamic_mask] = opacity[dynamic_mask] * track_scale[
                    gaussians.instance_id[dynamic_mask]
                ]
                rendered_features, alpha, _ = render_features(
                    means=posed["means"], quats=posed["quats"],
                    scales=posed["scales"], opacities=opacity,
                    colors=posed["colors"], intrinsics=targets.intrinsics,
                    world_to_cam=torch.linalg.inv(targets.cam_to_world),
                    height=height, width=width, render_depth=False,
                )
                rendered = model.refiner(rendered_features, alpha)
                truth = batch["images"][camera_index:camera_index + 1].to(device)
                squared = (rendered - truth).square()
                full_mse = squared.mean()
                mask = box_dynamic_masks(
                    evaluation_boxes, torch.tensor([slot], device=device),
                    targets.intrinsics, torch.linalg.inv(targets.cam_to_world),
                    height, width,
                ).unsqueeze(1)
                dynamic_mse = (squared * mask).sum() / (mask.sum() * 3.0).clamp_min(1.0)
                objective = (full_mse + args.dynamic_bbox_weight * dynamic_mse) / len(refiner_states)
                objective.backward()
                losses.append(float(dynamic_mse.detach()))
                del targets, posed, opacity, rendered_features, alpha, rendered
            anchor_raw = torch.tanh(raw[slot])
            anchor = anchor_raw.square().mean()
            (args.anchor_weight * anchor).backward()
            torch.nan_to_num_(raw.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw,), 1.0)
            optimizer.step()
            records.append({
                "epoch": epoch + 1, "frame": frame,
                "dynamic_bbox_mse": float(np.mean(losses)),
                "track_scale_mean": float(track_scale.mean().detach()),
                "track_scale_p95": float(torch.quantile(track_scale.detach(), 0.95)),
                "grad_norm": float(grad_norm),
            })
            print(json.dumps(records[-1]), flush=True)
            del batch

    state_path = out / "dynamic_frame_opacity_state.pt"
    with torch.no_grad():
        scales = torch.exp(torch.tanh(raw) * args.max_log_scale).cpu()
    torch.save({
        "frame_track_opacity_scale": scales,
        "frames": frames,
        "track_count": track_count,
        "source_payload": str(source),
    }, state_path)
    manifest = {
        "status": "completed", "scene": args.scene,
        "source_payload": str(source), "state": str(state_path),
        "optimized": "per-frame per-track dynamic opacity multiplier",
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "max_log_scale": args.max_log_scale,
        "dynamic_bbox_weight": args.dynamic_bbox_weight,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Probe per-source-frame dynamic Gaussian mean deformation with RGB supervision.

Only dynamic Gaussian means are corrected, independently for a registered set of
source frames.  All colors/features, covariance, opacity, boxes, cameras and
network weights remain frozen.  The correction is applied after the normal
box-based pose, so it represents residual non-rigid/canonical misalignment rather
than another rigid box translation.
"""

from __future__ import annotations

import copy
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.models import TargetViews, box_dynamic_masks, render_features
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--frames", required=True, help="Comma-separated source frame IDs")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-xyz-delta-m", type=float, default=0.15)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-refiner-states", nargs="+")
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
    all_frames = [int(v) for v in metadata["frames"]]
    frames = [int(v) for v in args.frames.split(",") if v.strip()]
    unknown = sorted(set(frames) - set(all_frames))
    if unknown:
        parser.error(f"--frames contains unknown source frames: {unknown}")
    if not frames:
        parser.error("--frames must not be empty")
    if args.camera_index < 0 or args.camera_index >= len(dataset.cameras):
        parser.error("--camera-index is out of range")
    normalized_times = dataset._normalised_times(all_frames)
    slots = {int(frame): slot for slot, frame in enumerate(all_frames)}
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source_dir / "final_static_128d.pt", device)
    dynamic = load_payload(source_dir / "final_dynamic_128d.pt", device)
    full = combine(static, dynamic)
    dynamic_count = dynamic.means.shape[0]
    dynamic_mask = full.instance_id >= 0
    dynamic_indices = torch.nonzero(dynamic_mask, as_tuple=False).flatten()
    if dynamic_indices.numel() != dynamic_count:
        raise RuntimeError("combined payload dynamic ordering/count is inconsistent")
    meta = torch.load(source_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    assert boxes is not None and evaluation_boxes is not None
    model = build_model(config, device, args.ckpt).eval().requires_grad_(False)
    camera_refiners = None
    if args.camera_refiner_states:
        if len(args.camera_refiner_states) != len(dataset.cameras):
            parser.error("--camera-refiner-states must provide one state per camera")
        camera_refiners = {}
        for camera_id, state_path in zip(dataset.cameras, args.camera_refiner_states):
            refiner = copy.deepcopy(model.refiner).to(device).eval()
            state = torch.load(Path(state_path).resolve(), map_location="cpu", weights_only=True)
            refiner.load_state_dict(state.get("refiner", state), strict=True)
            camera_refiners[camera_id] = refiner

    # One bounded correction tensor per registered source frame and dynamic point.
    raw_delta = torch.nn.Parameter(torch.zeros((len(frames), dynamic_count, 3), device=device))
    optimizer = torch.optim.Adam((raw_delta,), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []
    started = time.perf_counter()

    def bounded_delta(frame_index: int) -> torch.Tensor:
        frame_slot = frames.index(frame_index)
        return torch.tanh(raw_delta[frame_slot]) * args.max_xyz_delta_m

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
            if camera_refiners is not None:
                model.refiner = camera_refiners[dataset.cameras[args.camera_index]]
            intrinsics = batch["intrinsics"][args.camera_index:args.camera_index + 1].to(device)
            cam_to_world = batch["cam_to_world"][args.camera_index:args.camera_index + 1].to(device)
            targets = TargetViews(
                intrinsics=intrinsics,
                cam_to_world=cam_to_world,
                times=normalized_times[slot:slot + 1].to(device),
                frame_index=torch.tensor([slot], dtype=torch.long, device=device),
            )
            posed = model.pose_gaussians(full, boxes, slot, normalized_times[slot].to(device))
            # Freeze the baseline ownership support for supervision.  This
            # avoids the known box-background RGB gradient confound: the probe
            # may improve only pixels already owned by dynamic Gaussians in the
            # RF-v1-c baseline, while support itself is evaluated separately.
            with torch.no_grad():
                baseline_ownership = dynamic_mask.to(posed["colors"].dtype).unsqueeze(1)
                baseline_colors129 = torch.cat((posed["colors"], baseline_ownership), dim=1)
                baseline_feature129, _, _ = render_features(
                    posed["means"], posed["quats"], posed["scales"], posed["opacities"],
                    baseline_colors129, intrinsics,
                    torch.linalg.inv(cam_to_world), height, width,
                )
                baseline_support = baseline_feature129[:, 128:129] >= 0.05
            correction = bounded_delta(frame)
            posed_means = posed["means"].clone()
            posed_means[dynamic_indices] = posed_means[dynamic_indices] + correction
            feature, alpha, _ = render_features(
                posed_means, posed["quats"], posed["scales"], posed["opacities"],
                posed["colors"], targets.intrinsics,
                torch.linalg.inv(targets.cam_to_world), height, width,
            )
            rendered = model.refiner(feature, alpha)
            box_mask = box_dynamic_masks(
                evaluation_boxes, torch.tensor([slot], dtype=torch.long, device=device),
                intrinsics, torch.linalg.inv(cam_to_world), height, width,
            ).unsqueeze(1)
            mask = baseline_support & box_mask
            if not bool(mask.any()):
                continue
            truth = batch["images"][args.camera_index:args.camera_index + 1].to(device)
            mse = ((rendered - truth).square() * mask).sum() / (mask.sum() * 3.0).clamp_min(1.0)
            anchor = (correction / args.max_xyz_delta_m).square().mean()
            loss = mse + args.anchor_weight * anchor
            loss.backward()
            torch.nan_to_num_(raw_delta.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_((raw_delta,), 1.0)
            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(raw_delta, nan=0.0, posinf=0.0, neginf=0.0)
            step = epoch * len(frames) + order_index + 1
            row = {
                "step": step, "epoch": epoch + 1, "frame": frame,
                "valid_views": 1, "bbox_rgb_mse": float(mse.detach()),
                "anchor": float(anchor.detach()), "grad_norm": float(grad_norm),
                "mean_xyz_delta_m": float(correction.detach().norm(dim=-1).mean()),
                "p95_xyz_delta_m": float(torch.quantile(correction.detach().norm(dim=-1), 0.95)),
                "max_xyz_delta_m": float(correction.detach().norm(dim=-1).max()),
                "baseline_dynamic_support_pixels": int(baseline_support.sum()),
                "loss_mask": "baseline_dynamic_support_intersection_evaluation_box",
                "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            del batch, targets, posed, posed_means, feature, alpha, rendered, mask, box_mask, truth, loss

    with torch.no_grad():
        corrections = torch.tanh(raw_delta) * args.max_xyz_delta_m
    static_target = refined_dir / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source_dir / "final_static_128d.pt").resolve(), static_target)
    dynamic_target = refined_dir / "final_dynamic_128d.pt"
    if not dynamic_target.exists():
        os.symlink((source_dir / "final_dynamic_128d.pt").resolve(), dynamic_target)
    meta_target = refined_dir / "global_dynamic_meta.pt"
    if not meta_target.exists():
        os.symlink((source_dir / "global_dynamic_meta.pt").resolve(), meta_target)
    torch.save({
        "deformation_mode": "per-frame-dynamic-mean-world-after-box-pose",
        "source_frames": frames,
        "dynamic_xyz_delta_world_m": corrections.cpu(),
        "max_xyz_delta_component_m": args.max_xyz_delta_m,
    }, output_dir / "temporal_mean_deformation.pt")
    manifest = {
        "status": "completed", "scene": args.scene, "source_frames": frames,
        "source_payload": str(source_dir), "refined_payload": str(refined_dir),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "optimized": "per-source-frame dynamic Gaussian world means after box pose",
        "frozen": "colors/features, scales, rotations, opacities, boxes, cameras, network and UNet weights",
        "training_camera_index": args.camera_index,
        "camera_refiner_states": args.camera_refiner_states,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "max_xyz_delta_component_m": args.max_xyz_delta_m,
        "anchor_weight": args.anchor_weight,
        "dynamic_gaussian_count": int(dynamic_count),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

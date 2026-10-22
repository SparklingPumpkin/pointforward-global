"""Run a controlled overlapping-window audit for PointForward.

The audit compares one window replayed with the same and a different query
sampling seed against an overlapping window rendering the exact same target
frames. All windows share one world coordinate anchor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from pointforward.losses import psnr, ssim
from pointforward.models import Gaussians, InstanceBoxes, SceneInput, TargetViews
from scripts.eval import build_model
from scripts.train import build_dataset


def boxes_to(boxes: InstanceBoxes | None, device: torch.device) -> InstanceBoxes | None:
    if boxes is None:
        return None
    return InstanceBoxes(
        centers=boxes.centers.to(device),
        dims=boxes.dims.to(device),
        rots=boxes.rots.to(device),
        times=boxes.times.to(device),
        valid=boxes.valid.to(device),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: torch.Tensor) -> str:
    array = values.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def tensor_summary(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"shape": list(values.shape), "count": 0}
    return {
        "shape": list(values.shape),
        "count": int(flat.numel()),
        "min": float(flat.min().cpu()),
        "max": float(flat.max().cpu()),
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
    }


def compact_hash(values: torch.Tensor, max_rows: int = 8192) -> str:
    if values.ndim == 0 or values.shape[0] <= max_rows:
        return tensor_sha256(values)
    indices = torch.linspace(0, values.shape[0] - 1, max_rows, device=values.device).long()
    return tensor_sha256(torch.index_select(values, 0, indices))


def region_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict | None:
    pred = torch.as_tensor(pred)
    target = torch.as_tensor(target, device=pred.device)
    mask = torch.as_tensor(mask, device=pred.device).squeeze().bool()
    if pred.ndim != 3 or target.shape != pred.shape or mask.shape != pred.shape[-2:]:
        raise ValueError(
            f"expected matching CHW images and an HW mask, got {pred.shape}, {target.shape}, {mask.shape}"
        )
    pixels = int(mask.sum().item())
    if pixels == 0:
        return None
    batch_mask = mask.unsqueeze(0).unsqueeze(0)
    pred_batch = pred.unsqueeze(0)
    target_batch = target.unsqueeze(0)
    difference = (pred - target).abs()
    return {
        "pixels": pixels,
        "l1": float(difference.permute(1, 2, 0)[mask].mean()),
        "psnr": float(psnr(pred_batch, target_batch, batch_mask)),
        "ssim": float(ssim(pred_batch, target_batch, mask=mask.unsqueeze(0))),
    }


def pair_metrics(left: dict, right: dict, masks: dict[str, torch.Tensor]) -> dict:
    result = {"targets": []}
    for index, frame in enumerate(left["target_frames"]):
        row = {"target_frame": frame, "regions": {}}
        alpha_left = left["alpha"][index, 0].float()
        alpha_right = right["alpha"][index, 0].float()
        depth_left = left["render_depth"][index, 0].float()
        depth_right = right["render_depth"][index, 0].float()
        feature_difference = left["features"][index].float() - right["features"][index].float()
        for name, mask_stack in masks.items():
            region_mask = mask_stack[index].bool()
            metrics = region_metrics(left["rgb"][index], right["rgb"][index], region_mask)
            if metrics is not None:
                metrics["alpha_l1"] = float((alpha_left - alpha_right).abs()[region_mask].mean())
                metrics["feature_l1"] = float(
                    feature_difference.permute(1, 2, 0)[region_mask].abs().mean()
                )
                valid = (
                    region_mask
                    & torch.isfinite(depth_left)
                    & torch.isfinite(depth_right)
                    & (depth_left > 0)
                    & (depth_right > 0)
                    & (alpha_left >= 0.01)
                    & (alpha_right >= 0.01)
                )
                depth_values = (depth_left - depth_right)[valid]
                metrics["depth_mae_alpha_0.01_m"] = (
                    None if not depth_values.numel() else float(depth_values.abs().mean())
                )
            row["regions"][name] = metrics
        row["alpha_l1"] = float((alpha_left - alpha_right).abs().mean())
        row["alpha_max_abs"] = float((alpha_left - alpha_right).abs().max())
        for threshold in (0.01, 0.5):
            valid = (
                torch.isfinite(depth_left)
                & torch.isfinite(depth_right)
                & (depth_left > 0)
                & (depth_right > 0)
                & (alpha_left >= threshold)
                & (alpha_right >= threshold)
            )
            values = (depth_left - depth_right)[valid]
            row[f"depth_alpha_{threshold:g}"] = None if not values.numel() else {
                "pixels": int(values.numel()),
                "coverage": float(valid.float().mean()),
                "mae_m": float(values.abs().mean()),
                "rmse_m": float(torch.sqrt((values ** 2).mean())),
                "median_absolute_m": float(values.abs().median()),
            }
        row["feature_l1"] = float(feature_difference.abs().mean())
        row["feature_rmse"] = float(torch.sqrt((feature_difference ** 2).mean()))
        result["targets"].append(row)
    return result


def metrics_to_gt(run: dict, gt: torch.Tensor, lidar_depth: torch.Tensor, masks: dict) -> list[dict]:
    rows = []
    for index, frame in enumerate(run["target_frames"]):
        row = {"target_frame": frame, "regions": {}}
        for name, mask_stack in masks.items():
            row["regions"][name] = region_metrics(run["rgb"][index], gt[index], mask_stack[index])
        alpha = run["alpha"][index, 0].float()
        predicted_depth = run["render_depth"][index, 0].float()
        valid = masks["lidar"][index] & (alpha >= 0.01) & torch.isfinite(predicted_depth)
        values = predicted_depth[valid] - lidar_depth[index, 0][valid]
        row["lidar_depth"] = None if not values.numel() else {
            "pixels": int(values.numel()),
            "mae_m": float(values.abs().mean()),
            "rmse_m": float(torch.sqrt((values ** 2).mean())),
        }
        rows.append(row)
    return rows


def shared_depth_metrics(left: dict, right: dict) -> list[dict]:
    left_lookup = {key: index for index, key in enumerate(left["context_keys"])}
    right_lookup = {key: index for index, key in enumerate(right["context_keys"])}
    rows = []
    for key in sorted(set(left_lookup) & set(right_lookup)):
        a = left["context_depth"][left_lookup[key]].float()
        b = right["context_depth"][right_lookup[key]].float()
        difference = a - b
        rows.append({
            "frame": key[0],
            "camera": key[1],
            "mae_m": float(difference.abs().mean()),
            "rmse_m": float(torch.sqrt((difference ** 2).mean())),
            "max_absolute_m": float(difference.abs().max()),
        })
    return rows


def save_visual(prefix: Path, target_frames: list[int], gt: torch.Tensor, runs: dict) -> None:
    columns = ["GT", "A1", "A2 same seed", "A3 different seed", "B1 overlap", "|A1-B1|"]
    fig, axes = plt.subplots(len(target_frames), len(columns), figsize=(22, 4 * len(target_frames)))
    if len(target_frames) == 1:
        axes = axes[None, :]
    for row, frame in enumerate(target_frames):
        images = [
            gt[row], runs["A1"]["rgb"][row], runs["A2"]["rgb"][row],
            runs["A3"]["rgb"][row], runs["B1"]["rgb"][row],
        ]
        difference = (runs["A1"]["rgb"][row] - runs["B1"]["rgb"][row]).abs().mean(0)
        for column, image in enumerate(images):
            axes[row, column].imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
        diff_artist = axes[row, 5].imshow(difference.numpy(), cmap="inferno", vmin=0, vmax=0.25)
        for column, title in enumerate(columns):
            axes[row, column].set_title(f"frame {frame}: {title}")
            axes[row, column].set_axis_off()
    fig.colorbar(diff_artist, ax=axes[:, 5], label="Mean absolute RGB difference", shrink=0.8)
    fig.suptitle("Controlled overlapping-window pilot; identical target RGB/K/c2w")
    fig.savefig(str(prefix) + "_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scene", default="005")
    parser.add_argument("--window-a-start", type=int, default=0)
    parser.add_argument("--window-b-start", type=int, default=5)
    parser.add_argument(
        "--target-offsets",
        default="6,12,19",
        help="offsets in window A; every target must be inside window B and outside both context sets",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset = build_dataset(config, "val")
    scene_index = next((i for i, item in enumerate(dataset.scenes) if item["name"] == args.scene), None)
    if scene_index is None:
        raise ValueError(f"scene {args.scene} is not in validation split")
    metadata = dataset.scenes[scene_index]
    all_frames = metadata["frames"]
    length = dataset.clip_length
    window_a = all_frames[args.window_a_start:args.window_a_start + length]
    window_b = all_frames[args.window_b_start:args.window_b_start + length]
    if len(window_a) != length or len(window_b) != length:
        raise ValueError("Both audit windows must have full clip length")
    target_offsets = [int(value) for value in args.target_offsets.split(",")]
    target_frames = [window_a[offset] for offset in target_offsets]
    if not set(target_frames) <= set(window_b):
        raise ValueError("Every target frame must occur in both windows")
    context_a = {
        window_a[min(index * dataset.frame_stride, len(window_a) - 1)]
        for index in range(dataset.num_context)
    }
    context_b = {
        window_b[min(index * dataset.frame_stride, len(window_b) - 1)]
        for index in range(dataset.num_context)
    }
    leaked_targets = set(target_frames) & (context_a | context_b)
    if leaked_targets:
        raise ValueError(
            "Audit targets must be held out from both context sets; leaked frames: "
            f"{sorted(leaked_targets)}"
        )
    if args.camera not in dataset.cameras:
        raise ValueError("Camera is not enabled by the config")

    device = torch.device("cuda")
    world_from = np.linalg.inv(dataset._ego_pose(metadata, window_a[0]))
    resolution = dataset.resolution
    model = build_model(config, device, str(checkpoint_path))
    window_times = {
        "A": dataset._normalised_times(window_a),
        "B": dataset._normalised_times(window_b),
    }
    alternate_target_times = {
        "A": torch.tensor([window_times["A"][window_a.index(frame)] for frame in target_frames]),
        "B": torch.tensor([window_times["B"][window_b.index(frame)] for frame in target_frames]),
    }

    def build_window(window: list[int]) -> tuple[SceneInput, TargetViews, InstanceBoxes | None, dict, dict]:
        slot_of = {frame: index for index, frame in enumerate(window)}
        times = dataset._normalised_times(window)
        context_frames = sorted({
            window[min(index * dataset.frame_stride, len(window) - 1)]
            for index in range(dataset.num_context)
        })
        context = dataset._collect(
            metadata, context_frames, slot_of, times, resolution, world_from, lidar=dataset.load_lidar
        )
        target = dataset._collect(
            metadata, target_frames, slot_of, times, resolution, world_from, lidar=True
        )
        camera_index = dataset.cameras.index(args.camera)
        selected = torch.arange(camera_index, len(target_frames) * len(dataset.cameras), len(dataset.cameras))
        boxes_cpu = dataset._build_boxes(metadata, window, world_from)
        boxes = boxes_to(boxes_cpu, device)
        scene = SceneInput(
            images=context["images"].to(device),
            intrinsics=context["intrinsics"].to(device),
            cam_to_world=context["cam_to_world"].to(device),
            times=context["times"].to(device),
            frame_index=context["frame_index"].to(device),
            boxes=boxes,
            sparse_depth=context.get("sparse_depth", None).to(device) if "sparse_depth" in context else None,
            sparse_mask=context.get("sparse_mask", None).to(device) if "sparse_mask" in context else None,
        )
        targets = TargetViews(
            intrinsics=target["intrinsics"][selected].to(device),
            cam_to_world=target["cam_to_world"][selected].to(device),
            times=target["times"][selected].to(device),
            frame_index=target["frame_index"][selected].to(device),
        )
        selected_target = {key: value[selected].cpu() for key, value in target.items()}
        metadata_out = {
            "window": list(map(int, window)),
            "context_frames": list(map(int, context_frames)),
            "context_keys": [(int(frame), int(camera)) for frame in context_frames for camera in dataset.cameras],
            "target_frames": list(map(int, target_frames)),
            "target_local_slots": [slot_of[frame] for frame in target_frames],
            "target_local_times": [float(times[slot_of[frame]]) for frame in target_frames],
            "box_instances": 0 if boxes_cpu is None else boxes_cpu.num_instances,
            "box_valid_count": 0 if boxes_cpu is None else int(boxes_cpu.valid.sum()),
            "box_canonical_slots": [] if boxes_cpu is None else boxes_cpu.canonical_index().tolist(),
        }
        return scene, targets, boxes, selected_target, metadata_out

    def run_once(
        label: str,
        window: list[int],
        seed: int,
        depth_override: torch.Tensor | None = None,
        measure_fresh_depth: bool = False,
        variants: bool = False,
        alternate_times: torch.Tensor | None = None,
    ) -> dict:
        seed_everything(seed)
        scene, targets, boxes, target, run_metadata = build_window(window)
        measured_depth = model.predict_depth(scene) if depth_override is None or measure_fresh_depth else None
        depth = measured_depth if depth_override is None else depth_override.to(device)
        if depth is None:
            raise RuntimeError("depth was neither predicted nor supplied")
        query_generator = torch.Generator(device=device).manual_seed(seed)
        sampled = model.build_queries(scene, depth, generator=query_generator)
        gaussians = model.decode(scene, depth, sampled)
        output = model.render(
            gaussians, boxes, targets, height=resolution[0], width=resolution[1], render_depth=True
        )
        variant_outputs = {}
        if variants:
            static_gaussians = Gaussians(
                means=gaussians.means,
                quats=gaussians.quats,
                scales=gaussians.scales,
                opacities=torch.where(
                    gaussians.instance_id < 0,
                    gaussians.opacities,
                    torch.zeros_like(gaussians.opacities),
                ),
                colors=gaussians.colors,
                sigma=gaussians.sigma,
                times=gaussians.times,
                instance_id=gaussians.instance_id,
                extras=gaussians.extras,
            )
            variant_outputs["static"] = model.render(
                static_gaussians, boxes, targets,
                height=resolution[0], width=resolution[1], render_depth=True,
            )
            if alternate_times is None:
                raise ValueError("alternate_times is required when variants are enabled")
            alternate_targets = TargetViews(
                intrinsics=targets.intrinsics,
                cam_to_world=targets.cam_to_world,
                times=alternate_times.to(device),
                frame_index=targets.frame_index,
            )
            variant_outputs["alternate_time"] = model.render(
                gaussians, boxes, alternate_targets,
                height=resolution[0], width=resolution[1], render_depth=True,
            )
            boxfree_scene = SceneInput(
                images=scene.images,
                intrinsics=scene.intrinsics,
                cam_to_world=scene.cam_to_world,
                times=scene.times,
                frame_index=scene.frame_index,
                boxes=None,
                sparse_depth=scene.sparse_depth,
                sparse_mask=scene.sparse_mask,
            )
            boxfree_generator = torch.Generator(device=device).manual_seed(seed)
            boxfree_sampled = model.build_queries(
                boxfree_scene, depth, generator=boxfree_generator
            )
            boxfree_decoded = model.decode(boxfree_scene, depth, boxfree_sampled)
            variant_outputs["boxfree_redecode"] = model.render(
                boxfree_decoded, None, targets,
                height=resolution[0], width=resolution[1], render_depth=True,
            )
            variant_outputs["boxfree_redecode_alternate_time"] = model.render(
                boxfree_decoded, None, alternate_targets,
                height=resolution[0], width=resolution[1], render_depth=True,
            )
            view_count = scene.num_views
            cameras_per_time = len(dataset.cameras)
            permutation = torch.roll(
                torch.arange(view_count, device=device), shifts=-cameras_per_time
            )
            permuted_scene = SceneInput(
                images=scene.images[permutation],
                intrinsics=scene.intrinsics[permutation],
                cam_to_world=scene.cam_to_world[permutation],
                times=scene.times[permutation],
                frame_index=scene.frame_index[permutation],
                boxes=scene.boxes,
                sparse_depth=scene.sparse_depth[permutation] if scene.sparse_depth is not None else None,
                sparse_mask=scene.sparse_mask[permutation] if scene.sparse_mask is not None else None,
            )
            permuted_decoded = model.decode(permuted_scene, depth[permutation], sampled)
            variant_outputs["context_permuted"] = model.render(
                permuted_decoded, boxes, targets,
                height=resolution[0], width=resolution[1], render_depth=True,
            )
            run_metadata["context_permutation"] = permutation.cpu().tolist()
            run_metadata["context_permuted_sigma_l1"] = float(
                (gaussians.sigma - permuted_decoded.sigma).abs().mean()
            )
            run_metadata["context_permuted_means_l1_m"] = float(
                (gaussians.means - permuted_decoded.means).abs().mean()
            )
        query_hash = hashlib.sha256()
        for key in ("sample_index", "queries", "view_id", "instance_id", "times", "box_index"):
            query_hash.update(compact_hash(sampled[key]).encode())
        gaussian_hash = hashlib.sha256()
        for values in (gaussians.means, gaussians.quats, gaussians.scales,
                       gaussians.opacities, gaussians.colors, gaussians.sigma,
                       gaussians.instance_id):
            gaussian_hash.update(compact_hash(values).encode())
        result = {
            "label": label,
            "seed": seed,
            "depth_source": "predicted" if depth_override is None else "reused:A1",
            **run_metadata,
            "context_depth_used": depth.float().cpu(),
            "context_depth_measured": (
                measured_depth.float().cpu() if measured_depth is not None else depth.float().cpu()
            ),
            "rgb": output["rgb"].float().cpu(),
            "features": output["features"].float().cpu(),
            "alpha": output["alpha"].float().cpu(),
            "render_depth": output["render_depth"].float().cpu(),
            "target": target,
            "query_hash": query_hash.hexdigest(),
            "sample_index_hash": tensor_sha256(sampled["sample_index"]),
            "gaussian_hash": gaussian_hash.hexdigest(),
            "query_count": int(sampled["queries"].shape[0]),
            "query_static_count": int((sampled["instance_id"] < 0).sum()),
            "query_dynamic_count": int((sampled["instance_id"] >= 0).sum()),
            "query_view_counts": torch.bincount(sampled["view_id"], minlength=scene.num_views).cpu().tolist(),
            "gaussian_means": tensor_summary(gaussians.means),
            "gaussian_scales": tensor_summary(gaussians.scales),
            "gaussian_opacities": tensor_summary(gaussians.opacities),
            "gaussian_sigma": tensor_summary(gaussians.sigma),
        }
        for variant_name, variant_output in variant_outputs.items():
            for field in ("rgb", "features", "alpha", "render_depth"):
                result[f"{variant_name}_{field}"] = variant_output[field].float().cpu()
        del scene, targets, boxes, depth, measured_depth, sampled, gaussians, output, variant_outputs
        torch.cuda.empty_cache()
        return result

    a1 = run_once(
        "A1", window_a, args.seed,
        variants=True, alternate_times=alternate_target_times["B"],
    )
    runs = {
        "A1": a1,
        "A2": run_once(
            "A2", window_a, args.seed,
            depth_override=a1["context_depth_used"], measure_fresh_depth=True,
        ),
        "A3": run_once("A3", window_a, args.seed + 1, depth_override=a1["context_depth_used"]),
        "B1": run_once(
            "B1", window_b, args.seed,
            variants=True, alternate_times=alternate_target_times["A"],
        ),
    }

    target = runs["A1"]["target"]
    for other in ("A2", "A3", "B1"):
        for key in (
            "images", "intrinsics", "cam_to_world", "sparse_depth", "sparse_mask",
            "dynamic_mask", "sky_mask",
        ):
            if not torch.equal(target[key], runs[other]["target"][key]):
                raise RuntimeError(f"Target {key} differs between A1 and {other}")
    gt = target["images"].float()
    lidar_depth = target["sparse_depth"].float()
    dynamic = target.get("dynamic_mask", torch.zeros_like(lidar_depth)).squeeze(1) > 0.5
    sky = target.get("sky_mask", torch.zeros_like(lidar_depth)).squeeze(1) > 0.5
    lidar_mask = target["sparse_mask"].squeeze(1) > 0.5
    masks = {
        "full": torch.ones_like(dynamic),
        "dynamic": dynamic,
        "static": ~dynamic,
        "sky": sky,
        "non_sky": ~sky,
        "lidar": lidar_mask,
        "dynamic_lidar": dynamic & lidar_mask,
        "static_lidar": (~dynamic) & lidar_mask,
    }

    comparisons = {
        "determinism_A1_A2": pair_metrics(runs["A1"], runs["A2"], masks),
        "sampling_A1_A3": pair_metrics(runs["A1"], runs["A3"], masks),
        "window_A1_B1": pair_metrics(runs["A1"], runs["B1"], masks),
        "depth_determinism_A1_A2": shared_depth_metrics(
            {**runs["A1"], "context_depth": runs["A1"]["context_depth_measured"]},
            {**runs["A2"], "context_depth": runs["A2"]["context_depth_measured"]},
        ),
        "depth_shared_context_A1_B1": shared_depth_metrics(
            {**runs["A1"], "context_depth": runs["A1"]["context_depth_measured"]},
            {**runs["B1"], "context_depth": runs["B1"]["context_depth_measured"]},
        ),
    }
    pseudo_runs = {}
    for source, label in (("A1", "A1_static"), ("B1", "B1_static")):
        pseudo_runs[label] = {
            **runs[source],
            **{
                field: runs[source][f"static_{field}"]
                for field in ("rgb", "features", "alpha", "render_depth")
            },
        }
    for source, label in (("A1", "A1_timeB"), ("B1", "B1_timeA")):
        pseudo_runs[label] = {
            **runs[source],
            **{
                field: runs[source][f"alternate_time_{field}"]
                for field in ("rgb", "features", "alpha", "render_depth")
            },
        }
    for source, label in (("A1", "A1_boxfree_redecode"), ("B1", "B1_boxfree_redecode")):
        pseudo_runs[label] = {
            **runs[source],
            **{
                field: runs[source][f"boxfree_redecode_{field}"]
                for field in ("rgb", "features", "alpha", "render_depth")
            },
        }
    for source, label in (
        ("A1", "A1_boxfree_redecode_timeB"),
        ("B1", "B1_boxfree_redecode_timeA"),
    ):
        pseudo_runs[label] = {
            **runs[source],
            **{
                field: runs[source][f"boxfree_redecode_alternate_time_{field}"]
                for field in ("rgb", "features", "alpha", "render_depth")
            },
        }
    for source, label in (("A1", "A1_context_permuted"), ("B1", "B1_context_permuted")):
        pseudo_runs[label] = {
            **runs[source],
            **{
                field: runs[source][f"context_permuted_{field}"]
                for field in ("rgb", "features", "alpha", "render_depth")
            },
        }
    comparisons["window_posthoc_static_A1_B1"] = pair_metrics(
        pseudo_runs["A1_static"], pseudo_runs["B1_static"], masks
    )
    comparisons["window_boxfree_redecode_A1_B1"] = pair_metrics(
        pseudo_runs["A1_boxfree_redecode"], pseudo_runs["B1_boxfree_redecode"], masks
    )
    comparisons["window_boxfree_redecode_time_aligned_to_B"] = pair_metrics(
        pseudo_runs["A1_boxfree_redecode_timeB"], pseudo_runs["B1_boxfree_redecode"], masks
    )
    comparisons["window_boxfree_redecode_time_aligned_to_A"] = pair_metrics(
        pseudo_runs["A1_boxfree_redecode"], pseudo_runs["B1_boxfree_redecode_timeA"], masks
    )
    comparisons["context_order_A"] = pair_metrics(runs["A1"], pseudo_runs["A1_context_permuted"], masks)
    comparisons["context_order_B"] = pair_metrics(runs["B1"], pseudo_runs["B1_context_permuted"], masks)
    comparisons["target_opacity_time_A_to_B"] = pair_metrics(
        runs["A1"], pseudo_runs["A1_timeB"], masks
    )
    comparisons["target_opacity_time_B_to_A"] = pair_metrics(
        runs["B1"], pseudo_runs["B1_timeA"], masks
    )
    comparisons["window_opacity_time_aligned_to_B"] = pair_metrics(
        pseudo_runs["A1_timeB"], runs["B1"], masks
    )
    comparisons["window_opacity_time_aligned_to_A"] = pair_metrics(
        runs["A1"], pseudo_runs["B1_timeA"], masks
    )
    to_gt = {label: metrics_to_gt(run, gt, lidar_depth, masks) for label, run in runs.items()}

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"scene{args.scene}_w{args.window_a_start:03d}_w{args.window_b_start:03d}"
    save_visual(prefix, target_frames, gt, runs)
    np.savez_compressed(
        str(prefix) + "_arrays.npz",
        gt=gt.numpy(),
        dynamic_mask=dynamic.numpy(),
        sky_mask=sky.numpy(),
        lidar_depth=lidar_depth.numpy(),
        lidar_mask=lidar_mask.numpy(),
        **{
            f"{label}_{field}": run[field].numpy()
            for label, run in runs.items()
            for field in ("rgb", "alpha", "render_depth")
        },
        **{
            f"{label}_{field}": run[field].numpy()
            for label, run in pseudo_runs.items()
            for field in ("rgb", "alpha", "render_depth")
        },
    )

    run_summaries = {}
    for label, run in runs.items():
        run_summaries[label] = {
            key: value for key, value in run.items()
            if key not in {
                "context_depth_used", "context_depth_measured", "rgb", "features",
                "alpha", "render_depth", "target", "static_rgb", "static_features",
                "static_alpha", "static_render_depth", "alternate_time_rgb",
                "alternate_time_features", "alternate_time_alpha", "alternate_time_render_depth",
                "boxfree_redecode_rgb", "boxfree_redecode_features", "boxfree_redecode_alpha",
                "boxfree_redecode_render_depth", "boxfree_redecode_alternate_time_rgb",
                "boxfree_redecode_alternate_time_features", "boxfree_redecode_alternate_time_alpha",
                "boxfree_redecode_alternate_time_render_depth",
                "context_permuted_rgb", "context_permuted_features",
                "context_permuted_alpha", "context_permuted_render_depth",
            }
        }
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_head, git_status = None, []
    manifest = {
        "schema_version": 1,
        "interpretation": "single-scene pilot; overlapping-window dependence, not general causal proof",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "audit_script_sha256": file_sha256(Path(__file__).resolve()),
        "git_head": git_head,
        "git_status": git_status,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "scene": args.scene,
        "camera": args.camera,
        "resolution_hw": list(resolution),
        "world_anchor_frame": int(window_a[0]),
        "world_from": world_from.tolist(),
        "target_frames": target_frames,
        "runs": run_summaries,
        "comparisons": comparisons,
        "metrics_to_gt": to_gt,
    }
    metrics_path = Path(str(prefix) + "_metrics.json")
    metrics_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))

    csv_path = Path(str(prefix) + "_per_target.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "comparison", "target_frame", "region", "pixels", "l1", "psnr", "ssim",
            "alpha_l1", "feature_l1", "depth_mae_alpha_0p01_m",
        ])
        writer.writeheader()
        for comparison, values in comparisons.items():
            if not isinstance(values, dict) or "targets" not in values:
                continue
            for row in values["targets"]:
                for region, metric in row["regions"].items():
                    if metric is None:
                        continue
                    writer.writerow({
                        "comparison": comparison,
                        "target_frame": row["target_frame"],
                        "region": region,
                        "pixels": metric["pixels"],
                        "l1": metric["l1"],
                        "psnr": metric["psnr"],
                        "ssim": metric["ssim"],
                        "alpha_l1": metric["alpha_l1"],
                        "feature_l1": metric["feature_l1"],
                        "depth_mae_alpha_0p01_m": metric["depth_mae_alpha_0.01_m"],
                    })
    print(json.dumps({
        "metrics": str(metrics_path),
        "arrays": str(prefix) + "_arrays.npz",
        "comparison": str(prefix) + "_comparison.png",
        "per_target": str(csv_path),
    }, indent=2))


if __name__ == "__main__":
    main()
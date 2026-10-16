"""Audit dynamic Gaussian image support and visibility without changing a model.

This is a diagnostic-only evaluation.  Ground-truth fine vehicle masks are
used exclusively for measurements and a conditional counterfactual; they are
never fed to the reconstruction or written back to a runtime payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import TargetViews, box_dynamic_masks
from scripts.deliver_continuous_scene import horizontal_strip, label, to_uint8, vertical_strip
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def hash_file_set(paths: list[Path], root: Path) -> str:
    """Hash both relative names and bytes so a renamed mask changes identity."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def psnr_from_sse(sse: float, scalar_count: int) -> float | None:
    if scalar_count == 0:
        return None
    mse = max(sse / scalar_count, 1e-10)
    return -10.0 * math.log10(mse)


def mask_scores(counts: dict[str, int]) -> dict[str, float | int | None]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return counts | {
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "iou": safe_ratio(tp, tp + fp + fn),
    }


def add_binary_counts(
    accumulator: dict[str, int], prediction: torch.Tensor, target: torch.Tensor
) -> None:
    prediction = prediction.bool()
    target = target.bool()
    accumulator["tp"] += int((prediction & target).sum())
    accumulator["fp"] += int((prediction & ~target).sum())
    accumulator["fn"] += int((~prediction & target).sum())
    accumulator["tn"] += int((~prediction & ~target).sum())


def add_rgb_region(
    accumulator: dict[str, float | int],
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    pixel_count = int(mask.sum())
    if not pixel_count:
        return
    error = (prediction.float() - target.float()).square()
    accumulator["pixel_count"] += pixel_count
    accumulator["scalar_count"] += pixel_count * prediction.shape[0]
    accumulator["sse"] += float(error[:, mask].sum())


def rgb_region_result(value: dict[str, float | int]) -> dict[str, float | int | None]:
    scalar_count = int(value["scalar_count"])
    return value | {
        "mse": value["sse"] / scalar_count if scalar_count else None,
        "psnr_db": psnr_from_sse(float(value["sse"]), scalar_count),
    }


def gray_rgb(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().unsqueeze(0).expand(3, -1, -1)


def depth_rgb(depth: torch.Tensor, alpha: torch.Tensor, threshold: float) -> torch.Tensor:
    valid = (alpha > threshold) & torch.isfinite(depth) & (depth > 0)
    output = torch.zeros((3, *depth.shape), dtype=torch.float32)
    if not bool(valid.any()):
        return output
    values = depth[valid].float().cpu().numpy()
    lo, hi = np.percentile(values, (2.0, 98.0))
    denominator = max(float(hi - lo), 1e-6)
    normalised = ((depth.float().cpu() - float(lo)) / denominator).clamp(0, 1)
    valid_cpu = valid.cpu()
    # Compact blue->cyan->yellow map, sufficient for a diagnostic panel.
    output[0] = normalised
    output[1] = (1.0 - (2.0 * normalised - 1.0).abs()).clamp(0, 1)
    output[2] = 1.0 - normalised
    output *= valid_cpu.unsqueeze(0)
    return output


def parse_keyframes(text: str, valid_frames: list[int]) -> set[int]:
    if not text.strip():
        return set()
    selected = {int(value) for value in text.split(",") if value.strip()}
    unknown = selected - set(valid_frames)
    if unknown:
        raise ValueError(f"Keyframes are outside this scene: {sorted(unknown)}")
    return selected


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--camera-refiner-states", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--keyframes", default="118,120,139,141")
    args = parser.parse_args()

    if not 0.0 <= args.alpha_threshold <= 1.0:
        parser.error("--alpha-threshold must be in [0,1]")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact 128-D rendering")

    started = time.perf_counter()
    device = torch.device(args.device)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    payload_dir = Path(args.payload_dir).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, args.split)
    if len(args.camera_refiner_states) != len(dataset.cameras):
        parser.error(
            "--camera-refiner-states must match configured cameras: "
            f"{dataset.cameras}"
        )
    matches = [item for item in dataset.scenes if item["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    all_frames = [int(value) for value in metadata["frames"]]
    frames = all_frames[::args.frame_stride]
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    keyframes = parse_keyframes(args.keyframes, all_frames) & set(frames)
    all_slots = {frame: slot for slot, frame in enumerate(all_frames)}
    all_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    static_path = payload_dir / "final_static_128d.pt"
    dynamic_path = payload_dir / "final_dynamic_128d.pt"
    dynamic_meta_path = payload_dir / "global_dynamic_meta.pt"
    for path in (static_path, dynamic_path, dynamic_meta_path):
        if not path.exists():
            raise FileNotFoundError(path)
    meta = torch.load(dynamic_meta_path, map_location="cpu", weights_only=True)
    payload_frames = [int(value) for value in meta["frames"]]
    if payload_frames != all_frames:
        raise ValueError("Global dynamic payload frame list does not match dataset")
    static = load_payload(static_path, device)
    dynamic = load_payload(dynamic_path, device)
    full = combine(static, dynamic)
    render_boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    if render_boxes is None or evaluation_boxes is None:
        raise ValueError("Dynamic runtime payload has no box graph")

    model = build_model(config, device, str(checkpoint_path)).eval()
    camera_refiners = {}
    refiner_paths = [Path(value).resolve() for value in args.camera_refiner_states]
    for camera, state_path in zip(dataset.cameras, refiner_paths):
        refiner = copy.deepcopy(model.refiner).to(device).eval()
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        refiner.load_state_dict(state.get("refiner", state), strict=True)
        camera_refiners[camera] = refiner

    binary_by_camera = {
        str(camera): {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for camera in dataset.cameras
    }
    binary_all = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    regions = {
        name: {"pixel_count": 0, "scalar_count": 0, "sse": 0.0}
        for name in (
            "base_global", "counterfactual_global", "base_bbox",
            "counterfactual_bbox", "base_bbox_vehicle",
            "base_bbox_background", "base_bbox_background_alpha_positive",
        )
    }
    frame_records = []
    fine_mask_paths: list[Path] = []
    visual_paths = []

    for frame_index, frame in enumerate(frames):
        slot = all_slots[frame]
        target = dataset._collect(
            metadata, [frame], all_slots, all_times, dataset.resolution,
            world_from, lidar=False,
        )
        frame_base, frame_static, frame_alpha, frame_depth = [], [], [], []
        frame_fine, frame_bbox, frame_counterfactual = [], [], []
        frame_base_bbox_sse = 0.0
        frame_base_bbox_scalars = 0
        frame_counter_bbox_sse = 0.0
        frame_counter_bbox_scalars = 0
        frame_base_global_sse = 0.0
        frame_counter_global_sse = 0.0
        frame_global_scalars = 0

        for camera_index, camera in enumerate(dataset.cameras):
            model.refiner = camera_refiners[camera]
            intrinsics = target["intrinsics"][camera_index:camera_index + 1].to(device)
            cam_to_world = target["cam_to_world"][camera_index:camera_index + 1].to(device)
            views = TargetViews(
                intrinsics=intrinsics,
                cam_to_world=cam_to_world,
                times=torch.tensor([float(all_times[slot])], device=device),
                frame_index=torch.tensor([slot], dtype=torch.long, device=device),
            )

            base_result = model.render(
                full, render_boxes, views, height=height, width=width,
                render_depth=False,
            )
            base = base_result["rgb"][0]
            del base_result
            static_result = model.render(
                static, None, views, height=height, width=width,
                render_depth=False,
            )
            static_rgb = static_result["rgb"][0]
            del static_result
            dynamic_result = model.render(
                dynamic, render_boxes, views, height=height, width=width,
                render_depth=True,
            )
            dynamic_alpha = dynamic_result["alpha"][0, 0]
            dynamic_depth = dynamic_result["render_depth"][0, 0]
            del dynamic_result

            relative = f"fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png"
            fine_path = Path(metadata["path"]) / relative
            fine_mask_paths.append(fine_path)
            fine, _ = dataset._load_calibrated_image(
                metadata, relative, dataset.resolution, camera, gray=True,
            )
            fine_mask = fine.squeeze(0).to(device) > 0.5
            bbox_mask = box_dynamic_masks(
                evaluation_boxes,
                torch.tensor([slot], dtype=torch.long, device=device),
                intrinsics,
                torch.linalg.inv(cam_to_world),
                height,
                width,
            )[0]
            alpha_positive = dynamic_alpha >= args.alpha_threshold
            restoration_mask = bbox_mask & ~fine_mask & alpha_positive
            counterfactual = torch.where(
                restoration_mask.unsqueeze(0), static_rgb, base
            )
            gt = target["images"][camera_index].to(device)

            add_binary_counts(binary_by_camera[str(camera)], alpha_positive, fine_mask)
            add_binary_counts(binary_all, alpha_positive, fine_mask)
            add_rgb_region(regions["base_global"], base, gt, torch.ones_like(fine_mask))
            add_rgb_region(
                regions["counterfactual_global"], counterfactual, gt,
                torch.ones_like(fine_mask),
            )
            add_rgb_region(regions["base_bbox"], base, gt, bbox_mask)
            add_rgb_region(regions["counterfactual_bbox"], counterfactual, gt, bbox_mask)
            add_rgb_region(regions["base_bbox_vehicle"], base, gt, bbox_mask & fine_mask)
            add_rgb_region(
                regions["base_bbox_background"], base, gt, bbox_mask & ~fine_mask
            )
            add_rgb_region(
                regions["base_bbox_background_alpha_positive"], base, gt,
                bbox_mask & ~fine_mask & alpha_positive,
            )

            error_base = (base.float() - gt.float()).square()
            error_counter = (counterfactual.float() - gt.float()).square()
            frame_base_global_sse += float(error_base.sum())
            frame_counter_global_sse += float(error_counter.sum())
            frame_global_scalars += int(error_base.numel())
            bbox_scalar_count = int(bbox_mask.sum()) * 3
            frame_base_bbox_sse += float(error_base[:, bbox_mask].sum())
            frame_counter_bbox_sse += float(error_counter[:, bbox_mask].sum())
            frame_base_bbox_scalars += bbox_scalar_count
            frame_counter_bbox_scalars += bbox_scalar_count

            if frame in keyframes:
                frame_base.append(base.cpu())
                frame_static.append(static_rgb.cpu())
                frame_alpha.append(dynamic_alpha.cpu())
                frame_depth.append(dynamic_depth.cpu())
                frame_fine.append(fine_mask.cpu())
                frame_bbox.append(bbox_mask.cpu())
                frame_counterfactual.append(counterfactual.cpu())

        frame_records.append({
            "frame": frame,
            "base_global_psnr_db": psnr_from_sse(
                frame_base_global_sse, frame_global_scalars
            ),
            "counterfactual_global_psnr_db": psnr_from_sse(
                frame_counter_global_sse, frame_global_scalars
            ),
            "global_scalar_count": frame_global_scalars,
            "base_global_sse": frame_base_global_sse,
            "counterfactual_global_sse": frame_counter_global_sse,
            "base_bbox_psnr_db": psnr_from_sse(
                frame_base_bbox_sse, frame_base_bbox_scalars
            ),
            "counterfactual_bbox_psnr_db": psnr_from_sse(
                frame_counter_bbox_sse, frame_counter_bbox_scalars
            ),
            "bbox_scalar_count": frame_base_bbox_scalars,
            "base_bbox_sse": frame_base_bbox_sse,
            "counterfactual_bbox_sse": frame_counter_bbox_sse,
        })

        if frame in keyframes:
            rows = []
            gt_panels, base_panels, counter_panels = [], [], []
            static_panels, alpha_panels, mask_panels, depth_panels = [], [], [], []
            for index, camera in enumerate(dataset.cameras):
                gt_panels.append(label(to_uint8(target["images"][index]), f"GT | camera {camera}"))
                base_panels.append(label(to_uint8(frame_base[index]), f"base full | camera {camera}"))
                counter_panels.append(label(to_uint8(frame_counterfactual[index]), f"conditional restoration | camera {camera}"))
                static_panels.append(label(to_uint8(frame_static[index]), f"static only | camera {camera}"))
                alpha_panels.append(label(to_uint8(gray_rgb(frame_alpha[index])), f"dynamic alpha | camera {camera}"))
                mask_overlay = torch.zeros((3, height, width))
                mask_overlay[0] = frame_bbox[index].float()
                mask_overlay[1] = frame_fine[index].float()
                mask_overlay[2] = (frame_alpha[index] >= args.alpha_threshold).float()
                mask_panels.append(label(to_uint8(mask_overlay), "R=bbox G=fine B=alpha+"))
                depth_panels.append(label(to_uint8(depth_rgb(
                    frame_depth[index], frame_alpha[index], args.alpha_threshold
                )), f"dynamic depth | camera {camera}"))
            rows.extend((
                horizontal_strip(gt_panels), horizontal_strip(base_panels),
                horizontal_strip(counter_panels), horizontal_strip(static_panels),
                horizontal_strip(alpha_panels), horizontal_strip(mask_panels),
                horizontal_strip(depth_panels),
            ))
            visual_path = visual_dir / f"frame_{frame:04d}_support_visibility.jpg"
            Image.fromarray(vertical_strip(rows)).save(visual_path, quality=92)
            visual_paths.append(str(visual_path))

        if frame_index % 10 == 0 or frame_index + 1 == len(frames):
            print(json.dumps({
                "frame": frame,
                "progress": [frame_index + 1, len(frames)],
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)

    binary_results = {
        "all_cameras": mask_scores(binary_all),
        "per_camera": {
            camera: mask_scores(counts)
            for camera, counts in binary_by_camera.items()
        },
    }
    region_results = {
        name: rgb_region_result(value) for name, value in regions.items()
    }
    mean_frame_bbox = [
        value["base_bbox_psnr_db"] for value in frame_records
        if value["base_bbox_psnr_db"] is not None
    ]
    mean_frame_counter_bbox = [
        value["counterfactual_bbox_psnr_db"] for value in frame_records
        if value["counterfactual_bbox_psnr_db"] is not None
    ]
    mean_frame_global = [
        value["base_global_psnr_db"] for value in frame_records
        if value["base_global_psnr_db"] is not None
    ]
    mean_frame_counter_global = [
        value["counterfactual_global_psnr_db"] for value in frame_records
        if value["counterfactual_global_psnr_db"] is not None
    ]
    mask_root = Path(metadata["path"])
    input_paths = [
        config_path, checkpoint_path, static_path, dynamic_path,
        dynamic_meta_path, *refiner_paths,
    ]
    script_path = Path(__file__).resolve()
    manifest = {
        "status": "completed",
        "diagnostic_only": True,
        "ground_truth_mask_contract": (
            "fine_dynamic_masks/vehicle is used only for diagnostic metrics, "
            "visualisation, and the conditional counterfactual; it is not an "
            "input to reconstruction and no model/payload/PLY is modified"
        ),
        "scene": args.scene,
        "frames_evaluated": frames,
        "frame_count": len(frames),
        "cameras": list(dataset.cameras),
        "resolution_hw": [height, width],
        "alpha_threshold_inclusive": args.alpha_threshold,
        "gaussian_counts": {"static": static.count, "dynamic": dynamic.count},
        "definitions": {
            "alpha_positive": "dynamic-only rendered alpha >= alpha_threshold",
            "precision": "TP/(TP+FP), positive=alpha_positive, target=fine vehicle mask",
            "recall": "TP/(TP+FN), positive=alpha_positive, target=fine vehicle mask",
            "iou": "TP/(TP+FP+FN)",
            "rgb_sse": "sum over selected pixels and all 3 RGB channels of (prediction-GT)^2",
            "psnr_db": "-10*log10(rgb_sse/(3*selected_pixel_count)), RGB range [0,1]",
            "counterfactual": (
                "replace base full-scene RGB with static-only RGB only where "
                "evaluation_bbox_union AND NOT fine_vehicle_mask AND alpha_positive"
            ),
            "bbox": "union of projected evaluation_boxes rectangles",
        },
        "mask_support_metrics": binary_results,
        "rgb_region_metrics_aggregate_pixels": region_results,
        "mean_per_frame_bbox_psnr_db": {
            "base": float(np.mean(mean_frame_bbox)) if mean_frame_bbox else None,
            "counterfactual": (
                float(np.mean(mean_frame_counter_bbox))
                if mean_frame_counter_bbox else None
            ),
        },
        "mean_per_frame_global_psnr_db": {
            "base": float(np.mean(mean_frame_global)) if mean_frame_global else None,
            "counterfactual": (
                float(np.mean(mean_frame_counter_global))
                if mean_frame_counter_global else None
            ),
        },
        "frame_records": frame_records,
        "visualisations": visual_paths,
        "inputs": {str(path): sha256_file(path) for path in input_paths},
        "fine_vehicle_mask_set": {
            "root": str(mask_root),
            "file_count": len(fine_mask_paths),
            "sha256_names_and_contents": hash_file_set(fine_mask_paths, mask_root),
        },
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "timing": {"wall_seconds": time.perf_counter() - started},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "mask_support_metrics": binary_results,
        "rgb_region_metrics_aggregate_pixels": region_results,
        "mean_per_frame_bbox_psnr_db": manifest["mean_per_frame_bbox_psnr_db"],
        "mean_per_frame_global_psnr_db": manifest["mean_per_frame_global_psnr_db"],
        "wall_seconds": manifest["timing"]["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

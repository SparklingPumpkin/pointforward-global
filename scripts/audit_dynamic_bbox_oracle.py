"""Audit projected dynamic-box metrics against fine vehicle masks.

This is a diagnostics-only utility.  It never updates a model, Gaussian, box,
or RGB image.  It consumes the labelled GT/prediction comparison frames saved
by ``render_final_global_runtime.py`` and reports:

* how much of the projected 3-D-box union is actual fine-mask vehicle;
* fine-mask recall, sky/background contamination, and exact pixel/SSE counts;
* an appearance-only oracle in which fine vehicle pixels are replaced by GT;
* per-camera and all-camera PSNR under the same mean-over-source-frames rule as
  the runtime renderer.

The oracle is exact for the decoded comparison-frame artifact.  JPEG
quantisation means it is not an exact recomputation of the renderer's in-memory
float tensors; this provenance is recorded prominently in the output manifest.
GT replacement is an upper-bound diagnostic only and is not a deliverable or
an inference method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from scripts.render_final_global_runtime import load_boxes, projected_box_bounds
from scripts.train import build_dataset


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_sha256(paths: list[Path], root: Path | None = None) -> str:
    """Hash ordered ``relative path + NUL + file hash`` records."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = str(path.relative_to(root)) if root is not None else str(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def psnr(squared_error_sum: float, pixel_count: int) -> float:
    if pixel_count <= 0:
        return float("nan")
    mse = max(squared_error_sum / pixel_count, 1e-10)
    return -10.0 * math.log10(mse)


def add(accumulator: dict[str, float | int], squared_error: np.ndarray, mask: np.ndarray) -> None:
    accumulator["squared_error_sum"] += float(squared_error[mask].sum(dtype=np.float64))
    accumulator["pixel_count"] += int(mask.sum())


def metric_summary(per_frame: list[dict[str, float | int]]) -> dict:
    valid = [item for item in per_frame if int(item["pixel_count"]) > 0]
    total_sse = sum(float(item["squared_error_sum"]) for item in valid)
    total_pixels = sum(int(item["pixel_count"]) for item in valid)
    values = [psnr(float(item["squared_error_sum"]), int(item["pixel_count"])) for item in valid]
    return {
        "source_frame_count": len(valid),
        "pixel_count": total_pixels,
        "squared_error_sum": total_sse,
        "mean_source_frame_psnr_db": float(np.mean(values)),
        "pixel_weighted_psnr_db": psnr(total_sse, total_pixels),
    }


def projected_union(bounds: torch.Tensor, visible: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Rasterise an instance union via a difference image, inclusive bounds."""
    difference = np.zeros((height + 1, width + 1), dtype=np.int16)
    for track_slot in range(bounds.shape[0]):
        if not bool(visible[track_slot]):
            continue
        x0, y0, x1, y1 = (int(value) for value in bounds[track_slot])
        x1 += 1
        y1 += 1
        difference[y0, x0] += 1
        difference[y1, x0] -= 1
        difference[y0, x1] -= 1
        difference[y1, x1] += 1
    return np.cumsum(np.cumsum(difference[:-1, :-1], axis=0), axis=1) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer-manifest", required=True)
    parser.add_argument("--comparison-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--label-height", type=int, default=24)
    parser.add_argument("--target-dynamic-psnr", type=float, default=27.0)
    args = parser.parse_args()

    manifest_path = Path(args.renderer_manifest).resolve()
    renderer_manifest = json.loads(manifest_path.read_text())
    comparison_dir = (
        Path(args.comparison_dir).resolve()
        if args.comparison_dir
        else manifest_path.parent / "frames" / "comparison"
    )
    output_path = Path(args.out).resolve()

    config_path = Path(renderer_manifest["config"]).resolve()
    payload_dir = Path(renderer_manifest["payload_dir"]).resolve()
    dynamic_meta_path = payload_dir / "global_dynamic_meta.pt"
    config = yaml.safe_load(config_path.read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, args.split)
    scene_name = str(renderer_manifest["scene"])
    metadata = next(item for item in dataset.scenes if item["name"] == scene_name)
    frames = list(metadata["frames"])
    height, width = dataset.resolution
    cameras = list(dataset.cameras)
    if renderer_manifest["resolution_hw_per_camera"] != [height, width]:
        raise ValueError("Dataset resolution differs from renderer manifest")
    if renderer_manifest["cameras"] != cameras:
        raise ValueError("Dataset cameras differ from renderer manifest")

    dynamic_meta = torch.load(dynamic_meta_path, map_location="cpu", weights_only=True)
    boxes = load_boxes(
        dynamic_meta.get("evaluation_boxes", dynamic_meta["boxes"]),
        torch.device("cpu"),
    )
    if boxes is None:
        raise ValueError("Dynamic metadata does not contain evaluation boxes")

    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    comparison_paths = [comparison_dir / f"frame_{index:04d}.jpg" for index in range(len(frames))]
    missing = [str(path) for path in comparison_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing comparison images, first entries: {missing[:3]}")

    metric_names = ("global", "dynamic_bbox", "fine_vehicle", "bbox_background", "oracle_global", "oracle_dynamic_bbox")
    combined = {name: [] for name in metric_names}
    per_camera = {str(camera): {name: [] for name in metric_names} for camera in cameras}
    count = {
        "dynamic_bbox_union_pixels": 0,
        "fine_vehicle_pixels": 0,
        "fine_vehicle_inside_bbox_pixels": 0,
        "sky_inside_bbox_pixels": 0,
        "bbox_background_pixels": 0,
    }
    geometry_count = {
        "dynamic_bbox_union_pixels": 0,
        "fine_vehicle_pixels": 0,
        "fine_vehicle_inside_bbox_pixels": 0,
        "sky_inside_bbox_pixels": 0,
        "bbox_background_pixels": 0,
    }
    fine_mask_paths: list[Path] = []
    sky_mask_paths: list[Path] = []
    fine_sources: dict[str, int] = {}

    for source_index, (frame, comparison_path) in enumerate(zip(frames, comparison_paths)):
        image = np.asarray(Image.open(comparison_path).convert("RGB"), dtype=np.float64) / 255.0
        if image.shape != (2 * height, len(cameras) * width, 3):
            raise ValueError(f"Unexpected comparison layout {image.shape}: {comparison_path}")
        gt = image[:height].reshape(height, len(cameras), width, 3).transpose(1, 0, 2, 3)
        prediction = image[height:].reshape(height, len(cameras), width, 3).transpose(1, 0, 2, 3)
        squared_error = np.square(prediction - gt).mean(axis=-1)

        # Collection provides exactly the calibrated intrinsics/poses used by the renderer.
        all_slots = {int(value): index for index, value in enumerate(frames)}
        all_times = dataset._normalised_times(frames)
        target = dataset._collect(
            metadata, [frame], all_slots, all_times,
            dataset.resolution, world_from, lidar=False,
        )
        bounds, visible = projected_box_bounds(
            boxes,
            torch.full((len(cameras),), source_index, dtype=torch.long),
            target["intrinsics"],
            torch.linalg.inv(target["cam_to_world"]),
            height,
            width,
        )

        frame_acc = {
            name: {"squared_error_sum": 0.0, "pixel_count": 0}
            for name in metric_names
        }
        for camera_index, camera in enumerate(cameras):
            bbox_full = projected_union(bounds[camera_index], visible[camera_index], height, width)
            valid = np.ones((height, width), dtype=bool)
            # Saved comparison frames have a black label overlay in these rows.
            valid[: args.label_height] = False
            bbox = bbox_full & valid

            fine_relative = f"fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png"
            fine_path = Path(metadata["path"]) / fine_relative
            if not fine_path.exists():
                fine_relative = f"dynamic_masks/all/{frame:03d}_{camera}.png"
                fine_path = Path(metadata["path"]) / fine_relative
            fine_tensor, _ = dataset._load_calibrated_image(
                metadata, fine_relative, dataset.resolution, camera, gray=True,
            )
            fine_full = fine_tensor[0].numpy() > 0.5
            fine = fine_full & valid
            fine_mask_paths.append(fine_path.resolve())
            fine_sources[fine_relative.split("/")[0]] = fine_sources.get(fine_relative.split("/")[0], 0) + 1

            sky_relative = f"sky_masks/{frame:03d}_{camera}.png"
            sky_path = Path(metadata["path"]) / sky_relative
            sky_tensor, _ = dataset._load_calibrated_image(
                metadata, sky_relative, dataset.resolution, camera, gray=True,
            )
            sky_full = sky_tensor[0].numpy() > 0.5
            sky = sky_full & valid
            sky_mask_paths.append(sky_path.resolve())

            fine_inside = fine & bbox
            background = bbox & ~fine_inside
            outside_fine = valid & ~fine_inside
            masks = {
                "global": valid,
                "dynamic_bbox": bbox,
                "fine_vehicle": fine_inside,
                "bbox_background": background,
                # Oracle removes all SSE on fine-vehicle pixels.
                "oracle_global": outside_fine,
                "oracle_dynamic_bbox": background,
            }
            camera_acc = {}
            for name, mask in masks.items():
                # Oracle denominator stays global/bbox-sized although its SSE mask excludes fine pixels.
                denominator_mask = (
                    valid if name == "oracle_global"
                    else bbox if name == "oracle_dynamic_bbox"
                    else mask
                )
                camera_acc[name] = {
                    "squared_error_sum": float(squared_error[camera_index][mask].sum(dtype=np.float64)),
                    "pixel_count": int(denominator_mask.sum()),
                }
                frame_acc[name]["squared_error_sum"] += camera_acc[name]["squared_error_sum"]
                frame_acc[name]["pixel_count"] += camera_acc[name]["pixel_count"]
                per_camera[str(camera)][name].append(camera_acc[name])

            count["dynamic_bbox_union_pixels"] += int(bbox.sum())
            count["fine_vehicle_pixels"] += int(fine.sum())
            count["fine_vehicle_inside_bbox_pixels"] += int(fine_inside.sum())
            count["sky_inside_bbox_pixels"] += int((sky & bbox).sum())
            count["bbox_background_pixels"] += int(background.sum())
            geometry_fine_inside = fine_full & bbox_full
            geometry_count["dynamic_bbox_union_pixels"] += int(bbox_full.sum())
            geometry_count["fine_vehicle_pixels"] += int(fine_full.sum())
            geometry_count["fine_vehicle_inside_bbox_pixels"] += int(geometry_fine_inside.sum())
            geometry_count["sky_inside_bbox_pixels"] += int((sky_full & bbox_full).sum())
            geometry_count["bbox_background_pixels"] += int((bbox_full & ~fine_full).sum())
        for name in metric_names:
            combined[name].append(frame_acc[name])

    combined_summary = {name: metric_summary(records) for name, records in combined.items()}
    per_camera_summary = {
        camera: {name: metric_summary(records) for name, records in metrics.items()}
        for camera, metrics in per_camera.items()
    }

    bbox_pixels = count["dynamic_bbox_union_pixels"]
    fine_inside = count["fine_vehicle_inside_bbox_pixels"]
    fine_pixels = count["fine_vehicle_pixels"]
    background_sse = combined_summary["bbox_background"]["squared_error_sum"]
    target_mse = 10.0 ** (-args.target_dynamic_psnr / 10.0)
    target_sse = target_mse * bbox_pixels
    pixel_weighted_required_reduction = max(0.0, 1.0 - target_sse / background_sse)
    mean_oracle = combined_summary["oracle_dynamic_bbox"]["mean_source_frame_psnr_db"]
    uniform_background_sse_reduction_for_mean = max(
        0.0, 1.0 - 10.0 ** (-(args.target_dynamic_psnr - mean_oracle) / 10.0)
    )

    script_path = Path(__file__).resolve()
    output = {
        "status": "completed",
        "diagnostic_only": True,
        "gt_usage": "GT replaces fine-vehicle pixels only to compute an unattainable appearance upper bound; no model, Gaussian, box, image, or deliverable is modified",
        "exactness": {
            "scope": "exact deterministic counts and SSE over decoded comparison JPEG pixels",
            "renderer_float_exact": False,
            "reason": "render_final_global_runtime saved labelled JPEG comparison frames, not lossless float tensors",
            "label_rows_excluded": [0, args.label_height - 1],
        },
        "scene": scene_name,
        "frames": len(frames),
        "cameras": cameras,
        "resolution_hw_per_camera": [height, width],
        "bbox_union_rule": "union of inclusive axis-aligned image extents of all visible corners of all valid projected 3-D evaluation boxes; duplicate/overlapping pixels counted once",
        "fine_mask_source": {
            "primary": "fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png, calibrated through dataset loader",
            "fallback": "dynamic_masks/all/{frame:03d}_{camera}.png",
            "file_counts_by_source": fine_sources,
            "threshold": "> 0.5",
        },
        "source_renderer_metrics_float_tensor": {
            "mean_psnr_source_frames_all_cameras": renderer_manifest.get(
                "mean_psnr_source_frames_all_cameras"
            ),
            "mean_psnr_non_sky_source_frames_all_cameras": renderer_manifest.get(
                "mean_psnr_non_sky_source_frames_all_cameras"
            ),
            "mean_psnr_dynamic_bbox_source_frames_all_cameras": renderer_manifest.get(
                "mean_psnr_dynamic_bbox_source_frames_all_cameras"
            ),
        },
        "mask_geometry_counts_full_frame": geometry_count,
        "mask_geometry_fractions_full_frame": {
            "fine_vehicle_inside_bbox_over_bbox": (
                geometry_count["fine_vehicle_inside_bbox_pixels"]
                / geometry_count["dynamic_bbox_union_pixels"]
            ),
            "bbox_background_over_bbox": (
                geometry_count["bbox_background_pixels"]
                / geometry_count["dynamic_bbox_union_pixels"]
            ),
            "fine_vehicle_recall_inside_bbox": (
                geometry_count["fine_vehicle_inside_bbox_pixels"]
                / geometry_count["fine_vehicle_pixels"]
            ),
            "sky_inside_bbox_over_bbox": (
                geometry_count["sky_inside_bbox_pixels"]
                / geometry_count["dynamic_bbox_union_pixels"]
            ),
        },
        "counts": count,
        "fractions": {
            "fine_vehicle_inside_bbox_over_bbox": fine_inside / bbox_pixels,
            "bbox_background_over_bbox": count["bbox_background_pixels"] / bbox_pixels,
            "fine_vehicle_recall_inside_bbox": fine_inside / fine_pixels,
            "sky_inside_bbox_over_bbox": count["sky_inside_bbox_pixels"] / bbox_pixels,
        },
        "metrics_all_cameras": combined_summary,
        "metrics_per_camera": per_camera_summary,
        "target_analysis": {
            "target_dynamic_bbox_psnr_db": args.target_dynamic_psnr,
            "pixel_weighted_required_bbox_background_sse_reduction_after_perfect_fine_vehicle": pixel_weighted_required_reduction,
            "uniform_background_sse_reduction_required_for_mean_source_frame_psnr": uniform_background_sse_reduction_for_mean,
            "note": "The first value is the reported ~37% bound; it uses aggregate pixel-weighted SSE. The second follows the renderer acceptance aggregation (arithmetic mean of frame PSNR) under uniform SSE scaling.",
        },
        "provenance": {
            "renderer_manifest": str(manifest_path),
            "comparison_dir": str(comparison_dir),
            "config": str(config_path),
            "dynamic_meta": str(dynamic_meta_path),
            "script": str(script_path),
            "sha256": {
                "renderer_manifest": sha256_file(manifest_path),
                "config": sha256_file(config_path),
                "dynamic_meta": sha256_file(dynamic_meta_path),
                "script": sha256_file(script_path),
                "comparison_frames_aggregate": aggregate_sha256(comparison_paths, comparison_dir),
                "fine_masks_aggregate": aggregate_sha256(fine_mask_paths, Path(metadata["path"])),
                "sky_masks_aggregate": aggregate_sha256(sky_mask_paths, Path(metadata["path"])),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate per-track SAM2 masks for the frozen Round-62 scene017 probe.

The projected 3D box is used only as a SAM2 prompt and as the final safety
crop.  The script never reads dataset dynamic/fine masks.  Its production
outputs are calibrated 576x768 masks at::

    OUTPUT/track-{track_id}/{frame:03d}_{camera}.png

For geometry diagnostics, the uncropped raw SAM2 boolean masks are also saved
without changing production-mask selection or fallback behavior at::

    OUTPUT/raw-track-masks/track-{track_id}/{frame:03d}_{camera}.png

The input contract is deliberately strict: the Round-61 ungated manifest,
ten frozen tracks, twelve observations per track, 120 prompts, and 108 unique
RGB images.  ``--dry-run-manifest-validation`` verifies that contract (and all
recorded projected areas) without importing torch or SAM2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable


IMAGE_HEIGHT = 576
IMAGE_WIDTH = 768
EXPECTED_PROMPTS = 120
EXPECTED_UNIQUE_IMAGES = 108
EXPECTED_VIEWS_PER_TRACK = 12
FROZEN_TRACK_IDS = (47, 26, 0, 41, 29, 18, 34, 19, 24, 7)
FROZEN_MANIFEST_SHA256 = (
    "4a39a357a8b516bdd6dc47c1649ef06af5fc74ea9648d2ed4c686378ed8baebf"
)

DEFAULT_TRACK_MANIFEST = Path(
    "experiments/argoverse-inference/global-dynamic-consistency/"
    "dynamic-roi-refinement/rgb-foreground-gating/DR-v1-q/"
    "run-20260929T114105Z-ungated-control/rebuilt-top10/track_once_manifest.json"
)
DEFAULT_DATA_ROOT = Path("/mnt/data/dataset/argoverse/processed/training/017")
DEFAULT_SAM2_ROOT = Path("/mnt/fjn/projects/tools/sam2")
DEFAULT_CHECKPOINT = DEFAULT_SAM2_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_BATCHING_ROOT = Path(
    "/mnt/data/fjn/projects/fjn/pbr-vehicle/pbr-vehicle-sun/src"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict calibrated per-track masks with SAM2.1 Hiera Tiny."
    )
    parser.add_argument("--track-manifest", type=Path, default=DEFAULT_TRACK_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", "--out", dest="output_dir", type=Path)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sam2-batching-root", type=Path, default=DEFAULT_BATCHING_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--min-predicted-iou", type=float, default=0.70)
    parser.add_argument("--min-mask-bbox-ratio", type=float, default=0.02)
    parser.add_argument("--max-raw-spill", type=float, default=0.50)
    parser.add_argument("--min-bbox-edge-px", type=float, default=4.0)
    parser.add_argument("--min-bbox-area-px", type=float, default=64.0)
    parser.add_argument("--min-reliable-views-per-track", type=int, default=4)
    parser.add_argument("--area-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--generic-contract",
        action="store_true",
        help=(
            "Accept any manifest whose tracks each contain exactly 12 views with "
            "bbox_xyxy; disables only the Round-62 identity/count assertions."
        ),
    )
    parser.add_argument(
        "--dry-run-manifest-validation",
        action="store_true",
        help="Validate the frozen 120-prompt contract without loading torch/SAM2.",
    )
    args = parser.parse_args()
    if not args.dry_run_manifest_validation and args.output_dir is None:
        parser.error("--output-dir is required unless using --dry-run-manifest-validation")
    if args.image_batch_size != 4:
        parser.error("Round-62 freezes --image-batch-size at 4")
    if args.min_reliable_views_per_track < 1:
        parser.error("--min-reliable-views-per-track must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: Any) -> str:
    digest = hashlib.sha256()
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def resolved_file(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} is not a file: {result}")
    return result


def load_frozen_prompts(
    manifest_path: Path,
    data_root: Path,
    area_tolerance: float,
    generic_contract: bool = False,
) -> tuple[
    dict[str, Any], list[dict[str, Any]],
    dict[tuple[int, int], list[dict[str, Any]]], tuple[int, ...]
]:
    """Validate and normalize the exact Round-61 prompt records."""
    manifest_hash = sha256_file(manifest_path)
    if not generic_contract and manifest_hash != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            "Track manifest is not the frozen Round-61 ungated source: "
            f"expected {FROZEN_MANIFEST_SHA256}, got {manifest_hash}"
        )
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracks = source.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("Track manifest must contain a 'tracks' list")
    by_track: dict[int, dict[str, Any]] = {}
    for track in tracks:
        if not isinstance(track, dict) or "track_id" not in track:
            raise ValueError("Every track record must contain track_id")
        track_id = int(track["track_id"])
        if track_id in by_track:
            raise ValueError(f"Duplicate track_id in manifest: {track_id}")
        by_track[track_id] = track
    if not generic_contract and set(by_track) != set(FROZEN_TRACK_IDS):
        raise ValueError(
            f"Expected frozen tracks {list(FROZEN_TRACK_IDS)}, got {sorted(by_track)}"
        )

    track_ids = tuple(sorted(by_track)) if generic_contract else FROZEN_TRACK_IDS
    images_dir = (data_root / "images").resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Scene image directory does not exist: {images_dir}")

    prompts: list[dict[str, Any]] = []
    by_image: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen_track_views: set[tuple[int, int, int]] = set()
    for track_id in track_ids:
        selected = by_track[track_id].get("selected_views")
        if not isinstance(selected, list) or len(selected) != EXPECTED_VIEWS_PER_TRACK:
            raise ValueError(
                f"Track {track_id} must have {EXPECTED_VIEWS_PER_TRACK} selected views"
            )
        for view_index, view in enumerate(selected):
            if not isinstance(view, dict):
                raise ValueError(f"Track {track_id} view {view_index} is not an object")
            frame, camera = int(view["frame"]), int(view["camera"])
            identity = (track_id, frame, camera)
            if identity in seen_track_views and not generic_contract:
                raise ValueError(f"Duplicate selected track-view: {identity}")
            seen_track_views.add(identity)
            bbox = [float(value) for value in view["bbox_xyxy"]]
            if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                raise ValueError(f"Invalid bbox for track-view {identity}: {bbox}")
            x0, y0, x1, y1 = bbox
            if not (0.0 <= x0 <= x1 <= IMAGE_WIDTH - 1.0):
                raise ValueError(f"Out-of-range bbox x coordinates for {identity}: {bbox}")
            if not (0.0 <= y0 <= y1 <= IMAGE_HEIGHT - 1.0):
                raise ValueError(f"Out-of-range bbox y coordinates for {identity}: {bbox}")
            calculated_area = (x1 - x0) * (y1 - y0) / (IMAGE_WIDTH * IMAGE_HEIGHT)
            recorded_area = float(view["bbox_area_fraction"])
            area_error = abs(calculated_area - recorded_area)
            if area_error > area_tolerance:
                raise ValueError(
                    f"bbox area mismatch for {identity}: recorded={recorded_area:.17g}, "
                    f"calculated={calculated_area:.17g}, error={area_error:.3g}"
                )
            image_path = (images_dir / f"{frame:03d}_{camera}.jpg").resolve()
            try:
                image_path.relative_to(images_dir)
            except ValueError as exc:
                raise RuntimeError(f"RGB path escaped image root: {image_path}") from exc
            if not image_path.is_file():
                raise FileNotFoundError(f"Selected-view RGB is missing: {image_path}")
            prompt = {
                "prompt_index": len(prompts),
                "track_id": track_id,
                "track_view_index": view_index,
                "frame": frame,
                "camera": camera,
                "image_path": str(image_path),
                "bbox_xyxy": bbox,
                "bbox_area_fraction_recorded": recorded_area,
                "bbox_area_fraction_calculated": calculated_area,
                "bbox_area_absolute_error": area_error,
            }
            prompts.append(prompt)
            by_image[(frame, camera)].append(prompt)

    expected_prompts = len(track_ids) * EXPECTED_VIEWS_PER_TRACK
    if len(prompts) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} prompts, got {len(prompts)}")
    if not generic_contract and len(by_image) != EXPECTED_UNIQUE_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_IMAGES} unique RGBs, got {len(by_image)}"
        )
    return source, prompts, dict(by_image), track_ids


def resolve_sam2_config(sam2_root: Path, config_arg: str) -> tuple[str, Path]:
    """Return Hydra's config identifier and the physical file used for hashing."""
    supplied = Path(config_arg).expanduser()
    if supplied.is_absolute():
        physical = resolved_file(supplied, "SAM2 config")
        package_config = (sam2_root / "sam2").resolve()
        try:
            identifier = physical.relative_to(package_config).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Absolute SAM2 config must be below {package_config}: {physical}"
            ) from exc
        return identifier, physical
    identifier = supplied.as_posix()
    physical = resolved_file(sam2_root / "sam2" / supplied, "SAM2 config")
    return identifier, physical


def crop_parameters(
    original_width: int,
    original_height: int,
    principal_x: float,
    principal_y: float,
) -> tuple[int, int, int, int]:
    """Mirror ArgoverseDriveStudioClips._crop_parameters exactly."""
    target_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    original_ratio = original_width / original_height
    if original_ratio > target_ratio:
        crop_height = original_height
        crop_width = min(original_width, int(round(crop_height * target_ratio)))
        crop_x = int(round(principal_x - crop_width / 2))
        crop_x = max(0, min(crop_x, original_width - crop_width))
        crop_y = 0
    else:
        crop_width = original_width
        crop_height = min(original_height, int(round(crop_width / target_ratio)))
        crop_x = 0
        crop_y = int(round(principal_y - crop_height / 2))
        crop_y = max(0, min(crop_y, original_height - crop_height))
    return crop_x, crop_y, crop_width, crop_height


def read_principal_point(path: Path) -> tuple[float, float]:
    values = [float(token) for token in path.read_text(encoding="utf-8").split()]
    if len(values) < 4:
        raise ValueError(f"Intrinsic file has fewer than four values: {path}")
    return values[2], values[3]


def bbox_pixel_mask(bbox: Iterable[float], np_module: Any) -> Any:
    x0, y0, x1, y1 = bbox
    left = max(0, int(math.floor(x0)))
    top = max(0, int(math.floor(y0)))
    right = min(IMAGE_WIDTH, int(math.ceil(x1)) + 1)
    bottom = min(IMAGE_HEIGHT, int(math.ceil(y1)) + 1)
    mask = np_module.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np_module.bool_)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = True
    return mask


def synchronize(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


class TimedPredictor:
    """Transparent predictor proxy that times encoder and decoder separately."""

    def __init__(self, predictor: Any, torch_module: Any, device: Any) -> None:
        self.predictor = predictor
        self.torch = torch_module
        self.device = device
        self.encoder_seconds = 0.0
        self.decoder_seconds = 0.0

    def set_image_batch(self, *args: Any, **kwargs: Any) -> Any:
        synchronize(self.torch, self.device)
        started = time.perf_counter()
        result = self.predictor.set_image_batch(*args, **kwargs)
        synchronize(self.torch, self.device)
        self.encoder_seconds += time.perf_counter() - started
        return result

    def predict_batch(self, *args: Any, **kwargs: Any) -> Any:
        synchronize(self.torch, self.device)
        started = time.perf_counter()
        result = self.predictor.predict_batch(*args, **kwargs)
        synchronize(self.torch, self.device)
        self.decoder_seconds += time.perf_counter() - started
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.predictor, name)


def main() -> None:
    process_started = time.perf_counter()
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = args.track_manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = resolved_file(manifest_path, "track manifest")
    data_root = args.data_root.expanduser().resolve()

    validation_started = time.perf_counter()
    _, prompts, by_image, track_ids = load_frozen_prompts(
        manifest_path, data_root, args.area_tolerance, args.generic_contract
    )
    prompt_count = len(prompts)
    validation_seconds = time.perf_counter() - validation_started
    validation_summary = {
        "status": "validated",
        "track_manifest": str(manifest_path),
        "track_manifest_sha256": sha256_file(manifest_path),
        "contract": "generic-12-view" if args.generic_contract else "round62-frozen",
        "frozen_track_ids": None if args.generic_contract else list(FROZEN_TRACK_IDS),
        "track_ids": list(track_ids),
        "track_count": len(track_ids),
        "prompts": len(prompts),
        "unique_rgb_images": len(by_image),
        "views_per_track": EXPECTED_VIEWS_PER_TRACK,
        "maximum_bbox_area_absolute_error": max(
            prompt["bbox_area_absolute_error"] for prompt in prompts
        ),
        "area_tolerance": args.area_tolerance,
        "validation_seconds": validation_seconds,
        "sam2_loaded": False,
    }
    if args.dry_run_manifest_validation:
        validation_summary["total_seconds"] = time.perf_counter() - process_started
        print(json.dumps(validation_summary, indent=2, ensure_ascii=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sam2_root = args.sam2_root.expanduser().resolve()
    checkpoint = resolved_file(args.checkpoint, "SAM2 checkpoint")
    config_identifier, config_path = resolve_sam2_config(sam2_root, args.config)
    batching_root = args.sam2_batching_root.expanduser().resolve()
    batching_path = resolved_file(
        batching_root / "pbr_vehicle_sun" / "sam2_batching.py", "SAM2 batching helper"
    )

    model_load_started = time.perf_counter()
    sys.path.insert(0, str(sam2_root))
    sys.path.insert(0, str(batching_root))
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from pbr_vehicle_sun.sam2_batching import process_sam2_image_batch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    predictor = SAM2ImagePredictor(
        build_sam2(config_identifier, str(checkpoint), device=device)
    )
    timed_predictor = TimedPredictor(predictor, torch, device)
    synchronize(torch, device)
    model_load_seconds = time.perf_counter() - model_load_started

    principal_points: dict[int, tuple[float, float]] = {}
    for camera in sorted({prompt["camera"] for prompt in prompts}):
        intrinsic_path = resolved_file(
            data_root / "intrinsics" / f"{camera}.txt", f"camera {camera} intrinsics"
        )
        principal_points[camera] = read_principal_point(intrinsic_path)

    rgb_records: list[dict[str, Any]] = []
    prompt_results: dict[int, dict[str, Any]] = {}
    raw_masks: dict[int, Any] = {}
    production_masks: dict[int, Any] = {}
    bbox_masks: dict[int, Any] = {}
    rgb_io_preprocess_seconds = 0.0
    raw_postprocess_seconds = 0.0
    image_items = sorted(by_image.items())
    decoder_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if device.type == "cuda" else nullcontext
    )
    for batch_start in range(0, len(image_items), args.image_batch_size):
        io_started = time.perf_counter()
        loaded: list[tuple[str, Any, list[dict[str, Any]]]] = []
        for (frame, camera), records in image_items[
            batch_start:batch_start + args.image_batch_size
        ]:
            raw_path = Path(records[0]["image_path"])
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError(f"OpenCV could not read selected RGB: {raw_path}")
            original_height, original_width = raw.shape[:2]
            principal_x, principal_y = principal_points[camera]
            crop_x, crop_y, crop_width, crop_height = crop_parameters(
                original_width, original_height, principal_x, principal_y
            )
            calibrated = raw[
                crop_y:crop_y + crop_height,
                crop_x:crop_x + crop_width,
            ]
            calibrated = cv2.resize(
                calibrated, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
            if calibrated.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise RuntimeError(
                    f"Unexpected calibrated RGB shape {calibrated.shape}: {raw_path}"
                )
            loaded.append((str(raw_path), calibrated, records))
            rgb_records.append({
                "frame": frame,
                "camera": camera,
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "original_size_wh": [original_width, original_height],
                "crop_xywh": [crop_x, crop_y, crop_width, crop_height],
                "calibrated_size_wh": [IMAGE_WIDTH, IMAGE_HEIGHT],
                "calibrated_bgr_sha256": sha256_array(calibrated),
                "prompt_indices": [record["prompt_index"] for record in records],
                "track_ids": [record["track_id"] for record in records],
            })
        rgb_io_preprocess_seconds += time.perf_counter() - io_started

        with torch.inference_mode():
            predictions = process_sam2_image_batch(
                timed_predictor, loaded, decoder_context=decoder_context
            )
        post_started = time.perf_counter()
        for prompt, raw_mask, score in predictions:
            prompt_index = int(prompt["prompt_index"])
            raw_mask = np.asarray(raw_mask, dtype=np.bool_)
            if raw_mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise RuntimeError(
                    f"SAM2 mask for prompt {prompt_index} has shape {raw_mask.shape}"
                )
            bbox_mask = bbox_pixel_mask(prompt["bbox_xyxy"], np)
            raw_pixels = int(raw_mask.sum())
            inside_pixels = int(np.logical_and(raw_mask, bbox_mask).sum())
            bbox_pixels = int(bbox_mask.sum())
            spill_pixels = raw_pixels - inside_pixels
            raw_spill = spill_pixels / max(raw_pixels, 1)
            production = np.logical_and(raw_mask, bbox_mask)
            production_pixels = int(production.sum())
            production_bbox_ratio = production_pixels / max(bbox_pixels, 1)
            x0, y0, x1, y1 = prompt["bbox_xyxy"]
            bbox_width, bbox_height = x1 - x0, y1 - y0
            bbox_geometric_area = bbox_width * bbox_height
            reasons = []
            if raw_pixels == 0:
                reasons.append("empty_raw_mask")
            if not math.isfinite(float(score)):
                reasons.append("non_finite_predicted_iou")
            elif float(score) < args.min_predicted_iou:
                reasons.append("predicted_iou_below_threshold")
            if production_bbox_ratio < args.min_mask_bbox_ratio:
                reasons.append("production_mask_bbox_ratio_below_threshold")
            if min(bbox_width, bbox_height) < args.min_bbox_edge_px:
                reasons.append("bbox_short_edge_below_threshold")
            if bbox_geometric_area < args.min_bbox_area_px:
                reasons.append("bbox_area_below_threshold")
            if raw_spill > args.max_raw_spill:
                reasons.append("raw_spill_above_threshold")
            reliable = not reasons
            raw_masks[prompt_index] = raw_mask.copy()
            production_masks[prompt_index] = production if reliable else np.zeros_like(production)
            bbox_masks[prompt_index] = bbox_mask
            prompt_results[prompt_index] = {
                **prompt,
                "sam2_predicted_iou": float(score),
                "raw_mask_pixels": raw_pixels,
                "raw_mask_sha256": sha256_array(raw_mask),
                "raw_spill_pixels": spill_pixels,
                "raw_spill_fraction": raw_spill,
                "bbox_raster_pixels": bbox_pixels,
                "production_mask_pixels_before_track_fallback": production_pixels,
                "production_mask_bbox_ratio": production_bbox_ratio,
                "single_view_reliable": reliable,
                "single_view_rejection_reasons": reasons,
            }
        raw_postprocess_seconds += time.perf_counter() - post_started

    if set(prompt_results) != set(range(prompt_count)):
        missing = sorted(set(range(prompt_count)) - set(prompt_results))
        raise RuntimeError(f"SAM2 did not return every prompt; missing indices: {missing}")

    reliable_by_track: dict[int, int] = defaultdict(int)
    for result in prompt_results.values():
        reliable_by_track[result["track_id"]] += int(result["single_view_reliable"])
    track_fallback = {
        track_id: reliable_by_track[track_id] < args.min_reliable_views_per_track
        for track_id in track_ids
    }

    output_write_started = time.perf_counter()
    output_records: list[dict[str, Any]] = []
    for prompt_index in range(prompt_count):
        result = prompt_results[prompt_index]
        track_id = result["track_id"]
        fallback = track_fallback[track_id]
        final_mask = bbox_masks[prompt_index] if fallback else production_masks[prompt_index]
        track_dir = output_dir / f"track-{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        output_path = track_dir / f"{result['frame']:03d}_{result['camera']}.png"
        if not cv2.imwrite(str(output_path), final_mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to write mask: {output_path}")
        raw_mask = raw_masks[prompt_index]
        raw_track_dir = output_dir / "raw-track-masks" / f"track-{track_id}"
        raw_track_dir.mkdir(parents=True, exist_ok=True)
        raw_output_path = (
            raw_track_dir / f"{result['frame']:03d}_{result['camera']}.png"
        )
        if not cv2.imwrite(str(raw_output_path), raw_mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to write raw SAM2 mask: {raw_output_path}")
        result.update({
            "track_fallback_to_projected_bbox": fallback,
            "final_mask_semantics": (
                "projected_bbox_fallback" if fallback
                else ("sam2_bbox_intersection" if result["single_view_reliable"] else "empty")
            ),
            "final_mask_pixels": int(final_mask.sum()),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "raw_output_path": str(raw_output_path),
            "raw_output_png_sha256": sha256_file(raw_output_path),
            "raw_output_array_sha256": sha256_array(raw_mask),
        })
        output_records.append(result)
    output_write_seconds = time.perf_counter() - output_write_started
    synchronize(torch, device)
    total_seconds_through_mask_outputs = time.perf_counter() - process_started

    prompt_manifest_write_started = time.perf_counter()
    prompt_manifest = {
        "schema_version": 1,
        "contract": (
            "generic per-track projected-3D-bbox SAM2 prompts"
            if args.generic_contract else
            "Round-62 frozen per-track projected-3D-bbox SAM2 prompts"
        ),
        "production_inputs_only_rgb_and_projected_bbox": True,
        "track_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "calibrated_resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "frozen_track_ids": None if args.generic_contract else list(FROZEN_TRACK_IDS),
        "track_ids": list(track_ids),
        "prompt_count": len(prompts),
        "unique_rgb_count": len(by_image),
        "maximum_bbox_area_absolute_error": max(
            prompt["bbox_area_absolute_error"] for prompt in prompts
        ),
        "area_tolerance": args.area_tolerance,
        "prompts": prompts,
    }
    prompt_manifest_path = output_dir / "prompt_manifest.json"
    prompt_manifest_path.write_text(
        json.dumps(prompt_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prompt_manifest_write_seconds = time.perf_counter() - prompt_manifest_write_started
    total_seconds_before_final_manifest = time.perf_counter() - process_started

    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    gpu = {"device": str(device), "cuda_available": bool(torch.cuda.is_available())}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu.update({
            "index": int(device.index if device.index is not None else 0),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        })
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "model": "SAM2.1 Hiera Tiny",
        "production_inputs_only_rgb_and_projected_bbox": True,
        "mask_contract": {
            "raw_spill_computed_before_bbox_crop": True,
            "production_mask": "raw SAM2 mask intersected with projected bbox",
            "single_view_failure": "empty mask",
            "track_failure": (
                f"fewer than {args.min_reliable_views_per_track} reliable views makes all "
                "track views projected-bbox masks (ungated equivalent)"
            ),
            "output_encoding": "uint8 PNG; foreground=255, background=0",
            "output_layout": "track-{track_id}/{frame:03d}_{camera}.png",
        },
        "counts": {
            "tracks": len(track_ids),
            "prompts": len(prompts),
            "unique_rgb_images": len(by_image),
            "single_view_reliable": sum(
                int(item["single_view_reliable"]) for item in output_records
            ),
            "single_view_rejected": sum(
                not item["single_view_reliable"] for item in output_records
            ),
            "tracks_falling_back": sum(track_fallback.values()),
        },
        "thresholds": {
            "predicted_iou_minimum": args.min_predicted_iou,
            "production_mask_bbox_ratio_minimum": args.min_mask_bbox_ratio,
            "raw_spill_maximum": args.max_raw_spill,
            "bbox_short_edge_minimum_px": args.min_bbox_edge_px,
            "bbox_area_minimum_px": args.min_bbox_area_px,
            "reliable_views_per_track_minimum": args.min_reliable_views_per_track,
        },
        "tracks": [
            {
                "track_id": track_id,
                "selected_view_count": EXPECTED_VIEWS_PER_TRACK,
                "reliable_view_count": reliable_by_track[track_id],
                "fallback_to_projected_bbox": track_fallback[track_id],
            }
            for track_id in track_ids
        ],
        "inputs": {
            "track_manifest": {
                "path": str(manifest_path), "sha256": sha256_file(manifest_path)
            },
            "data_root": str(data_root),
            "rgb_files": rgb_records,
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "config": {
                "identifier": config_identifier,
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "sam2_source": {
                "root": str(sam2_root),
                "hashed_files": [
                    {
                        "path": str(resolved_file(sam2_root / "sam2" / name, "SAM2 source")),
                        "sha256": sha256_file(
                            resolved_file(sam2_root / "sam2" / name, "SAM2 source")
                        ),
                    }
                    for name in ("build_sam.py", "sam2_image_predictor.py")
                ],
            },
            "sam2_batching_helper": {
                "path": str(batching_path), "sha256": sha256_file(batching_path)
            },
        },
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "batching": {
            "image_batch_size": args.image_batch_size,
            "all_boxes_for_same_image_in_one_decoder_call": True,
            "multimask_selection": "highest SAM2 predicted IoU",
        },
        "outputs": {
            "directory": str(output_dir),
            "mask_contract": {
                "production": {
                    "layout": "track-{track_id}/{frame:03d}_{camera}.png",
                    "semantics": (
                        "existing reliable SAM2/projected-bbox intersection with "
                        "single-view empty and track-level projected-bbox fallback"
                    ),
                    "encoding": "uint8 PNG; foreground=255, background=0",
                },
                "raw_sam2": {
                    "layout": (
                        "raw-track-masks/track-{track_id}/"
                        "{frame:03d}_{camera}.png"
                    ),
                    "semantics": (
                        "uncropped raw SAM2 boolean mask before projected-bbox "
                        "intersection, reliability rejection, or track fallback"
                    ),
                    "encoding": "uint8 PNG; foreground=255, background=0",
                    "array_hash_semantics": (
                        "sha256 over the contiguous calibrated-resolution bool array"
                    ),
                },
            },
            "prompt_manifest": {
                "path": str(prompt_manifest_path),
                "sha256": sha256_file(prompt_manifest_path),
            },
            "mask_files": output_records,
        },
        "timing_seconds_cold_start": {
            "manifest_validation": validation_seconds,
            "model_import_and_load": model_load_seconds,
            "rgb_io_and_calibrated_preprocess": rgb_io_preprocess_seconds,
            "sam2_image_encoder": timed_predictor.encoder_seconds,
            "sam2_prompt_decoder": timed_predictor.decoder_seconds,
            "raw_spill_crop_and_reliability": raw_postprocess_seconds,
            "mask_output_write": output_write_seconds,
            "prompt_manifest_write": prompt_manifest_write_seconds,
            "total_through_mask_outputs": total_seconds_through_mask_outputs,
            "total_before_final_manifest_write": total_seconds_before_final_manifest,
        },
        "peak_cuda_memory": {
            "bytes": peak_cuda_bytes,
            "gib": peak_cuda_bytes / (1024 ** 3),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "opencv_version": cv2.__version__,
            "gpu": gpu,
        },
    }
    manifest_path_out = output_dir / "manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest_path_out),
        "prompts": len(prompts),
        "unique_rgb_images": len(by_image),
        "single_view_reliable": manifest["counts"]["single_view_reliable"],
        "tracks_falling_back": manifest["counts"]["tracks_falling_back"],
        "timing_seconds_cold_start": manifest["timing_seconds_cold_start"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

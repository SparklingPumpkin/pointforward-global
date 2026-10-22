#!/usr/bin/env python3
"""Build the fixed Round-64 RGB vehicle and swept-background proxy payload.

The production path consumes RGB, projected boxes, SAM2 masks, and the frozen
PointForward dense-depth model.  Evaluation masks are deliberately absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.audit_dynamic_depth_box_contract import (
    Box,
    collect_selected_views,
    predict_depths,
    prepare_dataset,
    ray_box_interval,
)


H, W = 576, 768
VOXEL_M = 0.02
OPACITY = 0.85
DEPTH_TOLERANCE_M = 0.10
BACKGROUND_MARGIN_M = 0.05
MASK_DILATION_PX = 3
MIN_RELIABLE_MASKS = 4
MIN_DYNAMIC_INPUT_PIXELS = 256
MIN_DYNAMIC_VOXELS = 64
MIN_SUPPORT = 2
BOX_SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    out = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            out.update(block)
    return out.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--track-manifest", required=True, type=Path)
    parser.add_argument("--sam2-dir", required=True, type=Path)
    parser.add_argument("--baseline-payload-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth-batch-size", type=int, default=12)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != (H, W):
        raise ValueError(f"Unexpected mask shape {mask.shape}: {path}")
    return mask


def box_mask(bounds: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    left, top = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    right, bottom = min(W, int(math.ceil(x1)) + 1), min(H, int(math.ceil(y1)) + 1)
    result = np.zeros((H, W), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def box_may_hit_image(
    box: Box, intrinsic: np.ndarray, camera_to_world: np.ndarray
) -> bool:
    """Conservative cheap cull before exact full-resolution ray/OBB testing."""
    corners = BOX_SIGNS * (box.dims * 0.5)
    world = corners @ box.rotation + box.center
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    front = camera[:, 2] > 1e-5
    if not front.any():
        return False
    # A box crossing the camera plane is rare but cannot be culled safely.
    if not front.all():
        return True
    uvw = camera @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return bool(
        uv[:, 0].max() >= 0 and uv[:, 0].min() < W
        and uv[:, 1].max() >= 0 and uv[:, 1].min() < H
    )


def world_points(
    ys: np.ndarray,
    xs: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    pixels = np.column_stack((xs.astype(np.float64) + 0.5,
                              ys.astype(np.float64) + 0.5,
                              np.ones(len(xs), dtype=np.float64)))
    camera = (pixels @ np.linalg.inv(intrinsic).T) * depth[:, None]
    return (camera @ camera_to_world[:3, :3].T
            + camera_to_world[:3, 3]).astype(np.float32)


def aggregate_one_support(
    means: np.ndarray, colors: np.ndarray, scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate a support image/view before cross-view support counting."""
    if len(means) == 0:
        empty3 = np.empty((0, 3), dtype=np.float32)
        return (np.empty((0, 3), dtype=np.int32), empty3, empty3.copy(),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64))
    keys = np.floor(means / VOXEL_M).astype(np.int32)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    count = np.bincount(inverse).astype(np.int64)
    mean_sum = np.column_stack([
        np.bincount(inverse, weights=means[:, channel]) for channel in range(3)
    ]).astype(np.float32)
    color_sum = np.column_stack([
        np.bincount(inverse, weights=colors[:, channel]) for channel in range(3)
    ]).astype(np.float32)
    scale_sum = np.bincount(inverse, weights=scales).astype(np.float32)
    return keys[first], mean_sum, color_sum, scale_sum, count


def finish_voxels(
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    track_ids: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    if not chunks:
        return {
            "means": np.empty((0, 3), np.float32),
            "colors": np.empty((0, 3), np.float32),
            "scales": np.empty(0, np.float32),
            "support": np.empty(0, np.int32),
            "track_id": np.empty(0, np.int32),
        }
    keys = np.concatenate([chunk[0] for chunk in chunks])
    if track_ids is not None:
        tracks = np.concatenate(track_ids).astype(np.int32)
        group_keys = np.column_stack((tracks, keys))
    else:
        tracks = None
        group_keys = keys
    unique, first, inverse = np.unique(
        group_keys, axis=0, return_index=True, return_inverse=True
    )
    support = np.bincount(inverse).astype(np.int32)
    pixel_count = np.bincount(
        inverse, weights=np.concatenate([chunk[4] for chunk in chunks])
    ).astype(np.float64)
    means_in = np.concatenate([chunk[1] for chunk in chunks])
    colors_in = np.concatenate([chunk[2] for chunk in chunks])
    scales_in = np.concatenate([chunk[3] for chunk in chunks])
    means_sum = np.column_stack([
        np.bincount(inverse, weights=means_in[:, channel]) for channel in range(3)
    ])
    colors_sum = np.column_stack([
        np.bincount(inverse, weights=colors_in[:, channel]) for channel in range(3)
    ])
    scales_sum = np.bincount(inverse, weights=scales_in)
    keep = support >= MIN_SUPPORT
    return {
        "means": (means_sum[keep] / pixel_count[keep, None]).astype(np.float32),
        "colors": (colors_sum[keep] / pixel_count[keep, None]).astype(np.float32),
        "scales": (scales_sum[keep] / pixel_count[keep]).astype(np.float32),
        "support": support[keep],
        "track_id": (
            unique[keep, 0].astype(np.int32) if tracks is not None
            else np.full(int(keep.sum()), -1, dtype=np.int32)
        ),
    }


def payload_from_arrays(data: dict[str, np.ndarray], instance_ids: np.ndarray) -> dict[str, torch.Tensor]:
    count = len(data["means"])
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    scales = np.repeat(data["scales"][:, None], 3, axis=1)
    return {
        "means": torch.from_numpy(data["means"]),
        "quats": torch.from_numpy(quats),
        "scales": torch.from_numpy(scales),
        "opacities": torch.full((count,), OPACITY, dtype=torch.float32),
        "colors": torch.from_numpy(data["colors"]),
        "sigma": torch.full((count,), 1e6, dtype=torch.float32),
        "times": torch.zeros(count, dtype=torch.float32),
        "instance_id": torch.from_numpy(instance_ids.astype(np.int64)),
        "extras": {"support_views": torch.from_numpy(data["support"])},
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.out.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    track_manifest_path = args.track_manifest.expanduser().resolve()
    sam_dir = args.sam2_dir.expanduser().resolve()
    baseline_dir = args.baseline_payload_dir.expanduser().resolve()
    source = json.loads(track_manifest_path.read_text(encoding="utf-8"))
    sam = json.loads((sam_dir / "manifest.json").read_text(encoding="utf-8"))

    cfg, dataset, metadata, world_from = prepare_dataset(
        args.config.expanduser().resolve(), args.scene
    )
    frames = [int(value) for value in metadata["frames"]]
    frame_slot = {frame: index for index, frame in enumerate(frames)}
    meta = torch.load(
        baseline_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    track_ids = [int(value) for value in meta["track_ids"]]
    track_slot = {track_id: index for index, track_id in enumerate(track_ids)}
    boxes = {name: meta["boxes"][name].cpu().numpy() for name in ("centers", "dims", "rots", "valid")}
    canonical_slot = boxes["valid"].argmax(axis=1)

    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    requested: set[tuple[int, int]] = set()
    for track in source["tracks"]:
        track_id = int(track["track_id"])
        for view in track["selected_views"]:
            key = (track_id, int(view["frame"]), int(view["camera"]))
            records[key] = view
            requested.add(key[1:])

    mask_records = {}
    for item in sam["outputs"]["mask_files"]:
        key = (int(item["track_id"]), int(item["frame"]), int(item["camera"]))
        mask_records[key] = item
    sam_tracks = {int(item["track_id"]): item for item in sam["tracks"]}
    initially_reliable = {
        track_id for track_id, item in sam_tracks.items()
        if not item["fallback_to_projected_bbox"]
        and int(item["reliable_view_count"]) >= MIN_RELIABLE_MASKS
    }

    collect_started = time.perf_counter()
    calibrated = collect_selected_views(dataset, metadata, world_from, requested)
    collect_seconds = time.perf_counter() - collect_started
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    depths, depth_timing = predict_depths(
        cfg, args.ckpt.expanduser().resolve(), device, calibrated, args.depth_batch_size
    )

    # Load only truly reliable masks.  Projected-box fallback masks are never
    # interpreted as vehicle silhouettes by this proxy builder.
    masks: dict[tuple[int, int, int], np.ndarray] = {}
    for key, item in mask_records.items():
        if key[0] in initially_reliable and item["single_view_reliable"]:
            masks[key] = load_mask(Path(item["output_path"]))

    geometry_started = time.perf_counter()
    dynamic_chunks = []
    dynamic_chunk_tracks = []
    dynamic_input_pixels: dict[int, int] = defaultdict(int)
    dynamic_candidate_pixels: dict[int, int] = defaultdict(int)
    dynamic_valid_depth_pixels: dict[int, int] = defaultdict(int)
    dynamic_view_count: dict[int, int] = defaultdict(int)

    # Cache all-track exact OBB intervals per selected RGB.  This is also the
    # all-track exclusion contract used by swept-background candidates.
    obb_cache: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for image_key in sorted(requested):
        frame, _ = image_key
        slot = frame_slot[frame]
        per_image = []
        intrinsic = calibrated[image_key]["intrinsic"]
        camera_to_world = calibrated[image_key]["camera_to_world"]
        for global_slot, track_id in enumerate(track_ids):
            if not boxes["valid"][global_slot, slot]:
                continue
            box = Box(
                center=boxes["centers"][global_slot, slot].astype(np.float64),
                dims=boxes["dims"][global_slot, slot].astype(np.float64),
                rotation=boxes["rots"][global_slot, slot].astype(np.float64),
            )
            if not box_may_hit_image(box, intrinsic, camera_to_world):
                continue
            hit, near, far = ray_box_interval(
                box, intrinsic, camera_to_world, return_depth=True
            )
            if hit.any():
                per_image.append((track_id, hit, near, far))
        obb_cache[image_key] = per_image

    for key in sorted(masks):
        track_id, frame, camera = key
        if track_id not in track_slot:
            continue
        image_key = (frame, camera)
        slot = frame_slot[frame]
        global_slot = track_slot[track_id]
        if not boxes["valid"][global_slot, slot]:
            continue
        target = next((item for item in obb_cache[image_key] if item[0] == track_id), None)
        if target is None:
            continue
        hit, near, far = target[1:]
        depth = depths[image_key]
        mask = masks[key]
        dynamic_candidate_pixels[track_id] += int(mask.sum())
        valid = (mask & hit & np.isfinite(depth) & (depth > 0)
                 & (depth >= near - DEPTH_TOLERANCE_M)
                 & (depth <= far + DEPTH_TOLERANCE_M))
        dynamic_valid_depth_pixels[track_id] += int(valid.sum())
        if not valid.any():
            continue
        clipped_depth = np.clip(depth[valid], near[valid], far[valid])
        ys, xs = np.nonzero(valid)
        datum = calibrated[image_key]
        points = world_points(
            ys, xs, clipped_depth, datum["intrinsic"], datum["camera_to_world"]
        )
        current_center = boxes["centers"][global_slot, slot]
        current_rotation = boxes["rots"][global_slot, slot]
        can_index = int(canonical_slot[global_slot])
        canonical_center = boxes["centers"][global_slot, can_index]
        canonical_rotation = boxes["rots"][global_slot, can_index]
        local = (points - current_center) @ current_rotation.T
        canonical = local @ canonical_rotation + canonical_center
        rgb = np.moveaxis(datum["image"], 0, -1)[ys, xs].astype(np.float32)
        focal = math.sqrt(float(datum["intrinsic"][0, 0] * datum["intrinsic"][1, 1]))
        scale = np.clip(1.5 * clipped_depth / focal, 0.005, 0.04).astype(np.float32)
        chunk = aggregate_one_support(canonical.astype(np.float32), rgb, scale)
        if len(chunk[0]):
            dynamic_chunks.append(chunk)
            dynamic_chunk_tracks.append(np.full(len(chunk[0]), track_id, np.int32))
            dynamic_input_pixels[track_id] += int(valid.sum())
            dynamic_view_count[track_id] += 1

    dynamic = finish_voxels(dynamic_chunks, track_ids=dynamic_chunk_tracks)

    # Static candidates are unioned and de-duplicated once per image so several
    # track boxes in one RGB never fake the distinct-view support requirement.
    static_chunks = []
    static_input_pixels = 0
    kernel = np.ones((2 * MASK_DILATION_PX + 1,) * 2, dtype=np.uint8)
    masks_by_image: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (track_id, frame, camera), mask in masks.items():
        masks_by_image[(frame, camera)].append((track_id, mask))
    for image_key, image_masks in sorted(masks_by_image.items()):
        region = np.zeros((H, W), dtype=bool)
        foreground = np.zeros((H, W), dtype=bool)
        for track_id, mask in image_masks:
            region |= box_mask(records[(track_id, *image_key)]["bbox_xyxy"])
            foreground |= cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        candidate = region & ~foreground
        depth = depths[image_key]
        allowed = candidate & np.isfinite(depth) & (depth > 0)
        for _, hit, _, far in obb_cache[image_key]:
            allowed &= (~hit) | (depth > far + BACKGROUND_MARGIN_M)
        if not allowed.any():
            continue
        ys, xs = np.nonzero(allowed)
        datum = calibrated[image_key]
        selected_depth = depth[allowed]
        points = world_points(
            ys, xs, selected_depth, datum["intrinsic"], datum["camera_to_world"]
        )
        rgb = np.moveaxis(datum["image"], 0, -1)[ys, xs].astype(np.float32)
        focal = math.sqrt(float(datum["intrinsic"][0, 0] * datum["intrinsic"][1, 1]))
        scale = np.clip(1.5 * selected_depth / focal, 0.005, 0.04).astype(np.float32)
        chunk = aggregate_one_support(points, rgb, scale)
        if len(chunk[0]):
            static_chunks.append(chunk)
            static_input_pixels += int(allowed.sum())

    static = finish_voxels(static_chunks)
    geometry_seconds = time.perf_counter() - geometry_started

    dynamic_voxels_by_track = {
        track_id: int((dynamic["track_id"] == track_id).sum()) for track_id in track_ids
    }
    reliable_tracks = [
        track_id for track_id in track_ids
        if track_id in initially_reliable
        and dynamic_input_pixels[track_id] >= MIN_DYNAMIC_INPUT_PIXELS
        and dynamic_voxels_by_track[track_id] >= MIN_DYNAMIC_VOXELS
    ]
    reliable_set = set(reliable_tracks)
    keep_dynamic = np.asarray(
        [track_id in reliable_set for track_id in dynamic["track_id"]], dtype=bool
    )
    dynamic = {name: values[keep_dynamic] for name, values in dynamic.items()}
    instance_ids = np.asarray(
        [track_slot[int(track_id)] for track_id in dynamic["track_id"]], np.int64
    )

    write_started = time.perf_counter()
    dynamic_path = output_dir / "dynamic_proxy.pt"
    static_path = output_dir / "static_proxy.pt"
    torch.save(payload_from_arrays(dynamic, instance_ids), dynamic_path)
    torch.save(payload_from_arrays(static, np.full(len(static["means"]), -1)), static_path)
    write_seconds = time.perf_counter() - write_started

    track_records = []
    for track_id in track_ids:
        candidates = dynamic_candidate_pixels[track_id]
        valid_depth = dynamic_valid_depth_pixels[track_id]
        track_records.append({
            "track_id": track_id,
            "global_slot": track_slot[track_id],
            "sam_reliable_views": int(sam_tracks[track_id]["reliable_view_count"]),
            "sam_track_fallback": bool(sam_tracks[track_id]["fallback_to_projected_bbox"]),
            "used_reliable_views": dynamic_view_count[track_id],
            "sam_candidate_pixels": candidates,
            "ray_obb_depth_valid_pixels": valid_depth,
            "ray_obb_depth_pass_fraction": valid_depth / candidates if candidates else None,
            "accepted_dynamic_input_pixels": dynamic_input_pixels[track_id],
            "support_ge2_dynamic_voxels": dynamic_voxels_by_track[track_id],
            "reliable": track_id in reliable_set,
        })
    total_candidates = sum(dynamic_candidate_pixels.values())
    total_valid_depth = sum(dynamic_valid_depth_pixels.values())
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "production_inputs_exclude_evaluation_masks": True,
        "contract": {
            "voxel_m": VOXEL_M,
            "opacity": OPACITY,
            "scale": "clamp(0.75 * 2 * z / sqrt(fx*fy), 0.005, 0.04) m",
            "dynamic_depth_tolerance_m": DEPTH_TOLERANCE_M,
            "background_behind_all_obb_margin_m": BACKGROUND_MARGIN_M,
            "mask_dilation_px": MASK_DILATION_PX,
            "minimum_distinct_support": MIN_SUPPORT,
            "minimum_reliable_masks_per_track": MIN_RELIABLE_MASKS,
            "minimum_dynamic_input_pixels": MIN_DYNAMIC_INPUT_PIXELS,
            "minimum_dynamic_voxels": MIN_DYNAMIC_VOXELS,
        },
        "counts": {
            "tracks": len(track_ids),
            "initial_sam_reliable_tracks": len(initially_reliable),
            "reliable_tracks": len(reliable_tracks),
            "dynamic_proxy_voxels": int(len(dynamic["means"])),
            "static_proxy_voxels": int(len(static["means"])),
            "static_input_pixels": static_input_pixels,
            "dynamic_sam_candidate_pixels": total_candidates,
            "dynamic_ray_obb_depth_valid_pixels": total_valid_depth,
        },
        "dynamic_ray_obb_depth_pass_fraction": (
            total_valid_depth / total_candidates if total_candidates else None
        ),
        "reliable_track_ids": reliable_tracks,
        "reliable_global_slots": [track_slot[item] for item in reliable_tracks],
        "tracks": track_records,
        "outputs": {
            "dynamic_proxy": {"path": str(dynamic_path), "sha256": sha256_file(dynamic_path)},
            "static_proxy": {"path": str(static_path), "sha256": sha256_file(static_path)},
        },
        "inputs": {
            "config": str(args.config.expanduser().resolve()),
            "checkpoint": str(args.ckpt.expanduser().resolve()),
            "track_manifest": {"path": str(track_manifest_path), "sha256": sha256_file(track_manifest_path)},
            "sam2_manifest": {"path": str(sam_dir / "manifest.json"), "sha256": sha256_file(sam_dir / "manifest.json")},
            "baseline_global_dynamic_meta": str(baseline_dir / "global_dynamic_meta.pt"),
        },
        "timing_seconds": {
            "dataset_collect_rgb_lidar": collect_seconds,
            **depth_timing,
            "geometry_and_voxel": geometry_seconds,
            "payload_write": write_seconds,
            "total": time.perf_counter() - started,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "counts": manifest["counts"],
        "depth_pass_fraction": manifest["dynamic_ray_obb_depth_pass_fraction"],
        "timing_seconds": manifest["timing_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()

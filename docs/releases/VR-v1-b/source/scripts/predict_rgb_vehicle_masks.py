#!/usr/bin/env python3
"""Predict Cityscapes vehicle masks for selected track views using RGB only.

This utility deliberately has a narrow production input contract: apart from
the track manifest and model files, image content is read only from
``DATA_ROOT/images/{frame:03d}_{camera}.jpg``.  In particular, no dataset mask
directory is inspected or consumed.

Run this script with the ``segformer`` conda environment and an NVlabs
SegFormer Cityscapes checkpoint, for example::

    /root/miniconda3/envs/segformer/bin/python scripts/predict_rgb_vehicle_masks.py \
      --track-manifest track_once_manifest.json \
      --track-ids 47 26 0 \
      --data-root /mnt/data/dataset/argoverse/processed/training/017 \
      --config /root/src/SegFormer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py \
      --checkpoint /path/to/checkpoint.pth \
      --out /path/to/predicted_vehicle_masks --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable


VEHICLE_CLASS_INDICES = (13, 14, 15)
DEFAULT_SEGFORMER_ROOT = Path("/root/src/SegFormer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RGB-only SegFormer inference on selected track views."
    )
    parser.add_argument("--track-manifest", type=Path, required=True)
    parser.add_argument(
        "--track-ids",
        nargs="+",
        required=True,
        help="Track IDs separated by spaces and/or commas.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Scene directory whose images are in DATA_ROOT/images.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--segformer-root",
        type=Path,
        default=DEFAULT_SEGFORMER_ROOT,
        help="Local NVlabs SegFormer checkout (default: /root/src/SegFormer).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_track_ids(values: Iterable[str]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            track_id = int(token)
            if track_id not in seen:
                result.append(track_id)
                seen.add(track_id)
    if not result:
        raise ValueError("--track-ids did not contain any track IDs")
    return result


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def load_selected_views(
    manifest_path: Path, selected_track_ids: list[int]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[int, int], list[int]]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("Track manifest must contain a list at key 'tracks'")

    by_id: dict[int, dict[str, Any]] = {}
    for track in tracks:
        if not isinstance(track, dict) or "track_id" not in track:
            raise ValueError("Every track entry must be an object with 'track_id'")
        track_id = int(track["track_id"])
        if track_id in by_id:
            raise ValueError(f"Duplicate track_id in manifest: {track_id}")
        by_id[track_id] = track

    missing = [track_id for track_id in selected_track_ids if track_id not in by_id]
    if missing:
        raise KeyError(f"Requested track IDs absent from manifest: {missing}")

    selected_tracks: list[dict[str, Any]] = []
    view_to_tracks: dict[tuple[int, int], list[int]] = {}
    for track_id in selected_track_ids:
        source = by_id[track_id]
        views = source.get("selected_views")
        if not isinstance(views, list):
            raise ValueError(f"Track {track_id} has no selected_views list")
        normalized_views = []
        for view in views:
            if not isinstance(view, dict) or "frame" not in view or "camera" not in view:
                raise ValueError(
                    f"Track {track_id} has a selected view without frame/camera"
                )
            frame = int(view["frame"])
            camera = int(view["camera"])
            normalized_views.append({"frame": frame, "camera": camera})
            view_to_tracks.setdefault((frame, camera), []).append(track_id)
        selected_tracks.append(
            {
                "track_id": track_id,
                "selected_view_count": len(normalized_views),
                "selected_views": normalized_views,
            }
        )

    for track_ids in view_to_tracks.values():
        track_ids.sort()
    return manifest, selected_tracks, view_to_tracks


def synchronize(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def gpu_metadata(torch_module: Any, device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "cuda_available": bool(torch_module.cuda.is_available()),
            "device": str(device),
            "name": None,
        }
    properties = torch_module.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "device": str(device),
        "index": int(device.index if device.index is not None else 0),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "capability": [int(properties.major), int(properties.minor)],
    }


def main() -> None:
    process_start = time.perf_counter()
    args = parse_args()

    track_manifest = require_file(args.track_manifest, "track manifest")
    config = require_file(args.config, "SegFormer config")
    checkpoint = require_file(args.checkpoint, "SegFormer checkpoint")
    script_path = Path(__file__).resolve()
    segformer_root = args.segformer_root.expanduser().resolve()
    if not (segformer_root / "mmseg" / "apis" / "inference.py").is_file():
        raise FileNotFoundError(
            f"Not an NVlabs SegFormer checkout (missing mmseg APIs): {segformer_root}"
        )

    selected_track_ids = parse_track_ids(args.track_ids)
    _, selected_tracks, view_to_tracks = load_selected_views(
        track_manifest, selected_track_ids
    )
    unique_views = sorted(view_to_tracks)
    if not unique_views:
        raise ValueError("The requested tracks have no selected views")

    images_dir = (args.data_root.expanduser().resolve() / "images").resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"RGB image directory does not exist: {images_dir}")
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports are intentionally inside the measured cold model-load interval.
    model_load_start = time.perf_counter()
    sys.path.insert(0, str(segformer_root))
    import numpy as np
    import torch
    from PIL import Image
    from mmseg.apis import inference_segmentor, init_segmentor
    import mmcv
    import mmseg

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(torch, device)
    model = init_segmentor(str(config), str(checkpoint), device=str(device))
    synchronize(torch, device)
    model_load_seconds = time.perf_counter() - model_load_start

    processing_start = time.perf_counter()
    input_records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    for frame, camera in unique_views:
        # No glob/fallback is used: this exact RGB path is the sole image input.
        image_path = (images_dir / f"{frame:03d}_{camera}.jpg").resolve()
        try:
            image_path.relative_to(images_dir)
        except ValueError as exc:
            raise RuntimeError(f"RGB path escaped data-root/images: {image_path}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"Selected-view RGB is missing: {image_path}")

        input_hash = sha256_file(image_path)
        result = inference_segmentor(model, str(image_path))
        if not isinstance(result, (list, tuple)) or not result:
            raise RuntimeError(f"Unexpected SegFormer output for {image_path}: {type(result)}")
        labels = np.asarray(result[0])
        if labels.ndim == 3 and labels.shape[0] == 1:
            labels = labels[0]
        if labels.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D semantic label map, got {labels.shape} for {image_path}"
            )
        vehicle_mask = np.isin(labels, VEHICLE_CLASS_INDICES).astype(np.uint8) * 255
        output_path = out_dir / f"{frame:03d}_{camera}.png"
        Image.fromarray(vehicle_mask, mode="L").save(output_path, format="PNG")

        input_records.append(
            {
                "frame": frame,
                "camera": camera,
                "path": str(image_path),
                "sha256": input_hash,
                "selected_by_track_ids": view_to_tracks[(frame, camera)],
            }
        )
        output_records.append(
            {
                "frame": frame,
                "camera": camera,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "foreground_pixels": int(np.count_nonzero(vehicle_mask)),
                "height": int(vehicle_mask.shape[0]),
                "width": int(vehicle_mask.shape[1]),
            }
        )

    synchronize(torch, device)
    processing_seconds = time.perf_counter() - processing_start
    total_seconds = time.perf_counter() - process_start
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    manifest = {
        "status": "completed",
        "production_inputs_only_rgb": True,
        "mask_semantics": {
            "dataset_label_space": "Cityscapes train IDs",
            "vehicle_class_indices": list(VEHICLE_CLASS_INDICES),
            "output_encoding": "uint8 PNG; vehicle=255, background=0",
        },
        "selected_track_ids": selected_track_ids,
        "selected_tracks": selected_tracks,
        "unique_rgb_count": len(unique_views),
        "inputs": {
            "track_manifest": {
                "path": str(track_manifest),
                "sha256": sha256_file(track_manifest),
            },
            "data_root": str(args.data_root.expanduser().resolve()),
            "images_dir": str(images_dir),
            "rgb_files": input_records,
        },
        "model": {
            "implementation": "local NVlabs SegFormer mmseg.apis",
            "segformer_root": str(segformer_root),
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
        },
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "outputs": {"directory": str(out_dir), "mask_files": output_records},
        "timing_seconds": {
            "model_load_cold": model_load_seconds,
            "rgb_io_inference_postprocess": processing_seconds,
            "total_cold": total_seconds,
        },
        "peak_cuda_memory": {
            "bytes": peak_cuda_bytes,
            "gib": peak_cuda_bytes / (1024**3),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "mmcv_version": mmcv.__version__,
            "mmseg_version": mmseg.__version__,
            "gpu": gpu_metadata(torch, device),
        },
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest["timing_seconds"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline compatibility audit for per-track SAM2 masks.

The production SAM2 path remains RGB plus projected boxes only.  This script
reads the SegFormer-derived fine vehicle masks after production solely to
report the non-independent Round-62 compatibility diagnostic.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.predict_sam2_track_masks import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    crop_parameters,
    read_principal_point,
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def blank() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "bbox_pixels": 0}


def add(dst: dict[str, int], src: dict[str, int]) -> None:
    for key in dst:
        dst[key] += int(src[key])


def ratio(a: int, b: int) -> float | None:
    return float(a / b) if b else None


def metrics(counts: dict[str, int]) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "iou": ratio(tp, tp + fp + fn),
    }


def main() -> None:
    parsed = args()
    manifest_path = parsed.mask_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("production_inputs_only_rgb_and_projected_bbox") is not True:
        raise ValueError("Not an RGB plus projected-box production manifest")
    data_root = Path(manifest["inputs"]["data_root"])
    fine_root = data_root / "fine_dynamic_masks" / "vehicle"
    if not fine_root.is_dir():
        raise FileNotFoundError(fine_root)

    principal_points = {}
    total = blank()
    track_counts: dict[int, dict[str, int]] = defaultdict(blank)
    per_view = []
    pred_empty = 0
    gt_empty = 0
    for record in manifest["outputs"]["mask_files"]:
        track_id = int(record["track_id"])
        frame, camera = int(record["frame"]), int(record["camera"])
        pred = np.asarray(Image.open(record["output_path"]).convert("L")) > 0
        if pred.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"Unexpected production mask shape: {pred.shape}")

        if camera not in principal_points:
            principal_points[camera] = read_principal_point(
                data_root / "intrinsics" / f"{camera}.txt"
            )
        gt_path = fine_root / f"{frame:03d}_{camera}.png"
        with Image.open(gt_path) as image:
            image = image.convert("L")
            width, height = image.size
            crop = crop_parameters(width, height, *principal_points[camera])
            x, y, w, h = crop
            image = image.crop((x, y, x + w, y + h))
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)
            gt = np.asarray(image) > 0

        x0f, y0f, x1f, y1f = record["bbox_xyxy"]
        x0 = max(0, int(np.floor(x0f)))
        y0 = max(0, int(np.floor(y0f)))
        x1 = min(IMAGE_WIDTH - 1, int(np.ceil(x1f)))
        y1 = min(IMAGE_HEIGHT - 1, int(np.ceil(y1f)))
        p = pred[y0:y1 + 1, x0:x1 + 1]
        g = gt[y0:y1 + 1, x0:x1 + 1]
        counts = {
            "tp": int(np.count_nonzero(p & g)),
            "fp": int(np.count_nonzero(p & ~g)),
            "fn": int(np.count_nonzero(~p & g)),
            "tn": int(np.count_nonzero(~p & ~g)),
            "bbox_pixels": int(p.size),
        }
        add(total, counts)
        add(track_counts[track_id], counts)
        pred_empty += int(not p.any())
        gt_empty += int(not g.any())
        per_view.append({
            "track_id": track_id,
            "frame": frame,
            "camera": camera,
            "counts": metrics(counts),
            "sam2_predicted_iou": record["sam2_predicted_iou"],
            "single_view_reliable": record["single_view_reliable"],
            "track_fallback_to_projected_bbox": record["track_fallback_to_projected_bbox"],
        })

    aggregate = metrics(total)
    report = {
        "schema_version": 1,
        "reference_is_independent_manual_gt": False,
        "reference_limitation": (
            "fine_dynamic_masks/vehicle contains a SegFormer-derived component; "
            "these metrics are compatibility diagnostics and cannot alone accept or reject SAM2"
        ),
        "mask_manifest": str(manifest_path),
        "track_view_count": len(per_view),
        "aggregate_micro": aggregate,
        "thresholds": {"precision": 0.60, "recall": 0.90, "iou": 0.55},
        "compatibility_thresholds_passed": bool(
            aggregate["precision"] is not None and aggregate["precision"] >= 0.60
            and aggregate["recall"] is not None and aggregate["recall"] >= 0.90
            and aggregate["iou"] is not None and aggregate["iou"] >= 0.55
        ),
        "empty_mask_counts": {"prediction": pred_empty, "reference": gt_empty},
        "per_track": [
            {"track_id": track_id, "micro": metrics(track_counts[track_id])}
            for track_id in sorted(track_counts)
        ],
        "per_view": per_view,
    }
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(parsed.out),
        "aggregate_micro": aggregate,
        "compatibility_thresholds_passed": report["compatibility_thresholds_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()

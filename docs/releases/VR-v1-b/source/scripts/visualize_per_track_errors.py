"""Create GT/predicted crops for the highest-error dynamic tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from scripts.render_final_global_runtime import (
    load_boxes,
    projected_box_bounds,
)
from scripts.train import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--metrics-manifest", required=True)
    parser.add_argument("--comparison-frames", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, "val")
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    frames = list(metadata["frames"])
    slots = {int(frame): slot for slot, frame in enumerate(frames)}
    times = dataset._normalised_times(frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    height, width = dataset.resolution

    payload_dir = Path(args.payload_dir)
    meta = torch.load(
        payload_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), torch.device("cpu"))
    assert boxes is not None
    metrics = json.loads(Path(args.metrics_manifest).read_text())
    ranked = metrics["per_track_dynamic_bbox_summary_by_error"][:args.top_k]
    frame_records = metrics["frame_records"]
    comparison_dir = Path(args.comparison_frames)

    rows = []
    records = []
    for summary in ranked:
        track_id = int(summary["track_id"])
        track_slot = int(summary["track_slot"])
        candidates = []
        for frame_record in frame_records:
            for track_record in frame_record.get("per_track_dynamic_bbox", []):
                if int(track_record["track_id"]) == track_id:
                    error_mass = track_record["pixel_count"] * 10.0 ** (
                        -float(track_record["psnr"]) / 10.0
                    )
                    candidates.append((error_mass, frame_record, track_record))
        if not candidates:
            continue
        _, frame_record, track_record = max(candidates, key=lambda item: item[0])
        frame = int(frame_record["frame"])
        source_index = int(frame_record["source_index"])
        slot = slots[frame]
        target = dataset._collect(
            metadata, [frame], slots, times, dataset.resolution,
            world_from, lidar=False,
        )
        bounds, visible = projected_box_bounds(
            boxes,
            torch.full((len(dataset.cameras),), slot, dtype=torch.long),
            target["intrinsics"],
            torch.linalg.inv(target["cam_to_world"]),
            height,
            width,
        )
        areas = []
        for camera_index in range(len(dataset.cameras)):
            if not bool(visible[camera_index, track_slot]):
                areas.append(-1)
                continue
            x0, y0, x1, y1 = [int(value) for value in bounds[camera_index, track_slot]]
            areas.append((x1 - x0 + 1) * (y1 - y0 + 1))
        camera_index = int(np.argmax(areas))
        x0, y0, x1, y1 = [int(value) for value in bounds[camera_index, track_slot]]
        margin = 24
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(width - 1, x1 + margin), min(height - 1, y1 + margin)
        comparison = Image.open(
            comparison_dir / f"frame_{source_index:04d}.jpg"
        ).convert("RGB")
        panel_x = camera_index * width
        gt = comparison.crop((panel_x + x0, y0, panel_x + x1 + 1, y1 + 1))
        pred = comparison.crop((
            panel_x + x0, height + y0,
            panel_x + x1 + 1, height + y1 + 1,
        ))
        gt = gt.resize((384, 256), Image.Resampling.LANCZOS)
        pred = pred.resize((384, 256), Image.Resampling.LANCZOS)
        row = Image.new("RGB", (768, 286), "white")
        row.paste(gt, (0, 30))
        row.paste(pred, (384, 30))
        draw = ImageDraw.Draw(row)
        draw.text((8, 8), (
            f"track {track_id} slot {track_slot} | frame {frame} cam {camera_index} | "
            f"pixel PSNR {summary['pixel_weighted_psnr']:.2f} dB | "
            f"error {100.0 * summary['error_contribution_fraction_over_tracks']:.1f}%"
        ), fill="black")
        draw.text((8, 34), "GT", fill="white", stroke_width=2, stroke_fill="black")
        draw.text((392, 34), "prediction", fill="white", stroke_width=2, stroke_fill="black")
        rows.append(row)
        records.append({
            "rank": len(records) + 1,
            "track_id": track_id,
            "track_slot": track_slot,
            "frame": frame,
            "camera": camera_index,
            "pixel_weighted_psnr": summary["pixel_weighted_psnr"],
            "error_contribution_fraction": summary["error_contribution_fraction_over_tracks"],
            "crop_xyxy": [x0, y0, x1, y1],
        })

    sheet = Image.new("RGB", (768, 286 * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, 286 * index))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=94)
    out.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"out": str(out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()


"""Freeze near-stationary dynamic tracks to a fixed canonical world pose.

This is an optional payload transformation for track-once global dynamic maps.
The Gaussian tensors remain byte-identical; only the full-scene box graph is
copied and adjusted.  Tracks are classified from world-space box-centre motion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def link_exact(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(target)
        return
    os.symlink(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-payload", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-center-span", type=float, required=True)
    parser.add_argument("--max-median-step", type=float, default=0.1)
    args = parser.parse_args()

    source = Path(args.source_payload).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("final_static_128d.pt", "final_dynamic_128d.pt"):
        link_exact(source / name, output / name)

    meta = torch.load(
        source / "global_dynamic_meta.pt", map_location="cpu", weights_only=False
    )
    boxes = {key: value.clone() for key, value in meta["boxes"].items()}
    track_ids = [int(value) for value in meta["track_ids"]]
    records = []
    frozen_slots = []
    for slot, track_id in enumerate(track_ids):
        valid = boxes["valid"][slot]
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        centers = boxes["centers"][slot, indices]
        if len(indices) > 1:
            span = torch.linalg.vector_norm(
                centers.max(dim=0).values - centers.min(dim=0).values
            ).item()
            steps = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=1)
            median_step = steps.median().item()
            path_length = steps.sum().item()
            endpoint_displacement = torch.linalg.vector_norm(
                centers[-1] - centers[0]
            ).item()
        else:
            span = median_step = path_length = endpoint_displacement = 0.0
        stationary = (
            span <= args.max_center_span
            and median_step <= args.max_median_step
        )
        if stationary and len(indices):
            canonical = int(indices[0])
            boxes["centers"][slot, indices] = boxes["centers"][slot, canonical].clone()
            boxes["dims"][slot, indices] = boxes["dims"][slot, canonical].clone()
            boxes["rots"][slot, indices] = boxes["rots"][slot, canonical].clone()
            frozen_slots.append(slot)
        records.append({
            "track_id": track_id,
            "global_slot": slot,
            "valid_frame_count": int(len(indices)),
            "center_span_m": span,
            "median_step_m": median_step,
            "path_length_m": path_length,
            "endpoint_displacement_m": endpoint_displacement,
            "stationary": stationary,
            "anchor": "earliest-valid canonical box" if stationary else None,
        })

    adjusted = dict(meta)
    adjusted["boxes"] = boxes
    adjusted["stationary_track_policy"] = {
        "mode": "freeze-to-earliest-valid-canonical-box",
        "max_center_span_m": args.max_center_span,
        "max_median_step_m": args.max_median_step,
        "frozen_track_count": len(frozen_slots),
        "moving_track_count": len(track_ids) - len(frozen_slots),
    }
    torch.save(adjusted, output / "global_dynamic_meta.pt")
    manifest = {
        "status": "completed",
        "source_payload": str(source),
        "gaussian_policy": "byte-identical symlinks to source static and dynamic payloads",
        "box_policy": adjusted["stationary_track_policy"],
        "tracks": records,
    }
    (output / "stationary_track_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["box_policy"], indent=2))


if __name__ == "__main__":
    main()

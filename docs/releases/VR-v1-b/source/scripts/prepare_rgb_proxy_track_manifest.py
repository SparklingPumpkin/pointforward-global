#!/usr/bin/env python3
"""Attach calibrated projected-box bounds to an existing track-view manifest.

The input view selection is preserved exactly.  This helper only recomputes the
projected 3-D box metadata needed by RGB-only SAM2 prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.build_track_once_dynamic import calibrated_intrinsics, track_view_candidates
from scripts.train import build_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--track-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    source_path = args.track_manifest.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = build_dataset(config, args.split)
    matches = [item for item in dataset.scenes if item["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    frames = [int(value) for value in metadata["frames"]]
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    intrinsics = calibrated_intrinsics(dataset, metadata, frames[0])
    source = json.loads(source_path.read_text(encoding="utf-8"))

    output_tracks = []
    unique_views: set[tuple[int, int]] = set()
    for record in source["tracks"]:
        track_id = int(record["track_id"])
        candidates = track_view_candidates(
            dataset, metadata, track_id, set(frames), world_from, intrinsics
        )
        lookup = {
            (int(item["frame"]), int(item["camera"])): item
            for item in candidates
        }
        selected = []
        for view in record["selected_views"]:
            key = (int(view["frame"]), int(view["camera"]))
            if key not in lookup:
                raise KeyError(f"Selected track-view is no longer visible: {track_id}, {key}")
            enriched = dict(view)
            enriched.update({
                "bbox_xyxy": lookup[key]["bbox_xyxy"],
                "bbox_area_fraction": lookup[key]["bbox_area_fraction"],
                "view_angle_rad": lookup[key]["view_angle_rad"],
            })
            selected.append(enriched)
            unique_views.add(key)
        updated = dict(record)
        updated["selected_views"] = selected
        output_tracks.append(updated)

    output = dict(source)
    output["tracks"] = output_tracks
    output["rgb_proxy_prompt_contract"] = {
        "source_manifest": str(source_path),
        "source_sha256": sha256_file(source_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "scene": args.scene,
        "track_count": len(output_tracks),
        "prompt_count": sum(len(item["selected_views"]) for item in output_tracks),
        "unique_rgb_count": len(unique_views),
        "selection_unchanged": True,
        "bbox_source": "calibrated projected 3-D boxes from track_view_candidates",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "out": str(output_path),
        **output["rgb_proxy_prompt_contract"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

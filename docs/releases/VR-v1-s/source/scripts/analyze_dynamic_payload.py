"""Quantify cross-clip association and duplication in a dynamic payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    observations = payload["extras"]["observations"]
    means = payload["means"].to(args.device)
    track = payload["instance_id"].to(args.device).long()
    result: dict[str, object] = {
        "dynamic_count": int(observations.numel()),
        "observation_count": {
            "exactly_one_fraction": float((observations == 1).float().mean()),
            "at_most_two_fraction": float((observations <= 2).float().mean()),
            "at_least_five_fraction": float((observations >= 5).float().mean()),
            "mean": float(observations.float().mean()),
        },
        "canonical_voxel_occupancy": {},
    }
    for voxel_size in (0.02, 0.05, 0.10, 0.20):
        coords = torch.floor(means / voxel_size).long()
        shifted = coords - coords.amin(dim=0)
        spans = shifted.amax(dim=0) + 1
        key = (((track * spans[0] + shifted[:, 0]) * spans[1] + shifted[:, 1])
               * spans[2] + shifted[:, 2])
        unique_count = int(torch.unique(key).numel())
        result["canonical_voxel_occupancy"][f"{voxel_size:.2f}m"] = {
            "unique_count": unique_count,
            "unique_over_points": unique_count / observations.numel(),
            "points_per_occupied_voxel": observations.numel() / unique_count,
        }

    manifest = json.loads(Path(args.build_manifest).read_text())
    stats = [window["global_dynamic"] for window in manifest["windows"]]
    decoded = sum(int(item["current_dynamic_count"]) for item in stats)
    added = sum(int(item["added"]) for item in stats)
    refreshed = sum(int(item["forward_refresh_updated"]) for item in stats)
    result["cross_clip_flow"] = {
        "decoded_current_points": decoded,
        "new_points_admitted": added,
        "new_over_decoded": added / decoded,
        "accepted_forward_refresh_queries": refreshed,
        "persisted_refresh_increments": int((observations - 1).sum()),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

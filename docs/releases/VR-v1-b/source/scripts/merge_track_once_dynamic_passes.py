"""Merge complementary track-once dynamic payloads in canonical object space."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


FIELDS = (
    "means", "quats", "scales", "opacities", "colors",
    "sigma", "times", "instance_id",
)


def first_per_voxel(means: torch.Tensor, voxel_size: float) -> torch.Tensor:
    if voxel_size <= 0 or means.shape[0] == 0:
        return torch.arange(means.shape[0], dtype=torch.long)
    coords = torch.floor(means / voxel_size).to(torch.int64)
    shifted = coords - coords.amin(dim=0, keepdim=True)
    spans = shifted.amax(dim=0) + 1
    keys = (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2]
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    count = int(inverse.max().item()) + 1
    positions = torch.arange(means.shape[0], dtype=torch.long)
    representatives = torch.full((count,), means.shape[0], dtype=torch.long)
    representatives.scatter_reduce_(
        0, inverse, positions, reduce="amin", include_self=True
    )
    return representatives.sort().values


def link_exact(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(target)
        return
    os.symlink(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    args = parser.parse_args()

    sources = [Path(value).resolve() for value in args.payloads]
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payloads = [
        torch.load(path / "final_dynamic_128d.pt", map_location="cpu", weights_only=False)
        for path in sources
    ]
    extra_names = set.intersection(*(set(value["extras"]) for value in payloads))
    merged = {
        field: torch.cat([value[field] for value in payloads], dim=0)
        for field in FIELDS
    }
    merged["extras"] = {
        name: torch.cat([value["extras"][name] for value in payloads], dim=0)
        for name in extra_names
    }
    before = int(merged["means"].shape[0])
    track_records = []
    selected_parts = []
    for instance_id in torch.unique(merged["instance_id"], sorted=True):
        candidates = torch.nonzero(
            merged["instance_id"] == instance_id, as_tuple=False
        ).flatten()
        local = first_per_voxel(merged["means"][candidates], args.voxel_size)
        chosen = candidates[local]
        selected_parts.append(chosen)
        track_records.append({
            "global_slot": int(instance_id),
            "before": int(candidates.numel()),
            "after": int(chosen.numel()),
        })
    selected = torch.cat(selected_parts).sort().values
    result = {
        field: merged[field][selected] for field in FIELDS
    }
    result["extras"] = {
        name: value[selected] for name, value in merged["extras"].items()
    }
    torch.save(result, output / "final_dynamic_128d.pt")
    link_exact(sources[0] / "final_static_128d.pt", output / "final_static_128d.pt")
    link_exact(sources[0] / "global_dynamic_meta.pt", output / "global_dynamic_meta.pt")
    manifest = {
        "status": "completed",
        "source_payloads": [str(path) for path in sources],
        "merge_policy": "first pass has priority; later passes add canonical voxels not already occupied",
        "voxel_size_m": args.voxel_size,
        "count_before": before,
        "count_after": int(selected.numel()),
        "tracks": track_records,
    }
    (output / "multi_pass_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({
        key: manifest[key]
        for key in ("voxel_size_m", "count_before", "count_after")
    }, indent=2))


if __name__ == "__main__":
    main()

"""Create a dynamic-only payload ablation while reusing the static payload."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def choose_best_per_track_voxel(
    payload: dict[str, object], voxel_size: float, device: torch.device
) -> torch.Tensor:
    means = payload["means"].to(device)
    track = payload["instance_id"].to(device).long()
    observations = payload["extras"]["observations"].to(device).float()
    distance = payload["extras"]["best_observation_distance"].to(device).float()
    coords = torch.floor(means / voxel_size).long()
    shifted = coords - coords.amin(dim=0)
    spans = shifted.amax(dim=0) + 1
    key = (((track * spans[0] + shifted[:, 0]) * spans[1] + shifted[:, 1])
           * spans[2] + shifted[:, 2])
    _, inverse = torch.unique(key, sorted=True, return_inverse=True)
    group_count = int(inverse.amax()) + 1
    score = observations * 1000.0 - distance.clamp_max(999.0)
    best_score = torch.full(
        (group_count,), -torch.inf, dtype=score.dtype, device=device
    )
    best_score.scatter_reduce_(
        0, inverse, score, reduce="amax", include_self=True
    )
    candidate = score >= best_score[inverse]
    indices = torch.arange(means.shape[0], device=device)
    representative = torch.full(
        (group_count,), means.shape[0], dtype=torch.long, device=device
    )
    representative.scatter_reduce_(
        0,
        inverse[candidate],
        indices[candidate],
        reduce="amin",
        include_self=True,
    )
    return representative.cpu()


def select_payload(payload: dict[str, object], index: torch.Tensor) -> dict[str, object]:
    output = {}
    for name, value in payload.items():
        if name == "extras":
            output[name] = {key: item[index] for key, item in value.items()}
        else:
            output[name] = value[index]
    return output


def link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(source.resolve(), target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--min-observations", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.voxel_size is None) == (args.min_observations is None):
        parser.error("Specify exactly one of --voxel-size or --min-observations")

    source = Path(args.source_payload_dir).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(
        source / "final_dynamic_128d.pt", map_location="cpu", weights_only=False
    )
    if args.voxel_size is not None:
        index = choose_best_per_track_voxel(
            payload, args.voxel_size, torch.device(args.device)
        )
    else:
        index = torch.nonzero(
            payload["extras"]["observations"] >= args.min_observations,
            as_tuple=False,
        ).flatten()
    selected = select_payload(payload, index)
    torch.save(selected, output / "final_dynamic_128d.pt")
    link(source / "final_static_128d.pt", output / "final_static_128d.pt")
    link(source / "global_dynamic_meta.pt", output / "global_dynamic_meta.pt")
    print({"before": int(payload["means"].shape[0]), "after": int(index.numel())})


if __name__ == "__main__":
    main()

"""Suppress static Gaussians that occupy the swept volume of dynamic boxes.

This is an optional payload transform.  The source payload is never modified;
all dynamic fields and the box graph are preserved exactly.  A configurable
fraction at the bottom of every box is retained to avoid removing road points.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin-m", type=float, default=0.0)
    parser.add_argument("--bottom-keep-fraction", type=float, default=0.20)
    parser.add_argument("--inside-opacity-multiplier", type=float, default=0.0)
    parser.add_argument("--box-batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0.0 <= args.bottom_keep_fraction < 1.0:
        parser.error("--bottom-keep-fraction must be in [0, 1)")
    if not 0.0 <= args.inside_opacity_multiplier <= 1.0:
        parser.error("--inside-opacity-multiplier must be in [0, 1]")

    started = time.perf_counter()
    source = Path(args.payload_dir).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    static_path = source / "final_static_128d.pt"
    meta_path = source / "global_dynamic_meta.pt"
    dynamic_path = source / "final_dynamic_128d.pt"
    static = torch.load(static_path, map_location="cpu", weights_only=True)
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    means = static["means"].to(device)
    box_payload = meta.get("evaluation_boxes", meta["boxes"])
    centers = box_payload["centers"].to(device)
    dims = box_payload["dims"].to(device)
    rots = box_payload["rots"].to(device)
    valid = box_payload["valid"].to(device)
    remove = torch.zeros(means.shape[0], dtype=torch.bool, device=device)

    track_records = []
    for track in range(centers.shape[0]):
        track_valid = valid[track]
        if not bool(track_valid.any()):
            continue
        track_centers = centers[track, track_valid]
        track_dims = dims[track, track_valid] + 2.0 * args.margin_m
        track_rots = rots[track, track_valid]
        radii = 0.5 * torch.linalg.vector_norm(track_dims, dim=-1)
        lower = (track_centers - radii[:, None]).amin(dim=0)
        upper = (track_centers + radii[:, None]).amax(dim=0)
        candidate_mask = ((means >= lower) & (means <= upper)).all(dim=-1)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        points = means[candidate_indices]
        hit = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
        for start in range(0, track_centers.shape[0], args.box_batch_size):
            end = min(start + args.box_batch_size, track_centers.shape[0])
            batch_centers = track_centers[start:end]
            batch_dims = track_dims[start:end]
            batch_rots = track_rots[start:end]
            delta = points[:, None, :] - batch_centers[None, :, :]
            local = torch.einsum("pbj,bij->pbi", delta, batch_rots)
            inside = (local.abs() <= 0.5 * batch_dims[None, :, :]).all(dim=-1)
            if args.bottom_keep_fraction > 0.0:
                bottom_cut = (
                    -0.5 + args.bottom_keep_fraction
                ) * batch_dims[None, :, 2]
                inside &= local[:, :, 2] > bottom_cut
            hit |= inside.any(dim=1)
        selected = candidate_indices[hit]
        remove[selected] = True
        track_records.append({
            "track": track,
            "valid_box_count": int(track_valid.sum()),
            "candidate_count": int(candidate_indices.numel()),
            "selected_count": int(selected.numel()),
        })

    remove_cpu = remove.cpu()
    original_opacity_mean = float(static["opacities"].float().mean())
    static["opacities"] = static["opacities"].clone()
    static["opacities"][remove_cpu] *= args.inside_opacity_multiplier
    torch.save(static, output / "final_static_128d.pt")
    os.symlink(dynamic_path, output / "final_dynamic_128d.pt")
    os.symlink(meta_path, output / "global_dynamic_meta.pt")
    manifest = {
        "status": "completed",
        "source_payload": str(source),
        "output_payload": str(output),
        "method": "static Gaussian suppression inside union of evaluation dynamic boxes",
        "margin_m": args.margin_m,
        "bottom_keep_fraction": args.bottom_keep_fraction,
        "inside_opacity_multiplier": args.inside_opacity_multiplier,
        "static_count": int(means.shape[0]),
        "selected_static_count": int(remove.sum()),
        "selected_static_fraction": float(remove.float().mean()),
        "original_opacity_mean": original_opacity_mean,
        "filtered_opacity_mean": float(static["opacities"].float().mean()),
        "track_records": track_records,
        "wall_seconds": time.perf_counter() - started,
    }
    (output.parent / "static_sweep_filter_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

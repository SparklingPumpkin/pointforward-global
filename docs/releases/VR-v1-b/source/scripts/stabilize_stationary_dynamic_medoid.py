"""Stabilize near-stationary dynamic tracks with a robust observed medoid.

Unlike the earlier first-valid-box freeze, this transform selects the observed
box that minimizes the joint translation/rotation distance to all valid boxes.
The canonical dynamic Gaussians are rigidly re-parameterized into that medoid
frame before the valid trajectory poses are made constant, so geometry is not
silently displaced when the canonical box changes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from pointforward.models.rotation import matrix_to_quaternion, quaternion_multiply
from pointforward.models.scene_graph import decanonicalise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-span-m", type=float, required=True)
    parser.add_argument("--max-median-step-m", type=float, default=0.10)
    parser.add_argument(
        "--rotation-weight-m-per-rad", type=float, default=0.50,
        help="Rotation contribution when choosing an actually observed medoid.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    source = Path(args.payload_dir).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    static_path = source / "final_static_128d.pt"
    dynamic_path = source / "final_dynamic_128d.pt"
    meta_path = source / "global_dynamic_meta.pt"
    dynamic = torch.load(dynamic_path, map_location="cpu", weights_only=True)
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    boxes = meta["boxes"]
    centers = boxes["centers"].clone()
    dims = boxes["dims"].clone()
    rots = boxes["rots"].clone()
    valid = boxes["valid"]
    instance_id = dynamic["instance_id"].long()
    records = []

    for track in range(centers.shape[0]):
        valid_slots = torch.nonzero(valid[track], as_tuple=False).flatten()
        if valid_slots.numel() < 2:
            continue
        observed_centers = centers[track, valid_slots]
        observed_rots = rots[track, valid_slots]
        span = float(torch.linalg.vector_norm(
            observed_centers.amax(dim=0) - observed_centers.amin(dim=0)
        ))
        steps = torch.linalg.vector_norm(
            observed_centers[1:] - observed_centers[:-1], dim=-1
        )
        median_step = float(steps.median())
        if span > args.max_span_m or median_step > args.max_median_step_m:
            continue

        translation_distance = torch.cdist(observed_centers, observed_centers)
        relative = torch.matmul(
            observed_rots[:, None].transpose(-1, -2), observed_rots[None]
        )
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
        rotation_distance = torch.acos(cosine.clamp(-1.0, 1.0))
        medoid_local = torch.argmin(
            (translation_distance + args.rotation_weight_m_per_rad * rotation_distance)
            .mean(dim=1)
        )
        medoid_slot = int(valid_slots[medoid_local])
        canonical_slot = int(valid_slots[0])
        old_center = centers[track, canonical_slot].clone()
        old_rot = rots[track, canonical_slot].clone()
        new_center = centers[track, medoid_slot].clone()
        new_dim = dims[track, medoid_slot].clone()
        new_rot = rots[track, medoid_slot].clone()

        point_mask = instance_id == track
        point_count = int(point_mask.sum())
        if point_count:
            dynamic["means"][point_mask] = decanonicalise(
                dynamic["means"][point_mask], new_center, new_rot,
                old_center, old_rot,
            )
            rigid_rotation = new_rot.transpose(-1, -2) @ old_rot
            dynamic["quats"][point_mask] = quaternion_multiply(
                matrix_to_quaternion(rigid_rotation).expand(point_count, -1),
                dynamic["quats"][point_mask],
            )

        centers[track, valid_slots] = new_center
        dims[track, valid_slots] = new_dim
        rots[track, valid_slots] = new_rot
        records.append({
            "track_slot": track,
            "track_id": int(meta.get("track_ids", list(range(centers.shape[0])))[track]),
            "valid_box_count": int(valid_slots.numel()),
            "span_m": span,
            "median_step_m": median_step,
            "canonical_slot_before": canonical_slot,
            "medoid_slot": medoid_slot,
            "canonical_to_medoid_translation_m": float(
                torch.linalg.vector_norm(new_center - old_center)
            ),
            "dynamic_point_count": point_count,
        })

    boxes["centers"] = centers
    boxes["dims"] = dims
    boxes["rots"] = rots
    meta["boxes"] = boxes
    torch.save(dynamic, output / "final_dynamic_128d.pt")
    torch.save(meta, output / "global_dynamic_meta.pt")
    os.symlink(static_path, output / "final_static_128d.pt")
    manifest = {
        "status": "completed",
        "source_payload": str(source),
        "output_payload": str(output),
        "method": "observed center+rotation medoid with canonical Gaussian rigid re-parameterization",
        "max_span_m": args.max_span_m,
        "max_median_step_m": args.max_median_step_m,
        "rotation_weight_m_per_rad": args.rotation_weight_m_per_rad,
        "track_count": int(centers.shape[0]),
        "stabilized_track_count": len(records),
        "stabilized_dynamic_point_count": sum(row["dynamic_point_count"] for row in records),
        "records": records,
        "wall_seconds": time.perf_counter() - started,
    }
    (output.parent / "stationary_medoid_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


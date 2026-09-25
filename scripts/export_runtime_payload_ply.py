"""Export a persistent runtime payload as one standard preview-color PLY.

The runtime payload keeps 128-D features, while a standard 3DGS PLY can only
carry RGB/degree-0 SH.  This exporter preserves the final global geometry and
uses the payload's cached ``preview_rgb`` for viewer-compatible colors.  The
dynamic subset is posed at one requested scene frame using the saved global
box graph.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pointforward.models import Gaussians, InstanceBoxes
from pointforward.models.gaussian_head import temporal_opacity
from pointforward.models.query import query_rgb
from pointforward.models.rotation import matrix_to_quaternion, quaternion_multiply
from pointforward.models.scene_graph import decanonicalise
from scripts.export_gaussians_ply import write_standard_ply


def load(path: Path) -> Gaussians:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return Gaussians(
        **{name: payload[name] for name in ("means", "quats", "scales", "opacities", "colors", "sigma", "times", "instance_id")},
        extras=payload.get("extras", {}),
    )


def boxes_from(payload: dict) -> InstanceBoxes:
    return InstanceBoxes(
        centers=payload["centers"], dims=payload["dims"], rots=payload["rots"],
        times=payload["times"], valid=payload["valid"],
    )


def pose_dynamic(dynamic: Gaussians, boxes: InstanceBoxes, frame: int, target_time: float):
    means = dynamic.means.clone()
    quats = dynamic.quats.clone()
    opacities = temporal_opacity(dynamic.opacities, dynamic.sigma, dynamic.times, target_time)
    if dynamic.count == 0:
        return means, quats, opacities
    inst = dynamic.instance_id.long()
    slot = min(frame, boxes.centers.shape[1] - 1)
    box_c = boxes.centers[inst, slot]
    box_r = boxes.rots[inst, slot]
    can_c, can_r = boxes.canonical()
    means = decanonicalise(dynamic.means, box_c, box_r, can_c[inst], can_r[inst])
    rel = torch.einsum("nji,njk->nik", box_r, can_r[inst])
    quats = quaternion_multiply(matrix_to_quaternion(rel), dynamic.quats)
    absent = ~boxes.valid[inst, slot]
    opacities = opacities.clone()
    opacities[absent] = 0.0
    return means, quats, opacities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    payload_dir = Path(args.payload_dir).resolve()
    out = Path(args.out).resolve()
    static = load(payload_dir / "final_static_128d.pt")
    dynamic = load(payload_dir / "final_dynamic_128d.pt")
    meta = torch.load(payload_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = boxes_from(meta["boxes"])
    frame = int(args.frame)
    target_time = float(meta["normalized_times"][frame])
    static_rgb = static.extras.get("preview_rgb")
    dynamic_rgb = dynamic.extras.get("preview_rgb")
    if static_rgb is None:
        static_rgb = query_rgb(static.colors)
    if dynamic_rgb is None:
        dynamic_rgb = query_rgb(dynamic.colors)
    static_opacities = temporal_opacity(static.opacities, static.sigma, static.times, target_time)
    dyn_means, dyn_quats, dyn_opacities = pose_dynamic(dynamic, boxes, frame, target_time)
    means = torch.cat((static.means, dyn_means), dim=0).numpy()
    quats = torch.cat((static.quats, dyn_quats), dim=0).numpy()
    scales = torch.cat((static.scales, dynamic.scales), dim=0).numpy()
    opacities = torch.cat((static_opacities, dyn_opacities), dim=0).numpy()
    rgb = torch.cat((static_rgb, dynamic_rgb), dim=0).numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_standard_ply(str(out), means, rgb, opacities, scales, quats)
    record = {
        "status": "completed", "payload_dir": str(payload_dir), "out": str(out),
        "scene_frame": frame, "target_time": target_time,
        "static_count": int(static.count), "dynamic_count": int(dynamic.count),
        "gaussian_count": int(static.count + dynamic.count),
        "coordinate_frame": "scene017 shared world frame from runtime payload",
        "colour": "cached preview_rgb encoded as degree-0 SH; not equivalent to 128-D+UNet output",
        "wall_seconds": time.perf_counter() - started,
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

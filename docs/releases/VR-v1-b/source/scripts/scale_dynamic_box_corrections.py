"""Apply a scalar fraction of saved DG-v1-r track pose corrections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from pointforward.models import InstanceBoxes
from scripts.deliver_continuous_scene import boxes_payload, save_gaussian_payload
from scripts.refine_dynamic_boxes_rgb import corrected_pose, rebase_payload
from scripts.render_final_global_runtime import load_boxes, load_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0.0 <= args.scale <= 1.0:
        parser.error("--scale must be in [0, 1]")

    device = torch.device(args.device)
    source = Path(args.payload_dir).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dynamic = load_payload(source / "final_dynamic_128d.pt", device)
    meta = torch.load(source / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes_cpu = load_boxes(meta["boxes"], torch.device("cpu"))
    assert boxes_cpu is not None
    boxes = InstanceBoxes(
        centers=boxes_cpu.centers.to(device), dims=boxes_cpu.dims.to(device),
        rots=boxes_cpu.rots.to(device), times=boxes_cpu.times.to(device),
        valid=boxes_cpu.valid.to(device),
    )
    refined = InstanceBoxes(
        centers=boxes.centers.clone(), dims=boxes.dims.clone(),
        rots=boxes.rots.clone(), times=boxes.times.clone(), valid=boxes.valid.clone(),
    )
    corrections = json.loads(Path(args.corrections).read_text())
    for item in corrections:
        track = int(item["track_index"])
        translation = torch.tensor(
            item["translation_local_m"], device=device, dtype=boxes.centers.dtype
        ) * args.scale
        yaw = torch.tensor(
            item["yaw_deg"], device=device, dtype=boxes.centers.dtype
        ).deg2rad() * args.scale
        for slot in torch.nonzero(boxes.valid[track], as_tuple=False).flatten().tolist():
            center, rotation = corrected_pose(
                boxes.centers[track, slot], boxes.rots[track, slot], translation, yaw
            )
            refined.centers[track, slot] = center
            refined.rots[track, slot] = rotation

    refined_dynamic = rebase_payload(dynamic, boxes, refined)
    static_target = output / "final_static_128d.pt"
    if not static_target.exists():
        os.symlink((source / "final_static_128d.pt").resolve(), static_target)
    save_gaussian_payload(output / "final_dynamic_128d.pt", refined_dynamic)
    refined_meta = dict(meta)
    refined_meta["boxes"] = boxes_payload(refined)
    refined_meta["evaluation_boxes"] = meta.get("evaluation_boxes", meta["boxes"])
    refined_meta["pose_correction_scale"] = args.scale
    torch.save(refined_meta, output / "global_dynamic_meta.pt")
    (output / "manifest.json").write_text(json.dumps({
        "status": "completed", "source_payload": str(source),
        "corrections": str(Path(args.corrections).resolve()),
        "scale": args.scale, "evaluation_boxes": "original unmodified boxes",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

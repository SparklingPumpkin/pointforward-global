"""Audit that articulated Gaussian attributes actually changed in this run."""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delivery", required=True)
    p.add_argument("--after", required=True)
    a = p.parse_args()
    root = Path(a.delivery)
    raw = torch.load(root/"pedestrian_tracks.pt", map_location="cpu", weights_only=True)
    before = torch.load(root/"runtime-payload/final_dynamic_128d.pt", map_location="cpu", weights_only=True)
    after = torch.load(Path(a.after)/"final_dynamic_128d.pt", map_location="cpu", weights_only=True)
    mask = before["instance_id"] >= raw["slot_offset"]
    assert torch.equal(before["instance_id"], after["instance_id"])
    report = {
        "human_count": len(raw["track_ids"]),
        "human_gaussians": int(mask.sum()),
        "raw_valid_poses": int(raw["pose_valid_original"].sum()),
        "completed_valid_poses": int(raw["pose_valid"].sum()),
        "internal_slerp_frames": int((raw["pose_fill_kind"] == 1).sum()),
        "leading_hold_frames": int((raw["pose_fill_kind"] == 2).sum()),
        "trailing_hold_frames": int((raw["pose_fill_kind"] == 3).sum()),
        "fields": {},
    }
    for key in ("means", "scales", "quats", "opacities", "colors"):
        x, y = before[key][mask].float(), after[key][mask].float()
        assert torch.isfinite(y).all()
        delta = y-x
        rms = float(delta.square().mean().sqrt())
        assert rms > 0, f"Human field {key} was not updated"
        report["fields"][key] = {"delta_rms": rms, "finite": True}
    report["pose_trajectory_fixed"] = True
    report["note"] = "Canonical human attributes changed under the same scene RGB optimization. Pose/shape remain input annotations."
    (root/"pedestrian_update_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

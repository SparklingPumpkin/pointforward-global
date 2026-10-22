"""Decode one persistent canonical human per global track, using twelve views."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from pedestrian_reference.models.smpl import NUM_VERTICES
from pointforward.models import Gaussians, SceneInput
from scripts.train import build_dataset
from scripts.eval import build_model
from scripts.build_track_once_dynamic import calibrated_intrinsics, track_view_candidates, select_diverse_large_views
from scripts.deliver_continuous_scene import save_gaussian_payload, concatenate_gaussians
from scripts.render_final_global_runtime import load_payload


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--payload-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="102")
    args = ap.parse_args()
    started = time.perf_counter()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dataset = build_dataset(cfg, "val")
    meta = next(m for m in dataset.scenes if m["name"] == args.scene)
    frames = meta["frames"]
    world_from = np.linalg.inv(dataset._ego_pose(meta, frames[0]))
    slots = {f: i for i, f in enumerate(frames)}
    times = dataset._normalised_times(frames)
    model = build_model(cfg, torch.device("cuda"), args.ckpt)
    source, out = Path(args.payload_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vehicle = load_payload(source / "final_dynamic_128d.pt", torch.device("cpu"))
    global_meta = torch.load(source / "global_dynamic_meta.pt", weights_only=True)
    offset = len(global_meta["boxes"]["valid"])
    parts, track_records, human_tracks = [], [], []
    intrinsics = calibrated_intrinsics(dataset, meta, frames[0])
    for tid, track in sorted(meta["humans"].items()):
        candidate_meta = dict(meta, instances={tid: track["boxes"]}, humans={tid: track})
        candidates = track_view_candidates(dataset, candidate_meta, tid, set(frames), world_from, intrinsics)
        candidates = [c for c in candidates if bool(track["valid"][c["frame"]])]
        if not candidates:
            continue
        chosen = sorted(select_diverse_large_views(candidates, 12, 0.65), key=lambda c:(c["frame"],c["camera"]))
        ctx_parts = []
        for view in chosen:
            old_cameras = dataset.cameras
            dataset.cameras = (view["camera"],)
            try:
                ctx_parts.append(dataset._collect(candidate_meta, [view["frame"]], slots, times,
                    dataset.resolution, world_from, lidar=True))
            finally:
                dataset.cameras = old_cameras
        ctx = {k: torch.cat([c[k] for c in ctx_parts]) for k in ctx_parts[0] if isinstance(ctx_parts[0][k], torch.Tensor)}
        humans = dataset._build_humans(candidate_meta, frames, world_from, ctx)
        if humans is None:
            continue
        human_tracks.append(humans)
        scene = SceneInput(**{k:ctx[k].cuda() for k in
            ("images","intrinsics","cam_to_world","times","frame_index","sparse_depth","sparse_mask")},
            humans=humans.to("cuda"))
        # Twelve view contract, locally normalized timestamps like the source head.
        scene.times = (scene.times-scene.times.min()) / (scene.times.max()-scene.times.min()).clamp_min(1e-8)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            depth = model.predict_depth(scene)
            features = model.backbone(scene.images)
            decoded = model._decode_humans(scene, depth, features, torch.linalg.inv(scene.cam_to_world))
        n = NUM_VERTICES
        slot = offset + len(parts)
        part = Gaussians(
            means=decoded["human_means"].float().cpu(), quats=decoded["human_quats"].float().cpu(),
            scales=decoded["human_scales"].float().cpu(), opacities=decoded["human_opacities"].float().cpu(),
            colors=decoded["human_colors"].float().cpu(), sigma=torch.full((n,),1e6),
            times=torch.zeros(n), instance_id=torch.full((n,),slot,dtype=torch.long),
            extras={"preview_rgb":torch.full((n,3),0.5)})
        parts.append(part)
        original = humans.pose_valid_original
        fill_kind = humans.pose_fill_kind
        bbox_valid = humans.boxes.valid
        record = {
            "track_id": tid,
            "slot": slot,
            "bbox_valid_frames": int(bbox_valid.sum()),
            "raw_valid_poses": int(original.sum()),
            "completed_valid_poses": int(humans.pose_valid.sum()),
            "internal_slerp_frames": int((fill_kind == 1).sum()),
            "leading_hold_frames": int((fill_kind == 2).sum()),
            "trailing_hold_frames": int((fill_kind == 3).sum()),
            "remaining_bbox_supported_invalid": int((bbox_valid & ~humans.pose_valid).sum()),
            "selected_views": chosen,
            "gaussians": n,
            "mean_opacity": float(part.opacities.mean()),
        }
        track_records.append(record)
        print(json.dumps(record), flush=True)
    if not parts:
        raise RuntimeError("No visible posed pedestrians")
    merged = concatenate_gaussians([vehicle]+parts)
    save_gaussian_payload(out / "final_dynamic_128d.pt", merged)
    # Placeholder rigid slots keep existing FFR loaders compatible. Actual posing
    # is replaced by SMPL before the single joint rasterization.
    bp = global_meta["boxes"]
    for key,value in bp.items():
        shape = (len(parts),)+value.shape[1:]
        pad = torch.zeros(shape, dtype=value.dtype)
        if key == "dims":
            pad.fill_(1)
        if key == "rots":
            pad[:] = torch.eye(3)
        bp[key] = torch.cat((value,pad))
    global_meta["track_ids"] = list(global_meta.get("track_ids", [])) + [r["track_id"] for r in track_records]
    torch.save(global_meta, out / "global_dynamic_meta.pt")
    raw = {"slot_offset":offset, "track_ids":[r["track_id"] for r in track_records],
        "pose":torch.cat([h.pose for h in human_tracks]),
        "pose_valid":torch.cat([h.pose_valid for h in human_tracks]),
        "pose_valid_original":torch.cat([h.pose_valid_original for h in human_tracks]),
        "pose_fill_kind":torch.cat([h.pose_fill_kind for h in human_tracks]),
        "betas":torch.cat([h.betas for h in human_tracks]),
        "centers":torch.cat([h.boxes.centers for h in human_tracks]),
        "dims":torch.cat([h.boxes.dims for h in human_tracks]),
        "rots":torch.cat([h.boxes.rots for h in human_tracks]),
        "valid":torch.cat([h.boxes.valid for h in human_tracks]), "frames":frames}
    torch.save(raw, Path(cfg["pedestrians"]["tracks_file"]))
    totals = {
        key: sum(int(record[key]) for record in track_records)
        for key in (
            "bbox_valid_frames", "raw_valid_poses", "completed_valid_poses",
            "internal_slerp_frames", "leading_hold_frames",
            "trailing_hold_frames", "remaining_bbox_supported_invalid",
        )
    }
    manifest = {
        "tracks": track_records,
        "totals": totals,
        "pose_completion": cfg["pedestrians"].get("pose_completion", {}),
        "seconds": time.perf_counter() - started,
    }
    (out/"pedestrian_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

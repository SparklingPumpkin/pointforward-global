"""Export a persistent runtime payload as one standard preview-color PLY.

The runtime payload keeps 128-D features, while a standard 3DGS PLY can only
carry RGB/degree-0 SH.  This exporter preserves the final global geometry and
uses the payload's cached ``preview_rgb`` for viewer-compatible colors.  The
dynamic subset is posed at one requested scene frame using the saved global
box graph.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--pedestrian-tracks", default=None)
    parser.add_argument("--smpl-model", default=None)
    parser.add_argument("--rgb-residual-state", default=None)
    parser.add_argument(
        "--rgb-surface-proxy-dir", default=None,
        help="Optional Round-64 proxy snapshot matching the runtime renderer.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    payload_dir = Path(args.payload_dir).resolve()
    out = Path(args.out).resolve()
    static = load(payload_dir / "final_static_128d.pt")
    frame = int(args.frame)
    final_dynamic_path = payload_dir / "final_dynamic_128d.pt"
    global_meta_path = payload_dir / "global_dynamic_meta.pt"
    if final_dynamic_path.exists() and global_meta_path.exists():
        dynamic = load(final_dynamic_path)
        meta = torch.load(global_meta_path, map_location="cpu", weights_only=True)
        boxes = boxes_from(meta["boxes"])
        pose_slot = frame
        target_time = float(meta["normalized_times"][frame])
        dynamic_semantics = "scene-global dynamic payload"
        dynamic_source = final_dynamic_path
    else:
        selected = None
        for meta_path in sorted(payload_dir.glob("dynamic_window_*_meta.pt")):
            candidate = torch.load(meta_path, map_location="cpu", weights_only=True)
            rendered_frames = [
                int(value) for value in candidate.get(
                    "rendered_frames", candidate["window_frames"]
                )
            ]
            if frame in rendered_frames:
                if selected is not None:
                    raise ValueError(f"Frame {frame} belongs to multiple dynamic windows")
                selected = (meta_path, candidate)
        if selected is None:
            raise ValueError(f"No dynamic window renders frame {frame}")
        meta_path, meta = selected
        dynamic_source = meta_path.with_name(meta_path.name.replace("_meta.pt", ".pt"))
        dynamic = load(dynamic_source)
        boxes = boxes_from(meta["boxes"])
        window_frames = [int(value) for value in meta["window_frames"]]
        pose_slot = window_frames.index(frame)
        target_time = float(meta["normalized_times"][pose_slot])
        dynamic_semantics = "mutually exclusive clip-local dynamic payload"
    static_rgb = static.extras.get("preview_rgb")
    dynamic_rgb = dynamic.extras.get("preview_rgb")
    if static_rgb is None:
        static_rgb = query_rgb(static.colors)
    if dynamic_rgb is None:
        dynamic_rgb = query_rgb(dynamic.colors)
    if args.rgb_residual_state:
        residual = torch.load(
            Path(args.rgb_residual_state).resolve(),
            map_location="cpu", weights_only=True,
        )
        static_rgb = (static_rgb + residual["static_rgb_residual"]).clamp(0.0, 1.0)
        dynamic_rgb = (dynamic_rgb + residual["dynamic_rgb_residual"]).clamp(0.0, 1.0)
    static_opacities = temporal_opacity(static.opacities, static.sigma, static.times, target_time)
    dyn_means, dyn_quats, dyn_opacities = pose_dynamic(
        dynamic, boxes, pose_slot, target_time
    )
    if args.pedestrian_tracks:
        if not args.smpl_model:
            parser.error("--pedestrian-tracks requires --smpl-model")
        from collections import OrderedDict
        from types import SimpleNamespace
        from pedestrian_reference.models.smpl import SMPL
        from pointforward.models.pedestrian_integration import skin_pedestrians

        raw = torch.load(args.pedestrian_tracks, map_location="cpu", weights_only=True)
        shim = SimpleNamespace(
            smpl=SMPL(args.smpl_model),
            pedestrian_runtime=raw,
            pedestrian_skin_cache=OrderedDict(),
        )
        posed = skin_pedestrians(
            shim,
            dynamic,
            {
                "means": dyn_means,
                "quats": dyn_quats,
                "opacities": dyn_opacities,
                "scales": dynamic.scales,
                "colors": dynamic.colors,
            },
            pose_slot,
        )
        dyn_means = posed["means"]
        dyn_quats = posed["quats"]
        dyn_opacities = posed["opacities"]
    proxy_record = None
    proxy_means = torch.empty((0, 3))
    proxy_quats = torch.empty((0, 4))
    proxy_scales = torch.empty((0, 3))
    proxy_opacities = torch.empty(0)
    proxy_rgb = torch.empty((0, 3))
    if args.rgb_surface_proxy_dir:
        proxy_dir = Path(args.rgb_surface_proxy_dir).resolve()
        proxy_manifest_path = proxy_dir / "manifest.json"
        proxy_manifest = json.loads(proxy_manifest_path.read_text(encoding="utf-8"))
        reliable_slots = torch.tensor(
            proxy_manifest["reliable_global_slots"], dtype=torch.long
        )
        suppressed = torch.isin(dynamic.instance_id.long(), reliable_slots)
        dyn_opacities = dyn_opacities.clone()
        dyn_opacities[suppressed] = 0.0
        proxy_static = load(proxy_dir / "static_proxy.pt")
        proxy_dynamic = load(proxy_dir / "dynamic_proxy.pt")
        proxy_dyn_means, proxy_dyn_quats, proxy_dyn_opacities = pose_dynamic(
            proxy_dynamic, boxes, pose_slot, target_time
        )
        proxy_static_opacities = temporal_opacity(
            proxy_static.opacities, proxy_static.sigma,
            proxy_static.times, target_time,
        )
        proxy_means = torch.cat((proxy_static.means, proxy_dyn_means), dim=0)
        proxy_quats = torch.cat((proxy_static.quats, proxy_dyn_quats), dim=0)
        proxy_scales = torch.cat((proxy_static.scales, proxy_dynamic.scales), dim=0)
        proxy_opacities = torch.cat((proxy_static_opacities, proxy_dyn_opacities), dim=0)
        proxy_rgb = torch.cat((proxy_static.colors, proxy_dynamic.colors), dim=0)
        proxy_record = {
            "directory": str(proxy_dir),
            "manifest_sha256": hashlib.sha256(proxy_manifest_path.read_bytes()).hexdigest(),
            "reliable_global_slots": proxy_manifest["reliable_global_slots"],
            "suppressed_baseline_dynamic_count": int(suppressed.sum()),
            "static_proxy_count": int(proxy_static.count),
            "dynamic_proxy_count": int(proxy_dynamic.count),
        }
    means = torch.cat((static.means, dyn_means, proxy_means), dim=0).numpy()
    quats = torch.cat((static.quats, dyn_quats, proxy_quats), dim=0).numpy()
    scales = torch.cat((static.scales, dynamic.scales, proxy_scales), dim=0).numpy()
    opacities = torch.cat((static_opacities, dyn_opacities, proxy_opacities), dim=0).numpy()
    rgb = torch.cat((static_rgb, dynamic_rgb, proxy_rgb), dim=0).numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_standard_ply(str(out), means, rgb, opacities, scales, quats)
    record = {
        "status": "completed", "payload_dir": str(payload_dir), "out": str(out),
        "scene_frame": frame, "target_time": target_time,
        "dynamic_source": str(dynamic_source),
        "dynamic_semantics": dynamic_semantics,
        "static_count": int(static.count), "dynamic_count": int(dynamic.count),
        "proxy": proxy_record,
        "gaussian_count": int(len(means)),
        "coordinate_frame": "scene017 shared world frame from runtime payload",
        "colour": (
            "cached preview_rgb plus additive Gaussian RGB residual encoded as degree-0 SH"
            if args.rgb_residual_state
            else "cached preview_rgb encoded as degree-0 SH; not equivalent to 128-D+UNet output"
        ),
        "rgb_residual_state": (
            str(Path(args.rgb_residual_state).resolve())
            if args.rgb_residual_state else None
        ),
        "pedestrian_tracks": (
            str(Path(args.pedestrian_tracks).resolve())
            if args.pedestrian_tracks else None
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

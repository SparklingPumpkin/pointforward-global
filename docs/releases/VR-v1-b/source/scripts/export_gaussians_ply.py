"""Export one PointForward scene snapshot as a standard 3DGS binary PLY.

PointForward stores a 128-D learned colour feature per Gaussian. The final RGB
is produced only after feature rasterisation and a spatial UNet, so there is no
exact per-Gaussian RGB/SH conversion. This exporter uses each query's source RGB
as a viewer-compatible preview colour while preserving the decoded geometry,
opacity, scale, and rotation at the requested target frame.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from plyfile import PlyData, PlyElement

from pointforward.models import InstanceBoxes, SceneInput
from pointforward.models.query import query_rgb
from scripts.eval import build_model
from scripts.train import build_dataset


SH_C0 = 0.28209479177387814


def boxes_to(boxes: InstanceBoxes | None, device: torch.device) -> InstanceBoxes | None:
    if boxes is None:
        return None
    return InstanceBoxes(
        centers=boxes.centers.to(device),
        dims=boxes.dims.to(device),
        rots=boxes.rots.to(device),
        times=boxes.times.to(device),
        valid=boxes.valid.to(device),
    )


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def write_standard_ply(
    path: str,
    means: np.ndarray,
    rgb: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
) -> None:
    """Write Graphdeco-style degree-0 SH fields in a binary little-endian PLY."""
    count = means.shape[0]
    fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    vertices = np.empty(count, dtype=np.dtype(fields))
    vertices["x"], vertices["y"], vertices["z"] = means.T
    vertices["nx"] = 0.0
    vertices["ny"] = 0.0
    vertices["nz"] = 0.0
    sh_dc = (np.clip(rgb, 0.0, 1.0) - 0.5) / SH_C0
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = sh_dc.T
    vertices["opacity"] = logit(opacities)
    log_scales = np.log(np.clip(scales, 1e-8, None))
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = log_scales.T
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = quats.T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="target frame offset inside the clip; dynamic Gaussians are posed at this frame",
    )
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, args.split)
    if not 0 <= args.clip_index < len(dataset):
        raise ValueError(f"clip index {args.clip_index} is outside [0, {len(dataset) - 1}]")

    scene_index, start = dataset.index[args.clip_index]
    metadata = dataset.scenes[scene_index]
    window = metadata["frames"][start : start + dataset.clip_length]
    if not 0 <= args.frame_offset < len(window):
        raise ValueError(f"frame offset {args.frame_offset} is outside [0, {len(window) - 1}]")

    world_from = np.linalg.inv(dataset._ego_pose(metadata, window[0]))
    normalized_times = dataset._normalised_times(window)
    slot_of = {frame: index for index, frame in enumerate(window)}
    context_frames = sorted(
        {
            window[min(index * dataset.frame_stride, len(window) - 1)]
            for index in range(dataset.num_context)
        }
    )
    context = dataset._collect(
        metadata,
        context_frames,
        slot_of,
        normalized_times,
        dataset.resolution,
        world_from,
        lidar=dataset.load_lidar,
    )
    boxes = boxes_to(dataset._build_boxes(metadata, window, world_from), device)
    scene = SceneInput(
        images=context["images"].to(device),
        intrinsics=context["intrinsics"].to(device),
        cam_to_world=context["cam_to_world"].to(device),
        times=context["times"].to(device),
        frame_index=context["frame_index"].to(device),
        boxes=boxes,
        sparse_depth=context.get("sparse_depth", None).to(device) if "sparse_depth" in context else None,
        sparse_mask=context.get("sparse_mask", None).to(device) if "sparse_mask" in context else None,
    )

    model = build_model(config, device, args.ckpt)
    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    gaussians = model.decode(scene, depth, sampled)

    target_time = normalized_times[args.frame_offset].to(device)
    posed = model.pose_gaussians(
        gaussians,
        boxes,
        frame=args.frame_offset,
        target_time=target_time,
    )
    preview_rgb = query_rgb(sampled["queries"])

    arrays = {
        "means": posed["means"].float().cpu().numpy(),
        "rgb": preview_rgb.float().cpu().numpy(),
        "opacities": posed["opacities"].float().cpu().numpy(),
        "scales": posed["scales"].float().cpu().numpy(),
        "quats": posed["quats"].float().cpu().numpy(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_standard_ply(args.out, **arrays)

    metadata_path = os.path.splitext(args.out)[0] + ".json"
    export_metadata = {
        "source_checkpoint": os.path.abspath(args.ckpt),
        "scene": metadata["name"],
        "clip_index": args.clip_index,
        "window_frames": [int(frame) for frame in window],
        "context_frames": [int(frame) for frame in context_frames],
        "snapshot_frame_offset": args.frame_offset,
        "snapshot_frame": int(window[args.frame_offset]),
        "snapshot_time": float(target_time),
        "gaussian_count": int(gaussians.count),
        "coordinate_frame": "world recentered at the first ego pose of this clip",
        "geometry": "decoded PointForward geometry posed at the snapshot frame",
        "colour": "degree-0 SH converted from source-query RGB; preview only, not the learned 128-D feature plus UNet output",
        "dynamic_semantics": "single temporal snapshot; export another frame-offset for a different dynamic pose",
    }
    with open(metadata_path, "w") as handle:
        json.dump(export_metadata, handle, indent=2)

    print(f"saved {args.out}")
    print(f"saved {metadata_path}")
    print(json.dumps(export_metadata, indent=2))


if __name__ == "__main__":
    main()
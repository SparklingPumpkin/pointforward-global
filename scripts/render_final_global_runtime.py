"""Render a final global map with PointForward's original feature/UNet path.

The runtime payload is produced by ``deliver_continuous_scene.py``.  It keeps
the final static global Gaussian features plus one dynamic Gaussian set and
box trajectory per reconstruction window.  This lets the final static map be
held fixed while dynamic Gaussians are posed at each requested frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import (
    Gaussians,
    InstanceBoxes,
    TargetViews,
    box_dynamic_masks,
)
from pointforward.models.rotation import matrix_to_quaternion, quaternion_to_matrix
from scripts.deliver_continuous_scene import (
    encode_video,
    horizontal_strip,
    interpolate_camera_pose,
    label,
    to_uint8,
    vertical_strip,
)
from scripts.eval import build_model
from scripts.train import build_dataset


_GAUSSIAN_FIELDS = (
    "means", "quats", "scales", "opacities", "colors", "sigma", "times", "instance_id"
)


def load_payload(path: Path, device: torch.device) -> Gaussians:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return Gaussians(
        **{name: payload[name].to(device) for name in _GAUSSIAN_FIELDS},
        extras={name: value.to(device) for name, value in payload.get("extras", {}).items()},
    )


def combine(static: Gaussians, dynamic: Gaussians) -> Gaussians:
    extra_names = set(static.extras) & set(dynamic.extras)
    return Gaussians(
        **{
            name: torch.cat((getattr(static, name), getattr(dynamic, name)), dim=0)
            for name in _GAUSSIAN_FIELDS
        },
        extras={
            name: torch.cat((static.extras[name], dynamic.extras[name]), dim=0)
            for name in extra_names
        },
    )


def load_boxes(payload: dict | None, device: torch.device) -> InstanceBoxes | None:
    if payload is None:
        return None
    return InstanceBoxes(
        centers=payload["centers"].to(device),
        dims=payload["dims"].to(device),
        rots=payload["rots"].to(device),
        times=payload["times"].to(device),
        valid=payload["valid"].to(device),
    )


def quaternion_slerp(
    first: torch.Tensor, second: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Shortest-path batched quaternion interpolation."""
    first = torch.nn.functional.normalize(first, dim=-1)
    second = torch.nn.functional.normalize(second, dim=-1)
    dot = (first * second).sum(dim=-1, keepdim=True)
    second = torch.where(dot < 0, -second, second)
    dot = dot.abs().clamp(max=1.0)
    near = dot > 0.9995
    theta = torch.acos(dot)
    denominator = torch.sin(theta).clamp_min(1e-8)
    spherical = (
        torch.sin((1.0 - alpha) * theta) / denominator * first
        + torch.sin(alpha * theta) / denominator * second
    )
    linear = (1.0 - alpha) * first + alpha * second
    return torch.nn.functional.normalize(
        torch.where(near, linear, spherical), dim=-1
    )


def interpolate_dynamic_boxes(
    boxes: InstanceBoxes | None, lower_slot: int, alpha: float
) -> tuple[InstanceBoxes | None, int]:
    """Build boxes with a fixed canonical slot and one interpolated pose."""
    if boxes is None:
        return None, lower_slot
    upper_slot = min(lower_slot + 1, boxes.centers.shape[1] - 1)
    canonical_slot = boxes.canonical_index()
    rows = torch.arange(boxes.num_instances, device=boxes.device)
    # A compact two-slot graph is sufficient: slot 0 preserves each track's
    # original canonical box and slot 1 holds the requested target pose.
    # Directly replacing an instance's first valid pose would also change its
    # canonical frame and cancel the intended motion for newly entering tracks.
    centers = torch.stack((
        boxes.centers[rows, canonical_slot], boxes.centers[:, lower_slot]
    ), dim=1)
    dims = torch.stack((
        boxes.dims[rows, canonical_slot], boxes.dims[:, lower_slot]
    ), dim=1)
    rots = torch.stack((
        boxes.rots[rows, canonical_slot], boxes.rots[:, lower_slot]
    ), dim=1)
    times = torch.stack((
        boxes.times[rows, canonical_slot], boxes.times[:, lower_slot]
    ), dim=1)
    valid = torch.stack((
        torch.ones(boxes.num_instances, dtype=torch.bool, device=boxes.device),
        boxes.valid[:, lower_slot],
    ), dim=1)
    render_slot = 1
    if alpha > 0.0:
        centers[:, render_slot] = torch.lerp(
            boxes.centers[:, lower_slot], boxes.centers[:, upper_slot], alpha
        )
        dims[:, render_slot] = torch.lerp(
            boxes.dims[:, lower_slot], boxes.dims[:, upper_slot], alpha
        )
        rots[:, render_slot] = quaternion_to_matrix(quaternion_slerp(
            matrix_to_quaternion(boxes.rots[:, lower_slot]),
            matrix_to_quaternion(boxes.rots[:, upper_slot]),
            alpha,
        ))
        times[:, render_slot] = torch.lerp(
            boxes.times[:, lower_slot], boxes.times[:, upper_slot], alpha
        )
    # Do not interpolate a missing annotation into a visible vehicle. Tracks
    # entering or leaving the scene switch visibility at the source frame.
    valid[:, render_slot] = (
        boxes.valid[:, lower_slot]
        if alpha == 0.0
        else boxes.valid[:, lower_slot] & boxes.valid[:, upper_slot]
    )
    return InstanceBoxes(
        centers=centers, dims=dims, rots=rots, times=times, valid=valid
    ), render_slot


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    squared_error = (prediction.float() - target.float()) ** 2
    if mask is None:
        mse = torch.mean(squared_error)
    else:
        valid = torch.broadcast_to(mask.bool(), squared_error.shape)
        if not valid.any():
            raise ValueError("PSNR mask contains no valid pixels")
        mse = squared_error[valid].mean()
    mse = mse.clamp_min(1e-10)
    return float((-10.0 * torch.log10(mse)).cpu())


def apply_camera_offset(
    cam_to_world: torch.Tensor,
    ego_to_world: torch.Tensor,
    lateral_left_m: float,
    vertical_m: float,
    pitch_down_deg: float,
) -> torch.Tensor:
    """Apply an ego-left/world-up translation and a local camera pitch."""
    output = cam_to_world.clone()
    left = torch.nn.functional.normalize(ego_to_world[:3, 1], dim=0)
    world_up = output.new_tensor((0.0, 0.0, 1.0))
    output[:3, 3] += left * lateral_left_m + world_up * vertical_m
    if pitch_down_deg:
        # OpenCV cameras use +Y down and +Z forward.  A negative local-X
        # rotation tilts the forward axis towards +Y, i.e. down in the image.
        angle = math.radians(-pitch_down_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = output.new_tensor((
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
            (0.0, sine, cosine),
        ))
        output[:3, :3] = output[:3, :3] @ rotation
    return output


def look_at_pose(
    eye: torch.Tensor, target: torch.Tensor, world_up: torch.Tensor
) -> torch.Tensor:
    """Construct an OpenCV cam-to-world pose looking from eye to target."""
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.linalg.cross(forward, world_up)
    if float(torch.linalg.vector_norm(right)) < 1e-6:
        raise ValueError("Orbit view direction is parallel to world up")
    right = torch.nn.functional.normalize(right, dim=0)
    down = torch.nn.functional.normalize(torch.linalg.cross(forward, right), dim=0)
    output = torch.eye(4, dtype=eye.dtype, device=eye.device)
    output[:3, :3] = torch.stack((right, down, forward), dim=1)
    output[:3, 3] = eye
    return output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--refiner-state",
        default=None,
        help="Optional target-domain refiner state overlay; the base checkpoint is unchanged.",
    )
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--camera-trajectory", choices=("source", "orbit"), default="source",
        help="Follow source cameras or render a fixed-time 360-degree scene orbit.",
    )
    parser.add_argument(
        "--camera-lateral-left-m", type=float, default=0.0,
        help="Translate every source camera along the interpolated ego left axis.",
    )
    parser.add_argument(
        "--camera-vertical-m", type=float, default=0.0,
        help="Translate every source camera along world +Z.",
    )
    parser.add_argument(
        "--camera-pitch-down-deg", type=float, default=0.0,
        help="Pitch every source camera down around its local right axis.",
    )
    parser.add_argument("--orbit-frame-count", type=int, default=180)
    parser.add_argument(
        "--orbit-source-frame", type=int, default=-1,
        help="Scene frame whose dynamic state is frozen during the orbit; -1 uses the middle frame.",
    )
    parser.add_argument("--orbit-radius-m", type=float, default=None)
    parser.add_argument("--orbit-height-m", type=float, default=None)
    parser.add_argument("--orbit-center-x", type=float, default=None)
    parser.add_argument("--orbit-center-y", type=float, default=None)
    parser.add_argument("--orbit-center-z", type=float, default=None)
    parser.add_argument("--orbit-start-angle-deg", type=float, default=0.0)
    parser.add_argument(
        "--camera-interpolation-factor",
        type=int,
        default=1,
        help=(
            "Number of evenly spaced camera-pose samples per source-frame "
            "interval. Dynamic Gaussian state is held at the lower source frame."
        ),
    )
    parser.add_argument(
        "--interpolate-dynamics",
        action="store_true",
        help=(
            "Interpolate dynamic box translation, rotation, and target time "
            "together with the camera. Without this flag dynamic objects are "
            "held at the lower source frame for backward compatibility."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--camera-batch-size",
        type=int,
        default=1,
        help="Number of cameras rendered together; larger values trade memory for speed.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute source-frame metrics without writing image frames or videos.",
    )
    parser.add_argument(
        "--dynamic-bbox-metrics",
        action="store_true",
        help=(
            "Also compute source-frame PSNR inside the union of projected "
            "dynamic 3D bounding-box rectangles."
        ),
    )
    parser.add_argument(
        "--dynamic-opacity-scale",
        type=float,
        default=1.0,
        help="Inference-only multiplier for dynamic Gaussian opacity.",
    )
    parser.add_argument(
        "--dynamic-scale-multiplier",
        type=float,
        default=1.0,
        help="Inference-only multiplier for dynamic Gaussian radii.",
    )
    parser.add_argument(
        "--static-scale-multiplier",
        type=float,
        default=1.0,
        help="Inference-only multiplier for static Gaussian radii.",
    )
    args = parser.parse_args()

    if args.camera_interpolation_factor <= 0:
        parser.error("--camera-interpolation-factor must be positive")
    if args.orbit_frame_count <= 0:
        parser.error("--orbit-frame-count must be positive")
    if args.camera_trajectory == "orbit" and args.camera_interpolation_factor != 1:
        parser.error("Orbit mode does not use --camera-interpolation-factor")
    if args.camera_batch_size <= 0:
        parser.error("--camera-batch-size must be positive")
    if args.dynamic_opacity_scale <= 0:
        parser.error("--dynamic-opacity-scale must be positive")
    if args.dynamic_scale_multiplier <= 0:
        parser.error("--dynamic-scale-multiplier must be positive")
    if args.static_scale_multiplier <= 0:
        parser.error("--static-scale-multiplier must be positive")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for 128-D feature rendering")
    device = torch.device(args.device)
    config_path = Path(args.config).resolve()
    payload_dir = Path(args.payload_dir).resolve()
    output_dir = Path(args.out).resolve()
    predicted_dir = output_dir / "frames" / "predicted"
    comparison_dir = output_dir / "frames" / "comparison"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    novel_view = (
        args.camera_trajectory == "orbit"
        or args.camera_lateral_left_m != 0.0
        or args.camera_vertical_m != 0.0
        or args.camera_pitch_down_deg != 0.0
    )
    if not novel_view:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, args.split)
    matches = [metadata for metadata in dataset.scenes if metadata["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    frames = list(metadata["frames"])
    if len(frames) < 2:
        raise ValueError(f"Expected at least two scene frames, found {len(frames)}")
    if len(dataset.cameras) != 3:
        raise ValueError(f"Expected exactly three cameras, found {dataset.cameras}")

    static_path = payload_dir / "final_static_128d.pt"
    meta_paths = sorted(payload_dir.glob("dynamic_window_*_meta.pt"))
    final_dynamic_path = payload_dir / "final_dynamic_128d.pt"
    global_dynamic_meta_path = payload_dir / "global_dynamic_meta.pt"
    has_global_dynamic = (
        final_dynamic_path.exists() and global_dynamic_meta_path.exists()
    )
    if not static_path.exists() or (not meta_paths and not has_global_dynamic):
        raise FileNotFoundError(f"Incomplete runtime payload in {payload_dir}")

    static = load_payload(static_path, device)
    static.scales = static.scales * args.static_scale_multiplier
    windows = []
    frame_to_window = {}
    if has_global_dynamic:
        meta = torch.load(
            global_dynamic_meta_path, map_location="cpu", weights_only=True
        )
        window_frames = [int(frame) for frame in meta["frames"]]
        dynamic = load_payload(final_dynamic_path, device)
        dynamic.opacities = (dynamic.opacities * args.dynamic_opacity_scale).clamp_max(1.0)
        dynamic.scales = dynamic.scales * args.dynamic_scale_multiplier
        window = {
            "frames": window_frames,
            "rendered_frames": window_frames,
            "times": meta["normalized_times"],
            "boxes": load_boxes(meta["boxes"], device),
            "evaluation_boxes": load_boxes(
                meta.get("evaluation_boxes", meta["boxes"]), device
            ),
            "dynamic": dynamic,
            "dynamic_path": final_dynamic_path,
        }
        windows.append(window)
        for frame in window["rendered_frames"]:
            frame_to_window[frame] = window
    else:
        for meta_path in meta_paths:
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)
            window_frames = [int(frame) for frame in meta["window_frames"]]
            dynamic_path = meta_path.with_name(meta_path.name.replace("_meta.pt", ".pt"))
            dynamic = load_payload(dynamic_path, device)
            dynamic.opacities = (dynamic.opacities * args.dynamic_opacity_scale).clamp_max(1.0)
            dynamic.scales = dynamic.scales * args.dynamic_scale_multiplier
            window = {
                "frames": window_frames,
                "rendered_frames": [
                    int(frame) for frame in meta.get("rendered_frames", window_frames)
                ],
                "times": meta["normalized_times"],
                "boxes": load_boxes(meta["boxes"], device),
                "evaluation_boxes": load_boxes(
                    meta.get("evaluation_boxes", meta["boxes"]), device
                ),
                "dynamic": dynamic,
                "dynamic_path": dynamic_path,
            }
            windows.append(window)
            for frame in window["rendered_frames"]:
                if frame in frame_to_window:
                    raise ValueError(
                        f"Frame {frame} is rendered by multiple payload windows"
                    )
                frame_to_window[frame] = window
    if sorted(frame_to_window) != sorted(frames):
        missing = sorted(set(frames) - set(frame_to_window))
        raise ValueError(f"Runtime payload does not cover all frames; missing {missing}")

    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    all_times = dataset._normalised_times(frames)
    all_slots = {frame: slot for slot, frame in enumerate(frames)}
    height, width = dataset.resolution
    model = build_model(config, device, args.ckpt)
    if args.refiner_state:
        refiner_payload = torch.load(
            Path(args.refiner_state).resolve(), map_location="cpu", weights_only=True
        )
        model.refiner.load_state_dict(
            refiner_payload.get("refiner", refiner_payload), strict=True
        )
    started = time.perf_counter()
    frame_records = []
    current_key = None
    current_gaussians = None

    ego_poses = torch.from_numpy(np.stack([
        world_from @ dataset._ego_pose(metadata, frame) for frame in frames
    ])).float()
    orbit = None
    output_samples = []
    if args.camera_trajectory == "orbit":
        source_index = (
            len(frames) // 2 if args.orbit_source_frame < 0
            else frames.index(args.orbit_source_frame)
        )
        frame = frames[source_index]
        positions = ego_poses[:, :3, 3]
        minimum, maximum = positions.amin(dim=0), positions.amax(dim=0)
        center = (minimum + maximum) * 0.5
        for axis, value in enumerate((
            args.orbit_center_x, args.orbit_center_y, args.orbit_center_z
        )):
            if value is not None:
                center[axis] = value
        planar_span = float(torch.linalg.vector_norm(maximum[:2] - minimum[:2]))
        radius = args.orbit_radius_m or (0.5 * planar_span + 50.0)
        height_above_center = args.orbit_height_m or max(35.0, 0.3 * planar_span)
        if radius <= 0 or height_above_center <= 0:
            parser.error("Orbit radius and height must be positive")
        orbit = {
            "center": center,
            "radius_m": radius,
            "height_m": height_above_center,
        }
        output_samples = [
            (source_index, frame, 0.0, index / args.orbit_frame_count)
            for index in range(args.orbit_frame_count)
        ]
    else:
        for source_index, frame in enumerate(frames):
            if source_index + 1 < len(frames):
                output_samples.extend(
                    (source_index, frame, step / args.camera_interpolation_factor, None)
                    for step in range(args.camera_interpolation_factor)
                )
            else:
                output_samples.append((source_index, frame, 0.0, None))

    cached_target_index = None
    cached_target = None
    cached_next_target = None
    for output_index, (source_index, frame, alpha, orbit_fraction) in enumerate(output_samples):
        window = frame_to_window[frame]
        key = tuple(window["frames"])
        if key != current_key:
            if current_gaussians is not None:
                del current_gaussians
                torch.cuda.empty_cache()
            current_gaussians = combine(static, window["dynamic"])
            current_key = key
        gaussians = current_gaussians
        local_slot = window["frames"].index(frame)
        if source_index != cached_target_index:
            if (
                cached_next_target is not None
                and cached_target_index is not None
                and source_index == cached_target_index + 1
            ):
                target = cached_next_target
            else:
                target = dataset._collect(
                    metadata, [frame], all_slots, all_times,
                    dataset.resolution, world_from, lidar=False,
                )
            cached_target = target
            cached_target_index = source_index
            if source_index + 1 < len(frames):
                cached_next_target = dataset._collect(
                    metadata, [frames[source_index + 1]], all_slots, all_times,
                    dataset.resolution, world_from, lidar=False,
                )
            else:
                cached_next_target = None
        else:
            target = cached_target

        if args.camera_trajectory == "orbit":
            angle = math.radians(args.orbit_start_angle_deg) + 2.0 * math.pi * orbit_fraction
            center = orbit["center"]
            eye = center.new_tensor((
                float(center[0]) + orbit["radius_m"] * math.cos(angle),
                float(center[1]) + orbit["radius_m"] * math.sin(angle),
                float(center[2]) + orbit["height_m"],
            ))
            target_cam_to_world = look_at_pose(
                eye, center, center.new_tensor((0.0, 0.0, 1.0))
            ).unsqueeze(0)
            active_camera_ids = [dataset.cameras[0]]
            active_intrinsics = target["intrinsics"][:1]
        elif alpha > 0.0:
            target_cam_to_world = torch.stack([
                interpolate_camera_pose(current, following, alpha)
                for current, following in zip(
                    target["cam_to_world"], cached_next_target["cam_to_world"]
                )
            ])
            ego_to_world = interpolate_camera_pose(
                ego_poses[source_index], ego_poses[source_index + 1], alpha
            )
            target_cam_to_world = torch.stack([
                apply_camera_offset(
                    pose, ego_to_world, args.camera_lateral_left_m,
                    args.camera_vertical_m, args.camera_pitch_down_deg,
                )
                for pose in target_cam_to_world
            ])
            active_camera_ids = list(dataset.cameras)
            active_intrinsics = target["intrinsics"]
        else:
            target_cam_to_world = target["cam_to_world"]
            ego_to_world = ego_poses[source_index]
            target_cam_to_world = torch.stack([
                apply_camera_offset(
                    pose, ego_to_world, args.camera_lateral_left_m,
                    args.camera_vertical_m, args.camera_pitch_down_deg,
                )
                for pose in target_cam_to_world
            ])
            active_camera_ids = list(dataset.cameras)
            active_intrinsics = target["intrinsics"]
        if args.interpolate_dynamics:
            render_boxes, render_slot = interpolate_dynamic_boxes(
                window["boxes"], local_slot, alpha
            )
            evaluation_boxes, _ = interpolate_dynamic_boxes(
                window["evaluation_boxes"], local_slot, alpha
            )
        else:
            render_boxes, render_slot = window["boxes"], local_slot
            evaluation_boxes = window["evaluation_boxes"]
        upper_slot = min(local_slot + 1, len(window["times"]) - 1)
        target_time = float(window["times"][local_slot])
        if args.interpolate_dynamics and alpha > 0.0:
            target_time = float(torch.lerp(
                window["times"][local_slot], window["times"][upper_slot], alpha
            ))
        # Render one camera at a time.  Every camera still sees the complete
        # global Gaussian set; this only avoids gsplat materialising the
        # N-by-128 colour buffer for all three cameras simultaneously.
        rendered_views = []
        for camera_start in range(0, len(active_camera_ids), args.camera_batch_size):
            camera_end = min(
                camera_start + args.camera_batch_size, len(active_camera_ids)
            )
            target_views = TargetViews(
                intrinsics=active_intrinsics[camera_start:camera_end].to(device),
                cam_to_world=target_cam_to_world[camera_start:camera_end].to(device),
                times=torch.full(
                    (camera_end - camera_start,), target_time, device=device,
                ),
                frame_index=torch.full(
                    (camera_end - camera_start,),
                    render_slot,
                    dtype=torch.long,
                    device=device,
                ),
            )
            rendered_views.append(model.render(
                gaussians, render_boxes, target_views,
                height=height, width=width, render_depth=False,
            )["rgb"])
        rendered = torch.cat(rendered_views, dim=0)
        sample_label = (
            f"orbit {360.0 * orbit_fraction:.1f} deg | frozen frame {frame:03d}"
            if args.camera_trajectory == "orbit"
            else
            f"frame {frame:03d}"
            if alpha == 0.0
            else f"frame {frame:03d}->{frames[source_index + 1]:03d} | alpha {alpha:.3f}"
        )
        if not args.metrics_only:
            prediction_panels = [
                label(to_uint8(rendered[index]), f"Final global 128D+UNet | {sample_label} | camera {camera}")
                for index, camera in enumerate(active_camera_ids)
            ]
            prediction_strip = horizontal_strip(prediction_panels)
            Image.fromarray(prediction_strip).save(predicted_dir / f"frame_{output_index:04d}.jpg", quality=92)
            if not novel_view:
                ground_truth_panels = [
                    label(to_uint8(target["images"][index]), f"GT RGB held at frame {frame:03d} | camera {camera}")
                    for index, camera in enumerate(active_camera_ids)
                ]
                comparison = vertical_strip((horizontal_strip(ground_truth_panels), prediction_strip))
                Image.fromarray(comparison).save(comparison_dir / f"frame_{output_index:04d}.jpg", quality=92)
        non_sky_mask = None
        dynamic_bbox_mask = None
        if alpha == 0.0 and not novel_view:
            if "sky_mask" not in target:
                raise KeyError("Dataset target does not provide sky_mask")
            non_sky_mask = target["sky_mask"].to(device) < 0.5
            if args.dynamic_bbox_metrics and evaluation_boxes is not None:
                dynamic_bbox_mask = box_dynamic_masks(
                    evaluation_boxes,
                    torch.full(
                        (len(dataset.cameras),),
                        render_slot,
                        dtype=torch.long,
                        device=device,
                    ),
                    target["intrinsics"].to(device),
                    torch.linalg.inv(target_cam_to_world.to(device)),
                    height,
                    width,
                ).unsqueeze(1)
                if not bool(dynamic_bbox_mask.any()):
                    dynamic_bbox_mask = None
        frame_records.append({
            "output_index": output_index,
            "frame": int(frame),
            "source_index": source_index,
            "camera_interpolation_alpha": alpha,
            "dynamic_interpolation": args.interpolate_dynamics,
            "window_frames": window["frames"],
            "local_slot": local_slot,
            "gaussian_count": gaussians.count,
            "psnr_all_cameras": (
                psnr(rendered, target["images"].to(device))
                if alpha == 0.0 and not novel_view else None
            ),
            "psnr_non_sky_all_cameras": (
                psnr(rendered, target["images"].to(device), non_sky_mask)
                if alpha == 0.0 and not novel_view else None
            ),
            "psnr_dynamic_bbox_all_cameras": (
                psnr(rendered, target["images"].to(device), dynamic_bbox_mask)
                if dynamic_bbox_mask is not None else None
            ),
        })
        if output_index % 25 == 0 or output_index == len(output_samples) - 1:
            print(json.dumps({
                "frame": int(frame),
                "index": output_index,
                "output_count": len(output_samples),
                "camera_interpolation_alpha": alpha,
                "gaussian_count": gaussians.count,
            }), flush=True)

    suffix = "" if args.camera_interpolation_factor == 1 else f"_{args.fps}fps"
    if args.camera_trajectory == "orbit":
        predicted_video = output_dir / "final_global_runtime_orbit.mp4"
    elif novel_view:
        predicted_video = output_dir / "final_global_runtime_novel_front3.mp4"
    else:
        predicted_video = output_dir / f"final_global_runtime_predicted_front3{suffix}.mp4"
    comparison_video = output_dir / f"final_global_runtime_gt_vs_predicted_front3{suffix}.mp4"
    if not args.metrics_only:
        encode_video(predicted_dir / "frame_%04d.jpg", predicted_video, args.fps)
        if not novel_view:
            encode_video(comparison_dir / "frame_%04d.jpg", comparison_video, args.fps)
    psnr_values = [
        record["psnr_all_cameras"] for record in frame_records
        if record["psnr_all_cameras"] is not None
    ]
    non_sky_psnr_values = [
        record["psnr_non_sky_all_cameras"] for record in frame_records
        if record["psnr_non_sky_all_cameras"] is not None
    ]
    manifest = {
        "status": "completed",
        "scene": args.scene,
        "frame_count": len(frames),
        "output_frame_count": len(output_samples),
        "frame_range": [int(frames[0]), int(frames[-1])],
        "fps": args.fps,
        "source_fps": args.fps / args.camera_interpolation_factor,
        "camera_interpolation_factor": args.camera_interpolation_factor,
        "camera_interpolation": (
            (
                "camera translation lerp and quaternion SLERP; dynamic box "
                "translation lerp, rotation SLERP, and target-time interpolation"
            )
            if args.camera_interpolation_factor > 1 and args.interpolate_dynamics
            else "translation lerp plus shortest-path quaternion SLERP; dynamic time and Gaussian state held at lower source frame"
            if args.camera_interpolation_factor > 1
            else "disabled"
        ),
        "dynamic_interpolation": args.interpolate_dynamics,
        "cameras": list(dataset.cameras) if args.camera_trajectory == "source" else [dataset.cameras[0]],
        "camera_trajectory": args.camera_trajectory,
        "novel_view": novel_view,
        "camera_transform": {
            "lateral_left_m": args.camera_lateral_left_m,
            "vertical_m": args.camera_vertical_m,
            "pitch_down_deg": args.camera_pitch_down_deg,
        },
        "orbit": None if orbit is None else {
            "center_world": [float(value) for value in orbit["center"]],
            "radius_m": orbit["radius_m"],
            "height_above_center_m": orbit["height_m"],
            "frozen_source_frame": int(frames[output_samples[0][0]]),
            "frame_count": args.orbit_frame_count,
        },
        "camera_batch_size": args.camera_batch_size,
        "resolution_hw_per_camera": [height, width],
        "config": str(config_path),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "refiner_state": (
            str(Path(args.refiner_state).resolve()) if args.refiner_state else None
        ),
        "payload_dir": str(payload_dir),
        "payload_sha256": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                [static_path, final_dynamic_path, global_dynamic_meta_path]
                if has_global_dynamic
                else [static_path, *[p for p in payload_dir.glob("dynamic_window_*.pt")]]
            )
        },
        "static_gaussian_count": static.count,
        "feature_dim": int(static.colors.shape[-1]),
        "rendering": (
            "final static global Gaussian + final scene-global dynamic Gaussian; model.pose_gaussians + 128-D render_features + refiner UNet"
            if has_global_dynamic
            else "final static global Gaussian + per-window dynamic payload; model.pose_gaussians + 128-D render_features + refiner UNet"
        ),
        "dynamic_semantics": (
            "one persistent dynamic Gaussian map uses stable scene track IDs and the full-scene box graph for every frame"
            if has_global_dynamic
            else "dynamic Gaussian set is selected by the corresponding clip and posed with that clip's box trajectory at each frame"
        ),
        "predicted_video": None if args.metrics_only else str(predicted_video),
        "comparison_video": (
            None if args.metrics_only or novel_view else str(comparison_video)
        ),
        "mean_psnr_source_frames_all_cameras": (
            float(np.mean(psnr_values)) if psnr_values else None
        ),
        "mean_psnr_non_sky_source_frames_all_cameras": (
            float(np.mean(non_sky_psnr_values)) if non_sky_psnr_values else None
        ),
        "dynamic_bbox_metrics_enabled": args.dynamic_bbox_metrics,
        "dynamic_opacity_scale": args.dynamic_opacity_scale,
        "dynamic_scale_multiplier": args.dynamic_scale_multiplier,
        "static_scale_multiplier": args.static_scale_multiplier,
        "mean_psnr_dynamic_bbox_source_frames_all_cameras": (
            float(np.mean([
                record["psnr_dynamic_bbox_all_cameras"]
                for record in frame_records
                if record["psnr_dynamic_bbox_all_cameras"] is not None
            ]))
            if any(
                record["psnr_dynamic_bbox_all_cameras"] is not None
                for record in frame_records
            )
            else None
        ),
        "frame_records": frame_records,
        "wall_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

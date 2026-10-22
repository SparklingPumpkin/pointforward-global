"""Render 60 FPS trajectories from stride-1 four-frame PointForward contexts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import SceneInput
from pointforward.models.gaussian_head import temporal_opacity
from pointforward.models.renderer import render_features
from pointforward.models.rotation import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix
from pointforward.models.scene_graph import decanonicalise
from scripts.deliver_continuous_scene import encode_video, label, to_uint8
from scripts.eval import build_model
from scripts.export_gaussians_ply import boxes_to
from scripts.train import build_dataset


def quaternion_slerp(first: torch.Tensor, second: torch.Tensor, alpha: float) -> torch.Tensor:
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
    return torch.nn.functional.normalize(torch.where(near, linear, spherical), dim=-1)


def interpolate_pose(first: torch.Tensor, second: torch.Tensor, alpha: float) -> torch.Tensor:
    output = torch.eye(4, dtype=first.dtype, device=first.device)
    first_quaternion = matrix_to_quaternion(first[:3, :3])
    second_quaternion = matrix_to_quaternion(second[:3, :3])
    output[:3, :3] = quaternion_to_matrix(quaternion_slerp(first_quaternion, second_quaternion, alpha))
    output[:3, 3] = torch.lerp(first[:3, 3], second[:3, 3], alpha)
    return output


def pose_gaussians_continuous(model, gaussians, boxes, lower_slot: int, alpha: float, target_time):
    means = gaussians.means
    quaternions = gaussians.quats
    opacities = temporal_opacity(gaussians.opacities, gaussians.sigma, gaussians.times, target_time)
    if boxes is None or boxes.num_instances == 0:
        return means, quaternions, opacities

    dynamic = gaussians.instance_id >= 0
    if not bool(dynamic.any()):
        return means, quaternions, opacities

    upper_slot = min(lower_slot + 1, boxes.centers.shape[1] - 1)
    instances = gaussians.instance_id[dynamic]
    center_first = boxes.centers[instances, lower_slot]
    center_second = boxes.centers[instances, upper_slot]
    rotation_first = boxes.rots[instances, lower_slot]
    rotation_second = boxes.rots[instances, upper_slot]
    center = torch.lerp(center_first, center_second, alpha)
    rotation = quaternion_to_matrix(quaternion_slerp(
        matrix_to_quaternion(rotation_first), matrix_to_quaternion(rotation_second), alpha
    ))
    canonical_center, canonical_rotation = boxes.canonical()

    means = means.clone()
    quaternions = quaternions.clone()
    opacities = opacities.clone()
    means[dynamic] = decanonicalise(
        gaussians.means[dynamic],
        center,
        rotation,
        canonical_center[instances],
        canonical_rotation[instances],
    )
    relative_rotation = torch.einsum(
        "nji,njk->nik", rotation, canonical_rotation[instances]
    )
    quaternions[dynamic] = quaternion_multiply(
        matrix_to_quaternion(relative_rotation), gaussians.quats[dynamic]
    )
    valid_first = boxes.valid[instances, lower_slot]
    valid_second = boxes.valid[instances, upper_slot]
    valid = valid_first if alpha == 0.0 else valid_first & valid_second
    dynamic_indices = torch.nonzero(dynamic, as_tuple=False).flatten()
    opacities[dynamic_indices[~valid]] = 0.0
    return means, quaternions, opacities


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--offsets", type=float, nargs="+", default=(1.0, 2.0, 3.0))
    parser.add_argument("--source-fps", type=int, default=10)
    parser.add_argument("--output-fps", type=int, default=60)
    parser.add_argument("--num-source-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--isotropic-max-scale", type=float, default=None)
    parser.add_argument("--context-window-length", type=int, default=4)
    args = parser.parse_args()

    if args.output_fps % args.source_fps:
        raise ValueError("output-fps must be an integer multiple of source-fps")
    samples_per_interval = args.output_fps // args.source_fps
    if samples_per_interval < 1:
        raise ValueError("output-fps must be >= source-fps")
    if any(offset < 0 for offset in args.offsets):
        raise ValueError("Offsets must be non-negative")
    if args.isotropic_max_scale is not None and args.isotropic_max_scale <= 0:
        raise ValueError("isotropic-max-scale must be positive")
    if args.context_window_length < 4 or args.context_window_length % 4:
        raise ValueError("context-window-length must be a positive multiple of four")
    context_stride = args.context_window_length // 4
    context_offsets = [index * context_stride for index in range(4)]

    config = yaml.safe_load(Path(args.config).read_text())
    if config["data"].get("num_context") != 4 or config["data"].get("frame_stride") != 1:
        raise ValueError("Dense-context rendering requires num_context=4 and frame_stride=1")
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = {}
    for offset in args.offsets:
        directory = output_dir / "frames" / f"left_{offset:g}m"
        directory.mkdir(parents=True, exist_ok=True)
        frame_dirs[offset] = directory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    dataset = build_dataset(config, args.split)
    matches = [metadata for metadata in dataset.scenes if metadata["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    source_frames = metadata["frames"]
    if args.num_source_frames is not None:
        source_frames = source_frames[: args.num_source_frames]
    if len(source_frames) < args.context_window_length:
        raise ValueError(f"At least {args.context_window_length} source frames are required")
    expected_output_frames = (len(source_frames) - 1) * samples_per_interval + 1

    world_from = np.linalg.inv(dataset._ego_pose(metadata, source_frames[0]))
    model = build_model(config, device, args.ckpt)
    height, width = dataset.resolution
    output_index = 0
    context_records = []
    frame_records = []
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    max_context_start = len(source_frames) - args.context_window_length
    for context_start in range(max_context_start + 1):
        full_window = source_frames[context_start:context_start + args.context_window_length]
        context_window = [full_window[offset] for offset in context_offsets]
        owned_interval_indices = [context_start]
        if context_start == max_context_start:
            owned_interval_indices = list(range(context_start, len(source_frames) - 1))
        slot_of = {frame: slot for slot, frame in enumerate(full_window)}
        normalized_times = dataset._normalised_times(full_window)
        context = dataset._collect(
            metadata,
            context_window,
            slot_of,
            normalized_times,
            dataset.resolution,
            world_from,
            lidar=dataset.load_lidar,
        )
        boxes_cpu = dataset._build_boxes(metadata, full_window, world_from)
        boxes = boxes_to(boxes_cpu, device)
        scene = SceneInput(
            images=context["images"].to(device),
            intrinsics=context["intrinsics"].to(device),
            cam_to_world=context["cam_to_world"].to(device),
            times=context["times"].to(device),
            frame_index=context["frame_index"].to(device),
            boxes=boxes,
            sparse_depth=context["sparse_depth"].to(device),
            sparse_mask=context["sparse_mask"].to(device),
        )
        generator = torch.Generator(device=device).manual_seed(args.seed + context_start)
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        depth = model.predict_depth(scene)
        sampled = model.build_queries(scene, depth, generator=generator)
        gaussians = model.decode(scene, depth, sampled)
        if args.isotropic_max_scale is not None:
            radius = gaussians.scales.float().square().mean(dim=-1, keepdim=True).sqrt()
            gaussians.scales = radius.clamp(max=args.isotropic_max_scale).expand_as(
                gaussians.scales
            ).to(gaussians.scales.dtype).contiguous()
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        context_output_count = 0

        for interval_index in owned_interval_indices:
            lower_frame = source_frames[interval_index]
            upper_frame = source_frames[interval_index + 1]
            lower_slot = interval_index - context_start
            lower_target = dataset._collect(
                metadata, [lower_frame], slot_of, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            upper_target = dataset._collect(
                metadata, [upper_frame], slot_of, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            intrinsic = lower_target["intrinsics"][0].to(device)
            camera_first = lower_target["cam_to_world"][0].to(device)
            camera_second = upper_target["cam_to_world"][0].to(device)
            ego_first = torch.from_numpy(world_from @ dataset._ego_pose(metadata, lower_frame)).float().to(device)
            ego_second = torch.from_numpy(world_from @ dataset._ego_pose(metadata, upper_frame)).float().to(device)

            for subframe in range(samples_per_interval):
                alpha = subframe / samples_per_interval
                target_time = (lower_slot + alpha) / (args.context_window_length - 1)
                camera_pose = interpolate_pose(camera_first, camera_second, alpha)
                ego_pose = interpolate_pose(ego_first, ego_second, alpha)
                left = ego_pose[:3, 1] / ego_pose[:3, 1].norm().clamp_min(1e-8)
                means, quaternions, opacities = pose_gaussians_continuous(
                    model, gaussians, boxes, lower_slot, alpha, target_time
                )
                shifted_poses = []
                for lateral_offset in args.offsets:
                    shifted = camera_pose.clone()
                    shifted[:3, 3] += left * lateral_offset
                    shifted_poses.append(shifted)
                poses = torch.stack(shifted_poses)
                features, alpha_map, _ = render_features(
                    means=means,
                    quats=quaternions,
                    scales=gaussians.scales,
                    opacities=opacities,
                    colors=gaussians.colors,
                    intrinsics=intrinsic.unsqueeze(0).expand(len(args.offsets), -1, -1),
                    world_to_cam=torch.linalg.inv(poses),
                    height=height,
                    width=width,
                )
                rgb = model.refiner(features, alpha_map)
                for offset_index, lateral_offset in enumerate(args.offsets):
                    image = label(
                        to_uint8(rgb[offset_index]),
                        f"PointForward | source {lower_frame:03d}->{upper_frame:03d} | alpha {alpha:.3f} | camera 0 | left {lateral_offset:g}m",
                    )
                    Image.fromarray(image).save(
                        frame_dirs[lateral_offset] / f"frame_{output_index:04d}.jpg", quality=92
                    )
                frame_records.append({
                    "output_index": output_index,
                    "source_interval": [int(lower_frame), int(upper_frame)],
                    "alpha": alpha,
                    "context_frames": [int(frame) for frame in context_window],
                    "target_time": target_time,
                })
                output_index += 1
                context_output_count += 1
                del means, quaternions, opacities, poses, features, alpha_map, rgb

        context_records.append({
            "context_start": context_start,
            "full_window_frames": [int(frame) for frame in full_window],
            "context_frames": [int(frame) for frame in context_window],
            "owned_intervals": [
                [int(source_frames[index]), int(source_frames[index + 1])]
                for index in owned_interval_indices
            ],
            "output_frame_count": context_output_count,
            "gaussian_count": int(gaussians.count),
            "decode_seconds": decode_seconds,
        })
        if context_start % 10 == 0 or context_start == max_context_start:
            print(json.dumps(context_records[-1]), flush=True)
        del context, scene, boxes, depth, sampled, gaussians
        torch.cuda.empty_cache()

    final_context_start = max_context_start
    final_full_window = source_frames[-args.context_window_length:]
    final_window = [final_full_window[offset] for offset in context_offsets]
    final_slot_of = {frame: slot for slot, frame in enumerate(final_full_window)}
    final_times = dataset._normalised_times(final_full_window)
    final_context = dataset._collect(
        metadata, final_window, final_slot_of, final_times,
        dataset.resolution, world_from, lidar=dataset.load_lidar,
    )
    final_boxes_cpu = dataset._build_boxes(metadata, final_full_window, world_from)
    final_boxes = boxes_to(final_boxes_cpu, device)
    final_scene = SceneInput(
        images=final_context["images"].to(device),
        intrinsics=final_context["intrinsics"].to(device),
        cam_to_world=final_context["cam_to_world"].to(device),
        times=final_context["times"].to(device),
        frame_index=final_context["frame_index"].to(device),
        boxes=final_boxes,
        sparse_depth=final_context["sparse_depth"].to(device),
        sparse_mask=final_context["sparse_mask"].to(device),
    )
    final_generator = torch.Generator(device=device).manual_seed(args.seed + final_context_start)
    final_depth = model.predict_depth(final_scene)
    final_sampled = model.build_queries(final_scene, final_depth, generator=final_generator)
    final_gaussians = model.decode(final_scene, final_depth, final_sampled)
    if args.isotropic_max_scale is not None:
        final_radius = final_gaussians.scales.float().square().mean(dim=-1, keepdim=True).sqrt()
        final_gaussians.scales = final_radius.clamp(max=args.isotropic_max_scale).expand_as(
            final_gaussians.scales
        ).to(final_gaussians.scales.dtype).contiguous()
    final_target = dataset._collect(
        metadata, [source_frames[-1]], final_slot_of, final_times,
        dataset.resolution, world_from, lidar=False,
    )
    final_camera = final_target["cam_to_world"][0].to(device)
    final_ego = torch.from_numpy(world_from @ dataset._ego_pose(metadata, source_frames[-1])).float().to(device)
    final_left = final_ego[:3, 1] / final_ego[:3, 1].norm().clamp_min(1e-8)
    means, quaternions, opacities = pose_gaussians_continuous(
        model, final_gaussians, final_boxes, args.context_window_length - 1, 0.0, 1.0
    )
    final_poses = []
    for lateral_offset in args.offsets:
        shifted = final_camera.clone()
        shifted[:3, 3] += final_left * lateral_offset
        final_poses.append(shifted)
    final_poses = torch.stack(final_poses)
    final_features, final_alpha, _ = render_features(
        means=means,
        quats=quaternions,
        scales=final_gaussians.scales,
        opacities=opacities,
        colors=final_gaussians.colors,
        intrinsics=final_target["intrinsics"][0].to(device).unsqueeze(0).expand(len(args.offsets), -1, -1),
        world_to_cam=torch.linalg.inv(final_poses),
        height=height,
        width=width,
    )
    final_rgb = model.refiner(final_features, final_alpha)
    for offset_index, lateral_offset in enumerate(args.offsets):
        image = label(
            to_uint8(final_rgb[offset_index]),
            f"PointForward | source {source_frames[-1]:03d} | camera 0 | left {lateral_offset:g}m",
        )
        Image.fromarray(image).save(
            frame_dirs[lateral_offset] / f"frame_{output_index:04d}.jpg", quality=92
        )
    frame_records.append({
        "output_index": output_index,
        "source_interval": [int(source_frames[-1]), int(source_frames[-1])],
        "alpha": 0.0,
        "context_frames": [int(frame) for frame in final_window],
        "target_time": 1.0,
    })
    output_index += 1
    if output_index != expected_output_frames:
        raise RuntimeError(f"Expected {expected_output_frames} output frames, generated {output_index}")

    videos = {}
    for lateral_offset in args.offsets:
        path = output_dir / f"pointforward_dense_context_left_{lateral_offset:g}m_camera0_60fps.mp4"
        encode_video(frame_dirs[lateral_offset] / "frame_%04d.jpg", path, args.output_fps)
        videos[str(lateral_offset)] = str(path)

    manifest = {
        "status": "completed",
        "scene": args.scene,
        "source_checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(Path(args.config).resolve()),
        "source_frame_count": len(source_frames),
        "source_frame_range": [int(source_frames[0]), int(source_frames[-1])],
        "source_fps": args.source_fps,
        "output_fps": args.output_fps,
        "samples_per_source_interval": samples_per_interval,
        "output_frame_count": output_index,
        "expected_output_formula": "(source_frame_count - 1) * (output_fps / source_fps) + 1",
        "reconstruction_cameras": list(dataset.cameras),
        "output_camera": 0,
        "context_policy": "four uniformly spaced source frames per decode, sliding by one source frame",
        "context_window_length": args.context_window_length,
        "context_offsets": context_offsets,
        "unique_context_count": len(context_records),
        "lateral_offsets_m": list(args.offsets),
        "gaussian_scale_postprocess": {
            "mode": "none" if args.isotropic_max_scale is None else "isotropic-trace",
            "isotropic_max_scale_m": args.isotropic_max_scale,
        },
        "offset_definition": "interpolated ego local +Y axis; camera orientation unchanged relative to interpolated source camera pose",
        "camera_interpolation": "translation lerp plus shortest-path quaternion SLERP",
        "dynamic_box_interpolation": "center lerp plus shortest-path quaternion SLERP; intermediate actor valid only when both endpoint boxes exist",
        "videos": videos,
        "contexts": context_records,
        "frame_records": frame_records,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "limitations": [
            "The fixed checkpoint was trained with wider temporal context spacing; consecutive-frame contexts are out of its training distribution.",
            "Each source interval is rendered from its assigned four-frame reconstruction, so context changes can remain visible at original-frame boundaries.",
            "The 60 FPS frames are model renders at interpolated poses/times, not optical-flow interpolation of output RGB frames.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "output_frame_count": output_index,
        "unique_context_count": len(context_records),
        "videos": videos,
        "wall_seconds": manifest["wall_seconds"],
        "peak_cuda_allocated_gib": manifest["peak_cuda_allocated_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()

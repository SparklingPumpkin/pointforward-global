"""Render 60 FPS trajectories from four-frame contexts with window stride three."""

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
from pointforward.models.renderer import render_features
from scripts.deliver_continuous_scene import encode_video, label, to_uint8
from scripts.eval import build_model
from scripts.export_gaussians_ply import boxes_to
from scripts.render_dense_context_60fps import interpolate_pose, pose_gaussians_continuous
from scripts.train import build_dataset


def context_schedule(frame_count: int) -> list[tuple[int, list[int]]]:
    """Return four-frame context starts and the source intervals each owns."""
    if frame_count < 4:
        raise ValueError("At least four source frames are required")
    max_start = frame_count - 4
    starts = list(range(0, max_start + 1, 3))
    if starts[-1] != max_start:
        starts.append(max_start)

    schedule = []
    next_interval = 0
    for start in starts:
        owned = [index for index in range(start, start + 3) if index >= next_interval]
        owned = [index for index in owned if index < frame_count - 1]
        if owned:
            schedule.append((start, owned))
            next_interval = owned[-1] + 1
    if next_interval != frame_count - 1:
        raise RuntimeError(f"Context schedule covers intervals [0,{next_interval}), expected [0,{frame_count - 1})")
    return schedule


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
    args = parser.parse_args()

    if args.output_fps % args.source_fps:
        raise ValueError("output-fps must be an integer multiple of source-fps")
    samples_per_interval = args.output_fps // args.source_fps
    if samples_per_interval < 1:
        raise ValueError("output-fps must be >= source-fps")
    if any(offset <= 0 for offset in args.offsets):
        raise ValueError("Offsets must be positive")

    config = yaml.safe_load(Path(args.config).read_text())
    if config["data"].get("num_context") != 4 or config["data"].get("frame_stride") != 1:
        raise ValueError("This renderer requires num_context=4 and frame_stride=1")
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
    schedule = context_schedule(len(source_frames))
    expected_output_frames = (len(source_frames) - 1) * samples_per_interval + 1

    world_from = np.linalg.inv(dataset._ego_pose(metadata, source_frames[0]))
    model = build_model(config, device, args.ckpt)
    height, width = dataset.resolution
    output_index = 0
    context_records = []
    frame_records = []
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for context_index, (context_start, owned_interval_indices) in enumerate(schedule):
        context_window = source_frames[context_start:context_start + 4]
        slot_of = {frame: slot for slot, frame in enumerate(context_window)}
        normalized_times = dataset._normalised_times(context_window)
        context = dataset._collect(
            metadata,
            context_window,
            slot_of,
            normalized_times,
            dataset.resolution,
            world_from,
            lidar=dataset.load_lidar,
        )
        boxes_cpu = dataset._build_boxes(metadata, context_window, world_from)
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
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        render_started = time.perf_counter()
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
                target_time = (lower_slot + alpha) / 3.0
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

        torch.cuda.synchronize()
        context_record = {
            "context_index": context_index,
            "context_start": context_start,
            "context_frames": [int(frame) for frame in context_window],
            "owned_intervals": [
                [int(source_frames[index]), int(source_frames[index + 1])]
                for index in owned_interval_indices
            ],
            "output_frame_count": context_output_count,
            "gaussian_count": int(gaussians.count),
            "decode_seconds": decode_seconds,
            "render_seconds": time.perf_counter() - render_started,
        }
        context_records.append(context_record)
        print(json.dumps(context_record), flush=True)
        del context, scene, boxes, depth, sampled, gaussians
        torch.cuda.empty_cache()

    final_context_start = len(source_frames) - 4
    final_window = source_frames[-4:]
    final_slot_of = {frame: slot for slot, frame in enumerate(final_window)}
    final_times = dataset._normalised_times(final_window)
    final_context = dataset._collect(
        metadata, final_window, final_slot_of, final_times,
        dataset.resolution, world_from, lidar=dataset.load_lidar,
    )
    final_boxes_cpu = dataset._build_boxes(metadata, final_window, world_from)
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
    final_target = dataset._collect(
        metadata, [source_frames[-1]], final_slot_of, final_times,
        dataset.resolution, world_from, lidar=False,
    )
    final_camera = final_target["cam_to_world"][0].to(device)
    final_ego = torch.from_numpy(world_from @ dataset._ego_pose(metadata, source_frames[-1])).float().to(device)
    final_left = final_ego[:3, 1] / final_ego[:3, 1].norm().clamp_min(1e-8)
    means, quaternions, opacities = pose_gaussians_continuous(
        model, final_gaussians, final_boxes, 3, 0.0, 1.0
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
        path = output_dir / f"pointforward_context4_stride3_left_{lateral_offset:g}m_camera0_60fps.mp4"
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
        "context_policy": "four consecutive source frames per decode, context-window stride three, one shared boundary anchor",
        "context_window_size": 4,
        "context_window_stride": 3,
        "unique_context_count": len(context_records),
        "lateral_offsets_m": list(args.offsets),
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
            "Adjacent reconstructions share one source-frame anchor but remain independently decoded Gaussian sets.",
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
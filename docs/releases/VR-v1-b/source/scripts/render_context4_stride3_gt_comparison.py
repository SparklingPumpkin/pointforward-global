"""Render unshifted camera 0 at 60 FPS and pair it with held source GT RGB."""

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
from scripts.render_context4_stride3_60fps import context_schedule
from scripts.render_dense_context_60fps import interpolate_pose, pose_gaussians_continuous
from scripts.train import build_dataset


def paired_frame(gt: np.ndarray, prediction: np.ndarray, frame: int, alpha: float) -> np.ndarray:
    left = label(gt, f"GT RGB | source frame {frame:03d} | held at 10 FPS")
    right = label(prediction, f"PointForward | camera 0 | offset 0m | alpha {alpha:.3f}")
    return np.concatenate((left, right), axis=1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--source-fps", type=int, default=10)
    parser.add_argument("--output-fps", type=int, default=60)
    parser.add_argument("--num-source-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output_fps % args.source_fps:
        raise ValueError("output-fps must be an integer multiple of source-fps")
    samples_per_interval = args.output_fps // args.source_fps
    config = yaml.safe_load(Path(args.config).read_text())
    if config["data"].get("num_context") != 4 or config["data"].get("frame_stride") != 1:
        raise ValueError("This renderer requires four consecutive context frames")

    output_dir = Path(args.out).resolve()
    frame_dir = output_dir / "frames" / "gt_vs_prediction"
    frame_dir.mkdir(parents=True, exist_ok=True)
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
            metadata, context_window, slot_of, normalized_times,
            dataset.resolution, world_from, lidar=dataset.load_lidar,
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
        context_output_count = 0
        render_started = time.perf_counter()

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
            gt = to_uint8(lower_target["images"][0])

            for subframe in range(samples_per_interval):
                alpha = subframe / samples_per_interval
                target_time = (lower_slot + alpha) / 3.0
                camera_pose = interpolate_pose(camera_first, camera_second, alpha)
                means, quaternions, opacities = pose_gaussians_continuous(
                    model, gaussians, boxes, lower_slot, alpha, target_time
                )
                features, alpha_map, _ = render_features(
                    means=means,
                    quats=quaternions,
                    scales=gaussians.scales,
                    opacities=opacities,
                    colors=gaussians.colors,
                    intrinsics=intrinsic.unsqueeze(0),
                    world_to_cam=torch.linalg.inv(camera_pose).unsqueeze(0),
                    height=height,
                    width=width,
                )
                prediction = model.refiner(features, alpha_map)[0]
                Image.fromarray(paired_frame(gt, to_uint8(prediction), lower_frame, alpha)).save(
                    frame_dir / f"frame_{output_index:04d}.jpg", quality=92
                )
                frame_records.append({
                    "output_index": output_index,
                    "source_interval": [int(lower_frame), int(upper_frame)],
                    "gt_source_frame": int(lower_frame),
                    "gt_policy": "hold lower 10 FPS source frame for six 60 FPS samples",
                    "alpha": alpha,
                    "context_frames": [int(frame) for frame in context_window],
                    "target_time": target_time,
                })
                output_index += 1
                context_output_count += 1
                del means, quaternions, opacities, features, alpha_map, prediction

        torch.cuda.synchronize()
        record = {
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
        context_records.append(record)
        print(json.dumps(record), flush=True)
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
    final_boxes = boxes_to(dataset._build_boxes(metadata, final_window, world_from), device)
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
    means, quaternions, opacities = pose_gaussians_continuous(
        model, final_gaussians, final_boxes, 3, 0.0, 1.0
    )
    features, alpha_map, _ = render_features(
        means=means,
        quats=quaternions,
        scales=final_gaussians.scales,
        opacities=opacities,
        colors=final_gaussians.colors,
        intrinsics=final_target["intrinsics"][0].to(device).unsqueeze(0),
        world_to_cam=torch.linalg.inv(final_camera).unsqueeze(0),
        height=height,
        width=width,
    )
    prediction = model.refiner(features, alpha_map)[0]
    gt = to_uint8(final_target["images"][0])
    Image.fromarray(paired_frame(gt, to_uint8(prediction), source_frames[-1], 0.0)).save(
        frame_dir / f"frame_{output_index:04d}.jpg", quality=92
    )
    frame_records.append({
        "output_index": output_index,
        "source_interval": [int(source_frames[-1]), int(source_frames[-1])],
        "gt_source_frame": int(source_frames[-1]),
        "gt_policy": "exact final 10 FPS source frame",
        "alpha": 0.0,
        "context_frames": [int(frame) for frame in final_window],
        "target_time": 1.0,
    })
    output_index += 1
    if output_index != expected_output_frames:
        raise RuntimeError(f"Expected {expected_output_frames} frames, generated {output_index}")

    video_path = output_dir / "pointforward_context4_stride3_camera0_gt_vs_pred_60fps.mp4"
    encode_video(frame_dir / "frame_%04d.jpg", video_path, args.output_fps)
    manifest = {
        "status": "completed",
        "scene": args.scene,
        "source_checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(Path(args.config).resolve()),
        "source_frame_count": len(source_frames),
        "source_frame_range": [int(source_frames[0]), int(source_frames[-1])],
        "source_fps": args.source_fps,
        "output_fps": args.output_fps,
        "output_frame_count": output_index,
        "reconstruction_cameras": list(dataset.cameras),
        "output_camera": 0,
        "lateral_offset_m": 0.0,
        "context_policy": "four consecutive source frames per decode, context-window stride three, one shared boundary anchor",
        "context_window_stride": 3,
        "prediction_interpolation": "camera translation lerp, camera rotation SLERP, dynamic box center lerp and rotation SLERP",
        "gt_policy": "left panel holds each real 10 FPS camera-0 RGB frame for the six corresponding 60 FPS samples; no synthetic GT interpolation",
        "layout": "GT RGB left | PointForward prediction right",
        "video": str(video_path),
        "contexts": context_records,
        "frame_records": frame_records,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "limitations": [
            "GT is available only at 10 FPS and is held between source timestamps; it is not a 60 FPS ground-truth capture.",
            "At alpha greater than zero, the right panel is an interpolated-time prediction while the left panel remains the lower source RGB frame.",
            "Independently decoded stride-three contexts can still produce boundary jitter.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "completed",
        "output_frame_count": output_index,
        "video": str(video_path),
        "wall_seconds": manifest["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
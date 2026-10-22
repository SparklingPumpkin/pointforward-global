"""Render parallel camera trajectories shifted along each frame's ego-left axis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import SceneInput, TargetViews
from scripts.deliver_continuous_scene import encode_video, horizontal_strip, label, to_uint8
from scripts.eval import build_model
from scripts.export_gaussians_ply import boxes_to
from scripts.train import build_dataset


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--offsets", type=float, nargs="+", default=(1.0, 2.0, 3.0))
    parser.add_argument(
        "--output-cameras",
        type=int,
        nargs="+",
        default=None,
        help="camera IDs to render; reconstruction still uses every camera in the config",
    )
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.offsets or any(offset <= 0 for offset in args.offsets):
        raise ValueError("All left offsets must be positive")
    if len(set(args.offsets)) != len(args.offsets):
        raise ValueError("Left offsets must be unique")

    config = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = {}
    for offset in args.offsets:
        frame_dir = output_dir / "frames" / f"left_{offset:g}m"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_dirs[offset] = frame_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    dataset = build_dataset(config, args.split)
    output_cameras = list(dataset.cameras) if args.output_cameras is None else args.output_cameras
    if len(set(output_cameras)) != len(output_cameras):
        raise ValueError("Output camera IDs must be unique")
    missing_cameras = [camera for camera in output_cameras if camera not in dataset.cameras]
    if missing_cameras:
        raise ValueError(f"Output cameras are absent from the reconstruction config: {missing_cameras}")
    output_camera_indices = [dataset.cameras.index(camera) for camera in output_cameras]
    matches = [metadata for metadata in dataset.scenes if metadata["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    frames = metadata["frames"]
    if args.num_frames is not None:
        frames = frames[:args.num_frames]
    if not frames:
        raise ValueError("No frames selected")

    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    model = build_model(config, device, args.ckpt)
    height, width = dataset.resolution
    camera_count = len(output_cameras)
    windows = []
    translation_checks = []
    output_frame = 0
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for window_index, offset_start in enumerate(range(0, len(frames), dataset.clip_length)):
        window = frames[offset_start:offset_start + dataset.clip_length]
        slot_of = {frame: slot for slot, frame in enumerate(window)}
        normalized_times = dataset._normalised_times(window)
        context_frames = sorted({
            window[min(index * dataset.frame_stride, len(window) - 1)]
            for index in range(dataset.num_context)
        })
        context = dataset._collect(
            metadata, context_frames, slot_of, normalized_times,
            dataset.resolution, world_from, lidar=dataset.load_lidar,
        )
        boxes_cpu = dataset._build_boxes(metadata, window, world_from)
        boxes = boxes_to(boxes_cpu, device)
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
        generator = torch.Generator(device=device).manual_seed(args.seed + window_index)
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        depth = model.predict_depth(scene)
        sampled = model.build_queries(scene, depth, generator=generator)
        gaussians = model.decode(scene, depth, sampled)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started

        render_started = time.perf_counter()
        alpha_means = {str(offset): [] for offset in args.offsets}
        for frame in window:
            target = dataset._collect(
                metadata, [frame], slot_of, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            camera_selection = torch.tensor(output_camera_indices, dtype=torch.long)
            base_c2w = target["cam_to_world"][camera_selection]
            base_intrinsics = target["intrinsics"][camera_selection]
            base_times = target["times"][camera_selection]
            base_frame_index = target["frame_index"][camera_selection]
            ego_to_local = world_from @ dataset._ego_pose(metadata, frame)
            left = torch.from_numpy(ego_to_local[:3, 1]).float()
            left = left / left.norm().clamp_min(1e-8)

            shifted_poses = []
            for lateral_offset in args.offsets:
                shifted = base_c2w.clone()
                shifted[:, :3, 3] += left.unsqueeze(0) * lateral_offset
                shifted_poses.append(shifted)
                actual_shift = torch.linalg.norm(shifted[:, :3, 3] - base_c2w[:, :3, 3], dim=-1)
                translation_checks.append({
                    "frame": int(frame),
                    "offset_m": lateral_offset,
                    "min_actual_shift_m": float(actual_shift.min()),
                    "max_actual_shift_m": float(actual_shift.max()),
                })

            target_views = TargetViews(
                intrinsics=base_intrinsics.repeat(len(args.offsets), 1, 1).to(device),
                cam_to_world=torch.cat(shifted_poses, dim=0).to(device),
                times=base_times.repeat(len(args.offsets)).to(device),
                frame_index=base_frame_index.repeat(len(args.offsets)).to(device),
            )
            rendered = model.render(
                gaussians, boxes, target_views, height=height, width=width, render_depth=False
            )
            rgb = rendered["rgb"]
            alpha = rendered["alpha"]
            for offset_index, lateral_offset in enumerate(args.offsets):
                begin = offset_index * camera_count
                end = begin + camera_count
                panels = [
                    label(
                        to_uint8(rgb[begin + camera_index]),
                        f"PointForward | frame {frame:03d} | camera {camera} | left {lateral_offset:g}m",
                    )
                    for camera_index, camera in enumerate(output_cameras)
                ]
                Image.fromarray(horizontal_strip(panels)).save(
                    frame_dirs[lateral_offset] / f"frame_{output_frame:04d}.jpg", quality=92
                )
                alpha_means[str(lateral_offset)].append(float(alpha[begin:end].mean().cpu()))
            output_frame += 1
            del target, target_views, shifted_poses, rendered, rgb, alpha

        torch.cuda.synchronize()
        window_record = {
            "window_index": window_index,
            "frames": [int(frame) for frame in window],
            "context_frames": [int(frame) for frame in context_frames],
            "gaussian_count": int(gaussians.count),
            "decode_seconds": decode_seconds,
            "render_seconds": time.perf_counter() - render_started,
            "mean_alpha_by_offset": {
                key: float(np.mean(values)) for key, values in alpha_means.items()
            },
        }
        windows.append(window_record)
        print(json.dumps(window_record), flush=True)
        del context, scene, boxes, depth, sampled, gaussians
        torch.cuda.empty_cache()

    videos = {}
    camera_label = "camera" + "-".join(str(camera) for camera in output_cameras)
    for lateral_offset in args.offsets:
        video_path = output_dir / f"pointforward_left_{lateral_offset:g}m_{camera_label}.mp4"
        encode_video(frame_dirs[lateral_offset] / "frame_%04d.jpg", video_path, args.fps)
        videos[str(lateral_offset)] = str(video_path)

    manifest = {
        "status": "completed",
        "scene": args.scene,
        "source_checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(Path(args.config).resolve()),
        "frame_count": len(frames),
        "frame_range": [int(frames[0]), int(frames[-1])],
        "fps": args.fps,
        "reconstruction_cameras": list(dataset.cameras),
        "output_cameras": output_cameras,
        "camera_semantics": {"0": "front", "1": "front_left", "2": "front_right"},
        "lateral_offsets_m": list(args.offsets),
        "offset_definition": "translation along each frame ego pose local +Y axis in the shared recentered world; camera rotations unchanged",
        "videos": videos,
        "windows": windows,
        "translation_checks": translation_checks,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "limitations": [
            "Novel trajectories are outside the training-camera path and have no aligned RGB ground truth.",
            "Each 20-frame window is reconstructed independently, matching training-time temporal normalization.",
            "Larger lateral offsets reveal disocclusions that the three input cameras may not observe.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
"""Build every global dynamic track exactly once from 12 uniformly spaced views.

The static payload is reused byte-for-byte from an existing global-map run.
For each scene-level track ID, four timestamps are selected uniformly from the
track's complete visible interval.  With the three configured cameras this is
the checkpoint's fixed 12-view input contract.  Only that track is exposed to
query assignment, its dynamic Gaussians are decoded once, rebased into the
scene-global canonical frame, and never updated or appended again.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import SceneInput
from pointforward.models.query import query_rgb
from scripts.deliver_continuous_scene import (
    boxes_payload,
    build_global_track_boxes,
    classify_stationary_tracks,
    concatenate_gaussians,
    postprocess_gaussian_scales,
    rebase_dynamic_to_global_tracks,
    sampled_camera_distance,
    save_gaussian_payload,
    select_gaussians,
    select_sampled,
)
from scripts.eval import build_model
from scripts.export_gaussians_ply import boxes_to
from scripts.train import build_dataset


SIGNS = np.asarray([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
], dtype=np.float64)


def uniform_timestamps(frames: list[int], count: int) -> list[int]:
    """Select ``count`` ordered timestamps across a complete visible interval.

    Very short tracks repeat their nearest available timestamp so the pretrained
    fusion network still receives exactly four timestamps x three cameras.
    """
    if not frames:
        raise ValueError("Cannot sample an empty track")
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(np.int64)
    return [int(frames[index]) for index in indices]


def calibrated_intrinsics(dataset, metadata: dict, frame: int) -> dict[int, np.ndarray]:
    """Return intrinsics after the Argoverse center crop and resize."""
    height, width = dataset.resolution
    output = {}
    for camera in dataset.cameras:
        path = Path(metadata["path"]) / "images" / f"{frame:03d}_{camera}.jpg"
        with Image.open(path) as image:
            original_width, original_height = image.size
        intrinsic = metadata["intrinsics"][camera].copy()
        crop_x, crop_y, crop_width, crop_height = dataset._crop_parameters(
            original_width,
            original_height,
            width,
            height,
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        intrinsic[0, 2] -= crop_x
        intrinsic[1, 2] -= crop_y
        intrinsic[0, :] *= width / crop_width
        intrinsic[1, :] *= height / crop_height
        output[int(camera)] = intrinsic
    return output


def track_front_camera_frames(
    dataset,
    metadata: dict,
    track_id: int,
    selected_set: set[int],
    world_from: np.ndarray,
    intrinsics: dict[int, np.ndarray],
) -> list[int]:
    """Frames where the track's projected 3D box overlaps a configured image."""
    height, width = dataset.resolution
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    visible = []
    for frame, (raw_pose, size) in metadata["instances"][track_id].items():
        if frame not in selected_set:
            continue
        object_to_world = world_from @ raw_pose
        local = signs * (size * 0.5)
        corners = local @ object_to_world[:3, :3].T + object_to_world[:3, 3]
        frame_visible = False
        for camera in dataset.cameras:
            world_to_camera = np.linalg.inv(
                dataset._cam_to_world(metadata, frame, camera, world_from)
            )
            camera_points = (
                corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            )
            front = camera_points[:, 2] > 1e-3
            if not front.any():
                continue
            points = camera_points[front]
            projected = (points / points[:, 2:3]) @ intrinsics[int(camera)].T
            lo = projected[:, :2].min(axis=0)
            hi = projected[:, :2].max(axis=0)
            intersects = (
                hi[0] >= 0 and lo[0] < width and hi[1] >= 0 and lo[1] < height
            )
            if intersects:
                frame_visible = True
                break
        if frame_visible:
            visible.append(int(frame))
    return sorted(visible)


def track_view_candidates(
    dataset,
    metadata: dict,
    track_id: int,
    selected_set: set[int],
    world_from: np.ndarray,
    intrinsics: dict[int, np.ndarray],
) -> list[dict]:
    """Enumerate visible frame-camera pairs with area and local view angle."""
    height, width = dataset.resolution
    candidates = []
    for frame, (raw_pose, size) in metadata["instances"][track_id].items():
        if frame not in selected_set:
            continue
        object_to_world = world_from @ raw_pose
        center = object_to_world[:3, 3]
        object_rotation = object_to_world[:3, :3]
        corners = SIGNS * (size * 0.5)
        corners = corners @ object_rotation.T + center
        for camera in dataset.cameras:
            camera_to_world = dataset._cam_to_world(
                metadata, frame, camera, world_from
            )
            world_to_camera = np.linalg.inv(camera_to_world)
            camera_points = (
                corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            )
            front = camera_points[:, 2] > 1e-3
            if not front.any():
                continue
            points = camera_points[front]
            uv = (points / points[:, 2:3]) @ intrinsics[int(camera)].T
            lo = uv[:, :2].min(axis=0)
            hi = uv[:, :2].max(axis=0)
            clipped_lo = np.maximum(lo, (0.0, 0.0))
            clipped_hi = np.minimum(hi, (width - 1.0, height - 1.0))
            extent = np.maximum(clipped_hi - clipped_lo, 0.0)
            area = float(extent[0] * extent[1] / (width * height))
            if area <= 0:
                continue
            camera_vector_world = camera_to_world[:3, 3] - center
            camera_vector_local = camera_vector_world @ object_rotation
            angle = float(np.arctan2(
                camera_vector_local[1], camera_vector_local[0]
            ))
            candidates.append({
                "frame": int(frame),
                "camera": int(camera),
                "bbox_area_fraction": area,
                "view_angle_rad": angle,
                "bbox_xyxy": [
                    float(clipped_lo[0]), float(clipped_lo[1]),
                    float(clipped_hi[0]), float(clipped_hi[1]),
                ],
            })
    return candidates


def annotate_fine_mask_visibility(
    dataset, metadata: dict, candidates_by_track: dict[int, list[dict]]
) -> None:
    """Attach visible moving-vehicle pixels to every projected track box."""
    cache: dict[tuple[int, int], np.ndarray] = {}
    root = Path(metadata["path"])
    for candidates in candidates_by_track.values():
        for item in candidates:
            key = (item["frame"], item["camera"])
            if key not in cache:
                relative = (
                    f"fine_dynamic_masks/vehicle/{key[0]:03d}_{key[1]}.png"
                )
                path = root / relative
                if path.exists():
                    mask, _ = dataset._load_calibrated_image(
                        metadata, relative, dataset.resolution, key[1], gray=True
                    )
                    cache[key] = mask.squeeze(0).numpy() > 0.5
                else:
                    cache[key] = np.zeros(dataset.resolution, dtype=bool)
            x0, y0, x1, y1 = item["bbox_xyxy"]
            x0, y0 = int(np.floor(x0)), int(np.floor(y0))
            x1, y1 = int(np.ceil(x1)) + 1, int(np.ceil(y1)) + 1
            crop = cache[key][y0:y1, x0:x1]
            pixels = int(crop.sum())
            item["fine_dynamic_pixels"] = pixels
            item["fine_dynamic_fill"] = pixels / max(int(crop.size), 1)
            item["visible_dynamic_area_fraction"] = (
                pixels / float(dataset.resolution[0] * dataset.resolution[1])
            )


def circular_distance(first: float, second: float) -> float:
    delta = abs(first - second) % (2.0 * np.pi)
    return float(min(delta, 2.0 * np.pi - delta))


def select_diverse_large_views(
    candidates: list[dict], count: int, angle_weight: float
) -> list[dict]:
    """Greedily maximise local azimuth coverage while retaining large boxes."""
    if not candidates:
        return []
    maximum_area = max(item["bbox_area_fraction"] for item in candidates)
    pool = sorted(candidates, key=lambda item: item["bbox_area_fraction"], reverse=True)
    selected = [pool.pop(0)]
    while pool and len(selected) < count:
        def score(item):
            diversity = min(
                circular_distance(item["view_angle_rad"], chosen["view_angle_rad"])
                for chosen in selected
            ) / np.pi
            area_quality = np.sqrt(item["bbox_area_fraction"] / maximum_area)
            return angle_weight * diversity + (1.0 - angle_weight) * area_quality

        best = max(range(len(pool)), key=lambda index: score(pool[index]))
        selected.append(pool.pop(best))
    while len(selected) < count:
        selected.append(dict(selected[len(selected) % len(selected)]))
    return selected


def select_diverse_large_view_pass(
    candidates: list[dict], count: int, angle_weight: float, pass_index: int
) -> list[dict]:
    """Select one complementary view group without changing the 12-view contract."""
    if pass_index < 0:
        raise ValueError("pass_index must be non-negative")
    remaining = list(candidates)
    selected = []
    for current in range(pass_index + 1):
        pool = remaining if remaining else list(candidates)
        selected = select_diverse_large_views(pool, count, angle_weight)
        if current == pass_index:
            break
        chosen = {(item["frame"], item["camera"]) for item in selected}
        remaining = [
            item for item in remaining
            if (item["frame"], item["camera"]) not in chosen
        ]
    return selected


def select_diverse_visible_views(
    candidates: list[dict], count: int, angle_weight: float
) -> list[dict]:
    """Prefer wide azimuth coverage and fine-mask-confirmed visible pixels."""
    visible = [item for item in candidates if item.get("fine_dynamic_pixels", 0) > 0]
    if len(visible) < count:
        return select_diverse_large_views(candidates, count, angle_weight)
    maximum = max(item["visible_dynamic_area_fraction"] for item in visible)
    pool = sorted(
        visible, key=lambda item: item["visible_dynamic_area_fraction"], reverse=True
    )
    selected = [pool.pop(0)]
    while pool and len(selected) < count:
        def score(item):
            diversity = min(
                circular_distance(item["view_angle_rad"], chosen["view_angle_rad"])
                for chosen in selected
            ) / np.pi
            visibility = np.sqrt(item["visible_dynamic_area_fraction"] / maximum)
            return angle_weight * diversity + (1.0 - angle_weight) * visibility

        best = max(range(len(pool)), key=lambda index: score(pool[index]))
        selected.append(pool.pop(best))
    return selected


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--static-payload", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestamps-per-track", type=int, default=4)
    parser.add_argument(
        "--view-selection",
        choices=("uniform-timestamps", "diverse-large", "diverse-visible"),
        default="uniform-timestamps",
    )
    parser.add_argument("--views-per-track", type=int, default=12)
    parser.add_argument(
        "--dynamic-queries-per-track",
        type=int,
        default=None,
        help="Override the checkpoint's per-instance query budget at inference.",
    )
    parser.add_argument("--view-angle-weight", type=float, default=0.65)
    parser.add_argument(
        "--view-pass-index",
        type=int,
        default=0,
        help="Zero-based complementary diverse-large group; 0 preserves DG-v1-f.",
    )
    parser.add_argument(
        "--gaussian-scale-postprocess",
        choices=("none", "isotropic-trace", "isotropic-volume"),
        default="isotropic-trace",
    )
    parser.add_argument("--isotropic-max-scale", type=float, default=0.05)
    parser.add_argument("--stationary-track-max-span", type=float, default=None)
    parser.add_argument("--stationary-track-max-median-step", type=float, default=0.1)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    static_path = Path(args.static_payload).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not static_path.exists():
        raise FileNotFoundError(static_path)

    config = yaml.safe_load(config_path.read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    device = torch.device("cuda")
    dataset = build_dataset(config, args.split)
    matches = [meta for meta in dataset.scenes if meta["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    evaluation_metadata = metadata
    selected_frames = [int(frame) for frame in metadata["frames"]]
    if args.views_per_track != 12:
        parser.error("The pretrained fusion network requires exactly 12 views")
    if not 0.0 <= args.view_angle_weight <= 1.0:
        parser.error("--view-angle-weight must be in [0, 1]")
    if args.view_pass_index < 0:
        parser.error("--view-pass-index must be non-negative")
    if (
        args.view_selection == "uniform-timestamps"
        and args.timestamps_per_track * len(dataset.cameras) != args.views_per_track
    ):
        parser.error(
            "The checkpoint requires 12 views; timestamps-per-track times "
            f"{len(dataset.cameras)} cameras must equal 12"
        )

    selected_set = set(selected_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, selected_frames[0]))
    stationary_track_ids, stationary_track_records = classify_stationary_tracks(
        metadata,
        args.stationary_track_max_span,
        args.stationary_track_max_median_step,
    )
    if stationary_track_ids:
        metadata = dict(metadata)
        stationary_set = set(stationary_track_ids)
        metadata["instances"] = {
            track_id: value
            for track_id, value in evaluation_metadata["instances"].items()
            if int(track_id) not in stationary_set
        }
    intrinsics = calibrated_intrinsics(
        dataset, metadata, selected_frames[0]
    )
    track_candidates = {
        int(track_id): track_view_candidates(
            dataset, metadata, int(track_id), selected_set, world_from, intrinsics
        )
        for track_id in metadata["instances"]
    }
    if args.view_selection == "diverse-visible":
        annotate_fine_mask_visibility(dataset, metadata, track_candidates)
    track_visible_frames = {
        int(track_id): track_front_camera_frames(
            dataset,
            metadata,
            int(track_id),
            selected_set,
            world_from,
            intrinsics,
        )
        for track_id in metadata["instances"]
    }
    track_visible_frames = {
        track_id: frames for track_id, frames in track_visible_frames.items() if frames
    }
    global_track_ids = sorted(
        track_id for track_id in track_visible_frames if track_candidates[track_id]
    )
    global_track_slot = {
        track_id: slot for slot, track_id in enumerate(global_track_ids)
    }
    global_boxes_cpu = build_global_track_boxes(
        dataset, metadata, selected_frames, world_from, global_track_ids
    )
    evaluation_track_ids = sorted(
        int(value) for value in evaluation_metadata["instances"]
    )
    evaluation_boxes_cpu = build_global_track_boxes(
        dataset,
        evaluation_metadata,
        selected_frames,
        world_from,
        evaluation_track_ids,
    )
    global_boxes = boxes_to(global_boxes_cpu, device)
    global_times = dataset._normalised_times(selected_frames)

    model = build_model(config, device, str(checkpoint_path))
    if args.dynamic_queries_per_track is not None:
        if args.dynamic_queries_per_track <= 0:
            parser.error("--dynamic-queries-per-track must be positive")
        model.num_dynamic = args.dynamic_queries_per_track
    depth_cache: dict[tuple[int, int], torch.Tensor] = {}
    feature_cache: dict[tuple[int, int], torch.Tensor] = {}
    dynamic_parts = []
    track_records = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for track_index, track_id in enumerate(global_track_ids):
        if args.view_selection == "diverse-visible":
            selected_views = select_diverse_visible_views(
                track_candidates[track_id],
                args.views_per_track,
                args.view_angle_weight,
            )
            context_frames = sorted({item["frame"] for item in selected_views})
        elif args.view_selection == "diverse-large":
            selected_views = select_diverse_large_view_pass(
                track_candidates[track_id],
                args.views_per_track,
                args.view_angle_weight,
                args.view_pass_index,
            )
            context_frames = sorted({item["frame"] for item in selected_views})
        else:
            context_frames = uniform_timestamps(
                track_visible_frames[track_id], args.timestamps_per_track
            )
            selected_views = [
                {"frame": int(frame), "camera": int(camera)}
                for frame in context_frames for camera in dataset.cameras
            ]
        slot_of = {frame: slot for slot, frame in enumerate(context_frames)}
        normalized_times = dataset._normalised_times(context_frames)
        context = dataset._collect(
            metadata,
            context_frames,
            slot_of,
            normalized_times,
            dataset.resolution,
            world_from,
            lidar=dataset.load_lidar,
        )
        local_boxes_cpu = build_global_track_boxes(
            dataset, metadata, context_frames, world_from, [track_id]
        )
        local_boxes = boxes_to(local_boxes_cpu, device)
        scene = SceneInput(
            images=context["images"].to(device),
            intrinsics=context["intrinsics"].to(device),
            cam_to_world=context["cam_to_world"].to(device),
            times=context["times"].to(device),
            frame_index=context["frame_index"].to(device),
            boxes=local_boxes,
            sparse_depth=(
                context["sparse_depth"].to(device)
                if "sparse_depth" in context else None
            ),
            sparse_mask=(
                context["sparse_mask"].to(device)
                if "sparse_mask" in context else None
            ),
        )

        all_view_keys = [
            (int(frame), int(camera))
            for frame in context_frames for camera in dataset.cameras
        ]
        selected_keys = [
            (item["frame"], item["camera"]) for item in selected_views
        ]
        selected_indices = torch.tensor(
            [all_view_keys.index(key) for key in selected_keys],
            dtype=torch.long,
            device=device,
        )
        scene = SceneInput(
            images=torch.index_select(scene.images, 0, selected_indices),
            intrinsics=torch.index_select(scene.intrinsics, 0, selected_indices),
            cam_to_world=torch.index_select(scene.cam_to_world, 0, selected_indices),
            times=torch.index_select(scene.times, 0, selected_indices),
            frame_index=torch.index_select(scene.frame_index, 0, selected_indices),
            boxes=scene.boxes,
            sparse_depth=(
                None if scene.sparse_depth is None
                else torch.index_select(scene.sparse_depth, 0, selected_indices)
            ),
            sparse_mask=(
                None if scene.sparse_mask is None
                else torch.index_select(scene.sparse_mask, 0, selected_indices)
            ),
        )

        view_keys = selected_keys
        missing = [index for index, key in enumerate(view_keys) if key not in depth_cache]
        if missing:
            index = torch.tensor(missing, dtype=torch.long, device=device)
            images = torch.index_select(scene.images, 0, index)
            sparse_depth = (
                None if scene.sparse_depth is None
                else torch.index_select(scene.sparse_depth, 0, index)
            )
            sparse_mask = (
                None if scene.sparse_mask is None
                else torch.index_select(scene.sparse_mask, 0, index)
            )
            height, width = scene.image_size
            depth_height = int(round(height * model.depth_scale))
            depth_width = int(round(width * model.depth_scale))
            depth = model.depth_model(
                images,
                height=depth_height,
                width=depth_width,
                sparse_depth=sparse_depth,
                sparse_mask=sparse_mask,
            )
            if (depth_height, depth_width) != (height, width):
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            features = model.backbone(images)
            for result_index, view_index in enumerate(missing):
                key = view_keys[view_index]
                depth_cache[key] = depth[result_index].detach()
                feature_cache[key] = features[result_index].detach()

        depth = torch.stack([depth_cache[key] for key in view_keys])
        features = torch.stack([feature_cache[key] for key in view_keys])
        generator = torch.Generator(device=device).manual_seed(args.seed + track_id)
        sampled = model.build_queries(scene, depth, generator=generator)
        dynamic_mask = sampled["instance_id"] == 0
        sampled = select_sampled(sampled, dynamic_mask)
        if sampled["queries"].shape[0] == 0:
            track_records.append({
                "track_id": track_id,
                "global_slot": global_track_slot[track_id],
                "visible_frame_count": len(track_visible_frames[track_id]),
                "selected_views": selected_views,
                "dynamic_count": 0,
            })
            continue

        decoded = model.decode(scene, depth, sampled, image_features=features)
        decoded.extras["preview_rgb"] = query_rgb(sampled["queries"])
        decoded.extras["observations"] = torch.ones(
            decoded.count, dtype=torch.int32, device=device
        )
        decoded.extras["best_observation_distance"] = sampled_camera_distance(
            sampled, scene
        )
        decoded.scales, _ = postprocess_gaussian_scales(
            decoded.scales,
            mode=args.gaussian_scale_postprocess,
            isotropic_max_scale=args.isotropic_max_scale,
        )
        dynamic = select_gaussians(decoded, decoded.instance_id >= 0)
        dynamic = rebase_dynamic_to_global_tracks(
            dynamic,
            local_boxes,
            [track_id],
            global_boxes,
            global_track_slot,
        )
        dynamic_parts.append(dynamic)
        track_records.append({
            "track_id": track_id,
            "global_slot": global_track_slot[track_id],
            "visible_frame_count": len(track_visible_frames[track_id]),
            "selected_views": selected_views,
            "sampled_view_count": len(selected_views),
            "dynamic_count": dynamic.count,
        })
        print(json.dumps({
            "track": track_index + 1,
            "track_count": len(global_track_ids),
            **track_records[-1],
        }), flush=True)
        del context, scene, local_boxes, sampled, decoded, dynamic
        torch.cuda.empty_cache()

    if not dynamic_parts:
        raise RuntimeError("No dynamic Gaussians were decoded")
    final_dynamic = concatenate_gaussians(dynamic_parts)
    dynamic_path = output_dir / "final_dynamic_128d.pt"
    meta_path = output_dir / "global_dynamic_meta.pt"
    save_gaussian_payload(dynamic_path, final_dynamic)
    torch.save(
        {
            "frames": selected_frames,
            "normalized_times": global_times.cpu(),
            "track_ids": global_track_ids,
            "boxes": boxes_payload(global_boxes_cpu),
            "evaluation_track_ids": evaluation_track_ids,
            "evaluation_boxes": boxes_payload(evaluation_boxes_cpu),
            "stationary_track_reclassification": {
                "max_center_span_m": args.stationary_track_max_span,
                "max_median_step_m": args.stationary_track_max_median_step,
                "track_ids": stationary_track_ids,
                "tracks": stationary_track_records,
            },
        },
        meta_path,
    )
    static_link = output_dir / "final_static_128d.pt"
    if static_link.exists() or static_link.is_symlink():
        if static_link.resolve() != static_path:
            raise FileExistsError(static_link)
    else:
        os.symlink(static_path, static_link)

    manifest = {
        "status": "completed",
        "scene": args.scene,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "static_payload": str(static_path),
        "static_policy": "byte-identical reuse of GD-v1-f final static payload",
        "dynamic_policy": "one independent 12-view clip per global track; decode once; no later refresh or append",
        "view_contract": {
            "selection": args.view_selection,
            "timestamps_per_track": args.timestamps_per_track,
            "cameras": list(dataset.cameras),
            "views_per_track": args.views_per_track,
            "angle_weight": args.view_angle_weight,
            "pass_index": args.view_pass_index,
            "selection_objective": (
                "greedy local-azimuth diversity and fine-dynamic-mask visible pixels"
                if args.view_selection == "diverse-visible"
                else "greedy weighted local-azimuth diversity and clipped bbox projected area"
                if args.view_selection == "diverse-large"
                else "uniform timestamps over every visible frame, all cameras"
            ),
        },
        "track_count": len(global_track_ids),
        "stationary_track_reclassification": {
            "enabled": args.stationary_track_max_span is not None,
            "max_center_span_m": args.stationary_track_max_span,
            "max_median_step_m": args.stationary_track_max_median_step,
            "track_ids": stationary_track_ids,
            "track_count": len(stationary_track_ids),
        },
        "dynamic_count": final_dynamic.count,
        "dynamic_queries_per_track": model.num_dynamic,
        "tracks": track_records,
        "gaussian_scale_postprocess": {
            "mode": args.gaussian_scale_postprocess,
            "isotropic_max_scale_m": args.isotropic_max_scale,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    (output_dir / "track_once_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

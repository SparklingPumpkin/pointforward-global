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
from pointforward.models.scene_graph import InstanceBoxes
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
    """Return intrinsics after the dataset's crop and resize."""
    height, width = dataset.resolution
    output = {}
    for camera in dataset.cameras:
        path = Path(metadata["path"]) / "images" / f"{frame:03d}_{camera}.jpg"
        with Image.open(path) as image:
            original_width, original_height = image.size
        intrinsic = metadata["intrinsics"][camera].copy()
        crop_x, crop_y = 0, 0
        crop_width, crop_height = original_width, original_height
        if hasattr(dataset, "_crop_parameters"):
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


def load_calibrated_foreground_mask(
    dataset, metadata: dict, path: Path, camera: int
) -> torch.Tensor:
    """Load an external RGB-predicted mask with the dataset crop contract."""
    height, width = dataset.resolution
    intrinsic = metadata["intrinsics"][camera]
    with Image.open(path) as image:
        image = image.convert("L")
        original_width, original_height = image.size
        crop_x, crop_y, crop_width, crop_height = dataset._crop_parameters(
            original_width,
            original_height,
            width,
            height,
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        image = image.crop(
            (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
        )
        image = image.resize((width, height), Image.NEAREST)
        mask = np.asarray(image, dtype=np.uint8) > 0
    return torch.from_numpy(mask.copy())


def foreground_mask_path(
    root: Path,
    layout: str,
    track_id: int,
    frame: int,
    camera: int,
) -> Path:
    """Resolve shared semantic or per-track instance foreground masks."""
    filename = f"{frame:03d}_{camera}.png"
    if layout == "shared":
        return root / filename
    if layout == "per-track":
        return root / f"track-{track_id}" / filename
    raise ValueError(f"Unsupported foreground mask layout: {layout}")


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


def select_camera_balanced_views(
    candidates: list[dict], count: int, angle_weight: float
) -> list[dict]:
    """Select an equal camera quota while reusing the area/diversity score."""
    cameras = sorted({int(item["camera"]) for item in candidates})
    if not cameras:
        return []
    quota = count // len(cameras)
    selected: list[dict] = []
    used: set[tuple[int, int]] = set()
    for camera in cameras:
        pool = [item for item in candidates if int(item["camera"]) == camera]
        chosen = select_diverse_large_views(pool, min(quota, len(pool)), angle_weight)
        selected.extend(chosen)
        used.update((int(item["frame"]), int(item["camera"])) for item in chosen)
    if len(selected) < count:
        remaining = [
            item for item in candidates
            if (int(item["frame"]), int(item["camera"])) not in used
        ]
        remaining.sort(
            key=lambda item: (-float(item.get("bbox_area_fraction", 0.0)), int(item["frame"]), int(item["camera"]))
        )
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--static-payload", required=True)
    parser.add_argument(
        "--allow-pending-static-payload", action="store_true",
        help="For a joined parallel pipeline only: static input is merely linked, not read; caller must verify its completed producer before rendering.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--query-origin", choices=("scene", "first-ego", "canonical-centre"),
        default="scene",
        help="Optional translation of learned query XYZ/Plucker moment per independent track clip. Geometry used for projection and stored global canonical means remain unchanged in coordinate convention.",
    )
    parser.add_argument("--timestamps-per-track", type=int, default=4)
    parser.add_argument(
        "--view-selection",
        choices=("uniform-timestamps", "diverse-large", "diverse-visible", "camera-balanced"),
        default="uniform-timestamps",
    )
    parser.add_argument("--views-per-track", type=int, default=12)
    parser.add_argument(
        "--track-ids",
        default=None,
        help=(
            "Optional comma-separated scene track IDs to decode. Global box "
            "slots and metadata still use the complete scene ordering, so the "
            "result can replace those tracks in an existing global payload."
        ),
    )
    parser.add_argument(
        "--temporal-segments-per-track",
        type=int,
        default=1,
        help=(
            "Split each requested track into this many temporal canonical "
            "segments. Each segment is decoded independently and is valid only "
            "on its assigned frames; default 1 preserves the original path."
        ),
    )
    parser.add_argument(
        "--dynamic-queries-per-track",
        type=int,
        default=None,
        help="Override the checkpoint's per-instance query budget at inference.",
    )
    parser.add_argument(
        "--decode-box-scale", type=float, default=1.0,
        help=(
            "Scale only local box dimensions used for query assignment; "
            "canonical Gaussians are rebased to the unscaled global box."
        ),
    )
    parser.add_argument("--decode-box-length-margin", type=float, default=0.0)
    parser.add_argument("--decode-box-width-margin", type=float, default=0.0)
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
    parser.add_argument(
        "--foreground-mask-dir",
        default=None,
        help=(
            "Directory containing RGB-predicted {frame:03d}_{camera}.png masks. "
            "Ignored unless a foreground gate is enabled."
        ),
    )
    parser.add_argument(
        "--foreground-mask-layout",
        choices=("shared", "per-track"),
        default="shared",
        help=(
            "Mask path layout. 'shared' preserves {frame}_{camera}.png; "
            "'per-track' reads track-{track_id}/{frame}_{camera}.png."
        ),
    )
    parser.add_argument(
        "--foreground-query-gate",
        action="store_true",
        help="Require box-assigned source pixels to lie in RGB foreground.",
    )
    parser.add_argument(
        "--foreground-fusion-gate",
        action="store_true",
        help="Require dynamic projections to lie in RGB foreground and box masks.",
    )
    parser.add_argument(
        "--source-anchor-dynamic-fq",
        action="store_true",
        help="Use each dynamic query's originating source-view feature as f_q.",
    )
    parser.add_argument(
        "--neutralize-dynamic-depth-cue",
        action="store_true",
        help="Set depth-consistency residual to zero only for dynamic queries.",
    )
    parser.add_argument(
        "--dynamic-soft-occlusion-order",
        action="store_true",
        help="Softly attenuate behind-front dynamic evidence using a per-track projected z-order.",
    )
    parser.add_argument("--stationary-track-max-span", type=float, default=None)
    parser.add_argument("--stationary-track-max-median-step", type=float, default=0.1)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    static_path = Path(args.static_payload).resolve()
    output_dir = Path(args.out).resolve()
    foreground_mask_dir = (
        None if args.foreground_mask_dir is None
        else Path(args.foreground_mask_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not static_path.exists() and not args.allow_pending_static_payload:
        raise FileNotFoundError(static_path)
    if args.foreground_query_gate or args.foreground_fusion_gate:
        if foreground_mask_dir is None or not foreground_mask_dir.is_dir():
            parser.error(
                "foreground gates require an existing --foreground-mask-dir"
            )

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
    evaluation_metadata = dict(metadata, instances=metadata.get('_original_instances', metadata['instances']))
    selected_frames = [int(frame) for frame in metadata["frames"]]
    if args.views_per_track not in (8, 10, 12):
        parser.error("Supported source view budgets are 8, 10, or 12")
    if not 0.0 <= args.view_angle_weight <= 1.0:
        parser.error("--view-angle-weight must be in [0, 1]")
    if args.view_pass_index < 0:
        parser.error("--view-pass-index must be non-negative")
    if args.temporal_segments_per_track <= 0:
        parser.error("--temporal-segments-per-track must be positive")
    if args.decode_box_scale < 1.0:
        parser.error("--decode-box-scale must be >= 1")
    if args.temporal_segments_per_track > 1 and not args.track_ids:
        parser.error(
            "--temporal-segments-per-track > 1 requires explicit --track-ids"
        )
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
    requested_track_ids = None
    if args.track_ids:
        requested_track_ids = sorted({
            int(value) for value in args.track_ids.split(",") if value.strip()
        })
        missing_track_ids = sorted(set(requested_track_ids) - set(global_track_ids))
        if missing_track_ids:
            parser.error(
                "--track-ids contains tracks without valid visible candidates: "
                + ",".join(str(value) for value in missing_track_ids)
            )
        decode_track_ids = [
            track_id for track_id in global_track_ids
            if track_id in set(requested_track_ids)
        ]
    else:
        decode_track_ids = global_track_ids
    global_track_slot = {
        track_id: slot for slot, track_id in enumerate(global_track_ids)
    }
    global_boxes_cpu = build_global_track_boxes(
        dataset, metadata, selected_frames, world_from, global_track_ids
    )
    render_track_ids = list(global_track_ids)
    decode_jobs = []
    if args.temporal_segments_per_track == 1:
        decode_jobs = [
            {
                "track_id": track_id,
                "segment_index": 0,
                "global_slot": global_track_slot[track_id],
                "allowed_frames": None,
            }
            for track_id in decode_track_ids
        ]
    else:
        box_parts = {
            name: [getattr(global_boxes_cpu, name).clone()]
            for name in ("centers", "dims", "rots", "times", "valid")
        }
        base_valid = global_boxes_cpu.valid.clone()
        for track_id in decode_track_ids:
            original_slot = global_track_slot[track_id]
            valid_slots = torch.nonzero(
                base_valid[original_slot], as_tuple=False
            ).flatten()
            visible_frame_groups = np.array_split(
                np.asarray(sorted({
                    int(item["frame"]) for item in track_candidates[track_id]
                }), dtype=np.int64),
                args.temporal_segments_per_track,
            )
            if any(group.size == 0 for group in visible_frame_groups):
                parser.error(
                    f"track {track_id} has too few visible frames for "
                    f"{args.temporal_segments_per_track} temporal segments"
                )
            cutoffs = [
                0.5 * (float(left[-1]) + float(right[0]))
                for left, right in zip(
                    visible_frame_groups[:-1], visible_frame_groups[1:]
                )
            ]
            slot_groups = []
            for segment_index in range(args.temporal_segments_per_track):
                chosen = []
                for slot in valid_slots.tolist():
                    frame_value = int(selected_frames[int(slot)])
                    assigned = int(np.searchsorted(cutoffs, frame_value, side="right"))
                    if assigned == segment_index:
                        chosen.append(int(slot))
                slot_groups.append(torch.tensor(chosen, dtype=torch.long))
            global_boxes_cpu.valid[original_slot].zero_()
            global_boxes_cpu.valid[original_slot, slot_groups[0]] = True
            for segment_index, slots in enumerate(slot_groups):
                if segment_index == 0:
                    segment_slot = original_slot
                else:
                    segment_slot = len(render_track_ids)
                    render_track_ids.append(
                        -(int(track_id) * args.temporal_segments_per_track + segment_index + 1)
                    )
                    for name in ("centers", "dims", "rots", "times"):
                        source = getattr(global_boxes_cpu, name)[original_slot:original_slot + 1]
                        box_parts[name].append(source.clone())
                    segment_valid = torch.zeros_like(
                        global_boxes_cpu.valid[original_slot:original_slot + 1]
                    )
                    segment_valid[0, slots] = True
                    box_parts["valid"].append(segment_valid)
                allowed_frames = {
                    int(value) for value in visible_frame_groups[segment_index].tolist()
                }
                decode_jobs.append({
                    "track_id": track_id,
                    "segment_index": segment_index,
                    "global_slot": segment_slot,
                    "allowed_frames": allowed_frames,
                })
        box_parts["centers"][0] = global_boxes_cpu.centers
        box_parts["dims"][0] = global_boxes_cpu.dims
        box_parts["rots"][0] = global_boxes_cpu.rots
        box_parts["times"][0] = global_boxes_cpu.times
        box_parts["valid"][0] = global_boxes_cpu.valid
        global_boxes_cpu = InstanceBoxes(**{
            name: torch.cat(parts, dim=0) for name, parts in box_parts.items()
        })
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

    for track_index, job in enumerate(decode_jobs):
        track_id = job["track_id"]
        candidate_pool = track_candidates[track_id]
        visible_frames = track_visible_frames[track_id]
        if job["allowed_frames"] is not None:
            candidate_pool = [
                item for item in candidate_pool
                if item["frame"] in job["allowed_frames"]
            ]
            visible_frames = [
                frame for frame in visible_frames
                if frame in job["allowed_frames"]
            ]
        if args.view_selection == "diverse-visible":
            selected_views = select_diverse_visible_views(
                candidate_pool,
                args.views_per_track,
                args.view_angle_weight,
            )
            context_frames = sorted({item["frame"] for item in selected_views})
        elif args.view_selection == "diverse-large":
            selected_views = select_diverse_large_view_pass(
                candidate_pool,
                args.views_per_track,
                args.view_angle_weight,
                args.view_pass_index,
            )
            context_frames = sorted({item["frame"] for item in selected_views})
        elif args.view_selection == "camera-balanced":
            selected_views = select_camera_balanced_views(
                candidate_pool,
                args.views_per_track,
                args.view_angle_weight,
            )
            context_frames = sorted({item["frame"] for item in selected_views})
        else:
            context_frames = uniform_timestamps(
                visible_frames, args.timestamps_per_track
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
        if args.decode_box_scale != 1.0 or args.decode_box_length_margin != 0.0 or args.decode_box_width_margin != 0.0:
            dims = local_boxes_cpu.dims * args.decode_box_scale
            dims = dims.clone()
            # Local axes are length (x), width (y), height (z).
            dims[..., 0] += args.decode_box_length_margin
            dims[..., 1] += args.decode_box_width_margin
            local_boxes_cpu = InstanceBoxes(
                centers=local_boxes_cpu.centers,
                dims=dims,
                rots=local_boxes_cpu.rots,
                times=local_boxes_cpu.times,
                valid=local_boxes_cpu.valid,
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
        source_selected_keys = [
            (item["frame"], item["camera"]) for item in selected_views
        ]
        # The pretrained sigma head remains a 12-slot contract. Reduced-view
        # probes lower expensive source RGB/depth/backbone work, then pad the
        # selected source slots by repetition so the generic checkpoint is
        # never rebuilt or silently given a different view count.
        selected_keys = list(source_selected_keys)
        if len(selected_keys) < 12:
            selected_keys.extend(
                source_selected_keys[index % len(source_selected_keys)]
                for index in range(12 - len(selected_keys))
            )
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
        foreground_masks = None
        if args.foreground_query_gate or args.foreground_fusion_gate:
            mask_parts = []
            for frame, camera in view_keys:
                mask_path = foreground_mask_path(
                    foreground_mask_dir,
                    args.foreground_mask_layout,
                    track_id,
                    frame,
                    camera,
                )
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing RGB foreground mask for selected view: {mask_path}"
                    )
                mask_parts.append(
                    load_calibrated_foreground_mask(
                        dataset, metadata, mask_path, camera
                    )
                )
            foreground_masks = torch.stack(mask_parts).to(device=device)

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
        scene.dynamic_foreground_masks = (
            foreground_masks if args.foreground_query_gate else None
        )
        sampled = model.build_queries(scene, depth, generator=generator)
        dynamic_mask = sampled["instance_id"] == 0
        sampled = select_sampled(sampled, dynamic_mask)
        if sampled["queries"].shape[0] == 0:
            track_records.append({
                "track_id": track_id,
                "segment_index": job["segment_index"],
                "global_slot": job["global_slot"],
                "visible_frame_count": len(visible_frames),
                "selected_views": selected_views,
                "dynamic_count": 0,
            })
            continue

        scene.dynamic_foreground_masks = (
            foreground_masks if args.foreground_fusion_gate else None
        )
        query_origin = None
        if args.query_origin == "first-ego":
            first_ego = world_from @ dataset._ego_pose(metadata, context_frames[0])
            query_origin = torch.as_tensor(
                first_ego[:3, 3], device=device, dtype=sampled["base_xyz"].dtype,
            )
        elif args.query_origin == "canonical-centre":
            query_origin = local_boxes.canonical()[0][0]
        decoded = model.decode(
            scene,
            depth,
            sampled,
            image_features=features,
            query_origin=query_origin,
            source_anchor_dynamic_fq=args.source_anchor_dynamic_fq,
            neutralize_dynamic_depth_cue=args.neutralize_dynamic_depth_cue,
            dynamic_soft_occlusion_order=args.dynamic_soft_occlusion_order,
        )
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
            {track_id: job["global_slot"]},
        )
        dynamic_parts.append(dynamic)
        track_records.append({
            "track_id": track_id,
            "segment_index": job["segment_index"],
            "global_slot": job["global_slot"],
            "visible_frame_count": len(visible_frames),
            "selected_views": selected_views,
            "sampled_view_count": len(selected_keys),
            "source_view_count": len(source_selected_keys),
            "model_view_slot_count": len(selected_keys),
            "padded_source_slots": max(0, len(selected_keys) - len(source_selected_keys)),
            "learned_query_origin_policy": args.query_origin,
            "learned_query_origin_translation": (
                None if query_origin is None else query_origin.detach().cpu().tolist()
            ),
            "foreground_query_gate": args.foreground_query_gate,
            "foreground_fusion_gate": args.foreground_fusion_gate,
            "source_anchor_dynamic_fq": args.source_anchor_dynamic_fq,
            "neutralize_dynamic_depth_cue": args.neutralize_dynamic_depth_cue,
            "dynamic_soft_occlusion_order": args.dynamic_soft_occlusion_order,
            "foreground_pixels_per_view": (
                None if foreground_masks is None
                else [int(mask.sum()) for mask in foreground_masks]
            ),
            "dynamic_count": dynamic.count,
        })
        print(json.dumps({
            "track": track_index + 1,
            "track_count": len(decode_jobs),
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
            "track_ids": render_track_ids,
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
        "allow_pending_static_payload": args.allow_pending_static_payload,
        "decode_box_margins_m": {"length_total": args.decode_box_length_margin, "width_total": args.decode_box_width_margin, "height_total": 0.0},
        "static_policy": "byte-identical reuse of GD-v1-f final static payload",
        "dynamic_policy": "one independent source-view clip per global track; source views are padded to the pretrained 12-slot fusion contract; decode once; no later refresh or append",
        "learned_query_origin_policy": args.query_origin,
        "view_contract": {
            "selection": args.view_selection,
            "timestamps_per_track": args.timestamps_per_track,
            "cameras": list(dataset.cameras),
            "views_per_track": args.views_per_track,
            "model_view_slots": 12,
            "padding_policy": "repeat source slots to 12 for reduced-view probes",
            "angle_weight": args.view_angle_weight,
            "pass_index": args.view_pass_index,
            "selection_objective": (
                "greedy local-azimuth diversity and fine-dynamic-mask visible pixels"
                if args.view_selection == "diverse-visible"
                else "greedy weighted local-azimuth diversity and clipped bbox projected area"
                if args.view_selection == "diverse-large"
                else "fixed equal camera quota with diverse-large score"
                if args.view_selection == "camera-balanced"
                else "uniform timestamps over every visible frame, all cameras"
            ),
        },
        "track_count": len(decode_jobs),
        "scene_global_track_count": len(render_track_ids),
        "requested_track_ids": requested_track_ids,
        "temporal_segments_per_track": args.temporal_segments_per_track,
        "stationary_track_reclassification": {
            "enabled": args.stationary_track_max_span is not None,
            "max_center_span_m": args.stationary_track_max_span,
            "max_median_step_m": args.stationary_track_max_median_step,
            "track_ids": stationary_track_ids,
            "track_count": len(stationary_track_ids),
        },
        "dynamic_count": final_dynamic.count,
        "dynamic_queries_per_track": model.num_dynamic,
        "source_anchor_dynamic_fq": args.source_anchor_dynamic_fq,
        "dynamic_soft_occlusion_order": args.dynamic_soft_occlusion_order,
        "foreground_gate": {
            "mask_dir": (
                None if foreground_mask_dir is None else str(foreground_mask_dir)
            ),
            "query": args.foreground_query_gate,
            "fusion": args.foreground_fusion_gate,
            "layout": args.foreground_mask_layout,
            "contract": (
                "RGB-predicted semantic vehicle mask intersected with the "
                "projected local track box; no dataset fine/dynamic mask"
                if args.foreground_query_gate or args.foreground_fusion_gate
                else None
            ),
        },
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

"""Render a full scene in training-length windows and export each Gaussian snapshot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from pointforward.models import (
    Gaussians,
    InstanceBoxes,
    SceneInput,
    TargetViews,
    canonicalise,
    decanonicalise,
)
from pointforward.models.backbone import sample_features
from pointforward.models.fusion import project_points, box_dynamic_masks
from pointforward.models.query import query_rgb
from pointforward.models.rotation import (
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)
from scripts.eval import build_model
from scripts.export_gaussians_ply import boxes_to, write_standard_ply
from scripts.train import build_dataset


def to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return (array * 255).round().astype(np.uint8)


def label(image: np.ndarray, text: str) -> np.ndarray:
    output = Image.fromarray(image)
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 24), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(output)


def horizontal_strip(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def vertical_strip(rows: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(rows, axis=0)


def encode_video(frame_pattern: Path, output_path: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_pattern),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(output_path),
            "-loglevel", "error",
        ],
        check=True,
    )


def tensor_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction.float() - target.float()) ** 2).clamp_min(1e-10)
    return float((-10.0 * torch.log10(mse)).cpu())


def interpolate_camera_pose(
    first: torch.Tensor, second: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Interpolate camera translation and shortest-path camera rotation."""
    first_quat = torch.nn.functional.normalize(
        matrix_to_quaternion(first[:3, :3]), dim=-1
    )
    second_quat = torch.nn.functional.normalize(
        matrix_to_quaternion(second[:3, :3]), dim=-1
    )
    dot = (first_quat * second_quat).sum()
    if dot < 0:
        second_quat = -second_quat
        dot = -dot
    dot = dot.clamp(max=1.0)
    if dot > 0.9995:
        quat = torch.nn.functional.normalize(
            torch.lerp(first_quat, second_quat, alpha), dim=-1
        )
    else:
        theta = torch.acos(dot)
        denominator = torch.sin(theta).clamp_min(1e-8)
        quat = (
            torch.sin((1.0 - alpha) * theta) / denominator * first_quat
            + torch.sin(alpha * theta) / denominator * second_quat
        )
    output = torch.eye(4, dtype=first.dtype, device=first.device)
    output[:3, :3] = quaternion_to_matrix(quat)
    output[:3, 3] = torch.lerp(first[:3, 3], second[:3, 3], alpha)
    return output


_GAUSSIAN_FIELDS = (
    "means", "quats", "scales", "opacities", "colors", "sigma", "times", "instance_id"
)


def select_gaussians(gaussians: Gaussians, index: torch.Tensor) -> Gaussians:
    return Gaussians(
        **{name: getattr(gaussians, name)[index] for name in _GAUSSIAN_FIELDS},
        extras={name: value[index] for name, value in gaussians.extras.items()},
    )


def concatenate_gaussians(parts: list[Gaussians]) -> Gaussians:
    nonempty = [part for part in parts if part.count > 0]
    if not nonempty:
        raise ValueError("At least one non-empty Gaussian set is required")
    extra_names = set.intersection(*(set(part.extras) for part in nonempty))
    return Gaussians(
        **{
            name: torch.cat([getattr(part, name) for part in nonempty], dim=0)
            for name in _GAUSSIAN_FIELDS
        },
        extras={
            name: torch.cat([part.extras[name] for part in nonempty], dim=0)
            for name in extra_names
        },
    )


def move_gaussians(gaussians: Gaussians, device: torch.device) -> Gaussians:
    return Gaussians(
        **{name: getattr(gaussians, name).to(device) for name in _GAUSSIAN_FIELDS},
        extras={name: value.to(device) for name, value in gaussians.extras.items()},
    )


def replace_gaussians(base: Gaussians, index: torch.Tensor, values: Gaussians) -> Gaussians:
    for name in _GAUSSIAN_FIELDS:
        getattr(base, name)[index] = getattr(values, name)
    for name, value in values.extras.items():
        if name in base.extras:
            base.extras[name][index] = value
    return base


def pointwise_lerp(
    old: torch.Tensor, new: torch.Tensor, alpha: float | torch.Tensor
) -> torch.Tensor:
    """Lerp with either one scalar or one weight per Gaussian."""
    if not torch.is_tensor(alpha):
        return old.lerp(new, alpha)
    weight = alpha.to(device=old.device, dtype=old.dtype)
    while weight.ndim < old.ndim:
        weight = weight.unsqueeze(-1)
    return torch.lerp(old, new, weight)


def sampled_camera_distance(
    sampled: dict[str, torch.Tensor], scene: SceneInput
) -> torch.Tensor:
    origins = scene.cam_to_world[sampled["view_id"].long(), :3, 3]
    return torch.linalg.vector_norm(sampled["world_xyz"] - origins, dim=-1)


def distance_update_alpha(
    distance: torch.Tensor,
    policy: str,
    max_alpha: float,
    near_distance: float,
    far_distance: float,
    old_best_distance: torch.Tensor | None = None,
    min_improvement: float = 0.0,
) -> torch.Tensor:
    """Return the new-observation weight; far observations retain old state."""
    t = ((far_distance - distance) / (far_distance - near_distance)).clamp(0, 1)
    weight = t.square() * (3.0 - 2.0 * t)
    alpha = weight * max_alpha
    if policy == "closest-smoothstep":
        if old_best_distance is None:
            raise ValueError("closest-smoothstep requires persistent best distance")
        alpha = alpha * (
            distance < old_best_distance - min_improvement
        ).to(alpha.dtype)
    elif policy != "smoothstep":
        raise ValueError(f"Unsupported distance update policy: {policy}")
    return alpha


def select_sampled(
    sampled: dict[str, torch.Tensor], index: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: value[index] for name, value in sampled.items()}


def blend_static_gaussians(
    old: Gaussians,
    update: Gaussians,
    alpha: float | torch.Tensor,
    update_position: bool = True,
    refresh_fields: str = "all",
) -> Gaussians:
    """Blend a decoder refresh into persistent static Gaussians."""
    quat_update = update.quats
    opposite = (old.quats * quat_update).sum(dim=-1, keepdim=True) < 0
    quat_update = torch.where(opposite, -quat_update, quat_update)
    quats = torch.nn.functional.normalize(
        pointwise_lerp(old.quats, quat_update, alpha), dim=-1
    )
    scales = pointwise_lerp(
        old.scales.clamp_min(1e-8).log(),
        update.scales.clamp_min(1e-8).log(),
        alpha,
    ).exp()
    old_opacity = torch.logit(old.opacities.clamp(1e-6, 1.0 - 1e-6))
    new_opacity = torch.logit(update.opacities.clamp(1e-6, 1.0 - 1e-6))
    extras = {}
    for name in set(old.extras) & set(update.extras):
        if name == "observations":
            extras[name] = old.extras[name] + 1
        elif name in (
            "best_position_observation_distance",
            "position_observations",
        ):
            # Geometry quality belongs to the explicit dynamic position merge,
            # not to decoder attribute refresh.  Static points carry these
            # fields only so one heterogeneous global Gaussian container can
            # preserve the dynamic state.
            extras[name] = old.extras[name]
        elif name == "best_observation_distance":
            extras[name] = torch.minimum(old.extras[name], update.extras[name])
        elif old.extras[name].is_floating_point():
            extras[name] = pointwise_lerp(
                old.extras[name], update.extras[name], alpha
            )
        else:
            extras[name] = old.extras[name]
    if refresh_fields not in ("all", "colors"):
        raise ValueError(f"Unsupported static refresh fields: {refresh_fields}")
    return Gaussians(
        means=(
            pointwise_lerp(old.means, update.means, alpha)
            if update_position else old.means
        ),
        quats=quats if refresh_fields == "all" else old.quats,
        scales=scales if refresh_fields == "all" else old.scales,
        opacities=(torch.sigmoid(pointwise_lerp(old_opacity, new_opacity, alpha))
                   if refresh_fields == "all" else old.opacities),
        colors=pointwise_lerp(old.colors, update.colors, alpha),
        sigma=torch.full_like(old.sigma, 1e6),
        times=torch.zeros_like(old.times),
        instance_id=torch.full_like(old.instance_id, -1),
        extras=extras,
    )


def points_visible_in_history(
    points: torch.Tensor,
    history: list[dict[str, torch.Tensor]],
    depth_threshold: float,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """Return points lying in the union of all previously visible regions.

    A point is historically visible when it projects inside a prior context
    image, lies in front of its camera, and is no farther than the prior dense
    surface plus ``depth_threshold``.  Points behind that surface remain novel
    because they were occluded in the prior clip.
    """
    visible = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    remaining = torch.arange(points.shape[0], device=points.device)
    height = int(history[0]["depth"].shape[-2]) if history else 0
    width = int(history[0]["depth"].shape[-1]) if history else 0

    # Newest clips usually reject most overlap immediately; visit them first
    # and only test still-novel candidates against older visibility records.
    for record in reversed(history):
        if remaining.numel() == 0:
            break
        intrinsics = record["intrinsics"].to(points.device)
        world_to_cam = torch.linalg.inv(record["cam_to_world"].to(points.device))
        depth = record["depth"].to(points.device)
        observed_parts = []
        for lo in range(0, remaining.numel(), chunk_size):
            candidate_index = remaining[lo:lo + chunk_size]
            candidate = points[candidate_index]
            projected = candidate.unsqueeze(0).expand(intrinsics.shape[0], -1, -1)
            uv, projected_depth, in_front = project_points(
                projected, intrinsics, world_to_cam
            )
            in_bounds = (
                (uv[..., 0] >= 0)
                & (uv[..., 0] <= width - 1)
                & (uv[..., 1] >= 0)
                & (uv[..., 1] <= height - 1)
            )
            surface_depth = sample_features(
                depth.unsqueeze(1), uv, height, width
            ).squeeze(-1)
            view_visible = (
                in_front
                & in_bounds
                & surface_depth.isfinite()
                & (surface_depth > 0)
                & (projected_depth <= surface_depth + depth_threshold)
            )
            observed_parts.append(view_visible.any(dim=0))
        observed = torch.cat(observed_parts)
        visible[remaining[observed]] = True
        remaining = remaining[~observed]
        del intrinsics, world_to_cam, depth, observed_parts, observed
    return visible


def persistent_static(gaussians: Gaussians) -> Gaussians:
    static = select_gaussians(gaussians, gaussians.instance_id < 0)
    static.sigma = torch.full_like(static.sigma, 1e6)
    static.times = torch.zeros_like(static.times)
    if "observations" not in static.extras:
        static.extras["observations"] = torch.ones(
            static.count, dtype=torch.int32, device=static.means.device
        )
    return static


def gaussian_payload(gaussians: Gaussians) -> dict:
    """Return a CPU-only, pickleable representation of a runtime Gaussian set."""
    return {
        "means": gaussians.means.detach().cpu(),
        "quats": gaussians.quats.detach().cpu(),
        "scales": gaussians.scales.detach().cpu(),
        "opacities": gaussians.opacities.detach().cpu(),
        "colors": gaussians.colors.detach().cpu(),
        "sigma": gaussians.sigma.detach().cpu(),
        "times": gaussians.times.detach().cpu(),
        "instance_id": gaussians.instance_id.detach().cpu(),
        "extras": {
            name: value.detach().cpu() for name, value in gaussians.extras.items()
        },
    }


def save_gaussian_payload(path: Path, gaussians: Gaussians) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gaussian_payload(gaussians), path)


def boxes_payload(boxes) -> dict | None:
    if boxes is None:
        return None
    return {
        "centers": boxes.centers.detach().cpu(),
        "dims": boxes.dims.detach().cpu(),
        "rots": boxes.rots.detach().cpu(),
        "times": boxes.times.detach().cpu(),
        "valid": boxes.valid.detach().cpu(),
    }


def window_track_ids(dataset, metadata: dict, frames: list[int]) -> list[int]:
    """Return the scene-level track IDs in the same order as ``_build_boxes``.

    DriveStudio stores stable scene-wide instance keys, but ``InstanceBoxes``
    intentionally contains only dense clip-local slots.  Repeating the small
    deterministic selection here preserves the original decoder contract while
    recovering the stable identity needed by the optional global-dynamic path.
    """
    present = {}
    for slot, frame in enumerate(frames):
        for track_id, per_frame in metadata["instances"].items():
            if frame in per_frame:
                present.setdefault(track_id, {})[slot] = per_frame[frame]
    if not present:
        return []
    origin = dataset._ego_pose(metadata, frames[0])[:3, 3]

    def closeness(item) -> float:
        return min(
            float(np.linalg.norm(pose[:3, 3] - origin))
            for pose, _ in item[1].values()
        )

    chosen = sorted(present.items(), key=closeness)[: dataset.max_instances]
    return [int(track_id) for track_id, _ in chosen]


def build_global_track_boxes(
    dataset,
    metadata: dict,
    frames: list[int],
    world_from: np.ndarray,
    track_ids: list[int],
) -> InstanceBoxes | None:
    """Build one full-scene box trajectory for every selected global track."""
    if not track_ids:
        return None
    slot_of = {frame: slot for slot, frame in enumerate(frames)}
    num_tracks, num_frames = len(track_ids), len(frames)
    centers = np.zeros((num_tracks, num_frames, 3))
    dims = np.ones((num_tracks, num_frames, 3))
    rots = np.tile(np.eye(3), (num_tracks, num_frames, 1, 1))
    valid = np.zeros((num_tracks, num_frames), dtype=bool)
    for track_slot, track_id in enumerate(track_ids):
        for frame, (raw_pose, size) in metadata["instances"][track_id].items():
            if frame not in slot_of:
                continue
            frame_slot = slot_of[frame]
            obj_to_world = world_from @ raw_pose
            centers[track_slot, frame_slot] = obj_to_world[:3, 3]
            rots[track_slot, frame_slot] = obj_to_world[:3, :3].T
            dims[track_slot, frame_slot] = size
            valid[track_slot, frame_slot] = True
    times = dataset._normalised_times(frames).numpy()[None, :].repeat(
        num_tracks, axis=0
    )
    return InstanceBoxes(
        centers=torch.from_numpy(centers).float(),
        dims=torch.from_numpy(dims).float(),
        rots=torch.from_numpy(rots).float(),
        times=torch.from_numpy(times).float(),
        valid=torch.from_numpy(valid),
    )


def classify_stationary_tracks(
    metadata: dict,
    max_center_span: float | None,
    max_median_step: float,
) -> tuple[list[int], list[dict]]:
    """Classify annotated tracks from world-space bbox-centre motion."""
    if max_center_span is None:
        return [], []
    stationary = []
    records = []
    for track_id, per_frame in metadata["instances"].items():
        ordered = sorted(per_frame.items())
        centers = np.stack([pose[:3, 3] for _, (pose, _) in ordered])
        if len(centers) > 1:
            span = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
            steps = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
            median_step = float(np.median(steps))
        else:
            span = median_step = 0.0
        is_stationary = (
            span <= max_center_span and median_step <= max_median_step
        )
        if is_stationary:
            stationary.append(int(track_id))
        records.append({
            "track_id": int(track_id),
            "valid_frame_count": len(ordered),
            "center_span_m": span,
            "median_step_m": median_step,
            "reclassified_static": is_stationary,
        })
    return sorted(stationary), records


def rebase_dynamic_to_global_tracks(
    dynamic: Gaussians,
    local_boxes: InstanceBoxes,
    local_track_ids: list[int],
    global_boxes: InstanceBoxes,
    global_track_slot: dict[int, int],
) -> Gaussians:
    """Move clip-canonical dynamic Gaussians into scene-global track frames."""
    if dynamic.count == 0:
        return dynamic
    local_can_c, local_can_r = local_boxes.canonical()
    global_can_c, global_can_r = global_boxes.canonical()
    local_ids = dynamic.instance_id.long()
    global_ids = torch.tensor(
        [global_track_slot[local_track_ids[int(index)]] for index in local_ids.cpu()],
        device=dynamic.means.device,
        dtype=torch.long,
    )
    source_c = local_can_c[local_ids]
    source_r = local_can_r[local_ids]
    target_c = global_can_c[global_ids]
    target_r = global_can_r[global_ids]
    means = canonicalise(dynamic.means, source_c, source_r, target_c, target_r)
    canonical_rotation = torch.einsum("nji,njk->nik", target_r, source_r)
    quats = quaternion_multiply(
        matrix_to_quaternion(canonical_rotation), dynamic.quats
    )
    return Gaussians(
        means=means,
        quats=quats,
        scales=dynamic.scales,
        opacities=dynamic.opacities,
        colors=dynamic.colors,
        # Track validity, rather than a clip-local temporal envelope, controls
        # when a persistent actor is rendered.
        sigma=torch.full_like(dynamic.sigma, 1e6),
        times=torch.zeros_like(dynamic.times),
        instance_id=global_ids,
        extras=dynamic.extras,
    )


def global_dynamic_to_local_tracks(
    dynamic: Gaussians,
    local_boxes: InstanceBoxes,
    local_track_ids: list[int],
    global_boxes: InstanceBoxes,
    global_track_slot: dict[int, int],
) -> tuple[Gaussians, torch.Tensor]:
    """Move visible global-track Gaussians into the current clip canonical frames."""
    if dynamic.count == 0 or not local_track_ids:
        empty = dynamic.means.new_zeros((0,), dtype=torch.long)
        return select_gaussians(dynamic, empty), empty
    global_to_local = torch.full(
        (global_boxes.num_instances,), -1, dtype=torch.long, device=dynamic.means.device
    )
    for local_slot, track_id in enumerate(local_track_ids):
        global_to_local[global_track_slot[track_id]] = local_slot
    local_ids_all = global_to_local[dynamic.instance_id.long()]
    source_index = torch.nonzero(local_ids_all >= 0, as_tuple=False).flatten()
    if source_index.numel() == 0:
        return select_gaussians(dynamic, source_index), source_index
    selected = select_gaussians(dynamic, source_index)
    global_ids = selected.instance_id.long()
    local_ids = local_ids_all[source_index]
    global_c, global_r = global_boxes.canonical()
    local_c, local_r = local_boxes.canonical()
    means = canonicalise(
        selected.means,
        global_c[global_ids],
        global_r[global_ids],
        local_c[local_ids],
        local_r[local_ids],
    )
    canonical_rotation = torch.einsum(
        "nji,njk->nik", local_r[local_ids], global_r[global_ids]
    )
    quats = quaternion_multiply(
        matrix_to_quaternion(canonical_rotation), selected.quats
    )
    return Gaussians(
        means=means,
        quats=quats,
        scales=selected.scales,
        opacities=selected.opacities,
        colors=selected.colors,
        sigma=selected.sigma,
        times=selected.times,
        instance_id=local_ids,
        extras=selected.extras,
    ), source_index


def blend_dynamic_gaussians(
    old: Gaussians, update: Gaussians, alpha: float | torch.Tensor
) -> Gaussians:
    """Refresh persistent dynamic attributes while keeping canonical xyz fixed."""
    quat_update = torch.where(
        (old.quats * update.quats).sum(dim=-1, keepdim=True) < 0,
        -update.quats,
        update.quats,
    )
    extras = {}
    for name in set(old.extras) & set(update.extras):
        if name == "observations":
            extras[name] = old.extras[name] + 1
        elif name in (
            "best_position_observation_distance",
            "position_observations",
        ):
            extras[name] = old.extras[name]
        elif name == "best_observation_distance":
            extras[name] = torch.minimum(old.extras[name], update.extras[name])
        elif old.extras[name].is_floating_point():
            extras[name] = pointwise_lerp(
                old.extras[name], update.extras[name], alpha
            )
        else:
            extras[name] = old.extras[name]
    return Gaussians(
        means=old.means,
        quats=torch.nn.functional.normalize(
            pointwise_lerp(old.quats, quat_update, alpha), dim=-1
        ),
        scales=pointwise_lerp(
            old.scales.clamp_min(1e-8).log(),
            update.scales.clamp_min(1e-8).log(),
            alpha,
        ).exp(),
        opacities=torch.sigmoid(pointwise_lerp(
            torch.logit(old.opacities.clamp(1e-6, 1.0 - 1e-6)),
            torch.logit(update.opacities.clamp(1e-6, 1.0 - 1e-6)),
            alpha,
        )),
        colors=pointwise_lerp(old.colors, update.colors, alpha),
        sigma=torch.full_like(old.sigma, 1e6),
        times=torch.zeros_like(old.times),
        instance_id=old.instance_id,
        extras=extras,
    )


def merge_global_dynamic_gaussians(
    old: Gaussians | None,
    current: Gaussians,
    voxel_size: float,
    alpha: float,
) -> tuple[Gaussians, dict[str, int | float]]:
    """Fuse dynamics by ``(global track ID, canonical-space voxel)``."""
    if old is None or old.count == 0:
        return current, {
            "voxel_size_m": voxel_size,
            "before": int(current.count),
            "after": int(current.count),
            "updated": 0,
            "added": int(current.count),
        }
    if current.count == 0:
        return old, {
            "voxel_size_m": voxel_size,
            "before": int(old.count),
            "after": int(old.count),
            "updated": 0,
            "added": 0,
        }

    total = old.count + current.count
    means = torch.cat((old.means, current.means), dim=0)
    track = torch.cat((old.instance_id, current.instance_id), dim=0).long()
    coords = torch.floor(means / voxel_size).long()
    shifted = coords - coords.amin(dim=0, keepdim=True)
    spans = shifted.amax(dim=0) + 1
    key = (((track * spans[0] + shifted[:, 0]) * spans[1] + shifted[:, 1])
           * spans[2] + shifted[:, 2])
    _, inverse = torch.unique(key, sorted=True, return_inverse=True)
    unique_count = int(inverse.amax().item()) + 1
    old_rep = torch.full(
        (unique_count,), total, dtype=torch.long, device=means.device
    )
    current_rep = torch.full_like(old_rep, total)
    old_positions = torch.arange(old.count, device=means.device)
    current_positions = torch.arange(current.count, device=means.device) + old.count
    old_rep.scatter_reduce_(
        0, inverse[:old.count], old_positions, reduce="amin", include_self=True
    )
    current_rep.scatter_reduce_(
        0, inverse[old.count:], current_positions, reduce="amin", include_self=True
    )
    has_old = old_rep < old.count
    has_current = current_rep < total
    overlap = has_old & has_current
    update_overlaps = not (isinstance(alpha, float) and alpha == 0.0)
    if bool(overlap.any()) and update_overlaps:
        old_index = old_rep[overlap]
        current_index = current_rep[overlap] - old.count
        refreshed = blend_dynamic_gaussians(
            select_gaussians(old, old_index),
            select_gaussians(current, current_index),
            alpha,
        )
        old = replace_gaussians(old, old_index, refreshed)
    kept_old = select_gaussians(old, old_rep[has_old])
    new_only = (~has_old) & has_current
    parts = [kept_old]
    added = int(new_only.sum().item())
    if added:
        parts.append(select_gaussians(
            current, current_rep[new_only] - old.count
        ))
    output = concatenate_gaussians(parts)
    return output, {
        "voxel_size_m": voxel_size,
        "before": int(total),
        "after": int(output.count),
        "updated": int(overlap.sum().item()) if update_overlaps else 0,
        "added": added,
    }


def merge_global_dynamic_gaussians_nearest_update(
    old: Gaussians | None,
    current: Gaussians,
    association_radius: float,
    position_alpha: float,
    max_candidates_per_cell: int = 16,
    query_chunk_size: int = 8192,
) -> tuple[Gaussians, dict[str, int | float | list[float]]]:
    """Associate current dynamics to persistent points and update old xyz.

    Unlike voxel deduplication, this keeps every persistent point and its
    identity.  A current point searches the 27 neighbouring spatial-hash
    cells of the same global track.  Matched current points are observations
    of existing surface and are never appended.  At most one best current
    observation updates each old point; only unmatched current points are new
    surface and are appended.
    """
    if current.count == 0:
        output = current if old is None else old
        return output, {
            "policy": "nearest-update",
            "association_radius_m": association_radius,
            "position_alpha": position_alpha,
            "before": int(output.count),
            "after": int(output.count),
            "matched_current": 0,
            "distinct_old_matched": 0,
            "association_collisions": 0,
            "position_updated": 0,
            "added": 0,
            "position_displacement_m_p50_p90_p99_max": [0.0, 0.0, 0.0, 0.0],
        }

    current.extras.setdefault(
        "best_position_observation_distance",
        current.extras["best_observation_distance"].clone(),
    )
    current.extras.setdefault(
        "position_observations",
        torch.ones(
            current.count,
            dtype=torch.int32,
            device=current.means.device,
        ),
    )
    if old is None or old.count == 0:
        return current, {
            "policy": "nearest-update",
            "association_radius_m": association_radius,
            "position_alpha": position_alpha,
            "before": int(current.count),
            "after": int(current.count),
            "matched_current": 0,
            "distinct_old_matched": 0,
            "association_collisions": 0,
            "position_updated": 0,
            "added": int(current.count),
            "position_displacement_m_p50_p90_p99_max": [0.0, 0.0, 0.0, 0.0],
        }
    old.extras.setdefault(
        "best_position_observation_distance",
        old.extras["best_observation_distance"].clone(),
    )
    old.extras.setdefault(
        "position_observations",
        torch.ones(old.count, dtype=torch.int32, device=old.means.device),
    )

    old_coords = torch.floor(old.means / association_radius).long()
    current_coords = torch.floor(current.means / association_radius).long()
    combined_coords = torch.cat((old_coords, current_coords), dim=0)
    coord_min = combined_coords.amin(dim=0) - 1
    coord_max = combined_coords.amax(dim=0) + 1
    spans = coord_max - coord_min + 1

    def encode(track: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        shifted = coords - coord_min
        return (
            ((track.long() * spans[0] + shifted[..., 0]) * spans[1]
             + shifted[..., 1]) * spans[2]
            + shifted[..., 2]
        )

    old_key = encode(old.instance_id.long(), old_coords)
    old_quality = old.extras["best_position_observation_distance"]
    quality_order = torch.argsort(old_quality, stable=True)
    key_order = torch.argsort(old_key[quality_order], stable=True)
    old_order = quality_order[key_order]
    sorted_old_key = old_key[old_order]
    offsets = torch.cartesian_prod(
        *[
            torch.tensor((-1, 0, 1), device=old.means.device)
            for _ in range(3)
        ]
    )
    candidate_slots = torch.arange(
        max_candidates_per_cell, device=old.means.device
    )
    matched_current_parts = []
    matched_old_parts = []
    matched_distance_parts = []
    radius_squared = association_radius * association_radius

    for lo in range(0, current.count, query_chunk_size):
        hi = min(lo + query_chunk_size, current.count)
        count = hi - lo
        neighbour_coords = current_coords[lo:hi, None, :] + offsets[None, :, :]
        neighbour_track = current.instance_id[lo:hi, None].expand(-1, 27)
        neighbour_key = encode(neighbour_track, neighbour_coords).contiguous()
        left = torch.searchsorted(sorted_old_key, neighbour_key, right=False)
        right = torch.searchsorted(sorted_old_key, neighbour_key, right=True)
        positions = left.unsqueeze(-1) + candidate_slots.view(1, 1, -1)
        valid = positions < right.unsqueeze(-1)
        safe_positions = positions.clamp(max=old.count - 1)
        candidate_old = old_order[safe_positions]
        candidate_xyz = old.means[candidate_old]
        distance_squared = (
            candidate_xyz - current.means[lo:hi, None, None, :]
        ).square().sum(dim=-1)
        distance_squared.masked_fill_(~valid, float("inf"))
        flat_distance = distance_squared.view(count, -1)
        best_distance, best_flat = flat_distance.min(dim=1)
        matched = best_distance <= radius_squared
        if bool(matched.any()):
            flat_old = candidate_old.view(count, -1)
            matched_current_parts.append(
                torch.nonzero(matched, as_tuple=False).flatten() + lo
            )
            matched_old_parts.append(
                flat_old.gather(1, best_flat.unsqueeze(1)).squeeze(1)[matched]
            )
            matched_distance_parts.append(best_distance[matched])

    if matched_current_parts:
        matched_current = torch.cat(matched_current_parts)
        matched_old = torch.cat(matched_old_parts)
        matched_distance = torch.cat(matched_distance_parts)
    else:
        matched_current = torch.zeros(
            0, dtype=torch.long, device=current.means.device
        )
        matched_old = matched_current
        matched_distance = current.means.new_zeros((0,))

    matched_mask = torch.zeros(
        current.count, dtype=torch.bool, device=current.means.device
    )
    matched_mask[matched_current] = True
    distinct_old_matched = 0
    position_updated = 0
    displacements = current.means.new_zeros((0,))
    if matched_current.numel() > 0:
        current_quality = current.extras[
            "best_position_observation_distance"
        ][matched_current]
        # Quality is primary; geometric distance breaks equal-quality ties.
        order = torch.argsort(matched_distance, stable=True)
        order = order[torch.argsort(current_quality[order], stable=True)]
        representative_rank = torch.full(
            (old.count,),
            order.numel(),
            dtype=torch.long,
            device=old.means.device,
        )
        representative_rank.scatter_reduce_(
            0,
            matched_old[order],
            torch.arange(order.numel(), device=old.means.device),
            reduce="amin",
            include_self=True,
        )
        old_index = torch.nonzero(
            representative_rank < order.numel(), as_tuple=False
        ).flatten()
        chosen = order[representative_rank[old_index]]
        current_index = matched_current[chosen]
        distinct_old_matched = int(old_index.numel())
        improves_position = (
            current.extras["best_position_observation_distance"][current_index]
            < old.extras["best_position_observation_distance"][old_index]
        )
        old_index = old_index[improves_position]
        current_index = current_index[improves_position]
        if old_index.numel() > 0:
            previous = old.means[old_index].clone()
            old.means[old_index] = pointwise_lerp(
                old.means[old_index], current.means[current_index], position_alpha
            )
            displacements = torch.linalg.vector_norm(
                old.means[old_index] - previous, dim=-1
            )
            old.extras["best_position_observation_distance"][old_index] = (
                current.extras["best_position_observation_distance"][current_index]
            )
            old.extras["position_observations"][old_index] += 1
            position_updated = int(old_index.numel())

    unmatched = ~matched_mask
    added = int(unmatched.sum().item())
    output = concatenate_gaussians(
        [old, select_gaussians(current, unmatched)] if added else [old]
    )
    if displacements.numel() > 0:
        displacement_stats = [
            float(value)
            for value in torch.quantile(
                displacements.float(),
                torch.tensor((0.5, 0.9, 0.99), device=displacements.device),
            ).cpu()
        ] + [float(displacements.max().cpu())]
    else:
        displacement_stats = [0.0, 0.0, 0.0, 0.0]
    return output, {
        "policy": "nearest-update",
        "association_radius_m": association_radius,
        "position_alpha": position_alpha,
        "max_candidates_per_cell": max_candidates_per_cell,
        "before": int(old.count + current.count),
        "after": int(output.count),
        "matched_current": int(matched_current.numel()),
        "distinct_old_matched": distinct_old_matched,
        "association_collisions": int(
            matched_current.numel() - distinct_old_matched
        ),
        "position_updated": position_updated,
        "added": added,
        "position_displacement_m_p50_p90_p99_max": displacement_stats,
    }


def project_static_gaussians_to_sparse_depth(
    gaussians: Gaussians,
    intrinsics: torch.Tensor,
    cam_to_world: torch.Tensor,
    height: int,
    width: int,
    max_valid_pixels_per_view: torch.Tensor | None = None,
    chunk_size: int = 262_144,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
    """Z-buffer persistent Gaussian centres into sparse depth prompts.

    The complete persistent static map is projected into every context view.
    Each centre contributes to its nearest integer pixel and the nearest
    positive camera-z depth wins, matching the dataset LiDAR projection
    convention used by InfiniDepth_DepthSensor.
    """
    num_views = int(intrinsics.shape[0])
    world_to_cam = torch.linalg.inv(cam_to_world)
    flat_depth = torch.full(
        (num_views, height * width),
        float("inf"),
        dtype=torch.float32,
        device=gaussians.means.device,
    )
    flat_all = flat_depth.view(-1)
    view_offset = (
        torch.arange(num_views, device=gaussians.means.device).view(-1, 1)
        * (height * width)
    )

    for lo in range(0, gaussians.count, chunk_size):
        means = gaussians.means[lo:lo + chunk_size]
        points = means.unsqueeze(0).expand(num_views, -1, -1)
        uv, projected_depth, in_front = project_points(
            points, intrinsics, world_to_cam
        )
        pixel_x = uv[..., 0].round().long()
        pixel_y = uv[..., 1].round().long()
        valid = (
            in_front
            & (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
            & projected_depth.isfinite()
        )
        if bool(valid.any()):
            linear = view_offset + pixel_y * width + pixel_x
            flat_all.scatter_reduce_(
                0,
                linear[valid],
                projected_depth.float()[valid],
                reduce="amin",
                include_self=True,
            )

    projected_valid = flat_depth.isfinite()
    valid = projected_valid
    if max_valid_pixels_per_view is not None:
        if max_valid_pixels_per_view.numel() != num_views:
            raise ValueError(
                "Sparse-depth budget must contain one count per context view"
            )
        valid = torch.zeros_like(projected_valid)
        for view_index in range(num_views):
            candidates = torch.nonzero(
                projected_valid[view_index], as_tuple=False
            ).flatten()
            budget = int(max_valid_pixels_per_view[view_index].item())
            if budget <= 0 or candidates.numel() == 0:
                continue
            if candidates.numel() > budget:
                # Even spacing in raster order keeps the prompt deterministic
                # and spatially distributed without reading any new LiDAR.
                positions = torch.linspace(
                    0,
                    candidates.numel() - 1,
                    steps=budget,
                    device=candidates.device,
                ).round().long()
                candidates = candidates[positions]
            valid[view_index, candidates] = True

    depth = torch.where(valid, flat_depth, torch.zeros_like(flat_depth)).reshape(
        num_views, 1, height, width
    )
    mask = valid.reshape(num_views, 1, height, width).float()
    valid_pixels = int(valid.sum().item())
    return depth, mask, {
        "source_gaussian_count": int(gaussians.count),
        "projected_valid_pixels_before_sparsify": int(projected_valid.sum().item()),
        "valid_pixels": valid_pixels,
        "total_pixels": int(valid.numel()),
        "valid_fraction": valid_pixels / max(int(valid.numel()), 1),
    }


def voxel_deduplicate_static(
    parts: list[Gaussians], voxel_size: float, append: Gaussians | None = None,
    representative_policy: str = "first",
) -> tuple[Gaussians, dict[str, int | float]]:
    """Keep the first static Gaussian per voxel and optionally append dynamics."""
    nonempty = [part for part in parts if part.count > 0]
    if not nonempty:
        raise ValueError("At least one non-empty static Gaussian set is required")
    total = sum(part.count for part in nonempty)
    means = torch.cat([part.means for part in nonempty], dim=0)
    coords = torch.floor(means / voxel_size).to(torch.int64)
    shifted = coords - coords.amin(dim=0, keepdim=True)
    spans = shifted.amax(dim=0) + 1
    if int(spans[0]) * int(spans[1]) > torch.iinfo(torch.int64).max // max(int(spans[2]), 1):
        raise OverflowError("Voxel coordinate extent cannot be packed into int64")
    keys = (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2]
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    unique_count = int(inverse.amax()) + 1
    positions = torch.arange(total, device=means.device)
    representatives = torch.full(
        (unique_count,), total, dtype=torch.long, device=means.device
    )
    if representative_policy == "first":
        representatives.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    elif representative_policy == "closest":
        quality = torch.cat([part.extras["best_observation_distance"] for part in nonempty])
        quality = torch.nan_to_num(quality, nan=float("inf"), posinf=float("inf"))
        best = torch.full((unique_count,), float("inf"), device=quality.device, dtype=quality.dtype)
        best.scatter_reduce_(0, inverse, quality, reduce="amin", include_self=True)
        eligible = torch.where(quality == best[inverse], positions, torch.full_like(positions,total))
        representatives.scatter_reduce_(0, inverse, eligible, reduce="amin", include_self=True)
        del quality, best, eligible
    else:
        raise ValueError("Unsupported voxel representative policy")
    selected_indices = []
    offset = 0
    for part in nonempty:
        in_part = (representatives >= offset) & (representatives < offset + part.count)
        selected_indices.append(representatives[in_part] - offset)
        offset += part.count
    del means, coords, shifted, keys, inverse, positions, representatives

    output_parts = list(zip(nonempty, selected_indices))
    if append is not None and append.count > 0:
        output_parts.append((append, None))
    extra_names = set.intersection(*(set(part.extras) for part, _ in output_parts))

    def selected(tensor: torch.Tensor, index: torch.Tensor | None) -> torch.Tensor:
        return tensor if index is None else tensor[index]

    # Build this temporary off-device so the previous global set can be freed
    # before the next full-size GPU allocation.
    output = Gaussians(
        **{
            name: torch.cat(
                [selected(getattr(part, name), index).cpu() for part, index in output_parts],
                dim=0,
            )
            for name in _GAUSSIAN_FIELDS
        },
        extras={
            name: torch.cat(
                [selected(part.extras[name], index).cpu() for part, index in output_parts],
                dim=0,
            )
            for name in extra_names
        },
    )
    static_after = unique_count
    return output, {
        "policy": "first-per-voxel" if representative_policy == "first" else "closest-observation-per-voxel",
        "voxel_size_m": voxel_size,
        "before": int(total),
        "after": int(static_after),
        "removed": int(total - static_after),
    }


def build_queries_from_static_gaussians(
    gaussians: Gaussians,
    scene: SceneInput,
    depth: torch.Tensor,
    depth_threshold: float,
    chunk_size: int = 65_536,
    source_view_policy: str = "min-depth-error",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reproject persistent centres and rebuild their RGB/ray query fields."""
    height, width = scene.image_size
    world_to_cam = torch.linalg.inv(scene.cam_to_world)
    query_parts = []
    source_parts = []
    best_view_parts = []

    for lo in range(0, gaussians.count, chunk_size):
        hi = min(lo + chunk_size, gaussians.count)
        means = gaussians.means[lo:hi]
        points = means.unsqueeze(0).expand(scene.num_views, -1, -1)
        uv, projected_depth, in_front = project_points(points, scene.intrinsics, world_to_cam)
        in_bounds = (
            (uv[..., 0] >= 0)
            & (uv[..., 0] <= width - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= height - 1)
        )
        sampled_depth = sample_features(depth.unsqueeze(1), uv, height, width).squeeze(-1)
        depth_error = (projected_depth - sampled_depth).abs()
        valid = in_front & in_bounds & sampled_depth.isfinite() & (sampled_depth > 0)
        if source_view_policy == "min-depth-error":
            # Preserve the original source-camera choice exactly.
            score = depth_error.masked_fill(~valid, float("inf"))
            best_error, best_view = score.min(dim=0)
            observed = best_error <= depth_threshold
        elif source_view_policy == "nearest-depth-valid":
            depth_valid = valid & (depth_error <= depth_threshold)
            distance = torch.linalg.vector_norm(
                points - scene.cam_to_world[:, None, :3, 3], dim=-1
            )
            score = distance.masked_fill(~depth_valid, float("inf"))
            best_distance, best_view = score.min(dim=0)
            observed = best_distance.isfinite()
        else:
            raise ValueError(f"Unsupported static source-view policy: {source_view_policy}")
        if not bool(observed.any()):
            continue

        means = means[observed]
        best_view = best_view[observed]
        uv = uv[:, observed]
        sampled_rgb = sample_features(scene.images, uv, height, width)
        gather_index = best_view.view(1, -1, 1).expand(1, -1, 3)
        rgb = torch.gather(sampled_rgb, 0, gather_index).squeeze(0)
        origins = scene.cam_to_world[best_view, :3, 3]
        directions = torch.nn.functional.normalize(means - origins, dim=-1)
        rays = torch.cat((directions, torch.cross(origins, directions, dim=-1)), dim=-1)
        query_parts.append(torch.cat((means, rgb, rays), dim=-1))
        source_parts.append(torch.nonzero(observed, as_tuple=False).flatten() + lo)
        best_view_parts.append(best_view)

    if not query_parts:
        empty = gaussians.means.new_zeros((0,), dtype=torch.long)
        return {
            "sample_index": empty,
            "queries": gaussians.means.new_zeros((0, 12)),
            "world_xyz": gaussians.means.new_zeros((0, 3)),
            "base_xyz": gaussians.means.new_zeros((0, 3)),
            "view_id": empty,
            "instance_id": empty,
            "times": gaussians.times.new_zeros((0,)),
            "box_index": empty,
        }, empty

    queries = torch.cat(query_parts, dim=0)
    source_index = torch.cat(source_parts, dim=0)
    best_view = torch.cat(best_view_parts, dim=0)
    return {
        "sample_index": torch.full_like(source_index, -1),
        "queries": queries,
        "world_xyz": queries[:, :3],
        "base_xyz": queries[:, :3],
        "view_id": best_view,
        "instance_id": torch.full_like(best_view, -1),
        "times": scene.times[best_view],
        "box_index": scene.frame_index[best_view],
    }, source_index


def build_queries_from_dynamic_gaussians(
    gaussians: Gaussians,
    scene: SceneInput,
    depth: torch.Tensor,
    depth_threshold: float,
    chunk_size: int = 65_536,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reproject clip-canonical dynamic centres and rebuild RGB/ray queries."""
    if scene.boxes is None:
        raise ValueError("Dynamic forward refresh requires clip-local boxes")
    height, width = scene.image_size
    world_to_cam = torch.linalg.inv(scene.cam_to_world)
    query_parts = []
    world_parts = []
    instance_parts = []
    source_parts = []
    best_view_parts = []
    can_c, can_r = scene.boxes.canonical()

    for lo in range(0, gaussians.count, chunk_size):
        hi = min(lo + chunk_size, gaussians.count)
        base_xyz = gaussians.means[lo:hi]
        instance_id = gaussians.instance_id[lo:hi].long()
        view_count = scene.num_views
        inst_v = instance_id.view(1, -1).expand(view_count, -1)
        slot_v = scene.frame_index.view(-1, 1).expand(-1, instance_id.numel())
        points = decanonicalise(
            base_xyz.unsqueeze(0).expand(view_count, -1, 3),
            scene.boxes.centers[inst_v, slot_v],
            scene.boxes.rots[inst_v, slot_v],
            can_c[instance_id].unsqueeze(0),
            can_r[instance_id].unsqueeze(0),
        )
        point_valid = scene.boxes.valid[inst_v, slot_v]
        uv, projected_depth, in_front = project_points(
            points, scene.intrinsics, world_to_cam
        )
        in_bounds = (
            (uv[..., 0] >= 0)
            & (uv[..., 0] <= width - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= height - 1)
        )
        sampled_depth = sample_features(
            depth.unsqueeze(1), uv, height, width
        ).squeeze(-1)
        depth_error = (projected_depth - sampled_depth).abs()
        valid = (
            point_valid
            & in_front
            & in_bounds
            & sampled_depth.isfinite()
            & (sampled_depth > 0)
        )
        score = depth_error.masked_fill(~valid, float("inf"))
        best_error, best_view = score.min(dim=0)
        observed = best_error <= depth_threshold
        if not bool(observed.any()):
            continue

        base_xyz = base_xyz[observed]
        instance_id = instance_id[observed]
        best_view = best_view[observed]
        uv = uv[:, observed]
        points = points[:, observed]
        sampled_rgb = sample_features(scene.images, uv, height, width)
        rgb = torch.gather(
            sampled_rgb,
            0,
            best_view.view(1, -1, 1).expand(1, -1, 3),
        ).squeeze(0)
        world_xyz = torch.gather(
            points,
            0,
            best_view.view(1, -1, 1).expand(1, -1, 3),
        ).squeeze(0)
        origins = scene.cam_to_world[best_view, :3, 3]
        directions = torch.nn.functional.normalize(
            world_xyz - origins, dim=-1
        )
        rays = torch.cat(
            (directions, torch.cross(origins, directions, dim=-1)), dim=-1
        )
        query_parts.append(torch.cat((base_xyz, rgb, rays), dim=-1))
        world_parts.append(world_xyz)
        instance_parts.append(instance_id)
        source_parts.append(
            torch.nonzero(observed, as_tuple=False).flatten() + lo
        )
        best_view_parts.append(best_view)

    if not query_parts:
        empty = gaussians.means.new_zeros((0,), dtype=torch.long)
        return {
            "sample_index": empty,
            "queries": gaussians.means.new_zeros((0, 12)),
            "world_xyz": gaussians.means.new_zeros((0, 3)),
            "base_xyz": gaussians.means.new_zeros((0, 3)),
            "view_id": empty,
            "instance_id": empty,
            "times": gaussians.times.new_zeros((0,)),
            "box_index": empty,
        }, empty

    queries = torch.cat(query_parts, dim=0)
    world_xyz = torch.cat(world_parts, dim=0)
    instance_id = torch.cat(instance_parts, dim=0)
    source_index = torch.cat(source_parts, dim=0)
    best_view = torch.cat(best_view_parts, dim=0)
    return {
        "sample_index": torch.full_like(source_index, -1),
        "queries": queries,
        "world_xyz": world_xyz,
        "base_xyz": queries[:, :3],
        "view_id": best_view,
        "instance_id": instance_id,
        "times": scene.times[best_view],
        "box_index": scene.frame_index[best_view],
    }, source_index


def concatenate_sampled(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat([part[name] for part in parts], dim=0) for name in parts[0]}


def gaussian_scale_stats(scales: torch.Tensor) -> dict[str, float | list[float]]:
    """Compact diagnostics for decoded positive Gaussian axis scales."""
    scales = scales.detach().float()
    shortest = scales.amin(dim=-1)
    longest = scales.amax(dim=-1)
    ratio = longest / shortest.clamp_min(1e-12)
    quantiles = torch.tensor((0.5, 0.9, 0.99), device=scales.device)

    def q(values: torch.Tensor) -> list[float]:
        return [float(value) for value in torch.quantile(values, quantiles).cpu()]

    return {
        "shortest_axis_m_p50_p90_p99": q(shortest),
        "longest_axis_m_p50_p90_p99": q(longest),
        "axis_ratio_p50_p90_p99": q(ratio),
        "axis_ratio_gt_10_fraction": float((ratio > 10.0).float().mean().cpu()),
        "axis_ratio_gt_100_fraction": float((ratio > 100.0).float().mean().cpu()),
    }


def postprocess_gaussian_scales(
    scales: torch.Tensor,
    mode: str,
    isotropic_max_scale: float | None,
) -> tuple[torch.Tensor, dict]:
    """Apply an inference-only covariance ablation without touching other attributes.

    ``isotropic-trace`` replaces the three axes by
    sqrt((sx^2 + sy^2 + sz^2) / 3), preserving trace of the 3D covariance
    before an optional radius cap. ``isotropic-volume`` uses the geometric mean
    and therefore preserves the determinant/volume before the optional cap.
    ``axis-cap`` retains decoded orientation and shorter axes, capping each
    axis independently; it bounds spatial extent without forcing a sphere.
    """
    before = gaussian_scale_stats(scales)
    if mode == "none":
        transformed = scales
        radius_before_cap = None
    elif mode == "isotropic-trace":
        radius_before_cap = scales.float().square().mean(dim=-1, keepdim=True).sqrt()
        transformed = radius_before_cap
    elif mode == "isotropic-volume":
        radius_before_cap = scales.float().clamp_min(1e-12).log().mean(dim=-1, keepdim=True).exp()
        transformed = radius_before_cap
    elif mode == "axis-cap":
        if isotropic_max_scale is None or isotropic_max_scale <= 0:
            raise ValueError("axis-cap requires a positive maximum axis scale")
        radius_before_cap = scales.float().amax(dim=-1, keepdim=True)
        transformed = scales.float()
    else:
        raise ValueError(f"Unsupported Gaussian scale postprocess mode: {mode}")

    capped_fraction = 0.0
    radius_quantiles = None
    if radius_before_cap is not None:
        quantiles = torch.tensor((0.5, 0.9, 0.99), device=scales.device)
        radius_quantiles = [
            float(value)
            for value in torch.quantile(radius_before_cap.squeeze(-1), quantiles).cpu()
        ]
        if isotropic_max_scale is not None:
            capped_fraction = float(
                (radius_before_cap > isotropic_max_scale).float().mean().cpu()
            )
            transformed = transformed.clamp(max=isotropic_max_scale)
        transformed = transformed.expand_as(scales).to(dtype=scales.dtype).contiguous()

    return transformed, {
        "mode": mode,
        "isotropic_max_scale_m": isotropic_max_scale,
        "scale_cap_semantics": "each_decoded_axis" if mode == "axis-cap" else "isotropic_radius",
        "radius_before_cap_m_p50_p90_p99": radius_quantiles,
        "capped_fraction": capped_fraction,
        "before": before,
        "after": gaussian_scale_stats(transformed),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--camera-interpolation-factor",
        type=int,
        default=1,
        help=(
            "Render this many evenly spaced camera-pose samples per source "
            "frame interval. Dynamic time and Gaussian state remain held at "
            "the lower source frame."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-ply", action="store_true")
    parser.add_argument(
        "--skip-frame-render",
        action="store_true",
        help="Skip per-window frame rendering when only runtime payloads are needed.",
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=("original", "global-gaussians"),
        default="original",
        help="Keep independent clip reconstructions or maintain one persistent Gaussian set.",
    )
    parser.add_argument(
        "--clip-order",
        choices=("forward", "reverse"),
        default="forward",
        help="Process clips from first to last or last to first; output frames stay chronological.",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=None,
        help="Frames between clip starts; defaults to clip_length. Smaller values create overlap.",
    )
    parser.add_argument(
        "--pad-final-window",
        action="store_true",
        help="Repeat the final frame when a tail window is shorter than window_length.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Frames covered by each clip; defaults to the dataset clip_length.",
    )
    parser.add_argument(
        "--context-frame-stride",
        type=int,
        default=None,
        help="Frame spacing between context timestamps; defaults to the dataset frame_stride.",
    )
    parser.add_argument("--global-voxel-size", type=float, default=0.05)
    parser.add_argument("--global-static-voxel-representative", choices=("first","closest"), default="first")
    parser.add_argument("--global-depth-threshold", type=float, default=1.0)
    parser.add_argument("--global-update-alpha", type=float, default=0.5)
    parser.add_argument(
        "--global-query-origin", choices=("scene", "clip", "clip-rigid"), default="scene",
        help="scene preserves original queries; clip translates queries; clip-rigid expresses static queries in the clip ego frame. Outputs remain in their original scene/canonical frames.",
    )
    parser.add_argument(
        "--global-update-distance-policy",
        choices=("uniform", "smoothstep", "closest-smoothstep"),
        default="uniform",
        help=(
            "Weight historical Gaussian attribute refreshes by camera range. "
            "closest-smoothstep additionally accepts only observations that "
            "improve each Gaussian's persistent best range."
        ),
    )
    parser.add_argument("--global-update-near-distance", type=float, default=12.0)
    parser.add_argument("--global-update-far-distance", type=float, default=60.0)
    parser.add_argument(
        "--global-update-min-improvement",
        type=float,
        default=0.0,
        help="Required range improvement in metres for closest-smoothstep.",
    )
    parser.add_argument(
        "--global-static-source-view",
        choices=("min-depth-error", "nearest-depth-valid"),
        default="min-depth-error",
        help="Select the source RGB/ray camera for old static queries. The optional "
        "nearest policy chooses the closest camera among depth-consistent views; "
        "all clip views still participate in neural feature fusion.",
    )
    parser.add_argument(
        "--global-static-refresh-fields", choices=("all", "colors"), default="all",
        help="Attributes refreshed for observed old static points; colors keeps geometry and opacity frozen.",
    )
    parser.add_argument("--global-static-offset-gain", type=float, default=1.0,
                        help="Scale the static decoder XYZ residual; 0 retains depth-lifted query centres.")
    parser.add_argument("--global-static-fusion-depth-tolerance", type=float, default=0.0,
                        help="Optional static view-depth consistency gate in metres; 0 preserves learned-only fusion.")
    parser.add_argument("--global-static-view-bank", choices=("disabled", "nearest4"), default="disabled",
                        help="Optional scene-wide closest reliable four-frame feature-only pass before runtime export.")
    parser.add_argument("--global-static-view-bank-alpha", type=float, default=1.0)
    parser.add_argument("--global-static-view-bank-depth-tolerance", type=float, default=1.0)
    parser.add_argument("--global-static-view-bank-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--global-growth-policy",
        choices=(
            "baseline",
            "visibility-difference-frozen-position",
            "two-clip-mutable-frozen-position",
            "three-clip-mutable-frozen-position",
            "unrestricted-frozen-position",
        ),
        default="baseline",
        help=(
            "baseline keeps AV-v1-m growth. visibility-difference-frozen-position "
            "freezes persistent means and only admits new static Gaussians outside "
            "the union of all previously visible clip regions. "
            "two-clip-mutable-frozen-position keeps persistent means frozen, permits "
            "new points and attribute refreshes in the current and immediately "
            "previous clip regions, and protects regions seen two or more clips ago. "
            "three-clip-mutable-frozen-position extends the mutable region to the "
            "two immediately previous clips and protects regions seen three or more "
            "clips ago. unrestricted-frozen-position admits new points across all "
            "clip ages before voxel deduplication, while keeping persistent means frozen."
        ),
    )
    parser.add_argument(
        "--global-dynamic-policy",
        choices=("per-window", "persistent-tracks"),
        default=None,
        help=(
            "per-window preserves the original clip-local dynamic replacement. "
            "persistent-tracks maps DriveStudio scene-level track IDs into one "
            "global box graph and fuses dynamic Gaussians in track-canonical voxels. "
            "Defaults to persistent-tracks for global-gaussians and per-window "
            "for the original reconstruction mode."
        ),
    )
    parser.add_argument(
        "--global-dynamic-forward-refresh",
        action="store_true",
        help=(
            "For persistent tracks, reproject every visible historical dynamic "
            "Gaussian into each new clip and pass it through RGB/depth fusion "
            "and Gaussian decoding before global canonical merging."
        ),
    )
    parser.add_argument(
        "--global-dynamic-merge-policy",
        choices=("voxel-frozen", "nearest-update"),
        default="voxel-frozen",
        help=(
            "voxel-frozen preserves the historical exact-voxel merge with "
            "immutable canonical xyz. nearest-update associates current points "
            "to same-track persistent neighbours, updates old canonical xyz "
            "from closer observations, and appends only unmatched points."
        ),
    )
    parser.add_argument(
        "--global-dynamic-association-radius",
        type=float,
        default=0.05,
        help="Same-track nearest-neighbour radius for nearest-update.",
    )
    parser.add_argument(
        "--global-dynamic-position-alpha",
        type=float,
        default=1.0,
        help="Lerp weight used when a closer observation updates old dynamic xyz.",
    )
    parser.add_argument(
        "--global-dynamic-max-candidates-per-cell",
        type=int,
        default=16,
        help="Quality-sorted old candidates retained per spatial-hash cell.",
    )
    parser.add_argument(
        "--sparse-depth-source",
        choices=("lidar", "global-gaussian-feedback"),
        default="lidar",
        help=(
            "Sparse prompt source for InfiniDepth. global-gaussian-feedback uses "
            "LiDAR only to bootstrap the first processed clip, then projects the "
            "accumulated static global Gaussian centres into later context views."
        ),
    )
    parser.add_argument(
        "--gaussian-scale-postprocess",
        choices=("none", "isotropic-trace", "isotropic-volume", "axis-cap"),
        default="none",
        help="Inference-only scale ablation applied after Gaussian decoding.",
    )
    parser.add_argument(
        "--isotropic-max-scale",
        type=float,
        default=None,
        help="Optional maximum isotropic radius in metres after scale postprocessing.",
    )
    parser.add_argument(
        "--global-static-source-pixel-radius", type=float, default=0.0,
        help="Optional static sphere scale = source depth * pixel radius / geometric-mean focal length; zero disables.",
    )
    parser.add_argument(
        "--global-static-source-min-scale", type=float, default=0.005,
        help="Physical lower bound in metres when source-pixel static spheres are enabled.",
    )
    parser.add_argument(
        "--save-runtime-payload",
        default=None,
        help=(
            "Save final 128-D static Gaussians and per-window dynamic Gaussians "
            "for faithful post-hoc rendering."
        ),
    )
    parser.add_argument(
        "--cache-overlap-image-networks",
        action="store_true",
        help=(
            "Cache frozen depth predictions and image-backbone features by "
            "scene frame/camera so overlapping windows do not recompute them."
        ),
    )
    parser.add_argument(
        "--exclude-ego-mask",
        action="store_true",
        help="Exclude dataset ego-mask pixels from Gaussian query sampling.",
    )
    parser.add_argument(
        "--stationary-track-max-span",
        type=float,
        default=None,
        help=(
            "Optional scene-wide bbox-centre span threshold in metres. Tracks "
            "below it and --stationary-track-max-median-step are removed from "
            "the scene graph before reconstruction, so they enter the static map."
        ),
    )
    parser.add_argument(
        "--stationary-track-max-median-step",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    if args.global_dynamic_policy is None:
        args.global_dynamic_policy = (
            "persistent-tracks"
            if args.reconstruction_mode == "global-gaussians"
            else "per-window"
        )

    if args.isotropic_max_scale is not None and args.isotropic_max_scale <= 0:
        parser.error("--isotropic-max-scale must be positive")
    if args.global_dynamic_association_radius <= 0:
        parser.error("--global-dynamic-association-radius must be positive")
    if not 0.0 < args.global_dynamic_position_alpha <= 1.0:
        parser.error("--global-dynamic-position-alpha must be in (0, 1]")
    if args.global_dynamic_max_candidates_per_cell <= 0:
        parser.error("--global-dynamic-max-candidates-per-cell must be positive")
    if (
        args.stationary_track_max_span is not None
        and args.stationary_track_max_span <= 0
    ):
        parser.error("--stationary-track-max-span must be positive")
    if args.stationary_track_max_median_step < 0:
        parser.error("--stationary-track-max-median-step must be non-negative")
    if args.camera_interpolation_factor <= 0:
        parser.error("--camera-interpolation-factor must be positive")
    if args.gaussian_scale_postprocess == "none" and args.isotropic_max_scale is not None:
        parser.error("--isotropic-max-scale requires an isotropic scale postprocess mode")
    if args.gaussian_scale_postprocess == "axis-cap" and args.isotropic_max_scale is None:
        parser.error("axis-cap requires --isotropic-max-scale as the maximum decoded axis length")
    if args.global_static_source_pixel_radius < 0:
        parser.error("--global-static-source-pixel-radius must be non-negative")
    if args.global_static_source_pixel_radius > 0 and (
        args.gaussian_scale_postprocess != "isotropic-trace"
        or args.isotropic_max_scale is None
        or not 0 < args.global_static_source_min_scale <= args.isotropic_max_scale
    ):
        parser.error("Source-pixel spheres require isotropic-trace and 0 < min-scale <= isotropic-max-scale")
    if args.global_voxel_size <= 0:
        parser.error("--global-voxel-size must be positive")
    if args.global_depth_threshold <= 0:
        parser.error("--global-depth-threshold must be positive")
    if not 0 <= args.global_static_offset_gain <= 1:
        parser.error("--global-static-offset-gain must be in [0, 1]")
    if args.global_static_fusion_depth_tolerance < 0:
        parser.error("--global-static-fusion-depth-tolerance must be non-negative")
    if not 0 <= args.global_static_view_bank_alpha <= 1:
        parser.error("--global-static-view-bank-alpha must be in [0, 1]")
    if args.global_static_view_bank_depth_tolerance <= 0 or args.global_static_view_bank_chunk_size <= 0:
        parser.error("Static view bank requires positive depth tolerance and chunk size")
    if args.global_static_view_bank != "disabled" and (
        args.reconstruction_mode != "global-gaussians" or not args.cache_overlap_image_networks
        or args.save_runtime_payload is None or not args.skip_frame_render
    ):
        parser.error("Static view bank requires global-gaussians, overlap cache, runtime payload export and skipped per-clip rendering")
    if not 0 <= args.global_update_alpha <= 1:
        parser.error("--global-update-alpha must be in [0, 1]")
    if args.global_update_near_distance < 0:
        parser.error("--global-update-near-distance must be non-negative")
    if args.global_update_far_distance <= args.global_update_near_distance:
        parser.error(
            "--global-update-far-distance must exceed the near distance"
        )
    if args.global_update_min_improvement < 0:
        parser.error("--global-update-min-improvement must be non-negative")
    if (
        args.sparse_depth_source == "global-gaussian-feedback"
        and args.reconstruction_mode != "global-gaussians"
    ):
        parser.error(
            "--sparse-depth-source global-gaussian-feedback requires "
            "--reconstruction-mode global-gaussians"
        )
    if (
        args.global_growth_policy != "baseline"
        and args.reconstruction_mode != "global-gaussians"
    ):
        parser.error(
            "Non-baseline --global-growth-policy requires "
            "--reconstruction-mode global-gaussians"
        )
    if (
        args.global_dynamic_policy != "per-window"
        and args.reconstruction_mode != "global-gaussians"
    ):
        parser.error(
            "--global-dynamic-policy persistent-tracks requires "
            "--reconstruction-mode global-gaussians"
        )
    if (
        args.global_dynamic_forward_refresh
        and args.global_dynamic_policy != "persistent-tracks"
    ):
        parser.error(
            "--global-dynamic-forward-refresh requires "
            "--global-dynamic-policy persistent-tracks"
        )
    if (
        args.cache_overlap_image_networks
        and args.sparse_depth_source != "lidar"
    ):
        parser.error(
            "--cache-overlap-image-networks currently requires "
            "--sparse-depth-source lidar so each view has a stable prompt identity"
        )

    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    output_dir = Path(args.out).resolve()
    predicted_dir = output_dir / "frames" / "predicted"
    comparison_dir = output_dir / "frames" / "comparison"
    gaussian_dir = output_dir / "gaussians"
    runtime_payload_dir = None
    if args.save_runtime_payload is not None:
        runtime_payload_dir = Path(args.save_runtime_payload)
        if not runtime_payload_dir.is_absolute():
            runtime_payload_dir = output_dir / runtime_payload_dir
        runtime_payload_dir = runtime_payload_dir.resolve()
        runtime_payload_dir.mkdir(parents=True, exist_ok=True)
    for directory in (predicted_dir, comparison_dir, gaussian_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for PointForward scene delivery")
    dataset = build_dataset(config, args.split)
    window_length = dataset.clip_length if args.window_length is None else args.window_length
    context_frame_stride = (
        dataset.frame_stride
        if args.context_frame_stride is None
        else args.context_frame_stride
    )
    window_step = window_length if args.window_step is None else args.window_step
    if window_length <= 0:
        parser.error("--window-length must be positive")
    if context_frame_stride <= 0:
        parser.error("--context-frame-stride must be positive")
    if not 0 < window_step <= window_length:
        parser.error("--window-step must be in [1, window_length]")
    minimum_window_frames = (dataset.num_context - 1) * context_frame_stride + 1
    if window_length < minimum_window_frames:
        parser.error(
            "--window-length must cover all context timestamps: "
            f"need at least {minimum_window_frames} frames"
        )
    matches = [metadata for metadata in dataset.scenes if metadata["name"] == args.scene]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one scene {args.scene!r}, found {len(matches)}")
    metadata = matches[0]
    evaluation_metadata = metadata
    available_frames = metadata["frames"]
    first = available_frames[0] if args.start_frame is None else args.start_frame
    if first not in available_frames:
        raise ValueError(f"Start frame {first} is not available")
    first_index = available_frames.index(first)
    selected_frames = available_frames[first_index:]
    if args.num_frames is not None:
        selected_frames = selected_frames[: args.num_frames]
    if not selected_frames:
        raise ValueError("No frames selected")

    world_from = np.linalg.inv(dataset._ego_pose(metadata, selected_frames[0]))
    stationary_track_ids, stationary_track_records = classify_stationary_tracks(
        metadata,
        args.stationary_track_max_span,
        args.stationary_track_max_median_step,
    )
    stationary_metadata = None
    if stationary_track_ids:
        metadata = dict(metadata)
        stationary_set = set(stationary_track_ids)
        stationary_metadata = dict(evaluation_metadata)
        stationary_metadata["instances"] = {
            track_id: value
            for track_id, value in evaluation_metadata["instances"].items()
            if int(track_id) in stationary_set
        }
        metadata["instances"] = {
            track_id: value
            for track_id, value in evaluation_metadata["instances"].items()
            if int(track_id) not in stationary_set
        }
    model = build_model(config, device, args.ckpt)
    height, width = dataset.resolution
    num_cameras = len(dataset.cameras)
    timings = []
    windows = []
    frame_output_index = 0
    peak_allocated = 0
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    global_gaussians = None
    feedback_prompt_budget = None
    visibility_history = []
    depth_cache: dict[tuple[int, int], torch.Tensor] = {}
    image_feature_cache: dict[tuple[int, int], torch.Tensor] = {}
    image_network_cache_hits = 0
    image_network_cache_misses = 0
    frame_to_output_index = {frame: index for index, frame in enumerate(selected_frames)}
    if window_step == window_length:
        chronological_offsets = list(range(0, len(selected_frames), window_length))
    else:
        if len(selected_frames) < minimum_window_frames:
            raise ValueError(
                f"Overlapping windows require at least {minimum_window_frames} frames"
            )
        final_offset = len(selected_frames) - minimum_window_frames
        chronological_offsets = list(range(0, final_offset + 1, window_step))
        if chronological_offsets[-1] != final_offset:
            chronological_offsets.append(final_offset)
    owned_end_by_offset = {
        offset: (
            chronological_offsets[index + 1]
            if index + 1 < len(chronological_offsets)
            else len(selected_frames)
        )
        for index, offset in enumerate(chronological_offsets)
    }
    window_offsets = chronological_offsets.copy()
    if args.clip_order == "reverse":
        window_offsets.reverse()

    global_frame_slot = {
        frame: index for index, frame in enumerate(selected_frames)
    }
    global_normalized_times = dataset._normalised_times(selected_frames)
    window_tracks_by_offset = {}
    global_track_ids = []
    global_track_slot = {}
    global_boxes_cpu = None
    global_boxes = None
    evaluation_track_ids = sorted(int(value) for value in evaluation_metadata["instances"])
    evaluation_boxes_cpu = build_global_track_boxes(
        dataset,
        evaluation_metadata,
        selected_frames,
        world_from,
        evaluation_track_ids,
    )
    if args.global_dynamic_policy == "persistent-tracks":
        window_tracks_by_offset = {
            offset: window_track_ids(
                dataset,
                metadata,
                selected_frames[offset:offset + window_length],
            )
            for offset in chronological_offsets
        }
        global_track_ids = sorted({
            track_id
            for track_ids in window_tracks_by_offset.values()
            for track_id in track_ids
        })
        global_track_slot = {
            track_id: index for index, track_id in enumerate(global_track_ids)
        }
        global_boxes_cpu = build_global_track_boxes(
            dataset, metadata, selected_frames, world_from, global_track_ids
        )
        global_boxes = boxes_to(global_boxes_cpu, device)

    for window_index, offset in enumerate(window_offsets):
        window = selected_frames[offset:offset + window_length]
        if args.pad_final_window and len(window) < window_length and window:
            window = window + [window[-1]] * (window_length - len(window))
        owned_frames = selected_frames[offset:owned_end_by_offset[offset]]
        slot_of = {frame: slot for slot, frame in enumerate(window)}
        normalized_times = dataset._normalised_times(window)
        context_frames = sorted({
            window[min(index * context_frame_stride, len(window) - 1)]
            for index in range(dataset.num_context)
        })
        uses_feedback_prompt = (
            args.sparse_depth_source == "global-gaussian-feedback"
            and global_gaussians is not None
        )
        context = dataset._collect(
            metadata, context_frames, slot_of, normalized_times,
            dataset.resolution, world_from,
            lidar=dataset.load_lidar and not uses_feedback_prompt,
        )
        boxes_cpu = dataset._build_boxes(metadata, window, world_from)
        boxes = boxes_to(boxes_cpu, device)
        stationary_boxes_cpu = (
            dataset._build_boxes(stationary_metadata, window, world_from)
            if stationary_metadata is not None else None
        )
        stationary_boxes = boxes_to(stationary_boxes_cpu, device)
        local_track_ids = window_tracks_by_offset.get(offset, [])
        if (
            args.global_dynamic_policy == "persistent-tracks"
            and boxes is not None
            and boxes.num_instances != len(local_track_ids)
        ):
            raise RuntimeError(
                "Global track ID recovery disagrees with clip-local box ordering"
            )

        old_static_count = 0
        old_static_refresh_candidates = 0
        observed_old_count = 0
        rejected_old_static_by_distance = 0
        old_dynamic_refresh_candidates = 0
        observed_old_dynamic_count = 0
        rejected_old_dynamic_by_distance = 0
        old_static = None
        old_dynamic = None
        old_source_index = None
        old_dynamic_source_index = None
        static_refresh_alpha = None
        dynamic_refresh_alpha = None
        if args.reconstruction_mode == "global-gaussians" and global_gaussians is not None:
            old_static = persistent_static(global_gaussians)
            if args.global_dynamic_policy == "persistent-tracks":
                old_dynamic = select_gaussians(
                    global_gaussians, global_gaussians.instance_id >= 0
                )
            global_gaussians = None
            old_static_count = old_static.count

        scene = SceneInput(
            images=context["images"].to(device),
            intrinsics=context["intrinsics"].to(device),
            cam_to_world=context["cam_to_world"].to(device),
            times=context["times"].to(device),
            frame_index=context["frame_index"].to(device),
            boxes=boxes,
            stationary_boxes=stationary_boxes,
            sparse_depth=context.get("sparse_depth", None).to(device) if "sparse_depth" in context else None,
            sparse_mask=context.get("sparse_mask", None).to(device) if "sparse_mask" in context else None,
            ego_mask=context.get("ego_mask", None).to(device) if "ego_mask" in context else None,
        )
        if uses_feedback_prompt:
            scene.sparse_depth, scene.sparse_mask, sparse_depth_stats = (
                project_static_gaussians_to_sparse_depth(
                    old_static,
                    scene.intrinsics,
                    scene.cam_to_world,
                    height,
                    width,
                    max_valid_pixels_per_view=feedback_prompt_budget,
                )
            )
            sparse_depth_stats["source"] = "global-gaussian-feedback"
        else:
            valid_pixels = (
                int(scene.sparse_mask.sum().item())
                if scene.sparse_mask is not None
                else 0
            )
            total_pixels = (
                int(scene.sparse_mask.numel())
                if scene.sparse_mask is not None
                else scene.num_views * height * width
            )
            sparse_depth_stats = {
                "source": (
                    "lidar-bootstrap"
                    if args.sparse_depth_source == "global-gaussian-feedback"
                    else "lidar"
                ),
                "source_gaussian_count": 0,
                "valid_pixels": valid_pixels,
                "total_pixels": total_pixels,
                "valid_fraction": valid_pixels / max(total_pixels, 1),
            }
            if args.sparse_depth_source == "global-gaussian-feedback":
                if scene.sparse_mask is None:
                    raise RuntimeError(
                        "The bootstrap clip has no LiDAR prompt from which to "
                        "derive the Gaussian-feedback sparsity budget"
                    )
                feedback_prompt_budget = scene.sparse_mask.reshape(
                    scene.num_views, -1
                ).sum(dim=1).long()
        # Keep each clip's sampling identical to the forward baseline even when
        # the processing order is reversed.
        generator = torch.Generator(device=device).manual_seed(
            args.seed + offset // window_step
        )
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        view_keys = [
            (int(frame), int(camera))
            for frame in context_frames
            for camera in dataset.cameras
        ]
        if args.cache_overlap_image_networks:
            missing_indices = [
                index for index, key in enumerate(view_keys)
                if key not in depth_cache or key not in image_feature_cache
            ]
            image_network_cache_hits += len(view_keys) - len(missing_indices)
            image_network_cache_misses += len(missing_indices)
            if missing_indices:
                missing = torch.tensor(
                    missing_indices, dtype=torch.long, device=device
                )
                missing_images = torch.index_select(scene.images, 0, missing)
                missing_sparse_depth = (
                    None if scene.sparse_depth is None
                    else torch.index_select(scene.sparse_depth, 0, missing)
                )
                missing_sparse_mask = (
                    None if scene.sparse_mask is None
                    else torch.index_select(scene.sparse_mask, 0, missing)
                )
                h, w = scene.image_size
                dh = int(round(h * model.depth_scale))
                dw = int(round(w * model.depth_scale))
                missing_depth = model.depth_model(
                    missing_images,
                    height=dh,
                    width=dw,
                    sparse_depth=missing_sparse_depth,
                    sparse_mask=missing_sparse_mask,
                )
                if (dh, dw) != (h, w):
                    missing_depth = torch.nn.functional.interpolate(
                        missing_depth.unsqueeze(1),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                missing_features = model.backbone(missing_images)
                for result_index, view_index in enumerate(missing_indices):
                    key = view_keys[view_index]
                    depth_cache[key] = missing_depth[result_index].detach()
                    image_feature_cache[key] = missing_features[result_index].detach()
                del (
                    missing,
                    missing_images,
                    missing_sparse_depth,
                    missing_sparse_mask,
                    missing_depth,
                    missing_features,
                )
            depth = torch.stack([depth_cache[key] for key in view_keys])
            image_features = torch.stack([
                image_feature_cache[key] for key in view_keys
            ])
        else:
            depth = model.predict_depth(scene)
            image_features = None
        sampled = model.build_queries(scene, depth, generator=generator)
        current_query_count = int(sampled["queries"].shape[0])
        if old_static is not None:
            old_sampled, old_source_index = build_queries_from_static_gaussians(
                old_static, scene, depth, args.global_depth_threshold,
                source_view_policy=args.global_static_source_view,
            )
            old_static_refresh_candidates = int(old_source_index.shape[0])
            if (
                args.global_update_distance_policy != "uniform"
                and old_source_index.numel() > 0
            ):
                old_static_distance = sampled_camera_distance(old_sampled, scene)
                old_static_best = old_static.extras[
                    "best_observation_distance"
                ][old_source_index]
                static_refresh_alpha = distance_update_alpha(
                    old_static_distance,
                    args.global_update_distance_policy,
                    args.global_update_alpha,
                    args.global_update_near_distance,
                    args.global_update_far_distance,
                    old_static_best,
                    args.global_update_min_improvement,
                )
                keep = static_refresh_alpha > 0
                rejected_old_static_by_distance = int((~keep).sum().item())
                old_sampled = select_sampled(old_sampled, keep)
                old_source_index = old_source_index[keep]
                static_refresh_alpha = static_refresh_alpha[keep]
            observed_old_count = int(old_source_index.shape[0])
            if observed_old_count > 0:
                sampled = concatenate_sampled([sampled, old_sampled])
            del old_sampled
        if (
            args.global_dynamic_forward_refresh
            and old_dynamic is not None
            and old_dynamic.count > 0
        ):
            if boxes is None or global_boxes is None:
                raise RuntimeError(
                    "Dynamic forward refresh requires local and global boxes"
                )
            local_old_dynamic, candidate_source_index = (
                global_dynamic_to_local_tracks(
                    old_dynamic,
                    boxes,
                    local_track_ids,
                    global_boxes,
                    global_track_slot,
                )
            )
            old_dynamic_refresh_candidates = int(local_old_dynamic.count)
            old_dynamic_sampled, observed_local_index = (
                build_queries_from_dynamic_gaussians(
                    local_old_dynamic,
                    scene,
                    depth,
                    args.global_depth_threshold,
                )
            )
            if (
                args.global_update_distance_policy != "uniform"
                and observed_local_index.numel() > 0
            ):
                old_dynamic_distance = sampled_camera_distance(
                    old_dynamic_sampled, scene
                )
                old_dynamic_best = local_old_dynamic.extras[
                    "best_observation_distance"
                ][observed_local_index]
                dynamic_refresh_alpha = distance_update_alpha(
                    old_dynamic_distance,
                    args.global_update_distance_policy,
                    args.global_update_alpha,
                    args.global_update_near_distance,
                    args.global_update_far_distance,
                    old_dynamic_best,
                    args.global_update_min_improvement,
                )
                keep = dynamic_refresh_alpha > 0
                rejected_old_dynamic_by_distance = int((~keep).sum().item())
                old_dynamic_sampled = select_sampled(
                    old_dynamic_sampled, keep
                )
                observed_local_index = observed_local_index[keep]
                dynamic_refresh_alpha = dynamic_refresh_alpha[keep]
            old_dynamic_source_index = candidate_source_index[
                observed_local_index
            ]
            observed_old_dynamic_count = int(
                old_dynamic_source_index.numel()
            )
            if observed_old_dynamic_count > 0:
                sampled = concatenate_sampled([sampled, old_dynamic_sampled])
            del (
                local_old_dynamic,
                candidate_source_index,
                old_dynamic_sampled,
                observed_local_index,
            )

        static_refresh_start = current_query_count
        static_refresh_end = static_refresh_start + observed_old_count
        dynamic_refresh_start = static_refresh_end
        dynamic_refresh_end = (
            dynamic_refresh_start + observed_old_dynamic_count
        )

        query_origin = None
        query_frame = None
        if args.global_query_origin == "clip":
            clip_pose = world_from @ dataset._ego_pose(metadata, window[0])
            query_origin = torch.as_tensor(clip_pose[:3, 3], device=device, dtype=sampled["base_xyz"].dtype)
        elif args.global_query_origin == "clip-rigid":
            query_frame = torch.as_tensor(
                world_from @ dataset._ego_pose(metadata, window[0]),
                device=device, dtype=sampled["base_xyz"].dtype,
            )
        decoded = model.decode(
            scene, depth, sampled, image_features=image_features,
            query_origin=query_origin,
            query_frame=query_frame,
            static_offset_gain=args.global_static_offset_gain,
            static_depth_tolerance=(args.global_static_fusion_depth_tolerance
                                    if args.global_static_fusion_depth_tolerance > 0 else None),
        )
        if image_features is not None:
            del image_features
        decoded.extras["preview_rgb"] = query_rgb(sampled["queries"])
        decoded.extras["observations"] = torch.ones(
            decoded.count, dtype=torch.int32, device=device
        )
        observation_distance = sampled_camera_distance(sampled, scene)
        decoded.extras["best_observation_distance"] = observation_distance
        # Kept on static points as harmless carrier fields so concatenating
        # static and dynamic Gaussians does not discard persistent dynamic
        # geometry quality state.
        decoded.extras["best_position_observation_distance"] = (
            observation_distance.clone()
        )
        decoded.extras["position_observations"] = torch.ones(
            decoded.count, dtype=torch.int32, device=device
        )
        decoded.scales, scale_postprocess = postprocess_gaussian_scales(
            decoded.scales,
            mode=args.gaussian_scale_postprocess,
            isotropic_max_scale=args.isotropic_max_scale,
        )
        if args.global_static_source_pixel_radius > 0:
            # Construction-time camera evidence only.  The sphere remains a
            # fixed 3-D attribute for all future target cameras.
            source_view = sampled["view_id"]
            source_pose = scene.cam_to_world[source_view]
            source_depth = ((sampled["world_xyz"] - source_pose[:, :3, 3])
                            * source_pose[:, :3, 2]).sum(-1).clamp_min(0)
            source_focal = (scene.intrinsics[source_view, 0, 0]
                            * scene.intrinsics[source_view, 1, 1]).sqrt()
            radius = (source_depth * args.global_static_source_pixel_radius
                      / source_focal.clamp_min(1e-6)).clamp(
                          min=args.global_static_source_min_scale,
                          max=args.isotropic_max_scale)
            decoded.scales = torch.where(
                (sampled["instance_id"] < 0)[:, None],
                radius[:, None].expand(-1, 3), decoded.scales)
            scale_postprocess["static_source_pixel_override"] = {
                "pixel_radius": args.global_static_source_pixel_radius,
                "minimum_scale_m": args.global_static_source_min_scale,
                "maximum_scale_m": args.isotropic_max_scale,
                "definition": "source projected depth times pixel radius divided by sqrt(fx*fy); fixed 3-D spheres, static only",
            }

        if args.global_growth_policy == "visibility-difference-frozen-position":
            visibility_filter_history = visibility_history
        elif args.global_growth_policy == "two-clip-mutable-frozen-position":
            visibility_filter_history = visibility_history[:-1]
        elif args.global_growth_policy == "three-clip-mutable-frozen-position":
            visibility_filter_history = visibility_history[:-2]
        elif args.global_growth_policy == "unrestricted-frozen-position":
            visibility_filter_history = []
        else:
            visibility_filter_history = []

        voxel_stats = None
        dynamic_stats = None
        growth_stats = {
            "policy": args.global_growth_policy,
            "historical_clip_count": len(visibility_history),
            "protected_history_clip_count": len(visibility_filter_history),
            "mutable_recent_history_clip_count": (
                len(visibility_history) - len(visibility_filter_history)
            ),
            "old_static_refresh_candidates": old_static_refresh_candidates,
            "distance_policy": args.global_update_distance_policy,
            "rejected_old_refresh_by_distance": rejected_old_static_by_distance,
            "rejected_old_refresh_in_protected_history": 0,
            "updated_old_static_count": observed_old_count,
            "new_static_before_visibility_filter": 0,
            "rejected_in_historical_visible_union": 0,
            "new_static_after_visibility_filter": 0,
            "persistent_position_update": (
                "frozen"
                if args.global_growth_policy != "baseline"
                else "alpha-blended"
            ),
        }
        if args.reconstruction_mode == "global-gaussians":
            current = select_gaussians(
                decoded, torch.arange(current_query_count, device=device)
            )
            refreshed_dynamic_local = None
            if observed_old_dynamic_count > 0:
                refreshed_dynamic_local = select_gaussians(
                    decoded,
                    torch.arange(
                        dynamic_refresh_start,
                        dynamic_refresh_end,
                        device=device,
                    ),
                )
            if old_static is not None and observed_old_count > 0:
                refreshed = select_gaussians(
                    decoded,
                    torch.arange(
                        static_refresh_start,
                        static_refresh_end,
                        device=device,
                    ),
                )
                refreshed = persistent_static(refreshed)
                original_observed = select_gaussians(old_static, old_source_index)
                refresh_source_index = old_source_index
                if (
                    args.global_update_distance_policy == "uniform"
                    and
                    args.global_growth_policy in (
                        "two-clip-mutable-frozen-position",
                        "three-clip-mutable-frozen-position",
                    )
                    and visibility_filter_history
                ):
                    protected_refresh = points_visible_in_history(
                        original_observed.means,
                        visibility_filter_history,
                        args.global_depth_threshold,
                    )
                    rejected_refresh_count = int(protected_refresh.sum().item())
                    mutable_refresh = ~protected_refresh
                    growth_stats["rejected_old_refresh_in_protected_history"] = (
                        rejected_refresh_count
                    )
                    growth_stats["updated_old_static_count"] = int(
                        mutable_refresh.sum().item()
                    )
                    refreshed = select_gaussians(refreshed, mutable_refresh)
                    original_observed = select_gaussians(
                        original_observed, mutable_refresh
                    )
                    refresh_source_index = old_source_index[mutable_refresh]
                    if static_refresh_alpha is not None:
                        static_refresh_alpha = static_refresh_alpha[
                            mutable_refresh
                        ]
                    del protected_refresh, mutable_refresh
                if refresh_source_index.numel() > 0:
                    blended = blend_static_gaussians(
                        original_observed,
                        refreshed,
                        (
                            args.global_update_alpha
                            if static_refresh_alpha is None
                            else static_refresh_alpha
                        ),
                        update_position=(args.global_growth_policy == "baseline"),
                        refresh_fields=args.global_static_refresh_fields,
                    )
                    old_static = replace_gaussians(
                        old_static, refresh_source_index, blended
                    )
                    del blended
                del refreshed, original_observed, refresh_source_index

            del decoded, sampled
            current_static = persistent_static(current)
            current_dynamic = select_gaussians(current, current.instance_id >= 0)
            del current
            dynamic_stats = {
                "policy": args.global_dynamic_policy,
                "forward_refresh_enabled": args.global_dynamic_forward_refresh,
                "forward_refresh_candidates": old_dynamic_refresh_candidates,
                "forward_refresh_observed": observed_old_dynamic_count,
                "forward_refresh_updated": observed_old_dynamic_count,
                "distance_policy": args.global_update_distance_policy,
                "forward_refresh_rejected_by_distance": (
                    rejected_old_dynamic_by_distance
                ),
                "forward_refresh_history_policy": "all-visible-history",
                "scene_track_count": len(global_track_ids),
                "clip_track_count": len(local_track_ids),
                "current_dynamic_count": int(current_dynamic.count),
                "previous_global_dynamic_count": (
                    0 if old_dynamic is None else int(old_dynamic.count)
                ),
            }
            if args.global_dynamic_policy == "persistent-tracks":
                if refreshed_dynamic_local is not None:
                    refreshed_dynamic_global = rebase_dynamic_to_global_tracks(
                        refreshed_dynamic_local,
                        boxes,
                        local_track_ids,
                        global_boxes,
                        global_track_slot,
                    )
                    original_dynamic = select_gaussians(
                        old_dynamic, old_dynamic_source_index
                    )
                    blended_dynamic = blend_dynamic_gaussians(
                        original_dynamic,
                        refreshed_dynamic_global,
                        (
                            args.global_update_alpha
                            if dynamic_refresh_alpha is None
                            else dynamic_refresh_alpha
                        ),
                    )
                    old_dynamic = replace_gaussians(
                        old_dynamic,
                        old_dynamic_source_index,
                        blended_dynamic,
                    )
                    del (
                        refreshed_dynamic_local,
                        refreshed_dynamic_global,
                        original_dynamic,
                        blended_dynamic,
                    )
                if current_dynamic.count > 0:
                    if boxes is None or global_boxes is None:
                        raise RuntimeError(
                            "Dynamic Gaussians require both local and global boxes"
                        )
                    current_dynamic = rebase_dynamic_to_global_tracks(
                        current_dynamic,
                        boxes,
                        local_track_ids,
                        global_boxes,
                        global_track_slot,
                    )
                if args.global_dynamic_merge_policy == "nearest-update":
                    current_dynamic, dynamic_merge_stats = (
                        merge_global_dynamic_gaussians_nearest_update(
                            old_dynamic,
                            current_dynamic,
                            args.global_dynamic_association_radius,
                            args.global_dynamic_position_alpha,
                            args.global_dynamic_max_candidates_per_cell,
                        )
                    )
                    dynamic_merge_stats["overlap_refresh_policy"] = (
                        "forward-refresh-for-attributes; closest-so-far "
                        "nearest association for canonical xyz"
                    )
                else:
                    current_dynamic, dynamic_merge_stats = (
                        merge_global_dynamic_gaussians(
                            old_dynamic,
                            current_dynamic,
                            args.global_voxel_size,
                            (
                                args.global_update_alpha
                                if args.global_update_distance_policy == "uniform"
                                else 0.0
                            ),
                        )
                    )
                    dynamic_merge_stats["overlap_refresh_policy"] = (
                        "uniform-alpha"
                        if args.global_update_distance_policy == "uniform"
                        else "disabled-after-forward-refresh"
                    )
                dynamic_stats.update(dynamic_merge_stats)
                dynamic_stats["global_dynamic_count"] = int(current_dynamic.count)
            else:
                dynamic_stats["global_dynamic_count"] = int(current_dynamic.count)
            growth_stats["new_static_before_visibility_filter"] = current_static.count
            if (
                args.global_growth_policy != "baseline"
                and visibility_filter_history
                and current_static.count > 0
            ):
                historically_visible = points_visible_in_history(
                    current_static.means,
                    visibility_filter_history,
                    args.global_depth_threshold,
                )
                rejected_count = int(historically_visible.sum().item())
                current_static = select_gaussians(
                    current_static, ~historically_visible
                )
                growth_stats["rejected_in_historical_visible_union"] = rejected_count
                del historically_visible
            growth_stats["new_static_after_visibility_filter"] = current_static.count
            static_parts = [current_static] if old_static is None else [old_static, current_static]
            global_gaussians, voxel_stats = voxel_deduplicate_static(
                static_parts, args.global_voxel_size, append=current_dynamic,
                representative_policy=args.global_static_voxel_representative,
            )
            del current_static, static_parts, current_dynamic
            if old_dynamic is not None:
                del old_dynamic
            torch.cuda.empty_cache()
            global_gaussians = move_gaussians(global_gaussians, device)
            gaussians = global_gaussians
            if old_static is not None:
                del old_static
        else:
            gaussians = decoded

        if args.global_growth_policy != "baseline":
            visibility_history.append({
                "intrinsics": scene.intrinsics.detach().cpu(),
                "cam_to_world": scene.cam_to_world.detach().cpu(),
                "depth": depth.detach().cpu(),
            })

        if (
            runtime_payload_dir is not None
            and args.reconstruction_mode == "global-gaussians"
            and args.global_dynamic_policy == "per-window"
        ):
            # The final PLY intentionally contains only preview RGB.  Preserve
            # the actual feature-carrying dynamic set and its local box slots.
            dynamic = select_gaussians(gaussians, gaussians.instance_id >= 0)
            save_gaussian_payload(
                runtime_payload_dir / f"dynamic_window_{window_index:02d}.pt",
                dynamic,
            )
            torch.save(
                {
                    "window_index": window_index,
                    "window_frames": [int(frame) for frame in window],
                    "rendered_frames": [int(frame) for frame in owned_frames],
                    "context_frames": [int(frame) for frame in context_frames],
                    "normalized_times": normalized_times.cpu(),
                    "boxes": boxes_payload(boxes_cpu),
                },
                runtime_payload_dir / f"dynamic_window_{window_index:02d}_meta.pt",
            )
            del dynamic
        if args.reconstruction_mode == "global-gaussians":
            del depth, scene, context
        else:
            del decoded, sampled, depth, scene, context
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started

        ply_path = None
        is_last_window = window_index == len(window_offsets) - 1
        should_export_ply = not args.skip_ply and (
            args.reconstruction_mode == "original" or is_last_window
        )
        if should_export_ply:
            ply_boxes = (
                global_boxes
                if args.global_dynamic_policy == "persistent-tracks"
                else boxes
            )
            ply_frame = (
                global_frame_slot[window[0]]
                if args.global_dynamic_policy == "persistent-tracks"
                else 0
            )
            ply_time = (
                global_normalized_times[ply_frame]
                if args.global_dynamic_policy == "persistent-tracks"
                else normalized_times[0]
            ).to(device)
            posed = model.pose_gaussians(
                gaussians, ply_boxes, frame=ply_frame, target_time=ply_time
            )
            preview_rgb = gaussians.extras["preview_rgb"]
            ply_name = (
                f"global_final_window_{window_index:02d}_frame_{window[0]:03d}_pointforward_preview.ply"
                if args.reconstruction_mode == "global-gaussians"
                else f"window_{window_index:02d}_frame_{window[0]:03d}_pointforward_preview.ply"
            )
            ply_path = gaussian_dir / ply_name
            write_standard_ply(
                str(ply_path),
                posed["means"].float().cpu().numpy(),
                preview_rgb.float().cpu().numpy(),
                posed["opacities"].float().cpu().numpy(),
                posed["scales"].float().cpu().numpy(),
                posed["quats"].float().cpu().numpy(),
            )

        render_started = time.perf_counter()
        frame_psnr = []
        if args.skip_frame_render:
            render_seconds = 0.0
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            record = {
                "window_index": window_index,
                "frames": [int(frame) for frame in window],
                "rendered_frames": [int(frame) for frame in owned_frames],
                "context_frames": [int(frame) for frame in context_frames],
                "gaussian_count": int(gaussians.count),
                "current_query_count": current_query_count,
                "old_static_count": old_static_count,
                "observed_old_count": observed_old_count,
                "sparse_depth_prompt": sparse_depth_stats,
                "global_growth": growth_stats,
                "global_dynamic": dynamic_stats,
                "voxel_deduplication": voxel_stats,
                "instance_count": 0 if boxes_cpu is None else int(boxes_cpu.num_instances),
                "decode_seconds": decode_seconds,
                "render_seconds": render_seconds,
                "mean_frame_psnr": None,
                "ply": str(ply_path) if ply_path else None,
                "gaussian_scale_postprocess": scale_postprocess,
            }
            windows.append(record)
            timings.append(decode_seconds)
            print(json.dumps(record), flush=True)
            del gaussians, boxes
            torch.cuda.empty_cache()
            continue
        for frame in owned_frames:
            target = dataset._collect(
                metadata, [frame], slot_of, normalized_times,
                dataset.resolution, world_from, lidar=False,
            )
            ground_truth = target["images"]
            source_index = frame_to_output_index[frame]
            is_final_frame = source_index == len(selected_frames) - 1
            subframe_count = 1 if is_final_frame else args.camera_interpolation_factor
            upper_target = None
            if subframe_count > 1:
                upper_frame = selected_frames[source_index + 1]
                upper_target = dataset._collect(
                    metadata, [upper_frame], slot_of, normalized_times,
                    dataset.resolution, world_from, lidar=False,
                )

            for subframe in range(subframe_count):
                alpha = subframe / args.camera_interpolation_factor
                camera_poses = target["cam_to_world"].to(device)
                if upper_target is not None and alpha > 0:
                    upper_poses = upper_target["cam_to_world"].to(device)
                    camera_poses = torch.stack([
                        interpolate_camera_pose(first, second, alpha)
                        for first, second in zip(camera_poses, upper_poses)
                    ])
                targets = TargetViews(
                    intrinsics=target["intrinsics"].to(device),
                    cam_to_world=camera_poses,
                    # Camera-only interpolation: dynamics and temporal opacity
                    # are held at the lower real source frame.
                    times=(
                        torch.full(
                            (num_cameras,),
                            float(global_normalized_times[global_frame_slot[frame]]),
                            device=device,
                        )
                        if args.global_dynamic_policy == "persistent-tracks"
                        else target["times"].to(device)
                    ),
                    frame_index=(
                        torch.full(
                            (num_cameras,),
                            global_frame_slot[frame],
                            dtype=torch.long,
                            device=device,
                        )
                        if args.global_dynamic_policy == "persistent-tracks"
                        else target["frame_index"].to(device)
                    ),
                )
                rendered = model.render(
                    gaussians,
                    global_boxes
                    if args.global_dynamic_policy == "persistent-tracks"
                    else boxes,
                    targets,
                    height=height, width=width, render_depth=False,
                )["rgb"]
                if subframe == 0:
                    frame_psnr.append(tensor_psnr(rendered, ground_truth.to(device)))
                display_frame = source_index + alpha
                predicted_panels = []
                ground_truth_panels = []
                for camera_index, camera in enumerate(dataset.cameras):
                    predicted_panels.append(label(
                        to_uint8(rendered[camera_index]),
                        f"PointForward | frame {display_frame:06.3f} | camera {camera}",
                    ))
                    ground_truth_panels.append(label(
                        to_uint8(ground_truth[camera_index]),
                        f"Input sequence | frame {frame:03d} | camera {camera}",
                    ))
                prediction_strip = horizontal_strip(predicted_panels)
                comparison = vertical_strip((
                    horizontal_strip(ground_truth_panels), prediction_strip
                ))
                output_index = source_index * args.camera_interpolation_factor + subframe
                Image.fromarray(prediction_strip).save(
                    predicted_dir / f"frame_{output_index:04d}.jpg", quality=92
                )
                Image.fromarray(comparison).save(
                    comparison_dir / f"frame_{output_index:04d}.jpg", quality=92
                )
                del rendered, targets, camera_poses
            del ground_truth, target
            if upper_target is not None:
                del upper_target
        torch.cuda.synchronize()
        render_seconds = time.perf_counter() - render_started
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
        record = {
            "window_index": window_index,
            "frames": [int(frame) for frame in window],
            "rendered_frames": [int(frame) for frame in owned_frames],
            "context_frames": [int(frame) for frame in context_frames],
            "gaussian_count": int(gaussians.count),
            "current_query_count": current_query_count,
            "old_static_count": old_static_count,
            "observed_old_count": observed_old_count,
            "sparse_depth_prompt": sparse_depth_stats,
            "global_growth": growth_stats,
            "global_dynamic": dynamic_stats,
            "voxel_deduplication": voxel_stats,
            "instance_count": 0 if boxes_cpu is None else int(boxes_cpu.num_instances),
            "decode_seconds": decode_seconds,
            "render_seconds": render_seconds,
            "mean_frame_psnr": float(np.mean(frame_psnr)),
            "ply": str(ply_path) if ply_path else None,
            "gaussian_scale_postprocess": scale_postprocess,
        }
        windows.append(record)
        timings.append(decode_seconds + render_seconds)
        print(json.dumps(record), flush=True)
        del gaussians, boxes
        torch.cuda.empty_cache()

    runtime_payload_manifest = None
    static_view_bank_stats = None
    if runtime_payload_dir is not None and args.reconstruction_mode == "global-gaussians":
        static_final = persistent_static(global_gaussians)
        if args.global_static_view_bank != "disabled":
            from pointforward.models.static_view_bank import refresh_static_features
            bank_frames = sorted({frame for frame, _ in image_feature_cache})
            bank_keys = [(frame, int(camera)) for frame in bank_frames for camera in dataset.cameras]
            if any(key not in image_feature_cache or key not in depth_cache for key in bank_keys):
                raise RuntimeError("Static view bank has incomplete camera groups")
            bank_context = dataset._collect(metadata, bank_frames, global_frame_slot,
                                            global_normalized_times, dataset.resolution,
                                            world_from, lidar=False)
            bank_images = bank_context["images"].to(device)
            bank_intrinsics = bank_context["intrinsics"].to(device)
            bank_poses = bank_context["cam_to_world"].to(device)
            bank_masks = None
            if global_boxes is not None:
                bank_masks = box_dynamic_masks(global_boxes, bank_context["frame_index"].to(device),
                                               bank_intrinsics, torch.linalg.inv(bank_poses),
                                               *bank_images.shape[-2:])
            ego_poses = torch.stack([torch.as_tensor(world_from @ dataset._ego_pose(metadata, frame),
                                                    device=device, dtype=bank_poses.dtype)
                                     for frame in bank_frames])
            static_view_bank_stats = refresh_static_features(
                model, static_final, images=bank_images,
                features=torch.stack([image_feature_cache[key] for key in bank_keys]),
                depth=torch.stack([depth_cache[key] for key in bank_keys]),
                intrinsics=bank_intrinsics, cam_to_world=bank_poses,
                frame_numbers=torch.tensor(bank_frames, device=device), ego_to_world=ego_poses,
                camera_count=len(dataset.cameras), dynamic_masks=bank_masks,
                alpha=args.global_static_view_bank_alpha,
                depth_tolerance=args.global_static_view_bank_depth_tolerance,
                chunk_size=args.global_static_view_bank_chunk_size,
            )
            print(json.dumps({"static_view_bank": static_view_bank_stats}), flush=True)
            del bank_context, bank_images, bank_intrinsics, bank_poses, bank_masks, ego_poses
        final_static_path = runtime_payload_dir / "final_static_128d.pt"
        save_gaussian_payload(final_static_path, static_final)
        runtime_payload_manifest = {
            "directory": str(runtime_payload_dir),
            "final_static": str(final_static_path),
            "feature_dim": int(static_final.colors.shape[-1]),
            "static_count": int(static_final.count),
        }
        if args.global_dynamic_policy == "persistent-tracks":
            dynamic_final = select_gaussians(
                global_gaussians, global_gaussians.instance_id >= 0
            )
            final_dynamic_path = runtime_payload_dir / "final_dynamic_128d.pt"
            global_dynamic_meta_path = runtime_payload_dir / "global_dynamic_meta.pt"
            save_gaussian_payload(final_dynamic_path, dynamic_final)
            torch.save(
                {
                    "frames": [int(frame) for frame in selected_frames],
                    "normalized_times": global_normalized_times.cpu(),
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
                global_dynamic_meta_path,
            )
            runtime_payload_manifest.update({
                "dynamic_policy": "persistent-tracks",
                "final_dynamic": str(final_dynamic_path),
                "global_dynamic_metadata": str(global_dynamic_meta_path),
                "dynamic_count": int(dynamic_final.count),
                "global_track_count": len(global_track_ids),
                "note": "Runtime tensors preserve one final 128-D static map, one final scene-global dynamic map, stable scene track IDs, and the full-scene box graph.",
            })
        else:
            runtime_payload_manifest.update({
                "dynamic_policy": "per-window",
                "dynamic_windows": [
                    str(runtime_payload_dir / f"dynamic_window_{index:02d}.pt")
                    for index in range(len(window_offsets))
                ],
                "dynamic_metadata": [
                    str(runtime_payload_dir / f"dynamic_window_{index:02d}_meta.pt")
                    for index in range(len(window_offsets))
                ],
                "note": "Runtime tensors preserve 128-D Gaussian colors and per-window dynamic instance metadata; PLY preview export does not.",
            })

    predicted_video = None
    comparison_video = None
    if not args.skip_frame_render:
        predicted_video = output_dir / "pointforward_predicted_front3.mp4"
        comparison_video = output_dir / "pointforward_gt_vs_predicted_front3.mp4"
        encode_video(predicted_dir / "frame_%04d.jpg", predicted_video, args.fps)
        encode_video(comparison_dir / "frame_%04d.jpg", comparison_video, args.fps)
    manifest = {
        "status": "completed",
        "scene": args.scene,
        "source_checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(Path(args.config).resolve()),
        "cameras": list(dataset.cameras),
        "camera_semantics": {"0": "front", "1": "front_left", "2": "front_right"},
        "frame_count": len(selected_frames),
        "frame_range": [int(selected_frames[0]), int(selected_frames[-1])],
        "fps": args.fps,
        "source_fps": args.fps / args.camera_interpolation_factor,
        "camera_interpolation_factor": args.camera_interpolation_factor,
        "output_frame_count": (
            (len(selected_frames) - 1) * args.camera_interpolation_factor + 1
        ),
        "camera_interpolation": (
            "translation lerp plus shortest-path quaternion SLERP; dynamic time and Gaussian state held at lower source frame"
            if args.camera_interpolation_factor > 1
            else "disabled"
        ),
        "resolution_hw_per_camera": [height, width],
        "predicted_video": str(predicted_video) if predicted_video else None,
        "comparison_video": str(comparison_video) if comparison_video else None,
        "reconstruction_mode": args.reconstruction_mode,
        "clip_order": args.clip_order,
        "window_length": window_length,
        "context_frame_stride": context_frame_stride,
        "window_step": window_step,
        "window_overlap": window_length - window_step,
        "sparse_depth_source": args.sparse_depth_source,
        "sparse_depth_policy": (
            "first processed clip uses dataset LiDAR; every later clip uses nearest-pixel z-buffer projection of all persistent static global Gaussian centres, deterministically sparsified to the bootstrap clip's per-view LiDAR point counts"
            if args.sparse_depth_source == "global-gaussian-feedback"
            else "dataset LiDAR for every clip"
        ),
        "processing_window_offsets": window_offsets,
        "global_growth_policy": args.global_growth_policy,
        "global_dynamic_policy": args.global_dynamic_policy,
        "stationary_track_reclassification": {
            "enabled": args.stationary_track_max_span is not None,
            "max_center_span_m": args.stationary_track_max_span,
            "max_median_step_m": args.stationary_track_max_median_step,
            "track_ids": stationary_track_ids,
            "track_count": len(stationary_track_ids),
            "tracks": stationary_track_records,
        },
        "global_dynamic_forward_refresh": args.global_dynamic_forward_refresh,
        "global_dynamic_merge": {
            "policy": args.global_dynamic_merge_policy,
            "association_radius_m": args.global_dynamic_association_radius,
            "position_alpha": args.global_dynamic_position_alpha,
            "max_candidates_per_cell": (
                args.global_dynamic_max_candidates_per_cell
            ),
            "position_update": (
                "same-track nearest association; keep persistent identity; "
                "closer observation updates canonical xyz; append unmatched only"
                if args.global_dynamic_merge_policy == "nearest-update"
                else "canonical xyz frozen; exact track-and-voxel merge"
            ),
        },
        "global_update_distance": {
            "policy": args.global_update_distance_policy,
            "static_voxel_representative": args.global_static_voxel_representative,
            "query_coordinate_frame": args.global_query_origin,
            "static_decoder_offset_gain": args.global_static_offset_gain,
            "static_source_pixel_radius": args.global_static_source_pixel_radius,
            "static_source_min_scale_m": args.global_static_source_min_scale,
            "static_fusion_depth_tolerance_m": args.global_static_fusion_depth_tolerance,
            "static_source_view_policy": args.global_static_source_view,
            "new_observation_max_alpha": args.global_update_alpha,
            "near_distance_m": args.global_update_near_distance,
            "far_cutoff_distance_m": args.global_update_far_distance,
            "min_improvement_m": args.global_update_min_improvement,
            "distance_definition": "Euclidean world distance from Gaussian centre to selected observing camera centre",
            "static_refresh_age_gate": (
                "original growth-policy gate"
                if args.global_update_distance_policy == "uniform"
                else "replaced by per-Gaussian distance policy"
            ),
            "dynamic_voxel_overlap_refresh": (
                "uniform alpha"
                if args.global_update_distance_policy == "uniform"
                else "disabled to avoid a second same-clip update after forward refresh"
            ),
        },
        "overlap_image_network_cache": {
            "enabled": args.cache_overlap_image_networks,
            "hits": image_network_cache_hits,
            "misses": image_network_cache_misses,
            "unique_views": len(depth_cache),
        },
        "global_dynamic_contract": (
            "DriveStudio scene-level instance keys define stable IDs; all currently visible historical dynamic Gaussians are transformed into the current clip frame, reprojected to RGB/depth, passed through fusion and the decoder without a clip-age update limit, transformed back to full-scene canonical tracks, same-track current observations are nearest-associated to persistent points, closer observations update persistent canonical xyz while preserving identity, and only unmatched surface is appended"
            if (
                args.global_dynamic_policy == "persistent-tracks"
                and args.global_dynamic_forward_refresh
                and args.global_dynamic_merge_policy == "nearest-update"
            )
            else "DriveStudio scene-level instance keys define stable IDs; all currently visible historical dynamic Gaussians are transformed into the current clip frame, reprojected to RGB/depth, passed through fusion and the decoder without a clip-age update limit, transformed back to full-scene canonical tracks, fused by track ID plus canonical voxel with frozen xyz and refreshed attributes, and rendered with one full-scene box graph"
            if (
                args.global_dynamic_policy == "persistent-tracks"
                and args.global_dynamic_forward_refresh
            )
            else "DriveStudio scene-level instance keys define stable IDs; every clip-canonical dynamic Gaussian is rebased to the track's earliest full-scene box, fused by track ID plus canonical voxel with frozen xyz and refreshed attributes, and rendered with one full-scene box graph"
            if args.global_dynamic_policy == "persistent-tracks"
            else "clip-local dynamic Gaussian set replaces the previous window set"
        ),
        "global_growth_contract": (
            "persistent static means are immutable; new static Gaussians are admitted only where the current visible surface is outside every prior clip visible-region union"
            if args.global_growth_policy == "visibility-difference-frozen-position"
            else "persistent static means are immutable; current and immediately previous clip regions permit new static Gaussians and attribute refreshes, while regions visible two or more processed clips ago are protected from both"
            if args.global_growth_policy == "two-clip-mutable-frozen-position"
            else "persistent static means are immutable; current and two immediately previous clip regions permit new static Gaussians and attribute refreshes, while regions visible three or more processed clips ago are protected from both"
            if args.global_growth_policy == "three-clip-mutable-frozen-position"
            else "persistent static means are immutable; all current static points are eligible for admission before voxel deduplication; attribute refresh still follows the selected distance policy"
            if args.global_growth_policy == "unrestricted-frozen-position"
            else "AV-v1-m baseline: persistent static means are alpha-blended and current static Gaussians are added before voxel deduplication"
        ),
        "global_gaussians": {
            "voxel_size_m": args.global_voxel_size,
            "depth_threshold_m": args.global_depth_threshold,
            "update_alpha": args.global_update_alpha,
            "static_policy": (
                "persistent positions frozen; other decoded attributes refreshed; new static points restricted to current-visible minus historical-visible union; then voxel-deduplicated"
                if args.global_growth_policy == "visibility-difference-frozen-position"
                else "persistent positions frozen; attribute refresh and new-point admission allowed only outside the visible-region union of clips two or more processing steps old; then voxel-deduplicated"
                if args.global_growth_policy == "two-clip-mutable-frozen-position"
                else "persistent positions frozen; attribute refresh and new-point admission allowed only outside the visible-region union of clips three or more processing steps old; then voxel-deduplicated"
                if args.global_growth_policy == "three-clip-mutable-frozen-position"
                else "persistent positions frozen; no historical visible-region admission exclusion; attribute refresh follows distance policy; then voxel-deduplicated"
                if args.global_growth_policy == "unrestricted-frozen-position"
                else "persistent, reprojected into each new clip, decoder-refreshed, then voxel-deduplicated"
            ),
            "dynamic_policy": (
                "scene-level track IDs; all visible history forward-refreshed; same-track nearest current-to-old association; closer observations update persistent canonical xyz; matched current suppressed; unmatched current appended; full-scene box graph"
                if (
                    args.global_dynamic_policy == "persistent-tracks"
                    and args.global_dynamic_forward_refresh
                    and args.global_dynamic_merge_policy == "nearest-update"
                )
                else "scene-level track IDs; all currently visible history reprojected into every clip for fusion and decoder refresh with no clip-age restriction; persistent full-scene canonical dynamic Gaussians; track-and-voxel fusion; full-scene box graph"
                if (
                    args.global_dynamic_policy == "persistent-tracks"
                    and args.global_dynamic_forward_refresh
                )
                else "scene-level track IDs; persistent full-scene canonical dynamic Gaussians; track-and-voxel fusion; full-scene box graph"
                if args.global_dynamic_policy == "persistent-tracks"
                else "current-window dynamic Gaussians replace the previous window dynamic set"
            ),
            "render_policy": "render the complete global set without explicit tile or frustum loading",
        } if args.reconstruction_mode == "global-gaussians" else None,
        "gaussian_export": (
            "one final global preview-color PLY snapshot"
            if args.reconstruction_mode == "global-gaussians" and not args.skip_ply
            else "one preview-color PLY snapshot per reconstruction window"
            if not args.skip_ply
            else "disabled"
        ),
        "gaussian_scale_postprocess": {
            "mode": args.gaussian_scale_postprocess,
            "isotropic_max_scale_m": args.isotropic_max_scale,
            "application_point": "after Gaussian decoding and before PLY export and rendering",
        },
        "runtime_payload": runtime_payload_manifest,
        "static_view_bank": static_view_bank_stats,
        "coordinate_frame": "shared world recentered at the first ego pose of the delivered sequence",
        "windows": windows,
        "mean_psnr_all_frames_cameras": (
            None
            if args.skip_frame_render
            else float(np.average(
                [window["mean_frame_psnr"] for window in windows],
                weights=[len(window["rendered_frames"]) for window in windows],
            ))
        ),
        "wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_gib": peak_allocated / (1024 ** 3),
        "limitations": [
            "The checkpoint was trained on Waymo; this is zero-shot Argoverse transfer, not an Argoverse-trained result.",
            (
                "Static and dynamic Gaussians persist globally; dynamic identity follows dataset-provided scene track IDs and therefore depends on the upstream tracking annotations."
                if args.global_dynamic_policy == "persistent-tracks"
                else "Static Gaussians persist globally; dynamic Gaussians are replaced by the current window because clip-local instance IDs are not stable track IDs."
                if args.reconstruction_mode == "global-gaussians"
                else "Each 20-frame window is reconstructed independently; all windows share one recentered world frame."
            ),
            "PLY colors are source-query RGB encoded as degree-0 SH previews, not the learned 128-D feature plus UNet output.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the dynamic RGB/depth/3-D-box geometry contract (Round 63).

This is a diagnostic probe, not a reconstruction path.  It freezes the ten
tracks and 120 observations selected by the DR-v1-r predecessor, predicts the
dense input depth once, and answers two independent questions:

* does a train-view affine correction of dense z-depth improve held-out LiDAR;
* does a shared vehicle-local translation/yaw correction (or extent-only
  correction) improve held-out *uncropped* SAM silhouettes.

The two anchors are intentionally independent: the primary depth test never
uses an OBB, and the box test never reads predicted depth.  The script also
reports the joint ray/OBB contract and a supporting cross-view transport test.

InstanceBoxes stores ``R = world -> local``.  Consequently the correction used
throughout this file is exactly::

    c' = c + R.T @ delta_local
    R' = D(yaw).T @ R

and cross-time transport is ``p_local = R_s @ (p_world-c_s)`` followed by
``p_target = R_t.T @ p_local + c_t``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import SceneInput
from scripts.eval import build_model
from scripts.train import build_dataset


FROZEN_TRACKS = (47, 26, 0, 41, 29, 18, 34, 19, 24, 7)
DEFAULT_MANIFEST = Path(
    "experiments/argoverse-inference/global-dynamic-consistency/"
    "dynamic-roi-refinement/rgb-foreground-gating/DR-v1-q/"
    "run-20260929T114105Z-ungated-control/rebuilt-top10/track_once_manifest.json"
)
H, W = 576, 768
SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Box:
    center: np.ndarray
    dims: np.ndarray
    rotation: np.ndarray  # world -> local


@dataclass
class View:
    track_id: int
    frame: int
    camera: int
    bbox: tuple[float, float, float, float]
    rgb: np.ndarray
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    sparse_depth: np.ndarray
    sparse_mask: np.ndarray
    dense_depth: np.ndarray
    raw_mask: np.ndarray
    box: Box

    @property
    def split(self) -> str:
        # Frame parity, rather than view order, prevents cameras at one timestamp
        # leaking across the train/held-out boundary.
        return "train" if self.frame % 2 else "heldout"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--track-manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--raw-mask-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene", default="017")
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth-batch-size", type=int, default=12)
    p.add_argument("--cross-view-max-points", type=int, default=2048)
    args = p.parse_args()
    if args.depth_batch_size < 1 or args.cross_view_max_points < 1:
        p.error("batch size and cross-view point cap must be positive")
    return args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def relative_gain(before: float, after: float) -> float | None:
    return float((before - after) / before) if before > 0 else None


def yaw_matrix(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def correct_box(box: Box, params: np.ndarray | None = None, scale: float = 1.0) -> Box:
    p = np.zeros(4, dtype=np.float64) if params is None else np.asarray(params, dtype=np.float64)
    d = yaw_matrix(float(p[3]))
    return Box(
        center=box.center + box.rotation.T @ p[:3],
        dims=box.dims * float(scale),
        rotation=d.T @ box.rotation,
    )


def camera_rays(k: np.ndarray, camera_to_world: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = roi
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pix = np.stack((xx + 0.5, yy + 0.5, np.ones_like(xx)), axis=-1).astype(np.float64)
    direction_cam = pix @ np.linalg.inv(k).T
    direction_cam /= np.maximum(np.linalg.norm(direction_cam, axis=-1, keepdims=True), 1e-15)
    direction_world = direction_cam @ camera_to_world[:3, :3].T
    origin = camera_to_world[:3, 3].astype(np.float64)
    forward = camera_to_world[:3, 2].astype(np.float64)
    cosine = direction_world @ forward
    return origin, direction_world, cosine


def projected_roi(box: Box, k: np.ndarray, camera_to_world: np.ndarray) -> tuple[int, int, int, int]:
    corners = SIGNS * (box.dims * 0.5)
    world = corners @ box.rotation + box.center  # local @ R + c
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    if np.any(camera[:, 2] <= 1e-5):
        return (0, 0, W, H)
    uvw = camera @ k.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    x0 = max(0, int(math.floor(float(uv[:, 0].min()))) - 2)
    y0 = max(0, int(math.floor(float(uv[:, 1].min()))) - 2)
    x1 = min(W, int(math.ceil(float(uv[:, 0].max()))) + 3)
    y1 = min(H, int(math.ceil(float(uv[:, 1].max()))) + 3)
    return (x0, y0, max(x0, x1), max(y0, y1))


def ray_box_interval(
    box: Box,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    return_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Rasterise an exact pixel-centre ray/OBB silhouette using slab intervals."""
    roi = projected_roi(box, intrinsic, camera_to_world)
    mask = np.zeros((H, W), dtype=bool)
    z_near_full = np.full((H, W), np.nan, dtype=np.float32) if return_depth else None
    z_far_full = np.full((H, W), np.nan, dtype=np.float32) if return_depth else None
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        return mask, z_near_full, z_far_full
    origin, direction, cosine = camera_rays(intrinsic, camera_to_world, roi)
    o_local = box.rotation @ (origin - box.center)
    d_local = direction @ box.rotation.T
    half = box.dims * 0.5
    parallel = np.abs(d_local) < 1e-12
    outside_parallel = np.any(parallel & (np.abs(o_local) > half), axis=-1)
    safe_d = np.where(parallel, 1.0, d_local)
    ta = (-half - o_local) / safe_d
    tb = (half - o_local) / safe_d
    lo = np.where(parallel, -np.inf, np.minimum(ta, tb))
    hi = np.where(parallel, np.inf, np.maximum(ta, tb))
    t_near = np.max(lo, axis=-1)
    t_far = np.min(hi, axis=-1)
    hit = (~outside_parallel) & (cosine > 1e-6) & (t_far > np.maximum(t_near, 0.0))
    mask[y0:y1, x0:x1] = hit
    if return_depth:
        zn = t_near * cosine
        zf = t_far * cosine
        local_near = np.where(hit, zn, np.nan).astype(np.float32)
        local_far = np.where(hit, zf, np.nan).astype(np.float32)
        z_near_full[y0:y1, x0:x1] = local_near
        z_far_full[y0:y1, x0:x1] = local_far
    return mask, z_near_full, z_far_full


def bbox_mask(bbox: Iterable[float]) -> np.ndarray:
    x0, y0, x1, y1 = [float(x) for x in bbox]
    left, top = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    right, bottom = min(W, int(math.ceil(x1)) + 1), min(H, int(math.ceil(y1)) + 1)
    out = np.zeros((H, W), dtype=bool)
    out[top:bottom, left:right] = True
    return out


def counts_iou(targets: list[np.ndarray], predictions: list[np.ndarray]) -> tuple[int, int, float]:
    inter = sum(int(np.logical_and(a, b).sum()) for a, b in zip(targets, predictions))
    union = sum(int(np.logical_or(a, b).sum()) for a, b in zip(targets, predictions))
    return inter, union, float(inter / union) if union else 0.0


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def read_manifest(path: Path) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    records: dict[int, list[dict[str, Any]]] = {}
    for track in source.get("tracks", []):
        track_id = int(track["track_id"])
        if track_id in FROZEN_TRACKS:
            selected = list(track.get("selected_views", []))
            if len(selected) != 12:
                raise ValueError(f"track {track_id} has {len(selected)} views, expected 12")
            records[track_id] = selected
    if set(records) != set(FROZEN_TRACKS):
        raise ValueError(f"frozen track mismatch: got {sorted(records)}")
    identities = [(t, int(v["frame"]), int(v["camera"])) for t, views in records.items() for v in views]
    if len(identities) != 120 or len(set(identities)) != 120:
        raise ValueError("manifest must contain exactly 120 distinct track-view prompts")
    return source, records


def prepare_dataset(config: Path, scene: str) -> tuple[dict[str, Any], Any, dict[str, Any], np.ndarray]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(cfg))
    data = cfg.setdefault("data", {})
    data.pop("split_file", None)
    data["val_scenes"] = [str(scene)]
    data["resolution"] = [H, W]
    data["load_lidar"] = True
    if data.get("layout") == "argoverse_drivestudio":
        data["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(cfg, "val")
    metadata = next((m for m in dataset.scenes if str(m["name"]) == str(scene)), None)
    if metadata is None:
        raise RuntimeError(f"scene {scene!r} was not loaded")
    first = int(metadata["frames"][0])
    world_from = np.linalg.inv(dataset._ego_pose(metadata, first))
    return cfg, dataset, metadata, world_from


def collect_selected_views(
    dataset: Any,
    metadata: dict[str, Any],
    world_from: np.ndarray,
    requested: set[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Use dataset._collect in bounded frame chunks and retain requested cameras."""
    frames = sorted({f for f, _ in requested})
    result: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    camera_slot = {int(c): i for i, c in enumerate(dataset.cameras)}
    for start in range(0, len(frames), 8):
        chunk = frames[start : start + 8]
        slot = {f: i for i, f in enumerate(chunk)}
        times = dataset._normalised_times(chunk)
        batch = dataset._collect(metadata, chunk, slot, times, (H, W), world_from, lidar=True)
        nc = len(dataset.cameras)
        for fi, frame in enumerate(chunk):
            for camera in dataset.cameras:
                key = (int(frame), int(camera))
                if key not in requested:
                    continue
                i = fi * nc + camera_slot[int(camera)]
                result[key] = {
                    "image": batch["images"][i].numpy(),
                    "intrinsic": batch["intrinsics"][i].numpy().astype(np.float64),
                    "camera_to_world": batch["cam_to_world"][i].numpy().astype(np.float64),
                    "sparse_depth": batch["sparse_depth"][i, 0].numpy(),
                    "sparse_mask": batch["sparse_mask"][i, 0].numpy() > 0.5,
                }
    if set(result) != requested:
        raise RuntimeError(f"missing calibrated views: {sorted(requested - set(result))[:5]}")
    return result


@torch.inference_mode()
def predict_depths(
    cfg: dict[str, Any], ckpt: Path, device: torch.device, calibrated: dict[tuple[int, int], dict[str, np.ndarray]], batch_size: int
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, float]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    model = build_model(cfg, device, str(ckpt))
    load_seconds = time.perf_counter() - load_start
    keys = sorted(calibrated)
    output: dict[tuple[int, int], np.ndarray] = {}
    infer_start = time.perf_counter()
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start : start + batch_size]
        images = torch.from_numpy(np.stack([calibrated[k]["image"] for k in batch_keys])).to(device)
        intrinsics = torch.from_numpy(np.stack([calibrated[k]["intrinsic"] for k in batch_keys])).float().to(device)
        cameras = torch.from_numpy(np.stack([calibrated[k]["camera_to_world"] for k in batch_keys])).float().to(device)
        sparse_depth = torch.from_numpy(np.stack([calibrated[k]["sparse_depth"] for k in batch_keys]))[:, None].to(device)
        sparse_mask = torch.from_numpy(np.stack([calibrated[k]["sparse_mask"] for k in batch_keys]))[:, None].float().to(device)
        n = len(batch_keys)
        scene = SceneInput(
            images=images,
            intrinsics=intrinsics,
            cam_to_world=cameras,
            times=torch.zeros(n, device=device),
            frame_index=torch.arange(n, device=device),
            sparse_depth=sparse_depth,
            sparse_mask=sparse_mask,
        )
        prediction = model.predict_depth(scene).float().cpu().numpy()
        for key, depth in zip(batch_keys, prediction):
            output[key] = depth
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_seconds = time.perf_counter() - infer_start
    peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
    del model
    return output, {"model_load": load_seconds, "dense_depth": infer_seconds, "peak_cuda_allocated_gib": peak_gib}


def build_views(
    records: dict[int, list[dict[str, Any]]], calibrated: dict[tuple[int, int], dict[str, np.ndarray]], depths: dict[tuple[int, int], np.ndarray],
    metadata: dict[str, Any], world_from: np.ndarray, raw_mask_dir: Path,
) -> dict[int, list[View]]:
    result: dict[int, list[View]] = defaultdict(list)
    for track_id in FROZEN_TRACKS:
        if track_id not in metadata["instances"]:
            raise KeyError(f"track {track_id} absent from scene metadata")
        for record in records[track_id]:
            frame, camera = int(record["frame"]), int(record["camera"])
            raw_path = raw_mask_dir / f"track-{track_id}" / f"{frame:03d}_{camera}.png"
            if not raw_path.is_file():
                raise FileNotFoundError(f"raw SAM mask missing: {raw_path}")
            raw = np.asarray(Image.open(raw_path).convert("L"), dtype=np.uint8) > 0
            if raw.shape != (H, W):
                raise ValueError(f"raw mask {raw_path} has shape {raw.shape}, expected {(H, W)}")
            pose, dims = metadata["instances"][track_id][frame]
            obj_to_world = world_from @ pose
            box = Box(
                center=obj_to_world[:3, 3].astype(np.float64),
                dims=np.asarray(dims, dtype=np.float64),
                rotation=obj_to_world[:3, :3].T.astype(np.float64),
            )
            datum = calibrated[(frame, camera)]
            result[track_id].append(View(
                track_id=track_id, frame=frame, camera=camera,
                bbox=tuple(float(x) for x in record["bbox_xyxy"]),
                rgb=np.moveaxis(datum["image"], 0, -1), intrinsic=datum["intrinsic"],
                camera_to_world=datum["camera_to_world"], sparse_depth=datum["sparse_depth"],
                sparse_mask=datum["sparse_mask"], dense_depth=depths[(frame, camera)], raw_mask=raw, box=box,
            ))
    return dict(result)


def fit_affine_depth(views_by_track: dict[int, list[View]]) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    for track_id, views in views_by_track.items():
        samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"train": [], "heldout": []}
        conditional_points = raw_points = 0
        for view in views:
            obb, _, _ = ray_box_interval(view.box, view.intrinsic, view.camera_to_world)
            primary = view.raw_mask & view.sparse_mask & np.isfinite(view.dense_depth) & (view.dense_depth > 0)
            conditional = primary & obb
            raw_points += int(primary.sum())
            conditional_points += int(conditional.sum())
            samples[view.split].append((view.dense_depth[primary].astype(np.float64), view.sparse_depth[primary].astype(np.float64)))
        joined = {k: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) if v else (np.empty(0), np.empty(0)) for k, v in samples.items()}
        x_train, y_train = joined["train"]
        x_test, y_test = joined["heldout"]
        design = np.column_stack((x_train, np.ones_like(x_train))) if len(x_train) else np.empty((0, 2))
        condition = float(np.linalg.cond(design)) if len(x_train) >= 2 else float("inf")
        eligible = len(x_train) >= 20 and len(x_test) >= 20 and float(np.std(x_train)) >= 0.25 and condition <= 1e4
        record: dict[str, Any] = {
            "track_id": track_id, "eligible": eligible, "train_points": len(x_train), "heldout_points": len(x_test),
            "train_dense_z_std": float(np.std(x_train)) if len(x_train) else None,
            "design_condition": condition if math.isfinite(condition) else None,
            "primary_raw_lidar_points": raw_points, "conditional_raw_obb_lidar_points": conditional_points,
            "conditional_coverage": ratio(conditional_points, raw_points),
        }
        if eligible:
            coeff = np.linalg.lstsq(design, y_train, rcond=None)[0]
            a, b = float(np.clip(coeff[0], 0.8, 1.2)), float(np.clip(coeff[1], -1.0, 1.0))
            before_errors, after_errors = np.abs(x_test - y_test), np.abs(a * x_test + b - y_test)
            before, after = float(before_errors.mean()), float(after_errors.mean())
            record.update({
                "a": a, "b_m": b, "heldout_mae_before": before, "heldout_mae_after": after,
                "heldout_relative_improvement": relative_gain(before, after),
                "heldout_abs_error_sum_before": float(before_errors.sum()),
                "heldout_abs_error_sum_after": float(after_errors.sum()),
            })
        tracks.append(record)
    eligible = [r for r in tracks if r["eligible"]]
    n = sum(r["heldout_points"] for r in eligible)
    before_sum = sum(r["heldout_abs_error_sum_before"] for r in eligible)
    after_sum = sum(r["heldout_abs_error_sum_after"] for r in eligible)
    point_before, point_after = ratio(before_sum, n), ratio(after_sum, n)
    macro_before = float(np.mean([r["heldout_mae_before"] for r in eligible])) if eligible else None
    macro_after = float(np.mean([r["heldout_mae_after"] for r in eligible])) if eligible else None
    improved = sum((r["heldout_relative_improvement"] or -math.inf) >= 0.15 for r in eligible)
    sufficient = len(eligible) >= 5 and n >= 100
    high = bool(sufficient and relative_gain(point_before, point_after) >= 0.25 and relative_gain(macro_before, macro_after) >= 0.20 and improved / len(eligible) >= 0.60)
    return {
        "tracks": tracks,
        "aggregate": {
            "eligible_tracks": len(eligible), "heldout_points": n, "point_micro_mae_before": point_before,
            "point_micro_mae_after": point_after, "point_micro_relative_improvement": relative_gain(point_before, point_after) if eligible else None,
            "track_macro_mae_before": macro_before, "track_macro_mae_after": macro_after,
            "track_macro_relative_improvement": relative_gain(macro_before, macro_after) if eligible else None,
            "tracks_improved_at_least_15pct": improved, "improved_fraction": ratio(improved, len(eligible)),
            "coverage_sufficient": sufficient, "evidence": "high" if high else ("low" if sufficient else "unknown"),
        },
    }


def parameter_tie_key(params: np.ndarray) -> tuple[Any, ...]:
    identity = bool(np.all(params == 0.0))
    l1 = float(np.abs(params[:3]).sum() / 0.5 + abs(params[3]) / 5.0)
    return (0 if identity else 1, l1, *[float(x) for x in params])


def better_counts(candidate: tuple[int, int], incumbent: tuple[int, int]) -> int:
    """Return positive iff candidate IoU is larger, without float tie noise."""
    ci, cu = candidate
    ii, iu = incumbent
    if cu == 0 or iu == 0:
        return (ci > ii) - (ci < ii)
    return (ci * iu > ii * cu) - (ci * iu < ii * cu)


def masks_for(views: list[View], params: np.ndarray | None = None, scale: float = 1.0) -> list[np.ndarray]:
    return [ray_box_interval(correct_box(v.box, params, scale), v.intrinsic, v.camera_to_world)[0] for v in views]


def fit_pose(views: list[View], targets: list[np.ndarray] | None = None, boxes: list[Box] | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    train = [v for v in views if v.split == "train"]
    target_by_id = {
        (v.frame, v.camera): (v.raw_mask if targets is None else targets[i])
        for i, v in enumerate(views)
    }
    box_by_id = {
        (v.frame, v.camera): (v.box if boxes is None else boxes[i])
        for i, v in enumerate(views)
    }

    def objective(params: np.ndarray) -> tuple[int, int]:
        predictions = [ray_box_interval(correct_box(box_by_id[(v.frame, v.camera)], params), v.intrinsic, v.camera_to_world)[0] for v in train]
        return counts_iou([target_by_id[(v.frame, v.camera)] for v in train], predictions)[:2]

    current = np.zeros(4, dtype=np.float64)
    score = objective(current)
    for translation_step, yaw_step in ((0.25, 2.5), (0.10, 1.0)):
        steps = (translation_step, translation_step, translation_step, yaw_step)
        for _ in range(2):
            for axis, step in enumerate(steps):
                options = []
                for delta in (-step, 0.0, step):
                    candidate = current.copy()
                    candidate[axis] += delta
                    options.append((candidate, objective(candidate)))
                best_p, best_s = options[0]
                for p, s in options[1:]:
                    comparison = better_counts(s, best_s)
                    if comparison > 0 or (comparison == 0 and parameter_tie_key(p) < parameter_tie_key(best_p)):
                        best_p, best_s = p, s
                current, score = best_p, best_s
    return current, score


def fit_extent(views: list[View], targets: list[np.ndarray] | None = None, boxes: list[Box] | None = None) -> tuple[float, tuple[int, int]]:
    train = [v for v in views if v.split == "train"]
    target_by_id = {
        (v.frame, v.camera): (v.raw_mask if targets is None else targets[i])
        for i, v in enumerate(views)
    }
    box_by_id = {
        (v.frame, v.camera): (v.box if boxes is None else boxes[i])
        for i, v in enumerate(views)
    }
    candidates = []
    for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
        pred = [ray_box_interval(correct_box(box_by_id[(v.frame, v.camera)], scale=scale), v.intrinsic, v.camera_to_world)[0] for v in train]
        candidates.append((scale, counts_iou([target_by_id[(v.frame, v.camera)] for v in train], pred)[:2]))
    best = candidates[0]
    for candidate in candidates[1:]:
        comparison = better_counts(candidate[1], best[1])
        candidate_key = (0 if candidate[0] == 1.0 else 1, abs(candidate[0] - 1.0), candidate[0])
        best_key = (0 if best[0] == 1.0 else 1, abs(best[0] - 1.0), best[0])
        if comparison > 0 or (comparison == 0 and candidate_key < best_key):
            best = candidate
    return best


def box_geometry(views_by_track: dict[int, list[View]], visual_dir: Path) -> dict[str, Any]:
    tracks = []
    visual_dir.mkdir(parents=True, exist_ok=True)
    for track_id, views in views_by_track.items():
        baseline = masks_for(views)
        eligible_train = [i for i, v in enumerate(views) if v.split == "train" and v.raw_mask.sum() >= 64 and baseline[i].sum() >= 64]
        eligible_test = [i for i, v in enumerate(views) if v.split == "heldout" and v.raw_mask.sum() >= 64 and baseline[i].sum() >= 64]
        eligible = len(eligible_train) >= 4 and len(eligible_test) >= 4
        record: dict[str, Any] = {"track_id": track_id, "eligible": eligible, "train_views": len(eligible_train), "heldout_views": len(eligible_test)}
        if eligible:
            fit_views = [views[i] for i in eligible_train + eligible_test]
            params, train_pose = fit_pose(fit_views)
            scale, train_extent = fit_extent(fit_views)
            heldout = [views[i] for i in eligible_test]
            target = [v.raw_mask for v in heldout]
            before = [baseline[i] for i in eligible_test]
            pose_after = masks_for(heldout, params=params)
            extent_after = masks_for(heldout, scale=scale)
            bi, bu, b_iou = counts_iou(target, before)
            pi, pu, p_iou = counts_iou(target, pose_after)
            ei, eu, e_iou = counts_iou(target, extent_after)
            record.update({
                "pose_params_dx_dy_dz_yaw_deg": params, "extent_scale": scale,
                "train_pose_intersection_union": train_pose, "train_extent_intersection_union": train_extent,
                "heldout_baseline_intersection": bi, "heldout_baseline_union": bu, "heldout_baseline_iou": b_iou,
                "heldout_pose_intersection": pi, "heldout_pose_union": pu, "heldout_pose_iou": p_iou,
                "heldout_pose_iou_gain": p_iou - b_iou,
                "heldout_extent_intersection": ei, "heldout_extent_union": eu, "heldout_extent_iou": e_iou,
                "heldout_extent_iou_gain": e_iou - b_iou,
            })
            make_track_comparison(track_id, heldout, before, pose_after, extent_after, visual_dir / f"track-{track_id}_heldout_before_after.jpg")
        tracks.append(record)

    def aggregate(branch: str) -> dict[str, Any]:
        eligible = [r for r in tracks if r["eligible"]]
        before_i = sum(r["heldout_baseline_intersection"] for r in eligible)
        before_u = sum(r["heldout_baseline_union"] for r in eligible)
        after_i = sum(r[f"heldout_{branch}_intersection"] for r in eligible)
        after_u = sum(r[f"heldout_{branch}_union"] for r in eligible)
        before_micro, after_micro = ratio(before_i, before_u), ratio(after_i, after_u)
        before_macro = float(np.mean([r["heldout_baseline_iou"] for r in eligible])) if eligible else None
        after_macro = float(np.mean([r[f"heldout_{branch}_iou"] for r in eligible])) if eligible else None
        improved = sum(r[f"heldout_{branch}_iou_gain"] >= 0.10 for r in eligible)
        sufficient = len(eligible) >= 5
        high = bool(sufficient and after_micro - before_micro >= 0.10 and after_macro - before_macro >= 0.08 and improved / len(eligible) >= 0.60)
        return {
            "eligible_tracks": len(eligible), "pixel_micro_iou_before": before_micro, "pixel_micro_iou_after": after_micro,
            "pixel_micro_iou_gain": (after_micro - before_micro) if eligible else None,
            "track_macro_iou_before": before_macro, "track_macro_iou_after": after_macro,
            "track_macro_iou_gain": (after_macro - before_macro) if eligible else None,
            "tracks_improved_at_least_0p10": improved, "improved_fraction": ratio(improved, len(eligible)),
            "coverage_sufficient": sufficient, "evidence": "high" if high else ("low" if sufficient else "unknown"),
        }
    return {"tracks": tracks, "shared_4dof": aggregate("pose"), "extent_only": aggregate("extent")}


def joint_contract(views_by_track: dict[int, list[View]]) -> dict[str, Any]:
    rows = []
    residuals = []
    inside_count = support_count = 0
    for views in views_by_track.values():
        for view in views:
            obb, near, far = ray_box_interval(view.box, view.intrinsic, view.camera_to_world, return_depth=True)
            raw, old = view.raw_mask, bbox_mask(view.bbox)
            support = raw & obb & np.isfinite(view.dense_depth)
            z = view.dense_depth[support]
            zn, zf = near[support], far[support]
            residual = np.where(z < zn, z - zn, np.where(z > zf, z - zf, 0.0))
            residuals.append(residual.astype(np.float64))
            support_count += len(residual)
            inside_count += int((residual == 0).sum())
            rows.append({
                "track_id": view.track_id, "frame": view.frame, "camera": view.camera,
                "raw_pixels": int(raw.sum()), "exact_obb_pixels": int(obb.sum()), "old_bbox_pixels": int(old.sum()),
                "raw_exact_intersection": int((raw & obb).sum()), "raw_old_bbox_intersection": int((raw & old).sum()),
                "raw_exact_iou": counts_iou([raw], [obb])[2], "raw_old_bbox_iou": counts_iou([raw], [old])[2],
                "depth_box_support_pixels": len(residual), "depth_inside_pixels": int((residual == 0).sum()),
            })
    all_res = np.concatenate(residuals) if residuals else np.empty(0)
    miss = np.abs(all_res[all_res != 0])
    return {
        "views": rows,
        "aggregate": {
            "views": len(rows), "raw_pixels": sum(r["raw_pixels"] for r in rows),
            "raw_exact_intersection": sum(r["raw_exact_intersection"] for r in rows),
            "raw_old_bbox_intersection": sum(r["raw_old_bbox_intersection"] for r in rows),
            "depth_box_support_pixels": support_count, "inside_rate": ratio(inside_count, support_count),
            "signed_residual_p10_p50_p90_m": np.percentile(all_res, [10, 50, 90]).tolist() if len(all_res) else None,
            "miss_only_abs_mean_m": float(miss.mean()) if len(miss) else 0.0,
            "miss_only_abs_p50_p90_m": np.percentile(miss, [50, 90]).tolist() if len(miss) else [0.0, 0.0],
        },
    }


def project_points(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(view.camera_to_world)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    uvw = camera @ view.intrinsic.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    u, v = np.rint(uv[:, 0]).astype(np.int64), np.rint(uv[:, 1]).astype(np.int64)
    valid = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack((u, v), axis=-1), np.linalg.norm(camera, axis=-1), valid


def source_world_points(view: View, lidar_only: bool, cap: int) -> np.ndarray:
    mask = view.raw_mask.copy()
    depth = view.sparse_depth if lidar_only else view.dense_depth
    if lidar_only:
        mask &= view.sparse_mask
    mask &= np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(mask)
    if len(xx) > cap:
        choose = np.linspace(0, len(xx) - 1, cap).round().astype(np.int64)
        yy, xx = yy[choose], xx[choose]
    if not len(xx):
        return np.empty((0, 3), dtype=np.float64)
    pix = np.stack((xx + 0.5, yy + 0.5, np.ones_like(xx)), axis=-1).astype(np.float64)
    cam = pix @ np.linalg.inv(view.intrinsic).T
    cam *= (depth[yy, xx] / cam[:, 2])[:, None]
    return cam @ view.camera_to_world[:3, :3].T + view.camera_to_world[:3, 3]


def cross_view_support(views_by_track: dict[int, list[View]], cap: int) -> dict[str, Any]:
    totals = {"dense_all": [0, 0], "dense_self_visible": [0, 0], "lidar_all": [0, 0], "lidar_self_visible": [0, 0]}
    pair_count = 0
    for views in views_by_track.values():
        interval_cache = {(v.frame, v.camera): ray_box_interval(v.box, v.intrinsic, v.camera_to_world, return_depth=True) for v in views}
        for source in views:
            for lidar_only, prefix in ((False, "dense"), (True, "lidar")):
                world = source_world_points(source, lidar_only, cap)
                if not len(world):
                    continue
                local = (world - source.box.center) @ source.box.rotation.T
                for target in views:
                    if target.frame == source.frame and target.camera == source.camera:
                        continue
                    target_world = local @ target.box.rotation + target.box.center
                    uv, ray_t, valid = project_points(target_world, target)
                    if not valid.any():
                        continue
                    u, v = uv[valid, 0], uv[valid, 1]
                    hits = target.raw_mask[v, u]
                    totals[f"{prefix}_all"][0] += int(hits.sum())
                    totals[f"{prefix}_all"][1] += len(hits)
                    obb, near, _ = interval_cache[(target.frame, target.camera)]
                    target_near_z = near[v, u]
                    # Convert target z-near back to unit-ray distance.  In
                    # calibrated camera coordinates cos(ray, forward) is just
                    # the normalised z component; evaluate only queried pixels
                    # instead of materialising another full-frame ray grid.
                    pix = np.stack((u + 0.5, v + 0.5, np.ones_like(u)), axis=-1).astype(np.float64)
                    rays_camera = pix @ np.linalg.inv(target.intrinsic).T
                    cos_uv = rays_camera[:, 2] / np.maximum(np.linalg.norm(rays_camera, axis=-1), 1e-12)
                    target_near_t = target_near_z / np.maximum(cos_uv, 1e-12)
                    self_visible = obb[v, u] & np.isfinite(target_near_t) & (np.abs(ray_t[valid] - target_near_t) <= np.maximum(0.10, 0.02 * target_near_t))
                    totals[f"{prefix}_self_visible"][0] += int(hits[self_visible].sum())
                    totals[f"{prefix}_self_visible"][1] += int(self_visible.sum())
            pair_count += max(0, len(views) - 1)
    return {"ordered_source_target_pairs": pair_count, "source_point_cap_per_view": cap, **{k: {"hits": v[0], "eligible": v[1], "hit_rate": ratio(v[0], v[1])} for k, v in totals.items()}}


def analytic_identity_tests() -> dict[str, Any]:
    source_a = yaw_matrix(31.0)
    target_a = yaw_matrix(-19.0)
    source = Box(np.asarray([2.0, -3.0, 1.0]), np.asarray([4.0, 2.0, 1.5]), source_a.T)
    target = Box(np.asarray([-1.0, 5.0, 0.2]), source.dims.copy(), target_a.T)
    local = np.asarray([0.7, -0.2, 0.1])
    world = source.rotation.T @ local + source.center
    recovered = source.rotation @ (world - source.center)
    target_world = target.rotation.T @ recovered + target.center
    target_roundtrip = target.rotation @ (target_world - target.center)
    corrected = correct_box(source, np.zeros(4), 1.0)
    error = max(
        float(np.max(np.abs(recovered - local))), float(np.max(np.abs(target_roundtrip - local))),
        float(np.max(np.abs(corrected.center - source.center))), float(np.max(np.abs(corrected.rotation - source.rotation))),
        float(np.max(np.abs(corrected.dims - source.dims))),
    )
    return {"max_abs_error": error, "threshold": 1e-9, "passed": error <= 1e-9}


def synthetic_depth_calibration() -> dict[str, Any]:
    # Ten analytic tracks with distinct clean ranges make the calibration
    # independent of both LiDAR values and SAM support while exercising the
    # formal point-micro, track-macro, and per-track-improvement decisions.
    clean_tracks = [
        (
            np.linspace(3.0 + i, 45.0 + 1.5 * i, 100, dtype=np.float64),
            np.linspace(3.05 + i, 44.95 + 1.5 * i, 100, dtype=np.float64),
        )
        for i in range(10)
    ]
    passes = []
    for kind, truth in (("bias", -0.5), ("bias", 0.5), ("scale", 0.90), ("scale", 1.10)):
        clean_train = np.concatenate([x for x, _ in clean_tracks])
        corrupt_train = clean_train - truth if kind == "bias" else clean_train / truth
        a, b = np.linalg.lstsq(np.column_stack((corrupt_train, np.ones_like(corrupt_train))), clean_train, rcond=None)[0]
        a, b = float(np.clip(a, 0.8, 1.2)), float(np.clip(b, -1.0, 1.0))
        track_metrics = []
        for _, clean_test in clean_tracks:
            corrupt_test = clean_test - truth if kind == "bias" else clean_test / truth
            before = float(np.mean(np.abs(corrupt_test - clean_test)))
            after = float(np.mean(np.abs(a * corrupt_test + b - clean_test)))
            track_metrics.append((len(clean_test), before, after, relative_gain(before, after)))
        total = sum(n for n, _, _, _ in track_metrics)
        point_before = sum(n * before for n, before, _, _ in track_metrics) / total
        point_after = sum(n * after for n, _, after, _ in track_metrics) / total
        macro_before = float(np.mean([before for _, before, _, _ in track_metrics]))
        macro_after = float(np.mean([after for _, _, after, _ in track_metrics]))
        point_gain = relative_gain(point_before, point_after)
        macro_gain = relative_gain(macro_before, macro_after)
        improved = sum(gain >= 0.15 for _, _, _, gain in track_metrics) / len(track_metrics)
        recovery = abs((b if kind == "bias" else a) - truth)
        tolerance = 0.11 if kind == "bias" else 0.01
        passes.append({
            "kind": kind, "truth": truth, "fit_a": a, "fit_b": b,
            "recovery_error": recovery, "point_micro_relative_improvement": point_gain,
            "track_macro_relative_improvement": macro_gain, "tracks_improved_fraction": improved,
            "passed": recovery <= tolerance and point_gain >= 0.25 and macro_gain >= 0.20 and improved >= 0.60,
        })
    identity = max(float(np.max(np.abs(clean - (1.0 * clean + 0.0)))) for pair in clean_tracks for clean in pair)
    return {"oracle": "analytic frozen clean z; no LiDAR/SAM target", "passes": passes, "identity_max_abs_error": identity, "passed": all(p["passed"] for p in passes) and identity <= 1e-9}


def identity_metric_noop(views_by_track: dict[int, list[View]]) -> dict[str, Any]:
    """Recompute identity depth/box metrics and require bit-exact no-op gain."""
    max_mask_difference = 0
    max_depth_difference = 0.0
    max_false_gain = 0.0
    for views in views_by_track.values():
        for view in views:
            baseline = ray_box_interval(view.box, view.intrinsic, view.camera_to_world)[0]
            identity = ray_box_interval(correct_box(view.box, np.zeros(4), 1.0), view.intrinsic, view.camera_to_world)[0]
            max_mask_difference = max(max_mask_difference, int(np.logical_xor(baseline, identity).sum()))
            depth_identity = 1.0 * view.dense_depth + 0.0
            max_depth_difference = max(max_depth_difference, float(np.max(np.abs(depth_identity - view.dense_depth))))
            if view.split == "heldout":
                target = view.raw_mask
                before = counts_iou([target], [baseline])[2]
                after = counts_iou([target], [identity])[2]
                max_false_gain = max(max_false_gain, abs(after - before))
    max_abs = max(float(max_mask_difference), max_depth_difference)
    return {
        "max_abs_stat_difference": max_abs, "heldout_max_abs_false_iou_gain": max_false_gain,
        "threshold": 1e-9, "passed": max_abs <= 1e-9 and max_false_gain <= 1e-9,
    }


def corrupt_box(box: Box, axis: str, truth: float) -> Box:
    if axis in ("dx", "dy", "dz"):
        delta = np.zeros(3)
        delta[("dx", "dy", "dz").index(axis)] = truth
        a = box.rotation.T
        return Box(box.center - a @ delta, box.dims.copy(), box.rotation.copy())
    if axis == "yaw":
        d = yaw_matrix(truth)
        # A_corrupt=A_clean D^T, therefore R_corrupt=D R_clean.
        return Box(box.center.copy(), box.dims.copy(), d @ box.rotation)
    if axis == "extent":
        return Box(box.center.copy(), box.dims / truth, box.rotation.copy())
    raise ValueError(axis)


def synthetic_camera_to_world(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Create an OpenCV camera pose looking at ``target`` with world z up."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = position
    return pose


def synthetic_box_views() -> dict[int, list[View]]:
    """Independent near-range oblique views, deliberately sensitive to 5° yaw."""
    intrinsic = np.asarray([[600.0, 0.0, W / 2], [0.0, 600.0, H / 2], [0.0, 0.0, 1.0]])
    # Four odd train and four even held-out timestamps.  Alternating shallow
    # oblique angles expose both the long and short faces of an elongated car;
    # An exaggerated-but-vehicle-shaped 7:1.4 planform makes 5 degree yaw and
    # 0.5 m translations decisive without a huge raster ROI.
    azimuths = (8.0, -10.0, 12.0, -14.0, 16.0, -18.0, 20.0, -22.0)
    frames = (1, 2, 3, 4, 5, 6, 7, 8)
    result: dict[int, list[View]] = {}
    zeros_rgb = np.zeros((H, W, 3), dtype=np.float32)
    zeros_float = np.zeros((H, W), dtype=np.float32)
    zeros_bool = np.zeros((H, W), dtype=bool)
    for track_id in range(10):
        views = []
        clean_box = Box(
            center=np.asarray([0.0, 0.0, 0.8]),
            dims=np.asarray([8.0 + 0.03 * track_id, 1.20, 1.55]),
            rotation=yaw_matrix(0.2 * track_id - 0.9).T,
        )
        for camera, (frame, azimuth) in enumerate(zip(frames, azimuths)):
            angle = math.radians(azimuth + 0.05 * track_id)
            distance = 15.0 + 0.05 * track_id
            position = clean_box.center + np.asarray([
                distance * math.cos(angle), distance * math.sin(angle), 1.15 + 0.08 * ((camera % 3) - 1),
            ])
            camera_to_world = synthetic_camera_to_world(position, clean_box.center)
            clean_mask = ray_box_interval(clean_box, intrinsic, camera_to_world)[0]
            views.append(View(
                track_id=track_id, frame=frame, camera=camera, bbox=(0.0, 0.0, W - 1.0, H - 1.0),
                rgb=zeros_rgb, intrinsic=intrinsic, camera_to_world=camera_to_world,
                sparse_depth=zeros_float, sparse_mask=zeros_bool, dense_depth=zeros_float,
                raw_mask=clean_mask, box=clean_box,
            ))
        result[track_id] = views
    return result


def fit_synthetic_single_axis(views: list[View], targets: list[np.ndarray], boxes: list[Box], axis: int) -> np.ndarray:
    """Formal coordinate descent restricted to the injected target axis."""
    train = [i for i, view in enumerate(views) if view.split == "train"]

    def objective(params: np.ndarray) -> tuple[int, int]:
        predictions = [
            ray_box_interval(correct_box(boxes[i], params), views[i].intrinsic, views[i].camera_to_world)[0]
            for i in train
        ]
        return counts_iou([targets[i] for i in train], predictions)[:2]

    current = np.zeros(4, dtype=np.float64)
    for translation_step, yaw_step in ((0.25, 2.5), (0.10, 1.0)):
        step = yaw_step if axis == 3 else translation_step
        for _ in range(2):
            options = []
            for delta in (-step, 0.0, step):
                candidate = current.copy()
                candidate[axis] += delta
                options.append((candidate, objective(candidate)))
            best_p, best_s = options[0]
            for params, score in options[1:]:
                comparison = better_counts(score, best_s)
                if comparison > 0 or (comparison == 0 and parameter_tie_key(params) < parameter_tie_key(best_p)):
                    best_p, best_s = params, score
            current = best_p
    return current


def synthetic_box_calibration(_real_views_by_track: dict[int, list[View]]) -> dict[str, Any]:
    views_by_track = synthetic_box_views()
    passes = []
    specifications = [(a, s) for a in ("dx", "dy", "dz") for s in (-0.5, 0.5)] + [("yaw", s) for s in (-5.0, 5.0)] + [("extent", s) for s in (0.90, 1.10)]
    for axis, truth in specifications:
        per_track = []
        for track_id, views in views_by_track.items():
            targets = masks_for(views)  # independent clean OBB oracle, never SAM
            eligible = sum(v.split == "train" and m.sum() >= 64 for v, m in zip(views, targets)) >= 4 and sum(v.split == "heldout" and m.sum() >= 64 for v, m in zip(views, targets)) >= 4
            if not eligible:
                continue
            boxes = [corrupt_box(v.box, axis, truth) for v in views]
            if axis == "extent":
                estimate, _ = fit_extent(views, targets, boxes)
                params = np.zeros(4)
                scale = estimate
                recovery = abs(estimate - truth)
                non_target = 0.0
            else:
                index = ("dx", "dy", "dz", "yaw").index(axis)
                params = fit_synthetic_single_axis(views, targets, boxes, index)
                scale = 1.0
                recovery = abs(params[index] - truth)
                non_target = float(np.max(np.abs(np.delete(params, index))))
            held = [i for i, v in enumerate(views) if v.split == "heldout" and targets[i].sum() >= 64]
            before = [ray_box_interval(boxes[i], views[i].intrinsic, views[i].camera_to_world)[0] for i in held]
            after = [ray_box_interval(correct_box(boxes[i], params, scale), views[i].intrinsic, views[i].camera_to_world)[0] for i in held]
            target = [targets[i] for i in held]
            bi, bu, b = counts_iou(target, before)
            ai, au, a = counts_iou(target, after)
            per_track.append({"track_id": track_id, "before_i": bi, "before_u": bu, "after_i": ai, "after_u": au, "before": b, "after": a, "gain": a - b, "recovery_error": recovery, "non_target_max_abs": non_target})
        before_i, before_u = sum(x["before_i"] for x in per_track), sum(x["before_u"] for x in per_track)
        after_i, after_u = sum(x["after_i"] for x in per_track), sum(x["after_u"] for x in per_track)
        micro_gain = ratio(after_i, after_u) - ratio(before_i, before_u) if per_track else None
        macro_gain = float(np.mean([x["after"] for x in per_track]) - np.mean([x["before"] for x in per_track])) if per_track else None
        improved = ratio(sum(x["gain"] >= 0.10 for x in per_track), len(per_track))
        tolerance = 1.1 if axis == "yaw" else (0.01 if axis == "extent" else 0.11)
        max_recovery = max((x["recovery_error"] for x in per_track), default=float("inf"))
        max_non_target = max((x["non_target_max_abs"] for x in per_track), default=float("inf"))
        passed = len(per_track) >= 5 and micro_gain >= 0.10 and macro_gain >= 0.08 and improved >= 0.60 and max_recovery <= tolerance and max_non_target <= 1e-12
        passes.append({"axis": axis, "truth": truth, "eligible_tracks": len(per_track), "pixel_micro_gain": micro_gain, "track_macro_gain": macro_gain, "improved_fraction": improved, "max_recovery_error": max_recovery, "max_non_target_abs": max_non_target, "passed": passed})
    return {
        "oracle": "independent deterministic clean synthetic OBB silhouettes; no real LiDAR/SAM target",
        "tracks": 10, "views_per_track": 8,
        "passes": passes, "passed": all(p["passed"] for p in passes),
    }


def overlay_image(view: View, obb: np.ndarray) -> np.ndarray:
    rgb = np.clip(view.rgb * 255.0, 0, 255).astype(np.uint8)
    out = rgb.copy()
    raw_boundary = mask_boundary(view.raw_mask)
    obb_boundary = mask_boundary(obb)
    out[raw_boundary] = (255, 40, 180)
    out[obb_boundary] = (30, 255, 70)
    finite = np.isfinite(view.dense_depth) & (view.dense_depth > 0)
    if finite.any():
        lo, hi = np.percentile(view.dense_depth[finite], [5, 95])
        heat = np.clip((view.dense_depth - lo) / max(hi - lo, 1e-6), 0, 1)
        tint = np.stack((255 * heat, 80 * np.ones_like(heat), 255 * (1 - heat)), axis=-1).astype(np.uint8)
        region = view.raw_mask
        out[region] = (0.65 * out[region] + 0.35 * tint[region]).astype(np.uint8)
    yy, xx = np.nonzero(view.sparse_mask & view.raw_mask)
    out[yy, xx] = (255, 255, 0)
    return out


def make_track_comparison(track_id: int, views: list[View], before: list[np.ndarray], pose: list[np.ndarray], extent: list[np.ndarray], path: Path) -> None:
    tiles = []
    for view, b, p, e in list(zip(views, before, pose, extent))[:6]:
        tile = np.clip(view.rgb * 255, 0, 255).astype(np.uint8)
        tile[mask_boundary(view.raw_mask)] = (255, 30, 180)
        tile[mask_boundary(b)] = (30, 220, 255)
        tile[mask_boundary(p)] = (30, 255, 70)
        tile[mask_boundary(e)] = (255, 150, 20)
        tiles.append(np.asarray(Image.fromarray(tile).resize((384, 288))))
    if not tiles:
        return
    canvas = np.zeros((288 * math.ceil(len(tiles) / 3), 384 * 3, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        canvas[(i // 3) * 288 : (i // 3 + 1) * 288, (i % 3) * 384 : (i % 3 + 1) * 384] = tile
    Image.fromarray(canvas).save(path, quality=92)


def create_overlays(views_by_track: dict[int, list[View]], output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for track_id in FROZEN_TRACKS:
        # Deterministically prefer the most informative (largest raw support)
        # observation, then break ties temporally.
        view = sorted(
            views_by_track[track_id],
            key=lambda v: (-int(v.raw_mask.sum()), v.frame, v.camera),
        )[0]
        obb = ray_box_interval(view.box, view.intrinsic, view.camera_to_world)[0]
        path = output / f"track-{track_id}_frame-{view.frame:03d}_camera-{view.camera}_overlay.jpg"
        Image.fromarray(overlay_image(view, obb)).save(path, quality=93)
        paths.append(str(path))
    return paths


def write_report(path: Path, result: dict[str, Any]) -> None:
    d = result["decision"]
    depth = result["depth_anchor"]["aggregate"]
    pose = result["box_geometry_anchor"]["shared_4dof"]
    extent = result["box_geometry_anchor"]["extent_only"]
    timing = result["timing_seconds"]
    lines = [
        "# Round 63 dynamic depth/box geometry contract probe", "",
        "> Diagnostic only: no Gaussian payload was changed; reconstruction acceptance is false.", "",
        "## Decision", "",
        f"- Probe validity: **{d['probe_valid']}**",
        f"- Depth evidence: **{d['depth']}**",
        f"- Box geometry evidence: **{d['box_geometry']}** (4DoF={pose['evidence']}, extent={extent['evidence']})",
        f"- 2×2 cell: **{d['cell']}**", "",
        "## Independent anchors", "",
        f"- Depth eligible tracks / held-out points: {depth['eligible_tracks']} / {depth['heldout_points']}",
        f"- Depth point-micro MAE: {depth['point_micro_mae_before']} → {depth['point_micro_mae_after']}",
        f"- Depth track-macro MAE: {depth['track_macro_mae_before']} → {depth['track_macro_mae_after']}",
        f"- 4DoF pixel-micro IoU gain: {pose['pixel_micro_iou_gain']}",
        f"- Extent pixel-micro IoU gain: {extent['pixel_micro_iou_gain']}", "",
        "## Calibration and timing", "",
        f"- Analytic identity/roundtrip: {result['calibration']['analytic']['passed']}",
        f"- Identity metric no-op: {result['calibration']['identity_metrics']['passed']}",
        f"- Synthetic depth: {result['calibration']['depth']['passed']}",
        f"- Synthetic box: {result['calibration']['box']['passed']}",
        f"- Total wall: {timing['total']:.3f} s (model load {timing['model_load']:.3f}, depth {timing['dense_depth']:.3f})", "",
        "Full per-view/per-track metrics are in `dynamic_depth_box_contract.json`. Visual legend: raw SAM magenta, original OBB green; correction panels additionally use baseline cyan, 4DoF green, extent orange.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    for p, label in ((args.config, "config"), (args.ckpt, "checkpoint"), (args.track_manifest, "track manifest")):
        if not p.is_file():
            raise FileNotFoundError(f"{label} missing: {p}")
    if not args.raw_mask_dir.is_dir():
        raise FileNotFoundError(f"raw mask directory missing: {args.raw_mask_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    _, records = read_manifest(args.track_manifest)
    cfg, dataset, metadata, world_from = prepare_dataset(args.config, args.scene)
    requested = {(int(v["frame"]), int(v["camera"])) for views in records.values() for v in views}
    calibrated = collect_selected_views(dataset, metadata, world_from, requested)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    depths, model_timing = predict_depths(cfg, args.ckpt, device, calibrated, args.depth_batch_size)
    views = build_views(records, calibrated, depths, metadata, world_from, args.raw_mask_dir)
    audit_start = time.perf_counter()
    joint = joint_contract(views)
    depth = fit_affine_depth(views)
    box = box_geometry(views, args.out / "heldout-box-corrections")
    cross = cross_view_support(views, args.cross_view_max_points)
    overlays = create_overlays(views, args.out / "overlays")
    analytic = analytic_identity_tests()
    identity_metrics = identity_metric_noop(views)
    synthetic_depth = synthetic_depth_calibration()
    synthetic_box = synthetic_box_calibration(views)
    audit_seconds = time.perf_counter() - audit_start
    depth_state = depth["aggregate"]["evidence"]
    pose_states = (box["shared_4dof"]["evidence"], box["extent_only"]["evidence"])
    box_state = "high" if "high" in pose_states else ("unknown" if "unknown" in pose_states else "low")
    valid = bool(analytic["passed"] and identity_metrics["passed"] and synthetic_depth["passed"] and synthetic_box["passed"])
    cell = f"depth-{depth_state}/box-{box_state}" if valid else "invalid-calibration"
    total = time.perf_counter() - started
    raw_mask_files = []
    for track_id in FROZEN_TRACKS:
        for record in records[track_id]:
            frame, camera = int(record["frame"]), int(record["camera"])
            path = args.raw_mask_dir / f"track-{track_id}" / f"{frame:03d}_{camera}.png"
            raw_mask_files.append({
                "track_id": track_id, "frame": frame, "camera": camera,
                "path": str(path.resolve()), "sha256": sha256_file(path),
            })
    result = {
        "schema_version": 1, "status": "completed" if valid else "invalid",
        "probe": "SOL-072 dynamic RGB/depth/OBB geometry contract", "scene": str(args.scene),
        "diagnostic_only": True, "gaussian_payload_modified": False,
        "acceptance": {"ACC_QUALITY_TIME_OR": False, "ACC_VIS": len(overlays) >= 10},
        "inputs": {
            "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
            "checkpoint": {"path": str(args.ckpt.resolve()), "sha256": sha256_file(args.ckpt)},
            "track_manifest": {"path": str(args.track_manifest.resolve()), "sha256": sha256_file(args.track_manifest)},
            "raw_mask_dir": str(args.raw_mask_dir.resolve()), "raw_mask_files": raw_mask_files,
            "tracks": list(FROZEN_TRACKS), "prompts": 120,
            "unique_images": len(requested), "split": "odd frame=train; even frame=heldout; all cameras at a timestamp stay together",
        },
        "decision": {"probe_valid": valid, "depth": depth_state if valid else "invalid", "box_geometry": box_state if valid else "invalid", "cell": cell},
        "joint_contract": joint, "depth_anchor": depth, "box_geometry_anchor": box, "cross_view_support": cross,
        "calibration": {"analytic": analytic, "identity_metrics": identity_metrics, "depth": synthetic_depth, "box": synthetic_box},
        "visualizations": {"overlays": overlays, "heldout_correction_dir": str((args.out / "heldout-box-corrections").resolve())},
        "timing_seconds": {**model_timing, "geometry_audit_visualization_and_synthetic": audit_seconds, "total": total},
    }
    json_path = args.out / "dynamic_depth_box_contract.json"
    json_path.write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.out / "report.md", result)
    print(json.dumps({"status": result["status"], "decision": result["decision"], "json": str(json_path), "report": str(args.out / "report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

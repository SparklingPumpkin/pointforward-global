"""Evaluate visible dynamic-Gaussian contribution from the production rasterization.

The 129th Gaussian color channel is 1 for dynamic Gaussians and 0 for static
Gaussians.  It is rasterized together with the production 128-D feature, so
the last rendered channel is the unnormalised visible dynamic contribution
``sum_i T_i * alpha_i * is_dynamic_i``.  Fine vehicle masks are diagnostic
only and never define support.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import TargetViews, box_dynamic_masks, render_features
from scripts.deliver_continuous_scene import horizontal_strip, label, to_uint8
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def psnr_from_sse(sse: float, scalar_count: int) -> float | None:
    if scalar_count == 0:
        return None
    return -10.0 * math.log10(max(sse / scalar_count, 1e-10))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_frames(text: str, available: list[int]) -> list[int]:
    if not text.strip():
        return available
    selected = [int(value) for value in text.split(",") if value.strip()]
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown frames: {sorted(unknown)}")
    return selected


def support_partitions(
    baseline_support: torch.Tensor, candidate_support: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the anti-gaming union and candidate-only newly claimed support."""
    return baseline_support | candidate_support, candidate_support & ~baseline_support


def comparison_numeric_test() -> dict:
    """Exercise union/partition SSE and ratio gates with a known fixture."""
    baseline_support = torch.tensor([True, True, False, False])
    candidate_support = torch.tensor([True, True, True, False])
    union, new = support_partitions(baseline_support, candidate_support)
    baseline_error = torch.tensor([0.01, 0.04, 0.09, 0.0], dtype=torch.float64)
    candidate_error = torch.tensor([0.01, 0.04, 0.36, 0.0], dtype=torch.float64)
    baseline_mass = 1.0
    candidate_mass = 1.5
    union_count = int(union.sum())
    frozen_count = int(baseline_support.sum())
    new_count = int(new.sum())
    observed = {
        "union_pixels": union_count,
        "frozen_pixels": frozen_count,
        "new_pixels": new_count,
        "baseline_union_psnr_db": psnr_from_sse(float(baseline_error[union].sum()), union_count),
        "candidate_union_psnr_db": psnr_from_sse(float(candidate_error[union].sum()), union_count),
        "baseline_frozen_psnr_db": psnr_from_sse(float(baseline_error[baseline_support].sum()), frozen_count),
        "candidate_frozen_psnr_db": psnr_from_sse(float(candidate_error[baseline_support].sum()), frozen_count),
        "baseline_new_psnr_db": psnr_from_sse(float(baseline_error[new].sum()), new_count),
        "candidate_new_psnr_db": psnr_from_sse(float(candidate_error[new].sum()), new_count),
        "area_ratio": float(candidate_support.sum()) / float(baseline_support.sum()),
        "contribution_ratio": candidate_mass / baseline_mass,
    }
    expected = {
        "union_pixels": 3, "frozen_pixels": 2, "new_pixels": 1,
        "baseline_union_psnr_db": -10.0 * math.log10(0.14 / 3.0),
        "candidate_union_psnr_db": -10.0 * math.log10(0.41 / 3.0),
        "baseline_frozen_psnr_db": -10.0 * math.log10(0.05 / 2.0),
        "candidate_frozen_psnr_db": -10.0 * math.log10(0.05 / 2.0),
        "baseline_new_psnr_db": -10.0 * math.log10(0.09),
        "candidate_new_psnr_db": -10.0 * math.log10(0.36),
        "area_ratio": 1.5, "contribution_ratio": 1.5,
    }
    max_error = max(
        abs(observed[key] - expected[key]) for key in expected
        if isinstance(expected[key], float)
    )
    dilution_rejected = (
        observed["candidate_union_psnr_db"] < observed["baseline_union_psnr_db"]
        and observed["area_ratio"] > 1.25
        and observed["contribution_ratio"] > 1.25
    )
    counts_match = all(observed[key] == expected[key] for key in (
        "union_pixels", "frozen_pixels", "new_pixels"
    ))
    return {
        "status": "passed" if counts_match and max_error <= 1e-9 and dilution_rejected else "failed",
        "observed": observed, "expected": expected,
        "max_absolute_numeric_error": max_error,
        "tolerance": 1e-9,
        "dilution_rejected": dilution_rejected,
    }


def file_identity(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved), "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def analytic_composition_test(device: torch.device) -> dict:
    """Check depth ordering against singleton alpha measured by the same kernel."""
    intrinsics = torch.tensor(
        [[[32.0, 0.0, 16.5], [0.0, 32.0, 16.5], [0.0, 0.0, 1.0]]],
        device=device,
    )
    world_to_cam = torch.eye(4, device=device).unsqueeze(0)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scale = torch.full((1, 3), 0.25, device=device)
    centre = (16, 16)

    def render(depths: list[float], opacities: list[float], dynamic: list[float]):
        means = torch.tensor([[0.0, 0.0, z] for z in depths], device=device)
        quats = quat.expand(len(depths), -1).contiguous()
        scales = scale.expand(len(depths), -1).contiguous()
        opacity = torch.tensor(opacities, device=device)
        colors = torch.tensor(dynamic, device=device).unsqueeze(1)
        feature, alpha, _ = render_features(
            means, quats, scales, opacity, colors, intrinsics, world_to_cam,
            height=33, width=33,
        )
        return float(feature[0, 0, centre[0], centre[1]]), float(
            alpha[0, 0, centre[0], centre[1]]
        )

    _, alpha_static = render([5.0], [0.8], [0.0])
    alpha_dynamic_near, _ = render([5.0], [0.6], [1.0])
    alpha_dynamic_far, _ = render([10.0], [0.6], [1.0])
    static_front, _ = render([5.0, 10.0], [0.8, 0.6], [0.0, 1.0])
    dynamic_front, _ = render([5.0, 10.0], [0.6, 0.8], [1.0, 0.0])
    _, alpha_static_mid = render([10.0], [0.5], [0.0])
    alpha_d1, _ = render([5.0], [0.5], [1.0])
    alpha_d2, _ = render([15.0], [0.5], [1.0])
    interleaved, _ = render(
        [5.0, 10.0, 15.0], [0.5, 0.5, 0.5], [1.0, 0.0, 1.0]
    )
    no_dynamic, _ = render([5.0, 10.0], [0.8, 0.6], [0.0, 0.0])
    expected = {
        "static_front": (1.0 - alpha_static) * alpha_dynamic_far,
        "dynamic_front": alpha_dynamic_near,
        "interleaved": alpha_d1 + (1.0 - alpha_d1) * (1.0 - alpha_static_mid) * alpha_d2,
        "no_dynamic": 0.0,
    }
    observed = {
        "static_front": static_front,
        "dynamic_front": dynamic_front,
        "interleaved": interleaved,
        "no_dynamic": no_dynamic,
    }
    errors = {key: abs(observed[key] - expected[key]) for key in expected}
    return {
        "status": "passed" if max(errors.values()) <= 1e-5 else "failed",
        "geometry": {
            "height_width": [33, 33], "fx_fy": [32.0, 32.0],
            "cx_cy": [16.5, 16.5], "scale": [0.25] * 3,
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "singleton_alpha": {
            "static_z5_opacity08": alpha_static,
            "dynamic_z5_opacity06": alpha_dynamic_near,
            "dynamic_z10_opacity06": alpha_dynamic_far,
        },
        "expected": expected, "observed": observed, "absolute_error": errors,
        "max_absolute_error": max(errors.values()), "tolerance": 1e-5,
        "scope": "production-alpha ordering/T-alpha formula; not boundary coverage",
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--ckpt")
    parser.add_argument("--payload-dir")
    parser.add_argument(
        "--candidate-payload-dir",
        default=None,
        help="Optional candidate payload evaluated on the union of baseline and candidate support.",
    )
    parser.add_argument(
        "--candidate-dynamic-opacity-scale",
        type=float,
        default=None,
        help="Controlled diagnostic candidate: multiply candidate dynamic opacity by this factor.",
    )
    parser.add_argument("--camera-refiner-states", nargs="+")
    parser.add_argument(
        "--candidate-camera-refiner-states",
        nargs="+",
        default=None,
        help="Optional candidate-specific refiner states; defaults to baseline states.",
    )
    parser.add_argument(
        "--candidate-rgb-residual-state", default=None,
        help="Optional per-frame/shared RGB residual state applied to candidate colors.",
    )
    parser.add_argument(
        "--candidate-temporal-mean-state", default=None,
        help=(
            "Optional per-source-frame dynamic XYZ correction state applied after "
            "the normal box pose. The state must contain source_frames and "
            "dynamic_xyz_delta_world_m."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", default="")
    parser.add_argument(
        "--camera-indices", default="",
        help="Comma-separated camera indices to evaluate; default is all cameras.",
    )
    parser.add_argument("--support-threshold", type=float, default=0.05)
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--diagnose-disabled", action="store_true")
    args = parser.parse_args()
    if (args.candidate_dynamic_opacity_scale is not None
            and not 0.0 <= args.candidate_dynamic_opacity_scale <= 2.0):
        parser.error("--candidate-dynamic-opacity-scale must be in [0,2]")
    device = torch.device(args.device)
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analytic = analytic_composition_test(device)
    (output_dir / "analytic_test.json").write_text(
        json.dumps(analytic, indent=2), encoding="utf-8"
    )
    if analytic["status"] != "passed":
        raise RuntimeError(f"Analytic ownership test failed: {analytic}")
    if args.analytic_only:
        print(json.dumps(analytic, indent=2))
        return
    required = (args.config, args.ckpt, args.payload_dir, args.camera_refiner_states)
    if any(value is None for value in required):
        parser.error("config, ckpt, payload-dir and camera-refiner-states are required")

    started = time.perf_counter()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, args.split)
    if len(args.camera_refiner_states) != len(dataset.cameras):
        parser.error("camera-refiner-states must match configured cameras")
    if (args.candidate_camera_refiner_states is not None
            and len(args.candidate_camera_refiner_states) != len(dataset.cameras)):
        parser.error("candidate-camera-refiner-states must match configured cameras")
    metadata = [item for item in dataset.scenes if item["name"] == args.scene][0]
    all_frames = [int(value) for value in metadata["frames"]]
    frames = parse_frames(args.frames, all_frames)
    if args.camera_indices.strip():
        camera_indices = [int(value) for value in args.camera_indices.split(",") if value.strip()]
        invalid = sorted(set(camera_indices) - set(range(len(dataset.cameras))))
        if invalid:
            parser.error(f"camera-indices out of range: {invalid}")
    else:
        camera_indices = list(range(len(dataset.cameras)))
    eval_cameras = [dataset.cameras[index] for index in camera_indices]
    all_slots = {frame: slot for slot, frame in enumerate(all_frames)}
    all_times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    height, width = dataset.resolution

    payload_dir = Path(args.payload_dir).resolve()
    static = load_payload(payload_dir / "final_static_128d.pt", device)
    dynamic = load_payload(payload_dir / "final_dynamic_128d.pt", device)
    full = combine(static, dynamic)
    meta = torch.load(
        payload_dir / "global_dynamic_meta.pt", map_location="cpu", weights_only=True
    )
    boxes = load_boxes(meta["boxes"], device)
    evaluation_boxes = load_boxes(meta.get("evaluation_boxes", meta["boxes"]), device)
    model = build_model(config, device, str(Path(args.ckpt).resolve())).eval()
    refiners = {}
    for camera, path in zip(dataset.cameras, args.camera_refiner_states):
        refiner = copy.deepcopy(model.refiner).to(device).eval()
        state = torch.load(path, map_location="cpu", weights_only=True)
        refiner.load_state_dict(state.get("refiner", state), strict=True)
        refiners[camera] = refiner
    candidate_refiner_paths = (
        args.candidate_camera_refiner_states or args.camera_refiner_states
    )
    candidate_refiners = {}
    for camera, path in zip(dataset.cameras, candidate_refiner_paths):
        if path == args.camera_refiner_states[dataset.cameras.index(camera)]:
            candidate_refiners[camera] = refiners[camera]
            continue
        refiner = copy.deepcopy(model.refiner).to(device).eval()
        state = torch.load(path, map_location="cpu", weights_only=True)
        refiner.load_state_dict(state.get("refiner", state), strict=True)
        candidate_refiners[camera] = refiner

    candidate_full = None
    candidate_boxes = None
    candidate_payload_dir = None
    if (args.candidate_payload_dir is not None
            or args.candidate_dynamic_opacity_scale is not None
            or args.candidate_rgb_residual_state is not None
            or args.candidate_temporal_mean_state is not None):
        candidate_payload_dir = Path(args.candidate_payload_dir or args.payload_dir).resolve()
        same_payload = candidate_payload_dir == payload_dir
        candidate_static = static if same_payload else load_payload(
            candidate_payload_dir / "final_static_128d.pt", device
        )
        candidate_dynamic = dynamic if same_payload else load_payload(
            candidate_payload_dir / "final_dynamic_128d.pt", device
        )
        if args.candidate_dynamic_opacity_scale is not None and not same_payload:
            candidate_dynamic = replace(
                candidate_dynamic,
                opacities=(
                    candidate_dynamic.opacities * args.candidate_dynamic_opacity_scale
                ).clamp(max=0.999),
            )
        candidate_full = full if same_payload else combine(candidate_static, candidate_dynamic)
        candidate_meta = torch.load(
            candidate_payload_dir / "global_dynamic_meta.pt",
            map_location="cpu",
            weights_only=True,
        )
        candidate_boxes = load_boxes(candidate_meta["boxes"], device)
        candidate_evaluation_boxes = load_boxes(
            candidate_meta.get("evaluation_boxes", candidate_meta["boxes"]), device
        )
    candidate_rgb_residual = None
    if args.candidate_rgb_residual_state is not None:
        candidate_rgb_residual = torch.load(
            Path(args.candidate_rgb_residual_state).resolve(),
            map_location="cpu", weights_only=True,
        )
        if candidate_rgb_residual.get("residual_mode") not in ("per-frame", "shared"):
            raise ValueError("candidate RGB residual state must be per-frame or shared")
    candidate_temporal_mean = None
    if args.candidate_temporal_mean_state is not None:
        candidate_temporal_mean = torch.load(
            Path(args.candidate_temporal_mean_state).resolve(),
            map_location="cpu", weights_only=True,
        )
        if candidate_temporal_mean.get("deformation_mode") != (
            "per-frame-dynamic-mean-world-after-box-pose"
        ):
            raise ValueError("unsupported candidate temporal mean deformation mode")
        source_frames = [int(value) for value in candidate_temporal_mean["source_frames"]]
        deltas = candidate_temporal_mean["dynamic_xyz_delta_world_m"]
        if deltas.ndim != 3 or deltas.shape[-1] != 3 or deltas.shape[0] != len(source_frames):
            raise ValueError("candidate temporal mean state has invalid delta shape")
        candidate_temporal_mean["source_frames"] = source_frames
        candidate_temporal_mean["dynamic_xyz_delta_world_m"] = deltas

    records, active_psnr, global_psnr, bbox_psnr = [], [], [], []
    candidate_global_psnr, candidate_bbox_psnr = [], []
    aggregate = {"support_pixels": 0, "contribution_sum": 0.0,
                 "fine_tp": 0, "fine_fp": 0, "fine_fn": 0,
                 "disabled_total_sse": 0.0, "disabled_outside_sse": 0.0}
    parity_max = {"feature": 0.0, "alpha": 0.0, "rgb": 0.0}
    comparison_records = []
    comparison_aggregate = {
        "baseline_union_sse": 0.0,
        "candidate_union_sse": 0.0,
        "union_scalar_count": 0,
        "baseline_frozen_sse": 0.0,
        "candidate_frozen_sse": 0.0,
        "frozen_scalar_count": 0,
        "baseline_new_sse": 0.0,
        "candidate_new_sse": 0.0,
        "new_scalar_count": 0,
        "baseline_support_pixels": 0,
        "candidate_support_pixels": 0,
        "union_support_pixels": 0,
        "candidate_new_support_pixels": 0,
        "baseline_contribution_sum": 0.0,
        "candidate_contribution_sum": 0.0,
        "rgb_max_abs": 0.0,
        "contribution_max_abs": 0.0,
        "support_xor_pixels": 0,
        "candidate_fine_tp": 0,
        "candidate_fine_fp": 0,
        "candidate_fine_fn": 0,
    }
    mask_paths: list[Path] = []
    visual_dir = output_dir / "visual"
    visual_dir.mkdir(exist_ok=True)

    for frame_index, frame in enumerate(frames):
        slot = all_slots[frame]
        target = dataset._collect(
            metadata, [frame], all_slots, all_times, dataset.resolution,
            world_from, lidar=False,
        )
        views = TargetViews(
            intrinsics=target["intrinsics"][camera_indices].to(device),
            cam_to_world=target["cam_to_world"][camera_indices].to(device),
            times=torch.full((len(camera_indices),), float(all_times[slot]), device=device),
            frame_index=torch.full((len(camera_indices),), slot, dtype=torch.long, device=device),
        )
        posed = model.pose_gaussians(full, boxes, slot, views.times[0])
        ownership = (full.instance_id >= 0).to(posed["colors"].dtype).unsqueeze(1)
        colors129 = torch.cat((posed["colors"], ownership), dim=1)
        feature129, alpha129, _ = render_features(
            posed["means"], posed["quats"], posed["scales"], posed["opacities"],
            colors129, views.intrinsics, torch.linalg.inv(views.cam_to_world),
            height, width,
        )
        features, contribution = feature129[:, :128], feature129[:, 128:129]
        rgbs = []
        for camera_index, camera in enumerate(eval_cameras):
            rgbs.append(refiners[camera](
                features[camera_index:camera_index + 1],
                alpha129[camera_index:camera_index + 1],
            ))
        rgb = torch.cat(rgbs, dim=0)
        support = contribution >= args.support_threshold
        truth = target["images"][camera_indices].to(device)
        error = (rgb.double() - truth.double()).square()
        # Candidate comparison performs a second 20M-Gaussian render. Release
        # baseline-only full-scene intermediates before allocating it; the
        # compact RGB/contribution/support tensors remain live for pairing.
        if candidate_full is not None and not args.diagnose_disabled:
            del posed, ownership, colors129, feature129, features, alpha129, rgbs

        candidate_support = None
        candidate_contribution = None
        candidate_rgb = None
        if candidate_full is not None:
            candidate_posed = model.pose_gaussians(
                candidate_full, candidate_boxes, slot, views.times[0]
            )
            if candidate_temporal_mean is not None:
                source_frames = candidate_temporal_mean["source_frames"]
                if int(frame) in source_frames:
                    delta_slot = source_frames.index(int(frame))
                    delta = candidate_temporal_mean[
                        "dynamic_xyz_delta_world_m"
                    ][delta_slot].to(device)
                    candidate_dynamic_indices = torch.nonzero(
                        candidate_full.instance_id >= 0, as_tuple=False
                    ).flatten()
                    if delta.shape[0] != candidate_dynamic_indices.numel():
                        raise ValueError(
                            "candidate temporal mean delta count does not match "
                            "candidate dynamic Gaussian count"
                        )
                    candidate_posed["means"] = candidate_posed["means"].clone()
                    candidate_posed["means"][candidate_dynamic_indices] += delta
            if args.candidate_dynamic_opacity_scale is not None and candidate_payload_dir == payload_dir:
                candidate_opacity = candidate_posed["opacities"].clone()
                candidate_opacity[candidate_full.instance_id >= 0] = (
                    candidate_opacity[candidate_full.instance_id >= 0]
                    * args.candidate_dynamic_opacity_scale
                ).clamp(max=0.999)
                candidate_posed["opacities"] = candidate_opacity
            candidate_ownership = (
                (candidate_full.instance_id >= 0)
                .to(candidate_posed["colors"].dtype)
                .unsqueeze(1)
            )
            candidate_colors129 = torch.cat(
                (candidate_posed["colors"], candidate_ownership), dim=1
            )
            candidate_feature129, candidate_alpha, _ = render_features(
                candidate_posed["means"], candidate_posed["quats"],
                candidate_posed["scales"], candidate_posed["opacities"],
                candidate_colors129, views.intrinsics,
                torch.linalg.inv(views.cam_to_world), height, width,
            )
            candidate_features = candidate_feature129[:, :128]
            candidate_contribution = candidate_feature129[:, 128:129]
            candidate_rgbs = []
            for camera_index, camera in enumerate(eval_cameras):
                candidate_rgbs.append(candidate_refiners[camera](
                    candidate_features[camera_index:camera_index + 1],
                    candidate_alpha[camera_index:camera_index + 1],
                ))
            candidate_rgb = torch.cat(candidate_rgbs, dim=0)
            if candidate_rgb_residual is not None:
                mode = candidate_rgb_residual["residual_mode"]
                static_residual = candidate_rgb_residual["static_rgb_residual"]
                dynamic_residual = candidate_rgb_residual["dynamic_rgb_residual"]
                if mode == "per-frame":
                    source_frames = [int(value) for value in candidate_rgb_residual["source_frames"]]
                    residual_slot = source_frames.index(int(frame))
                    dynamic_residual = dynamic_residual[residual_slot]
                residual_colors = torch.cat((static_residual, dynamic_residual), dim=0).to(device)
                residual_scene = replace(candidate_full, colors=residual_colors)
                posed_residual = model.pose_gaussians(
                    residual_scene, candidate_boxes, slot, views.times[0]
                )
                residual_view, _, _ = render_features(
                    posed_residual["means"], posed_residual["quats"],
                    posed_residual["scales"], posed_residual["opacities"],
                    posed_residual["colors"], views.intrinsics,
                    torch.linalg.inv(views.cam_to_world), height, width,
                    render_depth=False,
                )
                candidate_rgb = (candidate_rgb + residual_view).clamp(0.0, 1.0)
                del residual_colors, residual_scene, posed_residual, residual_view
            candidate_support = candidate_contribution >= args.support_threshold
            union, new_support = support_partitions(support, candidate_support)
            candidate_error = (candidate_rgb.double() - truth.double()).square()

            def partition_sse(values: torch.Tensor, mask: torch.Tensor) -> float:
                return float((values * mask).sum())

            union_count = int(union.sum()) * 3
            frozen_count = int(support.sum()) * 3
            new_count = int(new_support.sum()) * 3
            row = {
                "frame": frame,
                "baseline_union_psnr_db": psnr_from_sse(partition_sse(error, union), union_count),
                "candidate_union_psnr_db": psnr_from_sse(partition_sse(candidate_error, union), union_count),
                "baseline_frozen_support_psnr_db": psnr_from_sse(partition_sse(error, support), frozen_count),
                "candidate_frozen_support_psnr_db": psnr_from_sse(partition_sse(candidate_error, support), frozen_count),
                "baseline_new_support_psnr_db": psnr_from_sse(partition_sse(error, new_support), new_count),
                "candidate_new_support_psnr_db": psnr_from_sse(partition_sse(candidate_error, new_support), new_count),
                "baseline_support_pixels": int(support.sum()),
                "candidate_support_pixels": int(candidate_support.sum()),
                "union_support_pixels": int(union.sum()),
                "candidate_new_support_pixels": int(new_support.sum()),
                "baseline_contribution_sum": float(contribution.sum()),
                "candidate_contribution_sum": float(candidate_contribution.sum()),
                "candidate_to_baseline_area_ratio": (
                    int(candidate_support.sum()) / int(support.sum())
                    if int(support.sum()) else None
                ),
                "candidate_to_baseline_contribution_ratio": (
                    float(candidate_contribution.sum()) / float(contribution.sum())
                    if float(contribution.sum()) else None
                ),
            }
            comparison_records.append(row)
            comparison_aggregate["rgb_max_abs"] = max(
                comparison_aggregate["rgb_max_abs"],
                float((candidate_rgb - rgb).abs().max()),
            )
            comparison_aggregate["contribution_max_abs"] = max(
                comparison_aggregate["contribution_max_abs"],
                float((candidate_contribution - contribution).abs().max()),
            )
            comparison_aggregate["support_xor_pixels"] += int(
                (candidate_support ^ support).sum()
            )
            for key, value in (
                ("baseline_union_sse", partition_sse(error, union)),
                ("candidate_union_sse", partition_sse(candidate_error, union)),
                ("baseline_frozen_sse", partition_sse(error, support)),
                ("candidate_frozen_sse", partition_sse(candidate_error, support)),
                ("baseline_new_sse", partition_sse(error, new_support)),
                ("candidate_new_sse", partition_sse(candidate_error, new_support)),
            ):
                comparison_aggregate[key] += value
            comparison_aggregate["union_scalar_count"] += union_count
            comparison_aggregate["frozen_scalar_count"] += frozen_count
            comparison_aggregate["new_scalar_count"] += new_count
            for key, value in (
                ("baseline_support_pixels", int(support.sum())),
                ("candidate_support_pixels", int(candidate_support.sum())),
                ("union_support_pixels", int(union.sum())),
                ("candidate_new_support_pixels", int(new_support.sum())),
                ("baseline_contribution_sum", float(contribution.sum())),
                ("candidate_contribution_sum", float(candidate_contribution.sum())),
            ):
                comparison_aggregate[key] += value
            del candidate_posed, candidate_ownership, candidate_colors129
            del candidate_feature129, candidate_alpha, candidate_features
            del candidate_rgbs, union, new_support
        support_pixels = int(support.sum())
        scalar_count = support_pixels * 3
        support_sse = float((error * support).sum())
        frame_dynamic_psnr = psnr_from_sse(support_sse, scalar_count)
        frame_global_psnr = psnr_from_sse(float(error.sum()), error.numel())
        bbox = box_dynamic_masks(
            evaluation_boxes,
            torch.full((len(camera_indices),), slot, dtype=torch.long, device=device),
            views.intrinsics, torch.linalg.inv(views.cam_to_world), height, width,
        ).unsqueeze(1)
        frame_bbox_psnr = psnr_from_sse(float((error * bbox).sum()), int(bbox.sum()) * 3)
        if frame_dynamic_psnr is not None:
            active_psnr.append(frame_dynamic_psnr)
        global_psnr.append(frame_global_psnr)
        if frame_bbox_psnr is not None:
            bbox_psnr.append(frame_bbox_psnr)
        if candidate_full is not None:
            candidate_global_psnr.append(psnr_from_sse(
                float(candidate_error.sum()), candidate_error.numel()
            ))
            candidate_bbox = box_dynamic_masks(
                evaluation_boxes,
                torch.full((len(camera_indices),), slot, dtype=torch.long, device=device),
                views.intrinsics, torch.linalg.inv(views.cam_to_world), height, width,
            ).unsqueeze(1)
            candidate_frame_bbox_psnr = psnr_from_sse(
                float((candidate_error * candidate_bbox).sum()),
                int(candidate_bbox.sum()) * 3,
            )
            if candidate_frame_bbox_psnr is not None:
                candidate_bbox_psnr.append(candidate_frame_bbox_psnr)
            row["candidate_global_psnr_db"] = candidate_global_psnr[-1]
            row["candidate_dynamic_bbox_psnr_db"] = candidate_frame_bbox_psnr
        aggregate["support_pixels"] += support_pixels
        aggregate["contribution_sum"] += float(contribution.sum())

        fine_views = []
        for camera_index, camera in enumerate(eval_cameras):
            relative = f"fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png"
            path = Path(metadata["path"]) / relative
            if not path.exists():
                fine_views.append(torch.zeros((height, width), dtype=torch.bool, device=device))
                continue
            mask_paths.append(path.resolve())
            fine, _ = dataset._load_calibrated_image(
                metadata, relative, dataset.resolution, camera, gray=True,
            )
            fine_views.append(fine.squeeze(0).to(device) > 0.5)
        fine = torch.stack(fine_views).unsqueeze(1)
        aggregate["fine_tp"] += int((support & fine).sum())
        aggregate["fine_fp"] += int((support & ~fine).sum())
        aggregate["fine_fn"] += int((~support & fine).sum())
        if candidate_full is not None:
            comparison_aggregate["candidate_fine_tp"] += int((candidate_support & fine).sum())
            comparison_aggregate["candidate_fine_fp"] += int((candidate_support & ~fine).sum())
            comparison_aggregate["candidate_fine_fn"] += int((~candidate_support & fine).sum())

        disabled_out_fraction = None
        if args.diagnose_disabled:
            disabled_opacity = posed["opacities"].clone()
            disabled_opacity[full.instance_id >= 0] = 0.0
            off_features, off_alpha, _ = render_features(
                posed["means"], posed["quats"], posed["scales"], disabled_opacity,
                posed["colors"], views.intrinsics, torch.linalg.inv(views.cam_to_world),
                height, width,
            )
            off_rgbs = []
            for camera_index, camera in enumerate(eval_cameras):
                off_rgbs.append(refiners[camera](
                    off_features[camera_index:camera_index + 1],
                    off_alpha[camera_index:camera_index + 1],
                ))
            off_rgb = torch.cat(off_rgbs, dim=0)
            delta = (rgb.double() - off_rgb.double()).square().sum(dim=1, keepdim=True)
            total_delta = float(delta.sum())
            outside_delta = float((delta * ~support).sum())
            aggregate["disabled_total_sse"] += total_delta
            aggregate["disabled_outside_sse"] += outside_delta
            disabled_out_fraction = outside_delta / total_delta if total_delta else 0.0

            old_features, old_alpha, _ = render_features(
                posed["means"], posed["quats"], posed["scales"], posed["opacities"],
                posed["colors"], views.intrinsics, torch.linalg.inv(views.cam_to_world),
                height, width,
            )
            old_rgbs = []
            for camera_index, camera in enumerate(eval_cameras):
                old_rgbs.append(refiners[camera](
                    old_features[camera_index:camera_index + 1],
                    old_alpha[camera_index:camera_index + 1],
                ))
            old_rgb = torch.cat(old_rgbs, dim=0)
            parity_max["feature"] = max(parity_max["feature"], float((old_features - features).abs().max()))
            parity_max["alpha"] = max(parity_max["alpha"], float((old_alpha - alpha129).abs().max()))
            parity_max["rgb"] = max(parity_max["rgb"], float((old_rgb - rgb).abs().max()))

        records.append({
            "frame": frame, "support_pixels": support_pixels,
            "dynamic_support_psnr_db": frame_dynamic_psnr,
            "global_psnr_db": frame_global_psnr,
            "dynamic_bbox_psnr_db": frame_bbox_psnr,
            "contribution_sum": float(contribution.sum()),
            "disabled_rgb_energy_outside_fraction": disabled_out_fraction,
        })
        if args.diagnose_disabled:
            panels = []
            for camera_index, camera in enumerate(eval_cameras):
                overlay = rgb[camera_index].clone()
                overlay[0] = torch.where(support[camera_index, 0], torch.ones_like(overlay[0]), overlay[0])
                panels.append(label(to_uint8(overlay), f"frame {frame} camera {camera} | red=support"))
            Image.fromarray(horizontal_strip(panels)).save(visual_dir / f"frame_{frame:04d}_support.jpg", quality=92)
        elif candidate_full is not None:
            panels = []
            for camera_index, camera in enumerate(eval_cameras):
                base_view = rgb[camera_index].clamp(0.0, 1.0)
                candidate_view = candidate_rgb[camera_index].clamp(0.0, 1.0)
                diff_view = (candidate_view - base_view).abs().clamp(0.0, 1.0)
                panels.extend([
                    label(to_uint8(base_view), f"f{frame} c{camera} baseline"),
                    label(to_uint8(candidate_view), f"f{frame} c{camera} candidate"),
                    label(to_uint8(diff_view), f"f{frame} c{camera} | abs diff"),
                ])
            Image.fromarray(horizontal_strip(panels)).save(
                visual_dir / f"frame_{frame:04d}_candidate.jpg", quality=92
            )
        print(json.dumps({"frame": frame, "progress": [frame_index + 1, len(frames)],
                          "dynamic_psnr": frame_dynamic_psnr}), flush=True)
        if candidate_full is not None:
            del candidate_error, candidate_rgb, candidate_support, candidate_contribution

    tp, fp, fn = aggregate["fine_tp"], aggregate["fine_fp"], aggregate["fine_fn"]
    disabled_total = aggregate["disabled_total_sse"]
    mask_digest = hashlib.sha256()
    for path in sorted(set(mask_paths)):
        mask_digest.update(str(path).encode())
        mask_digest.update(bytes.fromhex(sha256_file(path)))
    manifest = {
        "status": "completed", "scene": args.scene, "frames_evaluated": frames,
        "all_source_frames_evaluated": frames == all_frames,
        "cameras": eval_cameras, "camera_indices": camera_indices,
        "support_threshold_inclusive": args.support_threshold,
        "definition": "c_dyn(p)=sum_i T_i(p)*alpha_i(p)*1[instance_id_i>=0] from the same 129-channel rasterization as production 128-D features",
        "aggregation": "per-frame selected-camera SSE/count, active-frame equal mean; double-empty frames NA",
        "analytic_test": analytic, "production_parity_max_abs": parity_max,
        "mean_psnr_dynamic_joint_support_active_frames": float(np.mean(active_psnr)) if active_psnr else None,
        "active_frame_count": len(active_psnr), "double_empty_frame_count": len(frames) - len(active_psnr),
        "mean_psnr_source_frames_all_cameras": float(np.mean(global_psnr)),
        "mean_psnr_dynamic_bbox_source_frames_all_cameras": float(np.mean(bbox_psnr)) if bbox_psnr else None,
        "support": {
            "pixel_count": aggregate["support_pixels"],
            "contribution_sum": aggregate["contribution_sum"],
            "fine_mask_tp": tp, "fine_mask_fp": fp, "fine_mask_fn": fn,
            "fine_mask_precision": tp / (tp + fp) if tp + fp else None,
            "fine_mask_recall": tp / (tp + fn) if tp + fn else None,
            "fine_mask_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "fine_mask_file_set_sha256": mask_digest.hexdigest(),
            "fine_mask_role": "diagnostic only; vehicle mask does not cover every configured rigid class",
        },
        "dynamic_disabled_rgb_energy_outside_support_fraction": (
            aggregate["disabled_outside_sse"] / disabled_total if disabled_total else None
        ),
        "frame_records": records,
        "wall_seconds": time.perf_counter() - started,
    }
    if candidate_full is not None:
        b_pixels = comparison_aggregate["baseline_support_pixels"]
        b_mass = comparison_aggregate["baseline_contribution_sum"]
        baseline_values = contribution.new_tensor(
            [row["baseline_contribution_sum"] for row in comparison_records]
        ).float()
        candidate_values = contribution.new_tensor(
            [row["candidate_contribution_sum"] for row in comparison_records]
        ).float()
        quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=device)
        area_ratio = (
            comparison_aggregate["candidate_support_pixels"] / b_pixels
            if b_pixels else None
        )
        mass_ratio = (
            comparison_aggregate["candidate_contribution_sum"] / b_mass
            if b_mass else None
        )
        mean_metric = lambda key: float(np.mean([
            row[key] for row in comparison_records if row[key] is not None
        ])) if any(row[key] is not None for row in comparison_records) else None
        active_means = {
            "baseline_union_psnr_db": mean_metric("baseline_union_psnr_db"),
            "candidate_union_psnr_db": mean_metric("candidate_union_psnr_db"),
            "baseline_frozen_support_psnr_db": mean_metric("baseline_frozen_support_psnr_db"),
            "candidate_frozen_support_psnr_db": mean_metric("candidate_frozen_support_psnr_db"),
            "baseline_new_support_psnr_db": mean_metric("baseline_new_support_psnr_db"),
            "candidate_new_support_psnr_db": mean_metric("candidate_new_support_psnr_db"),
        }
        area_frame_ratios = [
            row["candidate_to_baseline_area_ratio"] for row in comparison_records
            if row["candidate_to_baseline_area_ratio"] is not None
        ]
        mass_frame_ratios = [
            row["candidate_to_baseline_contribution_ratio"] for row in comparison_records
            if row["candidate_to_baseline_contribution_ratio"] is not None
        ]
        frame_ratio_violations = sum(
            not (0.75 <= row["candidate_to_baseline_area_ratio"] <= 1.25)
            or not (0.75 <= row["candidate_to_baseline_contribution_ratio"] <= 1.25)
            for row in comparison_records
            if row["candidate_to_baseline_area_ratio"] is not None
            and row["candidate_to_baseline_contribution_ratio"] is not None
        )
        aggregate_ratio_ok = (
            area_ratio is not None and mass_ratio is not None
            and 0.75 <= area_ratio <= 1.25 and 0.75 <= mass_ratio <= 1.25
        )
        union_improves = (
            active_means["candidate_union_psnr_db"]
            > active_means["baseline_union_psnr_db"]
        )
        frozen_non_degrades = (
            active_means["candidate_frozen_support_psnr_db"]
            >= active_means["baseline_frozen_support_psnr_db"]
        )
        new_non_degrades = (
            comparison_aggregate["new_scalar_count"] == 0
            or active_means["candidate_new_support_psnr_db"]
            >= active_means["baseline_new_support_psnr_db"]
        )
        quality_ok = union_improves and frozen_non_degrades and new_non_degrades
        if not quality_ok or not aggregate_ratio_ok:
            admission_status = "rejected"
        elif frame_ratio_violations:
            admission_status = "review_required"
        else:
            admission_status = "passed"
        identity_exact = (
            comparison_aggregate["rgb_max_abs"] == 0.0
            and comparison_aggregate["contribution_max_abs"] == 0.0
            and comparison_aggregate["support_xor_pixels"] == 0
        )
        identity_status = "identity_pass" if identity_exact else "identity_mismatch"
        candidate_tp = comparison_aggregate["candidate_fine_tp"]
        candidate_fp = comparison_aggregate["candidate_fine_fp"]
        candidate_fn = comparison_aggregate["candidate_fine_fn"]
        identity_paths = [
            Path(__file__), config_path, Path(args.ckpt),
            payload_dir / "final_static_128d.pt",
            payload_dir / "final_dynamic_128d.pt",
            payload_dir / "global_dynamic_meta.pt",
            candidate_payload_dir / "final_static_128d.pt",
            candidate_payload_dir / "final_dynamic_128d.pt",
            candidate_payload_dir / "global_dynamic_meta.pt",
            *(Path(path) for path in args.camera_refiner_states),
            *(Path(path) for path in candidate_refiner_paths),
        ]
        identities = {}
        for path in identity_paths:
            resolved = str(path.resolve())
            if resolved not in identities:
                identities[resolved] = file_identity(path)
        manifest["candidate_comparison"] = {
            "candidate_payload_dir": str(candidate_payload_dir),
            "candidate_dynamic_opacity_scale": args.candidate_dynamic_opacity_scale,
            "aggregation": "global diagnostic aggregate over evaluated views; frame_records retain paired frame scores",
            "baseline_union_psnr_db": psnr_from_sse(
                comparison_aggregate["baseline_union_sse"], comparison_aggregate["union_scalar_count"]
            ),
            "candidate_union_psnr_db": psnr_from_sse(
                comparison_aggregate["candidate_union_sse"], comparison_aggregate["union_scalar_count"]
            ),
            "baseline_frozen_support_psnr_db": psnr_from_sse(
                comparison_aggregate["baseline_frozen_sse"], comparison_aggregate["frozen_scalar_count"]
            ),
            "candidate_frozen_support_psnr_db": psnr_from_sse(
                comparison_aggregate["candidate_frozen_sse"], comparison_aggregate["frozen_scalar_count"]
            ),
            "baseline_new_support_psnr_db": psnr_from_sse(
                comparison_aggregate["baseline_new_sse"], comparison_aggregate["new_scalar_count"]
            ),
            "candidate_new_support_psnr_db": psnr_from_sse(
                comparison_aggregate["candidate_new_sse"], comparison_aggregate["new_scalar_count"]
            ),
            "active_frame_equal_mean": active_means,
            "candidate_global": {
                "mean_psnr_source_frames_all_cameras": float(np.mean(candidate_global_psnr)),
                "mean_psnr_dynamic_bbox_source_frames_all_cameras": float(np.mean(candidate_bbox_psnr)) if candidate_bbox_psnr else None,
                "fine_mask_tp": candidate_tp,
                "fine_mask_fp": candidate_fp,
                "fine_mask_fn": candidate_fn,
                "fine_mask_precision": candidate_tp / (candidate_tp + candidate_fp) if candidate_tp + candidate_fp else None,
                "fine_mask_recall": candidate_tp / (candidate_tp + candidate_fn) if candidate_tp + candidate_fn else None,
                "fine_mask_iou": candidate_tp / (candidate_tp + candidate_fp + candidate_fn) if candidate_tp + candidate_fp + candidate_fn else None,
            },
            "identity_diagnostics": {
                "rgb_max_abs": comparison_aggregate["rgb_max_abs"],
                "contribution_max_abs": comparison_aggregate["contribution_max_abs"],
                "support_xor_pixels": comparison_aggregate["support_xor_pixels"],
                "status": identity_status,
            },
            "anti_manipulation_gate": {
                "allowed_ratio_interval_inclusive": [0.75, 1.25],
                "area_ratio_ok": area_ratio is not None and 0.75 <= area_ratio <= 1.25,
                "contribution_ratio_ok": mass_ratio is not None and 0.75 <= mass_ratio <= 1.25,
                "frame_ratio_violation_count": frame_ratio_violations,
                "frame_ratio_review_required": frame_ratio_violations > 0,
                "union_strictly_improves": union_improves,
                "frozen_baseline_support_non_degrades": frozen_non_degrades,
                "new_support_non_degrades_or_na": new_non_degrades,
                "admission_status": admission_status,
            },
            "support": {
                **{key: comparison_aggregate[key] for key in (
                    "baseline_support_pixels", "candidate_support_pixels",
                    "union_support_pixels", "candidate_new_support_pixels",
                    "baseline_contribution_sum", "candidate_contribution_sum",
                )},
                "candidate_to_baseline_area_ratio": area_ratio,
                "candidate_to_baseline_contribution_ratio": mass_ratio,
                "per_frame_contribution_sum_quantiles": {
                    "q": [0.05, 0.25, 0.5, 0.75, 0.95],
                    "baseline": torch.quantile(baseline_values, quantiles).cpu().tolist(),
                    "candidate": torch.quantile(candidate_values, quantiles).cpu().tolist(),
                },
                "per_frame_ratio_quantiles": {
                    "q": [0.05, 0.25, 0.5, 0.75, 0.95],
                    "area": torch.quantile(torch.tensor(area_frame_ratios), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])).tolist(),
                    "contribution": torch.quantile(torch.tensor(mass_frame_ratios), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])).tolist(),
                },
            },
            "identity": {
                "config": str(config_path), "checkpoint": str(Path(args.ckpt).resolve()),
                "baseline_payload": str(payload_dir), "candidate_payload": str(candidate_payload_dir),
                "baseline_refiners": [str(Path(path).resolve()) for path in args.camera_refiner_states],
                "candidate_refiners": [str(Path(path).resolve()) for path in candidate_refiner_paths],
                "scene": args.scene, "split": args.split, "dataset_path": str(Path(metadata["path"]).resolve()),
                "frames": frames, "cameras": list(dataset.cameras), "files": identities,
            },
            "frame_records": comparison_records,
        }
        manifest["candidate_comparison"]["numeric_partition_test"] = comparison_numeric_test()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "mean_psnr_dynamic_joint_support_active_frames", "active_frame_count",
        "mean_psnr_source_frames_all_cameras", "production_parity_max_abs",
        "dynamic_disabled_rgb_energy_outside_support_fraction", "wall_seconds",
    )}, indent=2))


if __name__ == "__main__":
    main()

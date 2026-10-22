"""Merge native single-view InfiniDepth Gaussians into one calibrated scene.

Each PointForward context image is processed independently by InfiniDepth's
native DepthSensor + GS path. The resulting camera-local Gaussians are rigidly
transformed into the same recentered world frame used by PointForward. This is
pose-based concatenation, not multi-view feature fusion or optimization.
"""

from __future__ import annotations

import argparse
import __future__
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.train import build_dataset


def vertex_fields(sh_degree: int) -> list[tuple[str, str]]:
    if sh_degree not in (0, 2):
        raise ValueError(f"Only SH degree 0 or 2 is supported, got {sh_degree}")
    fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
    ]
    fields.extend((f"f_rest_{index}", "<f4") for index in range(3 * ((sh_degree + 1) ** 2 - 1)))
    fields.extend([
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    return fields


FIELDS = vertex_fields(0)
VERTEX_DTYPE = np.dtype(FIELDS)


class _FutureAnnotationsLoader(importlib.machinery.SourceFileLoader):
    """Compile selected upstream modules with postponed annotations on Python 3.9."""

    def get_code(self, fullname):
        source_path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(source_path), source_path)

    def source_to_code(self, data, path, *, _optimize=-1):
        return compile(
            data,
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
            optimize=_optimize,
        )


class _InfiniDepthGSCompatibilityFinder(importlib.abc.MetaPathFinder):
    def __init__(self, repo: Path) -> None:
        self.package_root = repo / "InfiniDepth"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "InfiniDepth" and not fullname.startswith("InfiniDepth."):
            return None
        relative = fullname.split(".")[1:]
        base = self.package_root.joinpath(*relative)
        package_source = base / "__init__.py"
        is_package = package_source.is_file()
        source = package_source if is_package else base.with_suffix(".py")
        if not source.is_file():
            return None
        loader = _FutureAnnotationsLoader(fullname, str(source))
        search_locations = [str(base)] if is_package else None
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            loader=loader,
            submodule_search_locations=search_locations,
        )


def install_python39_infinidepth_compatibility(repo: Path) -> None:
    if sys.version_info >= (3, 10):
        return
    if not any(isinstance(finder, _InfiniDepthGSCompatibilityFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _InfiniDepthGSCompatibilityFinder(repo))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert one proper rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    candidates = np.array([
        1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2],
        1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
        1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2],
        1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2],
    ])
    index = int(np.argmax(candidates))
    value = np.sqrt(max(candidates[index], 0.0)) * 0.5
    denominator = max(4.0 * value, 1e-12)
    if index == 0:
        quat = [value, (matrix[2, 1] - matrix[1, 2]) / denominator,
                (matrix[0, 2] - matrix[2, 0]) / denominator,
                (matrix[1, 0] - matrix[0, 1]) / denominator]
    elif index == 1:
        quat = [(matrix[2, 1] - matrix[1, 2]) / denominator, value,
                (matrix[0, 1] + matrix[1, 0]) / denominator,
                (matrix[0, 2] + matrix[2, 0]) / denominator]
    elif index == 2:
        quat = [(matrix[0, 2] - matrix[2, 0]) / denominator,
                (matrix[0, 1] + matrix[1, 0]) / denominator, value,
                (matrix[1, 2] + matrix[2, 1]) / denominator]
    else:
        quat = [(matrix[1, 0] - matrix[0, 1]) / denominator,
                (matrix[0, 2] + matrix[2, 0]) / denominator,
                (matrix[1, 2] + matrix[2, 1]) / denominator, value]
    quat = np.asarray(quat, dtype=np.float32)
    return quat / np.linalg.norm(quat)


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose quaternions so the resulting rotation is left @ right."""
    w1, x1, y1, z1 = left.unbind(-1)
    w2, x2, y2, z2 = right.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def sh_basis_degree2(directions: torch.Tensor) -> torch.Tensor:
    """Real SH basis used by gsplat, ordered from degree 0 through degree 2."""
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = directions.unbind(-1)
    return torch.stack((
        torch.full_like(x, 0.2820947917738781),
        -0.48860251190292 * y,
        0.48860251190292 * z,
        -0.48860251190292 * x,
        1.092548430592079 * x * y,
        -1.092548430592079 * y * z,
        0.9461746957575601 * z * z - 0.3153915652525201,
        -1.092548430592079 * x * z,
        0.5462742152960395 * (x * x - y * y),
    ), dim=-1)


def rotate_sh_degree2(harmonics: torch.Tensor, camera_to_world_rotation: torch.Tensor) -> torch.Tensor:
    """Rotate camera-basis SH coefficients so world-space viewing directions agree."""
    sample_count = 64
    index = torch.arange(sample_count, dtype=torch.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / sample_count
    radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    world_directions = torch.stack((radius * torch.cos(index * golden_angle),
                                    radius * torch.sin(index * golden_angle), z), dim=-1)
    rotation = camera_to_world_rotation.detach().double().cpu()
    local_directions = world_directions @ rotation
    world_basis = sh_basis_degree2(world_directions)
    local_basis = sh_basis_degree2(local_directions)
    transform = torch.linalg.lstsq(world_basis, local_basis).solution
    transform[transform.abs() < 1e-12] = 0.0
    transform = transform.to(device=harmonics.device, dtype=harmonics.dtype)
    return torch.einsum("ji,nci->ncj", transform, harmonics)


def make_vertex_array(means, harmonics, opacities, scales, rotations, sh_degree: int = 0) -> np.ndarray:
    count = means.shape[0]
    dtype = np.dtype(vertex_fields(sh_degree))
    vertices = np.empty(count, dtype=dtype)
    means_np = means.detach().float().cpu().numpy()
    harmonics_np = harmonics[..., : (sh_degree + 1) ** 2].detach().float().cpu().numpy()
    dc_np = harmonics_np[..., 0]
    opacity_np = opacities.detach().float().cpu().numpy()
    scales_np = scales.detach().float().cpu().numpy()
    rotations_np = rotations.detach().float().cpu().numpy()
    vertices["x"], vertices["y"], vertices["z"] = means_np.T
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0.0
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = dc_np.T
    if sh_degree > 0:
        # Graphdeco PLY stores RGB triplets per SH basis: k1-RGB, k2-RGB, ...
        rest = harmonics_np[..., 1:].transpose(0, 2, 1).reshape(count, -1)
        for index in range(rest.shape[1]):
            vertices[f"f_rest_{index}"] = rest[:, index]
    opacity_np = np.clip(opacity_np, 1e-6, 1 - 1e-6)
    vertices["opacity"] = np.log(opacity_np / (1 - opacity_np))
    log_scales = np.log(np.clip(scales_np, 1e-10, None))
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = log_scales.T
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = rotations_np.T
    return vertices


def write_binary_ply(path: Path, shards: list[Path], indices: list[np.ndarray] | None = None) -> int:
    first = np.load(shards[0], mmap_mode="r")
    dtype = first.dtype
    counts = [int(np.load(path, mmap_mode="r").shape[0]) for path in shards]
    if indices is None:
        output_count = sum(counts)
    else:
        output_count = sum(len(index) for index in indices)
    header = ["ply", "format binary_little_endian 1.0",
              "comment generated by native InfiniDepth per-view GS concatenation",
              f"element vertex {output_count}"]
    for name in dtype.names:
        header.append(f"property float {name}")
    header.append("end_header")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(("\n".join(header) + "\n").encode("ascii"))
        for shard_index, shard_path in enumerate(shards):
            vertices = np.load(shard_path, mmap_mode="r")
            if vertices.dtype != dtype:
                raise ValueError(f"Shard dtype mismatch: {shard_path}")
            selected = vertices if indices is None else vertices[indices[shard_index]]
            output.write(np.asarray(selected, dtype=dtype).tobytes(order="C"))
    os.replace(temporary, path)
    return output_count


def deterministic_sample_indices(count: int, take: int, seed: int) -> np.ndarray:
    if take > count:
        raise ValueError(f"Cannot retain {take} of only {count} Gaussians")
    return np.sort(np.random.default_rng(seed).choice(count, size=take, replace=False))


def spatial_coverage_indices(
    query_coord: torch.Tensor,
    take: int,
    height: int,
    width: int,
    block_size: int,
    seed: int,
) -> torch.Tensor:
    """Choose unique source pixels with balanced quotas across image blocks."""
    if query_coord.shape[0] != 1:
        raise ValueError("Spatial coverage selection expects one view")
    coordinates = query_coord[0].detach().float().cpu().numpy()
    pixel_y = np.floor((coordinates[:, 0] + 1.0) * height * 0.5).astype(np.int64)
    pixel_x = np.floor((coordinates[:, 1] + 1.0) * width * 0.5).astype(np.int64)
    pixel_y = np.clip(pixel_y, 0, height - 1)
    pixel_x = np.clip(pixel_x, 0, width - 1)
    pixel_ids = pixel_y * width + pixel_x
    _, unique_candidate_indices = np.unique(pixel_ids, return_index=True)
    if len(unique_candidate_indices) < take:
        raise RuntimeError(
            f"Only {len(unique_candidate_indices)} unique candidate pixels for requested {take} Gaussians"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_candidate_indices)
    unique_y = pixel_y[unique_candidate_indices]
    unique_x = pixel_x[unique_candidate_indices]
    blocks_w = (width + block_size - 1) // block_size
    block_ids = (unique_y // block_size) * blocks_w + (unique_x // block_size)
    block_count = ((height + block_size - 1) // block_size) * blocks_w
    base_quota, extra_quota = divmod(take, block_count)
    selected_parts = []
    selected_mask = np.zeros(len(unique_candidate_indices), dtype=bool)
    for block in range(block_count):
        positions = np.flatnonzero(block_ids == block)
        quota = base_quota + (1 if block < extra_quota else 0)
        chosen = positions[: min(quota, len(positions))]
        if len(chosen):
            selected_parts.append(unique_candidate_indices[chosen])
            selected_mask[chosen] = True

    selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    deficit = take - len(selected)
    if deficit:
        remaining = unique_candidate_indices[~selected_mask]
        selected = np.concatenate((selected, remaining[:deficit]))
    if len(selected) != take or len(np.unique(pixel_ids[selected])) != take:
        raise RuntimeError("Spatial coverage selection failed to produce exact unique-pixel count")
    return torch.from_numpy(selected.astype(np.int64)).to(query_coord.device)


def restore_sh_band_weights(harmonics: torch.Tensor) -> torch.Tensor:
    """Undo InfiniDepth's fixed 0.025/0.00625 attenuation for SH degrees 1/2."""
    if harmonics.shape[-1] < 9:
        raise ValueError("Second-order SH restoration requires at least 9 coefficients")
    restored = harmonics.clone()
    restored[..., 1:4] /= 0.025
    restored[..., 4:9] /= 0.00625
    return restored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/infinidepth_native_scene005"))
    parser.add_argument("--samples-per-view", type=int, default=2_000_000)
    parser.add_argument("--budget-total", type=int, default=1_440_000)
    parser.add_argument("--input-size", type=int, nargs=2, default=(768, 1024), metavar=("H", "W"))
    parser.add_argument("--sh-degree", type=int, choices=(0, 2), default=0)
    parser.add_argument("--candidate-samples-per-view", type=int, default=None)
    parser.add_argument("--spatial-block-size", type=int, default=16)
    parser.add_argument("--restore-sh-band-weights", action="store_true")
    parser.add_argument("--only-full", action="store_true", help="Write only the full exact-count PLY")
    parser.add_argument("--max-views", type=int, default=None, help="Testing only; omit for all context views")
    parser.add_argument("--disable-sky-mask", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.samples_per_view < 1 or args.budget_total < 1:
        raise ValueError("Point counts must be positive")
    candidate_samples = args.candidate_samples_per_view or args.samples_per_view
    if candidate_samples < args.samples_per_view:
        raise ValueError("candidate-samples-per-view must be >= samples-per-view")
    if args.spatial_block_size < 1:
        raise ValueError("spatial-block-size must be positive")
    if args.restore_sh_band_weights and args.sh_degree != 2:
        raise ValueError("SH band-weight restoration requires --sh-degree 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo = Path("/mnt/data/why/InfiniDepth")
    depth_checkpoint = repo / "checkpoints/depth/infinidepth_depthsensor.ckpt"
    gs_checkpoint = repo / "checkpoints/gs/infinidepth_depthsensor_gs.ckpt"
    sky_checkpoint = repo / "checkpoints/sky/skyseg.onnx"
    for required in (depth_checkpoint, gs_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    sys.path.insert(0, str(repo))
    install_python39_infinidepth_compatibility(repo)
    from InfiniDepth.gs import GSPixelAlignPredictor
    from InfiniDepth.utils.gs_utils import _build_sparse_uniform_gaussians
    from InfiniDepth.utils.inference_utils import run_optional_sampling_sky_mask
    from InfiniDepth.utils.io_utils import depth_to_disparity
    from InfiniDepth.utils.model_utils import build_model

    config = yaml.safe_load(Path(args.config).read_text())
    dataset = build_dataset(config, args.split)
    scene_index, start = dataset.index[args.clip_index]
    metadata = dataset.scenes[scene_index]
    window = metadata["frames"][start:start + dataset.clip_length]
    context_frames = sorted({window[min(i * dataset.frame_stride, len(window) - 1)]
                             for i in range(dataset.num_context)})
    world_from = np.linalg.inv(dataset._ego_pose(metadata, window[0]))
    slot_of = {frame: index for index, frame in enumerate(window)}
    normalized_times = dataset._normalised_times(window)
    batch = dataset._collect(metadata, context_frames, slot_of, normalized_times,
                             tuple(args.input_size), world_from, lidar=True)
    if "sparse_depth" not in batch:
        raise RuntimeError("Selected context has no LiDAR")
    view_records = [(frame, camera) for frame in context_frames for camera in dataset.cameras]
    if args.max_views is not None:
        view_records = view_records[:args.max_views]
    if not args.only_full and args.budget_total % len(view_records):
        raise ValueError("budget-total must be divisible by the number of processed views")
    budget_per_view = args.budget_total // len(view_records) if not args.only_full else 0
    if not args.only_full and budget_per_view > args.samples_per_view:
        raise ValueError("Budget per view exceeds full per-view sample count")

    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    sh_label = f"_sh{args.sh_degree}" if args.sh_degree else ""
    correction_label = "_coverage_restored" if candidate_samples > args.samples_per_view and args.restore_sh_band_weights else ""
    full_path = output_dir / f"scene{metadata['name']}_native_infinidepth{sh_label}{correction_label}_full_{args.samples_per_view * len(view_records)}.ply"
    budget_path = output_dir / f"scene{metadata['name']}_native_infinidepth{sh_label}_budget_{args.budget_total}.ply"
    expected_dtype = np.dtype(vertex_fields(args.sh_degree))
    device = torch.device("cuda")
    model = build_model("InfiniDepth_DepthSensor", model_path=str(depth_checkpoint)).to(device).eval()
    gs_predictor = GSPixelAlignPredictor(dino_feature_dim=1024).to(device)
    gs_predictor.load_from_infinidepth_gs_checkpoint(str(gs_checkpoint))
    gs_predictor.eval()
    view_metadata = []
    started = time.time()

    for view, (frame, camera) in enumerate(view_records):
        layout_label = "_graphdeco" if args.sh_degree else ""
        sampling_label = f"_coverage{candidate_samples}" if candidate_samples > args.samples_per_view else ""
        restore_label = "_restored" if args.restore_sh_band_weights else ""
        shard_path = shard_dir / f"view{view:02d}_frame{frame:03d}_cam{camera}_sh{args.sh_degree}{layout_label}{sampling_label}{restore_label}.npy"
        record = {"view_index": view, "frame": int(frame), "camera": int(camera),
                  "shard": str(shard_path), "requested_gaussians": args.samples_per_view}
        if args.resume and shard_path.is_file():
            vertices = np.load(shard_path, mmap_mode="r")
            if vertices.dtype == expected_dtype and len(vertices) == args.samples_per_view:
                record.update(status="reused", gaussian_count=len(vertices))
                view_metadata.append(record)
                print(f"[{view + 1}/{len(view_records)}] reused {shard_path.name}", flush=True)
                continue
        image = batch["images"][view:view + 1].to(device)
        intrinsics = batch["intrinsics"][view:view + 1].to(device)
        c2w = batch["cam_to_world"][view].to(device)
        metric_depth = batch["sparse_depth"][view:view + 1].to(device)
        depth_mask = (batch["sparse_mask"][view:view + 1].to(device) > 0.5)
        valid = depth_mask & (metric_depth > 1.0) & (metric_depth < 100.0)
        metric_depth = torch.where(valid, metric_depth, torch.zeros_like(metric_depth))
        valid_indices = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
        generator = torch.Generator(device=device).manual_seed(config.get("seed", 0) + view)
        if len(valid_indices) > 1500:
            selected = valid_indices[torch.randperm(len(valid_indices), generator=generator, device=device)[:1500]]
            prompt = torch.zeros_like(metric_depth)
            prompt.flatten()[selected] = metric_depth.flatten()[selected]
        else:
            prompt = metric_depth.clone()
        gt_disparity = depth_to_disparity(metric_depth)
        prompt_disparity = depth_to_disparity(prompt)
        sky_mask = run_optional_sampling_sky_mask(
            image=image,
            enable_skyseg_model=not args.disable_sky_mask,
            sky_model_ckpt_path=str(sky_checkpoint),
            dilate_px=0,
        )
        identity = torch.eye(4, device=device).unsqueeze(0)
        depthmap, dino_tokens, query_coord, sampled_depth = model.inference_for_gs(
            image=image,
            intrinsics=intrinsics,
            gt_depth=gt_disparity,
            gt_depth_mask=valid.float(),
            prompt_depth=prompt_disparity,
            prompt_mask=prompt > 0,
            sky_mask=sky_mask,
            sample_point_num=candidate_samples,
            coord_deterministic_sampling=True,
        )
        unique_candidate_pixels = None
        if candidate_samples > args.samples_per_view:
            coverage_indices = spatial_coverage_indices(
                query_coord,
                args.samples_per_view,
                image.shape[-2],
                image.shape[-1],
                args.spatial_block_size,
                int(config.get("seed", 0)) + view,
            )
            unique_candidate_pixels = int(torch.unique(coverage_indices).numel())
            query_coord = query_coord[:, coverage_indices]
            sampled_depth = sampled_depth[:, coverage_indices]
        dense = gs_predictor(image, depthmap, dino_tokens, intrinsics, identity)
        gaussians = _build_sparse_uniform_gaussians(
            dense, query_coord, sampled_depth, intrinsics, identity,
            image.shape[-2], image.shape[-1])
        means = gaussians.means[0]
        means = means @ c2w[:3, :3].T + c2w[:3, 3]
        world_quat = torch.from_numpy(matrix_to_quaternion_wxyz(
            c2w[:3, :3].float().cpu().numpy())).to(device)
        rotations = quaternion_multiply_wxyz(world_quat.expand_as(gaussians.rotations[0]),
                                              gaussians.rotations[0])
        rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        harmonics = gaussians.harmonics[0]
        if args.sh_degree == 2:
            if args.restore_sh_band_weights:
                harmonics = restore_sh_band_weights(harmonics)
            harmonics = rotate_sh_degree2(harmonics, c2w[:3, :3])
        vertices = make_vertex_array(means, harmonics, gaussians.opacities[0],
                                     gaussians.scales[0], rotations, args.sh_degree)
        temporary = shard_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, vertices, allow_pickle=False)
        os.replace(temporary, shard_path)
        record.update(status="generated", gaussian_count=len(vertices),
                      candidate_gaussians=candidate_samples,
                      spatial_selection="unique-pixel block-stratified" if candidate_samples > args.samples_per_view else "native",
                      spatial_block_size=args.spatial_block_size if candidate_samples > args.samples_per_view else None,
                      selected_unique_pixels=unique_candidate_pixels,
                      lidar_valid=int(valid.sum()), prompt_points=int((prompt > 0).sum()),
                      cam_to_world=c2w.float().cpu().numpy().tolist(),
                      xyz_min=[float(value) for value in means.min(dim=0).values.cpu()],
                      xyz_max=[float(value) for value in means.max(dim=0).values.cpu()])
        view_metadata.append(record)
        print(f"[{view + 1}/{len(view_records)}] generated frame {frame} cam {camera}: {len(vertices)}", flush=True)
        del image, metric_depth, prompt, depthmap, dino_tokens, query_coord, sampled_depth, dense, gaussians, vertices
        torch.cuda.empty_cache()

    shards = [Path(record["shard"]) for record in view_metadata]
    full_count = write_binary_ply(full_path, shards)
    seed = int(config.get("seed", 0))
    budget_count = None
    if not args.only_full:
        budget_indices = [
            deterministic_sample_indices(record["gaussian_count"], budget_per_view, seed + view)
            for view, record in enumerate(view_metadata)
        ]
        budget_count = write_binary_ply(budget_path, shards, budget_indices)
    manifest = {
        "status": "completed",
        "scene": metadata["name"],
        "clip_index": args.clip_index,
        "window_frames": [int(value) for value in window],
        "context_frames": [int(value) for value in context_frames],
        "cameras": [int(value) for value in dataset.cameras],
        "view_order": "frame-major then camera order",
        "view_count": len(view_records),
        "input_resolution_hw": list(args.input_size),
        "samples_per_view": args.samples_per_view,
        "candidate_samples_per_view": candidate_samples,
        "sampling_policy": (
            f"native {candidate_samples} candidates followed by unique-pixel block-stratified selection "
            f"to {args.samples_per_view} using {args.spatial_block_size}x{args.spatial_block_size} blocks"
            if candidate_samples > args.samples_per_view else "native deterministic triangle sampling"
        ),
        "sh_degree": args.sh_degree,
        "sh_coefficients_per_channel": (args.sh_degree + 1) ** 2,
        "sh_coordinate_frame": "world viewing-direction basis" if args.sh_degree else "constant DC only",
        "sh_rest_layout": "Graphdeco coefficient-major RGB triplets: k1-RGB, k2-RGB, ...",
        "sh_band_weight_restoration": (
            {"degree_1_divisor": 0.025, "degree_2_divisor": 0.00625}
            if args.restore_sh_band_weights else None
        ),
        "depth_model": "InfiniDepth_DepthSensor",
        "depth_checkpoint": {"path": str(depth_checkpoint), "sha256": sha256(depth_checkpoint)},
        "gs_checkpoint": {"path": str(gs_checkpoint), "sha256": sha256(gs_checkpoint)},
        "lidar_prompt": "same-frame projected LiDAR; native range (1,100)m; deterministic maximum 1500 prompts",
        "native_gs_process": "per-view inference_for_gs + GSPixelAlignPredictor + _build_sparse_uniform_gaussians",
        "merge": "rigid c2w transform to PointForward recentered world; concatenation only; no cross-view fusion",
        "full_output": {"path": str(full_path), "gaussian_count": full_count,
                        "policy": "all requested native samples; no statistical/opacity/voxel/dedup filtering"},
        "budget_output": None if args.only_full else {
            "path": str(budget_path), "gaussian_count": budget_count,
            "policy": f"deterministic uniform sampling without replacement, {budget_per_view} per view, seed {seed}+view_index",
        },
        "native_sampling_filters": "InfiniDepth triangle discontinuity pruning and camera-Z <90m face selection still apply before exact-count resampling",
        "dynamic_scene_warning": "Four timestamps are concatenated without motion compensation; moving objects may be duplicated/trailed",
        "coordinate_frame": "world recentered at the first ego pose of this clip",
        "vertex_schema": [name for name, _ in vertex_fields(args.sh_degree)],
        "shards": view_metadata,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"full": manifest["full_output"], "budget": manifest["budget_output"],
                      "manifest": str(manifest_path), "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
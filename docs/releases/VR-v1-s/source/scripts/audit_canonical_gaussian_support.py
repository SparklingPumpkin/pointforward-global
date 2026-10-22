"""Audit selected-view support of canonical dynamic Gaussian centres.

This script is evaluation-only.  For each rebuilt track it takes the final
canonical Gaussian centres, de-canonicalises them with the track box pose at
each of the track's fixed selected views, and projects the centres into that
camera.  Calibrated ``fine_dynamic_masks/vehicle`` images are read only to
measure support; they are never fed back to reconstruction.

The resulting centre metrics intentionally ignore Gaussian opacity, scale,
footprint, depth ordering, and occlusion.  They therefore answer a different
question from rendered-alpha support metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.models import InstanceBoxes, box_dynamic_masks, decanonicalise, project_points
from scripts.train import build_dataset


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_boxes(payload: dict) -> InstanceBoxes:
    return InstanceBoxes(
        centers=payload["centers"].cpu(),
        dims=payload["dims"].cpu(),
        rots=payload["rots"].cpu(),
        times=payload["times"].cpu(),
        valid=payload["valid"].cpu(),
    )


def empty_counts() -> dict[str, int]:
    return {
        "canonical_center_count": 0,
        "selected_view_count": 0,
        "center_view_pair_count": 0,
        "valid_projection_count": 0,
        "fine_mask_hit_count": 0,
        "bbox_hit_count": 0,
        "fine_and_bbox_hit_count": 0,
        "fine_mask_pixel_count_in_bbox": 0,
        "unique_projected_fine_pixel_count": 0,
    }


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key in target:
        target[key] += int(source[key])


def scored_counts(counts: dict[str, int]) -> dict[str, int | float | None]:
    return counts | {
        # Every valid projected centre is treated as a positive prediction.
        "center_support_precision": ratio(
            counts["fine_mask_hit_count"], counts["valid_projection_count"]
        ),
        "valid_projection_rate": ratio(
            counts["valid_projection_count"], counts["center_view_pair_count"]
        ),
        "projected_center_in_bbox_rate": ratio(
            counts["bbox_hit_count"], counts["valid_projection_count"]
        ),
        "fine_support_precision_within_bbox": ratio(
            counts["fine_and_bbox_hit_count"], counts["bbox_hit_count"]
        ),
        # Zero-radius point coverage: unique fine-mask pixels hit by at least
        # one rounded centre, divided by fine vehicle pixels in this track bbox.
        "fine_mask_zero_radius_pixel_coverage": ratio(
            counts["unique_projected_fine_pixel_count"],
            counts["fine_mask_pixel_count_in_bbox"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    rebuilt_dir = run_dir / "rebuilt-top10"
    manifest_path = rebuilt_dir / "track_once_manifest.json"
    dynamic_path = rebuilt_dir / "final_dynamic_128d.pt"
    meta_path = rebuilt_dir / "global_dynamic_meta.pt"
    output_dir = (
        Path(args.out).resolve()
        if args.out is not None else run_dir / "canonical-support-audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (config_path, manifest_path, dynamic_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(path)

    config = yaml.safe_load(config_path.read_text())
    dataset = build_dataset(config, args.split)
    scene_matches = [item for item in dataset.scenes if item["name"] == args.scene]
    if len(scene_matches) != 1:
        raise ValueError(f"Expected one scene {args.scene!r}, found {len(scene_matches)}")
    metadata = scene_matches[0]
    frames = [int(value) for value in metadata["frames"]]
    slot_of = {frame: slot for slot, frame in enumerate(frames)}
    camera_to_index = {int(camera): index for index, camera in enumerate(dataset.cameras)}
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    height, width = dataset.resolution

    track_manifest = json.loads(manifest_path.read_text())
    dynamic = torch.load(dynamic_path, map_location="cpu", weights_only=True)
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    if [int(value) for value in meta["frames"]] != frames:
        raise ValueError("Payload frame list does not match dataset scene")
    boxes = load_boxes(meta["boxes"])
    canonical_centers, canonical_rots = boxes.canonical()
    means = dynamic["means"].cpu()
    instance_id = dynamic["instance_id"].long().cpu()

    aggregate = empty_counts()
    track_results = []
    fine_paths: set[Path] = set()

    for record in track_manifest["tracks"]:
        global_slot = int(record["global_slot"])
        track_id = int(record["track_id"])
        track_means = means[instance_id == global_slot]
        if track_means.shape[0] != int(record["dynamic_count"]):
            raise ValueError(
                f"Track {track_id}: payload has {track_means.shape[0]} centres but "
                f"manifest records {record['dynamic_count']}"
            )
        track_counts = empty_counts()
        track_counts["canonical_center_count"] = int(track_means.shape[0])
        track_counts["selected_view_count"] = len(record["selected_views"])
        supported_centers = torch.zeros(track_means.shape[0], dtype=torch.bool)
        visible_centers = torch.zeros_like(supported_centers)
        view_results = []

        for view in record["selected_views"]:
            frame = int(view["frame"])
            camera = int(view["camera"])
            if frame not in slot_of or camera not in camera_to_index:
                raise ValueError(f"Unknown selected view {(frame, camera)}")
            slot = slot_of[frame]
            if not bool(boxes.valid[global_slot, slot]):
                raise ValueError(
                    f"Selected view {(frame, camera)} has invalid box for track {track_id}"
                )

            relative = f"fine_dynamic_masks/vehicle/{frame:03d}_{camera}.png"
            fine_path = Path(metadata["path"]) / relative
            fine_paths.add(fine_path.resolve())
            fine_image, intrinsic_np = dataset._load_calibrated_image(
                metadata, relative, dataset.resolution, camera, gray=True
            )
            fine = fine_image.squeeze(0) > 0.5
            intrinsic = torch.from_numpy(intrinsic_np).float().unsqueeze(0)
            cam_to_world = torch.from_numpy(
                dataset._cam_to_world(metadata, frame, camera, world_from)
            ).float().unsqueeze(0)
            world_to_cam = torch.linalg.inv(cam_to_world)

            world_points = decanonicalise(
                track_means,
                boxes.centers[global_slot, slot],
                boxes.rots[global_slot, slot],
                canonical_centers[global_slot],
                canonical_rots[global_slot],
            )
            uv, _, in_front = project_points(
                world_points.unsqueeze(0), intrinsic, world_to_cam
            )
            uv = uv[0]
            valid = in_front[0] & (
                (uv[:, 0] >= 0)
                & (uv[:, 0] <= width - 1)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] <= height - 1)
            )
            u = uv[:, 0].round().long().clamp(0, width - 1)
            v = uv[:, 1].round().long().clamp(0, height - 1)
            fine_hit = valid & fine[v, u]

            single_box = InstanceBoxes(
                centers=boxes.centers[global_slot : global_slot + 1],
                dims=boxes.dims[global_slot : global_slot + 1],
                rots=boxes.rots[global_slot : global_slot + 1],
                times=boxes.times[global_slot : global_slot + 1],
                valid=boxes.valid[global_slot : global_slot + 1],
            )
            bbox_mask = box_dynamic_masks(
                single_box,
                torch.tensor([slot], dtype=torch.long),
                intrinsic,
                world_to_cam,
                height,
                width,
            )[0]
            bbox_hit = valid & bbox_mask[v, u]
            fine_and_bbox = fine_hit & bbox_hit

            projected_fine_pixels = torch.zeros_like(fine)
            projected_fine_pixels[v[fine_and_bbox], u[fine_and_bbox]] = True
            fine_in_bbox = fine & bbox_mask
            unique_supported_pixels = projected_fine_pixels & fine_in_bbox

            view_counts = empty_counts()
            view_counts["canonical_center_count"] = int(track_means.shape[0])
            view_counts["selected_view_count"] = 1
            view_counts["center_view_pair_count"] = int(track_means.shape[0])
            view_counts["valid_projection_count"] = int(valid.sum())
            view_counts["fine_mask_hit_count"] = int(fine_hit.sum())
            view_counts["bbox_hit_count"] = int(bbox_hit.sum())
            view_counts["fine_and_bbox_hit_count"] = int(fine_and_bbox.sum())
            view_counts["fine_mask_pixel_count_in_bbox"] = int(fine_in_bbox.sum())
            view_counts["unique_projected_fine_pixel_count"] = int(
                unique_supported_pixels.sum()
            )
            add_counts(track_counts, {
                key: value if key not in ("canonical_center_count", "selected_view_count") else 0
                for key, value in view_counts.items()
            })
            supported_centers |= fine_hit
            visible_centers |= valid
            view_results.append({
                "frame": frame,
                "camera": camera,
                **scored_counts(view_counts),
            })

        track_result = {
            "track_id": track_id,
            "global_slot": global_slot,
            **scored_counts(track_counts),
            "canonical_center_any_fine_support_count": int(supported_centers.sum()),
            "canonical_center_any_fine_support_coverage": ratio(
                int(supported_centers.sum()), int(track_means.shape[0])
            ),
            "canonical_center_any_valid_projection_count": int(visible_centers.sum()),
            "canonical_center_any_valid_projection_coverage": ratio(
                int(visible_centers.sum()), int(track_means.shape[0])
            ),
            "views": view_results,
        }
        track_results.append(track_result)
        add_counts(aggregate, track_counts)

    # Track-level centre counts must not be multiplied by the number of views.
    aggregate["canonical_center_count"] = sum(
        int(value["canonical_center_count"]) for value in track_results
    )
    aggregate["selected_view_count"] = sum(
        int(value["selected_view_count"]) for value in track_results
    )
    supported_count = sum(
        int(value["canonical_center_any_fine_support_count"])
        for value in track_results
    )
    visible_count = sum(
        int(value["canonical_center_any_valid_projection_count"])
        for value in track_results
    )

    result = {
        "status": "completed",
        "diagnostic_only": True,
        "scene": args.scene,
        "run_dir": str(run_dir),
        "resolution_hw": [height, width],
        "track_count": len(track_results),
        "definitions": {
            "canonical_centres": (
                "final means in rebuilt-top10/final_dynamic_128d.pt, grouped by "
                "global instance slot; no opacity threshold is applied"
            ),
            "pose_and_projection": (
                "each canonical centre is de-canonicalised with that track's "
                "box pose at each fixed selected view and projected with the "
                "calibrated camera"
            ),
            "valid_projection": (
                "positive camera depth and rounded centre inside [0,W-1]x[0,H-1]"
            ),
            "center_support_precision": (
                "valid projected centre-view pairs landing in calibrated fine "
                "vehicle mask / all valid projected centre-view pairs"
            ),
            "canonical_center_any_fine_support_coverage": (
                "canonical centres landing in calibrated fine vehicle mask in at "
                "least one selected view / all canonical centres"
            ),
            "fine_mask_zero_radius_pixel_coverage": (
                "unique calibrated fine-vehicle pixels within this track's "
                "projected bbox hit by a rounded zero-radius centre / fine-vehicle "
                "pixels within that bbox"
            ),
            "projected_center_in_bbox_rate": (
                "valid projected centres within the same track's projected box / "
                "all valid projected centres"
            ),
            "important_exclusion": (
                "centre-only audit ignores opacity, Gaussian scale/footprint, "
                "occlusion, depth ordering and rasteriser compositing; it is not "
                "the rendered-alpha support metric"
            ),
            "fine_mask_usage": (
                "fine_dynamic_masks/vehicle is evaluation-only and never enters "
                "reconstruction or modifies payload/PLY"
            ),
        },
        "aggregate_micro": scored_counts(aggregate) | {
            "canonical_center_any_fine_support_count": supported_count,
            "canonical_center_any_fine_support_coverage": ratio(
                supported_count, aggregate["canonical_center_count"]
            ),
            "canonical_center_any_valid_projection_count": visible_count,
            "canonical_center_any_valid_projection_coverage": ratio(
                visible_count, aggregate["canonical_center_count"]
            ),
        },
        "tracks": track_results,
        "inputs": {
            str(path): sha256_file(path)
            for path in (config_path, manifest_path, dynamic_path, meta_path)
        },
        "fine_vehicle_masks": {
            "file_count": len(fine_paths),
            "files": [str(path) for path in sorted(fine_paths)],
            "sha256_names_and_contents": hashlib.sha256(
                "".join(
                    f"{path}:{sha256_file(path)}\n" for path in sorted(fine_paths)
                ).encode("utf-8")
            ).hexdigest(),
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "timing": {"wall_seconds": time.perf_counter() - started},
    }
    json_path = output_dir / "canonical_support_audit.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    aggregate_result = result["aggregate_micro"]
    rows = [
        "| Track | Centres | Valid projections | Centre precision | Any-view centre coverage | In-box rate | Zero-radius pixel coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in track_results:
        rows.append(
            "| {track_id} | {canonical_center_count} | {valid_projection_count} | "
            "{center_support_precision:.6f} | {canonical_center_any_fine_support_coverage:.6f} | "
            "{projected_center_in_bbox_rate:.6f} | {fine_mask_zero_radius_pixel_coverage:.6f} |".format(
                **item
            )
        )
    markdown = "\n".join([
        "# Canonical Gaussian center support audit",
        "",
        "This is an evaluation-only, centre-projection audit over each top10 track's fixed 12 selected views. Fine vehicle masks are calibrated through the dataset adapter. No fine mask is used by reconstruction.",
        "",
        "It deliberately ignores opacity, Gaussian scale/footprint, occlusion and compositing, so these values must not be compared as if they were rendered-alpha support.",
        "",
        "## Aggregate",
        "",
        f"- Center support precision: `{aggregate_result['center_support_precision']:.6f}`",
        f"- Canonical center any-view fine-support coverage: `{aggregate_result['canonical_center_any_fine_support_coverage']:.6f}`",
        f"- Projected center in own bbox rate: `{aggregate_result['projected_center_in_bbox_rate']:.6f}`",
        f"- Fine-mask zero-radius pixel coverage: `{aggregate_result['fine_mask_zero_radius_pixel_coverage']:.6f}`",
        f"- Canonical centers: `{aggregate_result['canonical_center_count']}`",
        f"- Valid center-view projections: `{aggregate_result['valid_projection_count']}`",
        "",
        "## Per track",
        "",
        *rows,
        "",
        "Full per-view counts and exact definitions are in `canonical_support_audit.json`.",
        "",
    ])
    markdown_path = output_dir / "canonical_support_audit.md"
    markdown_path.write_text(markdown)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "aggregate_micro": result["aggregate_micro"],
        "wall_seconds": result["timing"]["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

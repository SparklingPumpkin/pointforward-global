"""Independently verify saved multi-clip audit artifacts on CPU."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "visualization/multi_clip_audit_heldout"
CONTINUOUS = ROOT / "visualization/multi_clip_continuous"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = max(float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2)), 1e-10)
    return float(-10.0 * np.log10(mse))


def average_json_metric(manifest: dict, comparison: str, path: tuple[str, ...]) -> float:
    values = []
    for row in manifest["comparisons"][comparison]["targets"]:
        item = row
        for key in path:
            item = item[key]
        values.append(float(item))
    return float(np.mean(values))


def verify_heldout() -> tuple[list[dict], list[dict]]:
    source_hash = sha256(ROOT / "scripts/audit_multi_clip_overlap.py")
    summaries = []
    files = []
    metric_paths = sorted(HELDOUT.glob("*/scene*_metrics.json"))
    if len(metric_paths) != 5:
        raise AssertionError(f"expected five heldout runs, found {len(metric_paths)}")
    for metric_path in metric_paths:
        manifest = json.loads(metric_path.read_text())
        if manifest["audit_script_sha256"] != source_hash:
            raise AssertionError(f"stale script hash in {metric_path}")
        context_union = set(manifest["runs"]["A1"]["context_frames"]) | set(
            manifest["runs"]["B1"]["context_frames"]
        )
        if set(manifest["target_frames"]) & context_union:
            raise AssertionError(f"context target leakage in {metric_path}")
        if not (
            manifest["runs"]["A1"]["sample_index_hash"]
            == manifest["runs"]["A2"]["sample_index_hash"]
        ):
            raise AssertionError(f"sample replay mismatch in {metric_path}")

        array_path = next(metric_path.parent.glob("*_arrays.npz"))
        with np.load(array_path) as arrays:
            for key in arrays.files:
                if arrays[key].dtype.kind in "fc" and not np.isfinite(arrays[key]).all():
                    raise AssertionError(f"nonfinite array {key} in {array_path}")
            for field in ("rgb", "alpha", "render_depth"):
                if not np.array_equal(arrays[f"A1_{field}"], arrays[f"A2_{field}"]):
                    raise AssertionError(f"A1/A2 {field} mismatch in {array_path}")
            recomputed = {}
            for name, left, right in (
                ("sampling", "A1", "A3"),
                ("window", "A1", "B1"),
                ("context_order_A", "A1", "A1_context_permuted"),
                ("context_order_B", "B1", "B1_context_permuted"),
            ):
                values = np.mean(
                    np.abs(arrays[f"{left}_rgb"].astype(np.float64) - arrays[f"{right}_rgb"].astype(np.float64)),
                    axis=(1, 2, 3),
                )
                recomputed[name] = float(values.mean())
            gt_psnr = {
                label: float(np.mean([psnr(image, truth) for image, truth in zip(arrays[f"{label}_rgb"], arrays["gt"])]))
                for label in ("A1", "B1")
            }

        expected = {
            "sampling": average_json_metric(manifest, "sampling_A1_A3", ("regions", "full", "l1")),
            "window": average_json_metric(manifest, "window_A1_B1", ("regions", "full", "l1")),
            "context_order_A": average_json_metric(manifest, "context_order_A", ("regions", "full", "l1")),
            "context_order_B": average_json_metric(manifest, "context_order_B", ("regions", "full", "l1")),
        }
        for key in expected:
            if not np.isclose(recomputed[key], expected[key], rtol=1e-6, atol=1e-8):
                raise AssertionError(f"{key} mismatch in {metric_path}: {recomputed[key]} vs {expected[key]}")
        for label in ("A1", "B1"):
            recorded = float(np.mean([
                row["regions"]["full"]["psnr"] for row in manifest["metrics_to_gt"][label]
            ]))
            if not np.isclose(gt_psnr[label], recorded, rtol=1e-6, atol=1e-6):
                raise AssertionError(f"GT PSNR mismatch for {label} in {metric_path}")

        row = {
            "run": metric_path.parent.name,
            "scene": manifest["scene"],
            "shift": int(manifest["runs"]["B1"]["window"][0]),
            "targets": manifest["target_frames"],
            "sampling_rgb_l1": recomputed["sampling"],
            "window_rgb_l1": recomputed["window"],
            "window_over_sampling": recomputed["window"] / recomputed["sampling"],
            "context_order_A_rgb_l1": recomputed["context_order_A"],
            "context_order_B_rgb_l1": recomputed["context_order_B"],
            "context_order_A_over_window": recomputed["context_order_A"] / recomputed["window"],
            "context_order_B_over_window": recomputed["context_order_B"] / recomputed["window"],
            "A_gt_psnr": gt_psnr["A1"],
            "B_gt_psnr": gt_psnr["B1"],
        }
        summaries.append(row)
        for path in sorted(metric_path.parent.iterdir()):
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return summaries, files


def verify_continuous() -> tuple[dict, list[dict]]:
    rows = []
    files = []
    for scene_dir in sorted(path for path in CONTINUOUS.iterdir() if path.is_dir()):
        csv_path = next(scene_dir.glob("*_per_frame.csv"))
        json_path = next(scene_dir.glob("*_summary.json"))
        with csv_path.open() as handle:
            scene_rows = list(csv.DictReader(handle))
        if len(scene_rows) != 60:
            raise AssertionError(f"expected 60 rows in {csv_path}")
        rows.extend(scene_rows)
        for path in (csv_path, json_path):
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    if len(rows) != 180:
        raise AssertionError(f"expected 180 continuous rows, found {len(rows)}")
    boundaries = []
    for scene in sorted({row["scene"] for row in rows}):
        scene_rows = [row for row in rows if row["scene"] == scene]
        for index in range(1, len(scene_rows)):
            previous, current = scene_rows[index - 1], scene_rows[index]
            if current["window_index"] != previous["window_index"]:
                boundaries.append({
                    "scene": scene,
                    "from": int(previous["frame"]),
                    "to": int(current["frame"]),
                    "psnr_change_db": float(current["psnr"]) - float(previous["psnr"]),
                })
    if len(boundaries) != 6:
        raise AssertionError(f"expected six boundaries, found {len(boundaries)}")
    local_psnr = {
        str(index): float(np.mean([float(row["psnr"]) for row in rows if int(row["local_index"]) == index]))
        for index in range(20)
    }
    return {
        "frame_count": len(rows),
        "boundary_count": len(boundaries),
        "boundaries": boundaries,
        "boundaries_psnr_improved": sum(row["psnr_change_db"] > 0 for row in boundaries),
        "boundaries_psnr_degraded": sum(row["psnr_change_db"] < 0 for row in boundaries),
        "local_psnr": local_psnr,
    }, files


def main() -> None:
    heldout, heldout_files = verify_heldout()
    continuous, continuous_files = verify_continuous()
    report = {
        "status": "PASS",
        "method": "independent CPU recomputation from final NPZ/CSV artifacts",
        "source_files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                ROOT / "scripts/audit_multi_clip_overlap.py",
                ROOT / "scripts/audit_multi_clip_continuous.py",
                ROOT / "pointforward/models/pointforward.py",
                ROOT / "pointforward/data/drivestudio.py",
            )
        },
        "heldout": heldout,
        "continuous": continuous,
        "artifact_files": heldout_files + continuous_files,
    }
    output = ROOT / "docs/multi-clip/artifact_verification.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"status": report["status"], "output": str(output), "heldout_runs": len(heldout), **continuous}, indent=2))


if __name__ == "__main__":
    main()
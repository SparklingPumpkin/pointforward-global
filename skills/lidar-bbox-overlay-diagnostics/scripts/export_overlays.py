#!/usr/bin/env python3
"""Export unchanged RGB + 3-D box + raw LiDAR diagnostic overlays.

This is deliberately a diagnostic renderer: it does not calibrate, fit, deskew,
move, or rewrite any source geometry.  It replays the parent run's point labels
and image identities when those records are available.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
YELLOW = (255, 220, 0)
ORANGE = (255, 110, 35)
CYAN = (0, 225, 255)
SIGNS = np.asarray([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
], dtype=float)


@lru_cache(None)
def fonts(size: int):
    latin = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fallback = FONT if Path(FONT).exists() else latin
    return ImageFont.truetype(latin, size), ImageFont.truetype(fallback, size)


def draw_text(draw, position, message: str, size: int):
    latin, cjk = fonts(size)
    x, y = position
    for char in message:
        font = latin if ord(char) < 128 else cjk
        draw.text((x, y), char, font=font, fill="white")
        x += font.getlength(char)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def key(mapping, value):
    """Read a frame/track from dictionaries saved with int or string keys."""
    if value in mapping:
        return mapping[value]
    return mapping[str(value)]


def box_uv(actor, camera, size, intrinsic):
    cp = transform(transform(SIGNS * size * 0.5, actor), np.linalg.inv(camera))
    if (cp[:, 2] <= 0.1).any():
        return None
    return (cp @ intrinsic.T)[:, :2] / cp[:, 2:]


def draw_box(rgb: np.ndarray, uv):
    image = Image.fromarray(rgb.copy())
    if uv is None:
        return np.asarray(image)
    draw = ImageDraw.Draw(image)
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1:
                draw.line([tuple(uv[i]), tuple(uv[j])], fill=YELLOW, width=2)
    return np.asarray(image)


def panel(rgb: np.ndarray, title: str, crop=None, size=(768, 576)):
    image = Image.fromarray(rgb)
    if crop is not None:
        image = image.crop(crop)
    image = ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color="#171717")
    result = Image.new("RGB", (size[0], size[1] + 44), "#171717")
    result.paste(image, (0, 44))
    draw_text(ImageDraw.Draw(result), (12, 9), title, 22)
    return result


def sheet(panels, title, footer, panel_size=(768, 576)):
    pw, ph = panel_size
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    image = Image.new("RGB", (cols * pw, 46 + rows * (ph + 44) + 74), "#171717")
    draw = ImageDraw.Draw(image)
    draw_text(draw, (12, 9), title, 25)
    for index, item in enumerate(panels):
        image.paste(item, (index % cols * pw, 46 + index // cols * (ph + 44)))
    for index, line in enumerate(footer):
        draw_text(draw, (12, image.height - 55 + index * 25), line, 19)
    return image


def find_rgb(args, audit: Path, frame: int, camera: int):
    name = args.image_pattern.format(frame=frame, camera=camera)
    candidates = []
    if args.image_dir:
        candidates.append(Path(args.image_dir) / name)
    candidates.append(audit / "images" / name)
    if args.fallback_image_dir:
        candidates.append(Path(args.fallback_image_dir) / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"RGB image not found for frame={frame}, camera={camera}: {candidates}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default="/mnt/fjn/projects/PointForward",
                    help="PointForward checkout containing experiments/lidar-bbox-calibration/benchmark")
    ap.add_argument("--audit-dir", required=True, help="Audit scene directory containing data_contract.json and poses.pt")
    ap.add_argument("--parent-run", required=True, help="Parent run containing manifest.json and *_labels.npz")
    ap.add_argument("--output-dir", required=True, help="Directory for diagnostic images and manifest")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--track-id", type=int, required=True)
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--scene", default=None, help="Manifest/label scene name; defaults to contract spec.scene")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--fallback-image-dir", default=None)
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--poses", default=None)
    ap.add_argument("--contract", default=None)
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--label-pattern", default="{scene}_t{track:03d}_f{frame:03d}_c{camera}_labels.npz")
    ap.add_argument("--image-pattern", default="{frame:03d}_{camera}.png")
    ap.add_argument("--lidar-pattern", default="{frame:03d}.bin")
    ap.add_argument("--skip-replay-checks", action="store_true",
                    help="Skip parent-manifest/label identity checks (use only for a new diagnostic source)")
    args = ap.parse_args()

    project = Path(args.project_root).resolve()
    audit = Path(args.audit_dir).resolve()
    parent = Path(args.parent_run).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract_path = Path(args.contract).resolve() if args.contract else audit / "data_contract.json"
    poses_path = Path(args.poses).resolve() if args.poses else audit / "poses.pt"
    contract = json.loads(contract_path.read_text())
    poses = torch.load(poses_path, map_location="cpu", weights_only=False)
    raw_root = Path(args.raw_root).resolve() if args.raw_root else Path(contract["raw_root"]).resolve()
    label_dir = Path(args.label_dir).resolve() if args.label_dir else parent
    image_dir = Path(args.image_dir).resolve() if args.image_dir else None
    fallback = Path(args.fallback_image_dir).resolve() if args.fallback_image_dir else None

    benchmark = project / "experiments/lidar-bbox-calibration/benchmark"
    if str(benchmark) not in sys.path:
        sys.path.insert(0, str(benchmark))
    try:
        from calibration_core import SIGNS as _SIGNS, box_uv as _box_uv, transform as _transform
        # Keep the benchmark's projection convention when available.
        global SIGNS, box_uv, transform
        SIGNS, box_uv, transform = _SIGNS, _box_uv, _transform
    except ImportError:
        pass

    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text()) if parent_manifest_path.exists() else []
    manifest_scenes = sorted({r.get("scene") for r in parent_manifest
                              if r.get("kind") == "rectified_rgb" and r.get("scene")})
    scene_candidates = [candidate for candidate in manifest_scenes
                        if all(any(r.get("scene") == candidate and r.get("frame") == frame
                                   and r.get("camera") == args.camera for r in parent_manifest)
                               for frame in args.frames)]
    scene = (args.scene or (scene_candidates[0] if scene_candidates else None)
             or poses.get("spec", {}).get("scene") or contract.get("scene") or "scene")
    image_records = {}
    expected_files = {}
    for record in parent_manifest:
        if "path" in record and "sha256" in record:
            expected_files[record["path"]] = record["sha256"]
        if record.get("kind") == "rectified_rgb" and record.get("frame") is not None:
            if record.get("scene", scene) == scene and record.get("camera") == args.camera:
                image_records[(record["frame"], record["camera"])] = record
    if not args.skip_replay_checks and not image_records:
        raise RuntimeError(f"No rectified_rgb records for scene={scene!r}, camera={args.camera} in {parent_manifest_path}")

    intrinsic = np.asarray(contract["intrinsics"][str(args.camera)], dtype=float)
    extrinsic = np.asarray(contract["extrinsics"][str(args.camera)], dtype=float)
    records = []
    panels_for_overview = []
    overview_titles = []
    footer = [
        "黄色=原始3D框；橙色=框内原始点；青色=1.5倍框范围内的框外点；灰色=其余点。",
        "颜色只表示空间包含关系，不代表车身归属；同像素保留最近点，未做光学遮挡判定。",
    ]

    for frame in args.frames:
        name = args.image_pattern.format(frame=frame, camera=args.camera)
        image_path = find_rgb(args, audit, frame, args.camera)
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = rgb.shape[:2]
        if not args.skip_replay_checks:
            expected = image_records[(frame, args.camera)]
            content_sha = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            if content_sha != expected["sha256"]:
                raise AssertionError((frame, "RGB identity mismatch", image_path))
            if not np.allclose(intrinsic, expected["K"]):
                raise AssertionError((frame, "intrinsics mismatch"))
        else:
            content_sha = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

        lidar_path = raw_root / "lidar" / args.lidar_pattern.format(frame=frame)
        if not lidar_path.exists():
            raise FileNotFoundError(lidar_path)
        lidar_sha = sha(lidar_path)
        if not args.skip_replay_checks and str(lidar_path) in expected_files and lidar_sha != expected_files[str(lidar_path)]:
            raise AssertionError((frame, "LiDAR identity mismatch", lidar_path))
        raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)

        ego = np.asarray(key(poses["ego"], frame))
        actor = np.asarray(key(poses["actors"][args.track_id], frame))
        size = np.asarray(key(poses["sizes"][args.track_id], frame))
        camera = ego @ extrinsic
        world = transform(raw, ego)
        local = transform(world, np.linalg.inv(actor))
        inside = (abs(local) <= size * 0.5).all(1)
        ring = (abs(local) <= size * 0.75).all(1) & ~inside
        cp = transform(world, np.linalg.inv(camera))
        uv = np.full((len(cp), 2), np.nan)
        front = cp[:, 2] > 0.1
        uv[front] = (cp[front] @ intrinsic.T)[:, :2] / cp[front, 2:]
        xy = np.rint(np.where(np.isfinite(uv), uv, -1000)).astype(int)
        valid = (front & (cp[:, 2] >= 1) & (cp[:, 2] <= 100) &
                 (xy[:, 0] >= 0) & (xy[:, 0] < width) &
                 (xy[:, 1] >= 0) & (xy[:, 1] < height))
        ids = np.flatnonzero(valid)
        pix = xy[ids, 1] * width + xy[ids, 0]
        order = np.lexsort((ids, cp[ids, 2], pix))
        ids, pix = ids[order], pix[order]
        ids = ids[np.r_[True, pix[1:] != pix[:-1]]] if len(pix) else ids

        label_path = label_dir / args.label_pattern.format(scene=scene, track=args.track_id, frame=frame, camera=args.camera)
        if not args.skip_replay_checks:
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            old = np.load(label_path)
            np.testing.assert_array_equal(np.flatnonzero(inside), old["point_ids"][old["box"]])
            np.testing.assert_allclose(uv[old["point_ids"]], old["source_uv"], atol=1e-8, equal_nan=True)

        points = rgb.copy()
        context = ids[~(inside | ring)[ids]]
        if len(context):
            px = xy[context]
            points[px[:, 1], px[:, 0]] = np.rint(
                points[px[:, 1], px[:, 0]] * 0.55 + np.array([175, 175, 175]) * 0.45
            ).astype(np.uint8)
        for selection, color in ((ring, CYAN), (inside, ORANGE)):
            chosen = ids[selection[ids]]
            for x, y in xy[chosen]:
                cv2.circle(points, (int(x), int(y)), 1, color, -1)
        bbox = box_uv(actor, camera, size, intrinsic)
        boxed = draw_box(rgb, bbox)
        combined = draw_box(points, bbox)
        if bbox is None:
            lo, hi = np.array([0, 0]), np.array([width, height])
        else:
            lo = np.maximum(np.floor(bbox.min(0) - [100, 80]), [0, 0]).astype(int)
            hi = np.minimum(np.ceil(bbox.max(0) + [100, 80]), [width, height]).astype(int)
            near = ids[(inside | ring)[ids]]
            if len(near):
                lo = np.minimum(lo, np.maximum(xy[near].min(0) - 15, [0, 0]))
                hi = np.maximum(hi, np.minimum(xy[near].max(0) + 16, [width, height]))
        crop = tuple(map(int, (*lo, *hi)))
        variants = [(rgb, "GT 原图"), (boxed, "GT + 原始3D框"),
                    (points, "GT + 原始LiDAR"), (combined, "GT + 原始3D框 + 原始LiDAR")]
        title = f"frame {frame:03d} / camera {args.camera} / track {args.track_id}：未修正原始几何"
        full = out / f"raw_f{frame:03d}_camera{args.camera}_full.jpg"
        detail = out / f"raw_f{frame:03d}_camera{args.camera}_detail.jpg"
        sheet([panel(a, b, size=(width, height)) for a, b in variants], title, footer, (width, height)).save(full, quality=97)
        sheet([panel(a, b + "（同比例裁剪）", crop, size=(width, height)) for a, b in variants], title, footer, (width, height)).save(detail, quality=97)
        overlay = out / f"raw_f{frame:03d}_camera{args.camera}_overlay.png"
        Image.fromarray(combined).save(overlay)
        panels_for_overview.append(panel(combined, f"frame {frame:03d} / 同比例裁剪", crop, size=(width, height)))
        overview_titles.append(frame)
        records.append({
            "frame": frame, "camera": args.camera, "track_id": args.track_id,
            "original_lidar_points": len(raw), "raw_box_points": int(inside.sum()),
            "raw_ring_points": int(ring.sum()), "displayed_box": int(inside[ids].sum()),
            "displayed_ring": int(ring[ids].sum()), "crop_xyxy": list(crop),
            "rgb_path": str(image_path), "rgb_content_sha256": content_sha,
            "lidar_path": str(lidar_path), "lidar_sha256": lidar_sha,
            "box": actor.tolist(), "box_size": size.tolist(), "K": intrinsic.tolist(),
            "camera_pose": camera.tolist(), "membership_replay": not args.skip_replay_checks,
            "uv_replay": not args.skip_replay_checks,
            "outputs": [full.name, detail.name, overlay.name],
        })

    overview = out / "overview_six_frames.jpg"
    sheet(panels_for_overview, f"原始框 + 原始LiDAR + GT：camera {args.camera} / track {args.track_id}，未修正",
          footer, (768, 576)).save(overview, quality=97)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Unchanged raw RGB, 3-D box and LiDAR diagnostic overlays",
        "correction_applied": False, "new_inference": False,
        "aspect_ratio_preserved": True,
        "projection_convention": "same nominal ego-frame projection as source; no timing/rolling-shutter/point-deskew correction",
        "retention": "keep: requested diagnosis evidence and small reproducible visualizations",
        "source_script_sha256": sha(Path(__file__)), "source_audit": str(audit),
        "parent_run": str(parent), "scene": scene, "replay_checks_passed": not args.skip_replay_checks,
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "frames": args.frames,
                      "replay_checks_passed": not args.skip_replay_checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()

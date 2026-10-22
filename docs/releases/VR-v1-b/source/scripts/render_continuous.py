'''Render a continuous multi-window video from a single checkpoint.

A single decode() call only covers one ~2s (clip_length-frame) window -- the
model was trained with locally-normalised times in [0, 1] over that window,
so pushing target times far outside it is out-of-distribution.  To render a
longer continuous clip (e.g. 60 frames = 3 windows) without extrapolating,
this script decodes Gaussians independently once per consecutive window, but
shares ONE world_from (anchored at the very first frame of the whole
sequence) across all windows so the ego motion/world geometry stays
spatially continuous -- only the per-window time normalisation restarts at
each window boundary, matching how the model was trained.

Usage:
    python -m scripts.render_continuous --config configs/waymo_ds.yaml \
        --ckpt runs/waymo_ds_run1/ckpt_006000.pt --out runs/waymo_ds_run1/continuous_60 \
        --num-frames 60
'''

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from pointforward.data.drivestudio import WaymoDriveStudioClips
from pointforward.models import InstanceBoxes, SceneInput, TargetViews

from scripts.eval import build_model
from scripts.train import build_dataset


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.clamp(0, 1).detach().cpu().numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 0))
    return np.array(pil)


def boxes_to(boxes, device):
    if boxes is None:
        return None
    return InstanceBoxes(
        centers=boxes.centers.to(device),
        dims=boxes.dims.to(device),
        rots=boxes.rots.to(device),
        times=boxes.times.to(device),
        valid=boxes.valid.to(device),
    )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--num-frames', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-instances', type=int, default=1)
    ap.add_argument('--scene', default=None, help='force this scene name instead of random selection')
    args = ap.parse_args()

    with open(args.config, 'r') as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(cfg, args.split)
    assert isinstance(dataset, WaymoDriveStudioClips)
    print(f'{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes')

    rng = np.random.default_rng(args.seed)
    if args.scene is not None:
        candidates = [si for si, meta in enumerate(dataset.scenes)
                      if meta['name'] == args.scene and len(meta['frames']) >= args.num_frames]
        assert candidates, f'scene {args.scene!r} not found in split or has < {args.num_frames} contiguous frames'
    else:
        candidates = [si for si, meta in enumerate(dataset.scenes) if len(meta['frames']) >= args.num_frames]
        assert candidates, f'no scene has >= {args.num_frames} contiguous frames'

    chosen = None
    for si in rng.permutation(candidates):
        meta = dataset.scenes[si]
        n = len(meta['frames'])
        for start in rng.permutation(n - args.num_frames + 1)[:20]:
            window = meta['frames'][start : start + args.num_frames]
            present = sum(1 for per_frame in meta['instances'].values() if any(f in per_frame for f in window))
            if present >= args.min_instances:
                chosen = (int(si), int(start))
                break
        if chosen:
            break
    assert chosen is not None
    si, start = chosen
    meta = dataset.scenes[si]
    global_window = meta['frames'][start : start + args.num_frames]
    print(f'scene={meta["name"]} start={start} num_frames={len(global_window)}')

    resolution = dataset.resolution
    num_cams = len(dataset.cameras)
    world_from = np.linalg.inv(dataset._ego_pose(meta, global_window[0]))

    model = build_model(cfg, device, args.ckpt)

    os.makedirs(args.out, exist_ok=True)
    frame_counter = 0
    chunk_idx = 0
    pos = 0
    while pos < len(global_window):
        chunk = global_window[pos : pos + dataset.clip_length]
        pos += len(chunk)
        chunk_idx += 1

        slot_of = {f: i for i, f in enumerate(chunk)}
        norm_times = dataset._normalised_times(chunk)
        context_frames = sorted(
            {chunk[min(i * dataset.frame_stride, len(chunk) - 1)] for i in range(dataset.num_context)}
        )

        ctx = dataset._collect(meta, context_frames, slot_of, norm_times, resolution, world_from, lidar=dataset.load_lidar)
        boxes = dataset._build_boxes(meta, chunk, world_from)

        scene = SceneInput(
            images=ctx['images'].to(device),
            intrinsics=ctx['intrinsics'].to(device),
            cam_to_world=ctx['cam_to_world'].to(device),
            times=ctx['times'].to(device),
            frame_index=ctx['frame_index'].to(device),
            boxes=boxes_to(boxes, device),
            sparse_depth=ctx['sparse_depth'].to(device) if 'sparse_depth' in ctx else None,
            sparse_mask=ctx['sparse_mask'].to(device) if 'sparse_mask' in ctx else None,
        )

        depth = model.predict_depth(scene)
        sampled = model.build_queries(scene, depth)
        gaussians = model.decode(scene, depth, sampled)

        dense = dataset._collect(meta, chunk, slot_of, norm_times, resolution, world_from, lidar=False)
        sel = torch.arange(args.camera, dense['images'].shape[0], num_cams)
        targets = TargetViews(
            intrinsics=dense['intrinsics'][sel].to(device),
            cam_to_world=dense['cam_to_world'][sel].to(device),
            times=dense['times'][sel].to(device),
            frame_index=dense['frame_index'][sel].to(device),
        )

        h, w = resolution
        out = model.render(gaussians, boxes_to(boxes, device), targets, height=h, width=w, render_depth=False)
        pred = out['rgb']

        num_instances = 0 if boxes is None else boxes.num_instances
        for t in range(pred.shape[0]):
            img = to_uint8(pred[t])
            img = label(img, f'frame={chunk[t]}  window={chunk_idx}  local_t={float(norm_times[slot_of[chunk[t]]]):.2f}  inst={num_instances}')
            Image.fromarray(img).save(os.path.join(args.out, f'frame_{frame_counter:03d}.png'))
            frame_counter += 1
        print(f'window {chunk_idx}: frames {chunk[0]}..{chunk[-1]}  gaussians={gaussians.count}  instances={num_instances}')

    mp4_path = os.path.join(args.out, 'continuous.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '10', '-i', os.path.join(args.out, 'frame_%03d.png'),
        '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', mp4_path, '-loglevel', 'error',
    ], check=True)
    print(f'saved {mp4_path}  ({frame_counter} frames)')


if __name__ == '__main__':
    main()

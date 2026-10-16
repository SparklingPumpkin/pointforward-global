"""Optional frozen-geometry refinement with an exact low-dimensional raster.

The base 128D raster is cached in CPU memory within this invocation. Geometry,
opacity, dynamic poses and ordering stay fixed. Group deltas commute with the
linear color raster, so only G channels need a differentiable raster. A second
phase trains the original UNet from freshly cached final-feature rasters.
No cache is accepted as an input. Original refinement entry points are intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from pointforward.models import render_features
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--payload-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scene', default='017')
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--feature-epochs', type=int, default=3)
    p.add_argument('--refiner-epochs', type=int, default=6)
    p.add_argument('--feature-lr', type=float, default=0.01)
    p.add_argument('--refiner-lr', type=float, default=0.0002)
    p.add_argument('--refiner-state-in', default=None)
    p.add_argument('--anchor-weight', type=float, default=0.0001)
    p.add_argument('--frame-stride', type=int, default=1)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--parity-check', action='store_true')
    p.add_argument('--autocast-refiner', action='store_true')
    a = p.parse_args()
    if a.groups < 1 or 128 % a.groups or a.frame_stride < 1:
        p.error('groups must divide 128; stride must be positive')
    torch.manual_seed(a.seed)
    torch.set_num_threads(8)
    start = time.perf_counter()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(a.payload_dir).resolve()
    config = yaml.safe_load(Path(a.config).read_text())
    if config['data']['layout'] == 'argoverse_drivestudio':
        config['data']['rigid_classes'] = ['REGULAR_VEHICLE', 'LARGE_VEHICLE',
            'TRUCK', 'BUS', 'VEHICULAR_TRAILER', 'MOTORCYCLE', 'BICYCLE',
            'BICYCLIST', 'MOTORCYCLIST']
    dataset = build_dataset(config, 'val')
    metadata = next(x for x in dataset.scenes if x['name'] == a.scene)
    all_frames = list(metadata['frames'])
    frames = all_frames[::a.frame_stride]
    if a.max_frames:
        frames = frames[:a.max_frames]
    slots = {int(f): i for i, f in enumerate(all_frames)}
    times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, all_frames[0]))
    h, w = dataset.resolution
    device = torch.device('cuda')
    static = load_payload(source / 'final_static_128d.pt', device)
    dynamic = load_payload(source / 'final_dynamic_128d.pt', device)
    scene = combine(static, dynamic)
    meta = torch.load(source / 'global_dynamic_meta.pt', map_location='cpu', weights_only=True)
    boxes = load_boxes(meta['boxes'], device)
    model = build_model(config, device, a.ckpt).eval().requires_grad_(False)
    index = torch.arange(128, device=device) // (128 // a.groups)
    delta = torch.nn.Parameter(torch.zeros(static.count, a.groups, device=device))
    records = []
    frame_orders = []
    phases = []
    parity = None

    def raster(posed, colors, k, pose):
        return render_features(posed['means'], posed['quats'], posed['scales'],
            posed['opacities'], colors, k, torch.linalg.inv(pose), h, w,
            render_depth=False)

    def data(frame):
        slot = slots[int(frame)]
        batch = dataset._collect(metadata, [int(frame)], slots, times,
                                 dataset.resolution, world_from, lidar=False)
        return (batch['images'], batch['intrinsics'], batch['cam_to_world'], slot)

    cache = []
    phase_start = time.perf_counter()
    with torch.no_grad():
        for i, frame in enumerate(frames):
            truth, k, pose, slot = data(frame)
            k, pose = k.to(device), pose.to(device)
            posed = model.pose_gaussians(scene, boxes, slot, times[slot].to(device))
            feats, alpha, _ = raster(posed, posed['colors'], k, pose)
            cache.append({'frame': int(frame), 'slot': slot, 'truth': truth,
                          'k': k, 'pose': pose, 'features': feats.cpu(), 'alpha': alpha.cpu()})
            if i % 10 == 0:
                print(json.dumps({'phase': 'cache', 'frame': int(frame),
                    'completed': i+1, 'total': len(frames), 'seconds': time.perf_counter()-phase_start}), flush=True)
            del feats, alpha, posed
    phases.append({'phase': 'cache', 'seconds': time.perf_counter()-phase_start})

    if a.parity_check and a.feature_epochs:
        # Same geometry and same joint static/dynamic sort; use one camera to
        # keep the reference 128D backward bounded. This verifies real outputs
        # and gradient to grouped parameters, not just an algebraic mock.
        item = cache[0]
        posed = model.pose_gaussians(scene, boxes, item['slot'], times[item['slot']].to(device))
        test = (torch.randn_like(delta) * 0.01).requires_grad_()
        grouped = torch.cat((test, torch.zeros(dynamic.count, a.groups, device=device)))
        small, _, _ = raster(posed, grouped, item['k'][:1], item['pose'][:1])
        fast = item['features'][:1].to(device) + small[:, index]
        reference_colors = torch.cat((static.colors + test[:, index], dynamic.colors))
        reference, _, _ = raster(posed, reference_colors, item['k'][:1], item['pose'][:1])
        fast_loss = fast.square().mean()
        ref_loss = reference.square().mean()
        fast_grad = torch.autograd.grad(fast_loss, test, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(ref_loss, test)[0]
        error = float((fast-reference).abs().max())
        grad_rel = float((fast_grad-ref_grad).norm()/ref_grad.norm().clamp_min(1e-12))
        parity = {'feature_max_abs': error, 'gradient_relative_l2': grad_rel,
                  'passed': error <= 2e-5 and grad_rel <= 1e-4}
        (out/'parity.json').write_text(json.dumps(parity, indent=2)+'\n')
        print(json.dumps(parity), flush=True)
        if not parity['passed']:
            raise RuntimeError('Exact low-dimensional raster parity failed')
        del posed, test, grouped, small, fast, reference_colors, reference, fast_grad, ref_grad, fast_loss, ref_loss

    generator = torch.Generator().manual_seed(a.seed)
    optimizer = torch.optim.Adam([delta], lr=a.feature_lr)
    zeros = torch.zeros(dynamic.count, a.groups, device=device)
    phase_start = time.perf_counter()
    for epoch in range(a.feature_epochs):
        losses = []
        order = torch.randperm(len(cache), generator=generator).tolist()
        frame_orders.append({'phase':'feature', 'epoch':epoch+1,
            'sha256':hashlib.sha256(json.dumps([cache[j]['frame'] for j in order]).encode()).hexdigest()})
        for j in order:
            item = cache[j]
            optimizer.zero_grad(set_to_none=True)
            posed = model.pose_gaussians(scene, boxes, item['slot'], times[item['slot']].to(device))
            small, _, _ = raster(posed, torch.cat((delta, zeros)), item['k'], item['pose'])
            feats = item['features'].to(device) + small[:, index]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.autocast_refiner):
                rgb = model.refiner(feats, item['alpha'].to(device))
            mse = (rgb.float()-item['truth'].to(device)).square().mean()
            loss = mse + a.anchor_weight * delta.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], 1.0)
            optimizer.step()
            losses.append(float(mse.detach()))
            if len(losses) % 20 == 0:
                print(json.dumps({'phase': 'feature', 'epoch': epoch+1,
                    'completed': len(losses), 'total': len(cache), 'mse': losses[-1]}), flush=True)
            del posed, small, feats, rgb, mse, loss
        record = {'phase': 'feature', 'epoch': epoch+1, 'mean_mse': float(np.mean(losses)),
                  'seconds': time.perf_counter()-phase_start}
        records.append(record)
        print(json.dumps(record), flush=True)
    phases.append({'phase': 'feature', 'seconds': time.perf_counter()-phase_start})
    # Fold updates into normal 128D payload; no special render path is needed.
    if a.feature_epochs:
        with torch.no_grad():
            for item in cache:
                posed = model.pose_gaussians(scene, boxes, item['slot'], times[item['slot']].to(device))
                small, _, _ = raster(posed, torch.cat((delta, zeros)), item['k'], item['pose'])
                item['features'] += small[:, index].cpu()
            refined_static = replace(static, colors=static.colors + delta[:, index])
            payload = out / 'runtime-payload-refined'
            payload.mkdir(exist_ok=True)
            save_gaussian_payload(payload/'final_static_128d.pt', refined_static)
            for name in ('final_dynamic_128d.pt', 'global_dynamic_meta.pt'):
                os.symlink((source/name).resolve(), payload/name)
            del refined_static, posed, small
    else:
        payload = source
    del scene, static, dynamic, delta, optimizer, zeros
    torch.cuda.empty_cache()
    phase_start = time.perf_counter()
    if a.refiner_epochs:
        if a.refiner_state_in:
            saved = torch.load(Path(a.refiner_state_in).resolve(), map_location='cpu', weights_only=True)
            model.refiner.load_state_dict(saved.get('refiner', saved), strict=True)
        model.refiner.requires_grad_(True)
        parameters = list(model.refiner.parameters())
        initial = [x.detach().clone() for x in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=a.refiner_lr, weight_decay=0.0)
        refiner_generator = torch.Generator().manual_seed(a.seed)
        for epoch in range(a.refiner_epochs):
            losses = []
            order = torch.randperm(len(cache), generator=refiner_generator).tolist()
            frame_orders.append({'phase':'refiner', 'epoch':epoch+1,
                'sha256':hashlib.sha256(json.dumps([cache[j]['frame'] for j in order]).encode()).hexdigest()})
            for j in order:
                item = cache[j]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.autocast_refiner):
                    rgb = model.refiner(item['features'].to(device), item['alpha'].to(device))
                mse = (rgb.float()-item['truth'].to(device)).square().mean()
                anchor = torch.stack([(x-y).square().mean() for x,y in zip(parameters,initial)]).mean()
                (torch.log(mse.clamp_min(1e-8)) + a.anchor_weight*anchor).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                losses.append(float(mse.detach()))
            record = {'phase':'refiner', 'epoch':epoch+1, 'mean_mse':float(np.mean(losses)),
                      'seconds':time.perf_counter()-phase_start}
            records.append(record)
            print(json.dumps(record), flush=True)
        torch.save({'refiner':{k:v.detach().cpu() for k,v in model.refiner.state_dict().items()},
                    'source_checkpoint':str(Path(a.ckpt).resolve())}, out/'refiner_state.pt')
    phases.append({'phase':'refiner', 'seconds':time.perf_counter()-phase_start})
    torch.cuda.synchronize()
    manifest = {'status':'completed', 'source_payload':str(source), 'refined_payload':str(payload),
        'checkpoint':str(Path(a.ckpt).resolve()), 'settings':vars(a), 'parity':parity,
        'wall_seconds':time.perf_counter()-start, 'phases':phases, 'records':records, 'frame_orders':frame_orders,
        'training_frames':frames, 'cache_dtype':'float32', 'cache_external_input':False,
        'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__ == '__main__':
    main()

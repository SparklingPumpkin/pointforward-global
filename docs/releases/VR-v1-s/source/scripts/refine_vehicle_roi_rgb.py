"""Vehicle-only RGB refinement after the unchanged joint350 baseline stage.

Static Gaussians, all network weights, instance IDs and box trajectories stay
fixed. Equal total weight per actor across its visible source-frame ROIs avoids
large/frequently observed actors dominating the vehicle supervision.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pointforward.models import TargetViews
from pointforward.models.rotation import quaternion_multiply
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.refine_joint_fast_rgb import _load_state, _q_delta
from scripts.render_final_global_runtime import combine, load_boxes, load_payload, projected_box_bounds
from scripts.train import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'ckpt', 'payload-dir', 'refiner-state', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--scene', default='017')
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--feature-lr', type=float, default=0.01)
    p.add_argument('--geometry-lr', type=float, default=0.005)
    p.add_argument('--opacity-lr', type=float, default=0.005)
    p.add_argument('--bbox-pad-pixels', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    device = torch.device('cuda')
    source, out = Path(args.payload_dir).resolve(), Path(args.out).resolve()
    refined = out/'runtime-payload-refined'; refined.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get('data', {}).get('layout') == 'argoverse_drivestudio':
        config['data']['rigid_classes'] = ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'TRUCK', 'BUS',
            'VEHICULAR_TRAILER', 'MOTORCYCLE', 'BICYCLE', 'BICYCLIST', 'MOTORCYCLIST']
    dataset = build_dataset(config, 'val')
    meta_scene = next(x for x in dataset.scenes if x['name'] == args.scene)
    frames = list(meta_scene['frames']); slots = {int(f): i for i, f in enumerate(frames)}
    times = dataset._normalised_times(frames)
    world_from = np.linalg.inv(dataset._ego_pose(meta_scene, frames[0]))
    height, width = dataset.resolution
    static = load_payload(source/'final_static_128d.pt', device)
    dynamic = load_payload(source/'final_dynamic_128d.pt', device)
    meta = torch.load(source/'global_dynamic_meta.pt', map_location='cpu', weights_only=True)
    boxes = load_boxes(meta['boxes'], device)
    evaluation_boxes = load_boxes(meta.get('evaluation_boxes', meta['boxes']), device)
    model = build_model(config, device, args.ckpt).eval()
    _load_state(model, args.refiner_state)
    model.requires_grad_(False)
    batch_cache, roi_cache = {}, {}
    counts = np.zeros(evaluation_boxes.num_instances, dtype=np.int64)
    for frame in frames:
        slot = slots[int(frame)]
        batch = dataset._collect(meta_scene, [int(frame)], slots, times,
                                 dataset.resolution, world_from, lidar=False)
        batch_cache[int(frame)] = batch
        index = torch.full((len(dataset.cameras),), slot, device=device, dtype=torch.long)
        bounds, visible = projected_box_bounds(evaluation_boxes, index,
            batch['intrinsics'].to(device), torch.linalg.inv(batch['cam_to_world'].to(device)),
            height, width)
        bounds, visible = bounds.cpu().numpy(), visible.cpu().numpy()
        rois = []
        for view, track in np.argwhere(visible):
            x0, y0, x1, y1 = map(int, bounds[view, track])
            if (x1-x0+1)*(y1-y0+1) < 4:
                continue
            pad = args.bbox_pad_pixels
            rois.append((int(view), int(track), max(0,x0-pad), max(0,y0-pad),
                         min(width,x1+pad+1), min(height,y1+pad+1)))
            counts[track] += 1
        roi_cache[int(frame)] = rois
    active_count = max(1, int((counts > 0).sum()))
    params = {name: torch.nn.Parameter(torch.zeros_like(getattr(dynamic, field)))
              for name, field in [('feature','colors'), ('xyz','means'), ('scale','scales'),
                                  ('rotation','means'), ('opacity','opacities'),
                                  ('time','times'), ('sigma','sigma')]}
    optimizer = torch.optim.AdamW([
        {'params':[params['feature']], 'lr':args.feature_lr},
        {'params':[params[k] for k in ('xyz','scale','rotation')], 'lr':args.geometry_lr},
        {'params':[params[k] for k in ('opacity','time','sigma')], 'lr':args.opacity_lr},
    ], weight_decay=0, fused=True)
    base_q = F.normalize(dynamic.quats, dim=-1)
    base_o = torch.logit(dynamic.opacities.clamp(1e-5,1-1e-5))
    def current():
        q = {k:torch.tanh(v) for k,v in params.items()}
        return replace(dynamic,
            means=dynamic.means+0.02*q['xyz'],
            scales=dynamic.scales*torch.exp(0.25*q['scale']),
            quats=F.normalize(quaternion_multiply(_q_delta(q['rotation']*math.radians(5)),base_q),dim=-1),
            opacities=torch.sigmoid(base_o+0.5*q['opacity']),
            colors=dynamic.colors+0.5*q['feature'],
            times=dynamic.times+0.05*q['time'], sigma=dynamic.sigma*torch.exp(0.25*q['sigma']))
    records, seen_frames = [], set()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.max_steps):
        epoch, oi = divmod(step, len(frames))
        order = torch.randperm(len(frames), generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        frame = int(frames[order[oi]]); slot = slots[frame]; seen_frames.add(frame)
        batch = batch_cache[frame]
        targets = TargetViews(intrinsics=batch['intrinsics'].to(device),
            cam_to_world=batch['cam_to_world'].to(device),
            times=times[slot:slot+1].to(device).expand(len(dataset.cameras)),
            frame_index=torch.full((len(dataset.cameras),),slot,dtype=torch.long,device=device))
        if step == 140:
            for group in optimizer.param_groups: group['lr'] *= 0.4
        optimizer.zero_grad(set_to_none=True)
        cd = current()
        rgb = model.render(combine(static,cd), boxes, targets, height=height, width=width,
                           render_depth=False, static_source_rgb_blend=0.75)['rgb']
        error = (rgb-batch['images'].to(device)).square()
        terms, roi_mses = [], []
        for view,track,x0,y0,x1,y1 in roi_cache[frame]:
            mse = error[view,:,y0:y1,x0:x1].mean()
            terms.append(torch.log(mse.clamp_min(1e-8))/max(1,int(counts[track])))
            roi_mses.append(mse.detach())
        roi_loss = torch.stack(terms).sum()*len(frames)/active_count if terms else error.mean()*0
        anchor = torch.stack([torch.tanh(v).square().mean() for v in params.values()]).mean()
        loss = roi_loss + 0.05*torch.log(error.mean().clamp_min(1e-8)) + 1e-4*anchor
        loss.backward()
        for v in params.values(): torch.nan_to_num_(v.grad,nan=0,posinf=0,neginf=0) if v.grad is not None else None
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.0)
        optimizer.step()
        row = {'step':step+1,'frame':frame,'loss':float(loss.detach()),
               'global_mse':float(error.detach().mean()),
               'vehicle_roi_mse':float(torch.stack(roi_mses).mean()) if roi_mses else None,
               'roi_count':len(roi_mses),'seconds':time.perf_counter()-started}
        records.append(row)
        with (out/'loss_steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if step == 0 or (step+1)%10 == 0: print(json.dumps(row),flush=True)
        del rgb,error,loss,cd,roi_loss,anchor,terms,roi_mses
    with torch.no_grad(): save_gaussian_payload(refined/'final_dynamic_128d.pt',current())
    for filename in ('final_static_128d.pt','global_dynamic_meta.pt'):
        os.symlink((source/filename).resolve(),refined/filename)
    manifest = {'status':'completed','steps':args.max_steps,'stop_reason':'max_steps',
        'source_payload':str(source),'refined_payload':str(refined),
        'refiner_state':str(Path(args.refiner_state).resolve()),
        'static_parameters_frozen':True,'all_network_parameters_frozen':True,
        'seen_source_frames':sorted(seen_frames),'per_track_visible_roi_counts':counts.tolist(),
        'loss':'inverse-visible-count per-actor log ROI MSE +0.05 global log MSE +1e-4 anchor',
        'bbox_pad_pixels':args.bbox_pad_pixels,'records':records,
        'wall_seconds':time.perf_counter()-started,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='records'}),flush=True)


if __name__ == '__main__': main()

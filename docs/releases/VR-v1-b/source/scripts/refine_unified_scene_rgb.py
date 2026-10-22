"""One uninterrupted optimizer for scene Gaussians, decoder and vehicle poses.

Milestones only serialize/evaluate the current state: no group is frozen,
reinitialized or restored between them. Original evaluation boxes stay fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pointforward.models import InstanceBoxes, TargetViews, box_dynamic_masks
from pointforward.models.rotation import quaternion_multiply
from scripts.deliver_continuous_scene import boxes_payload, save_gaussian_payload
from scripts.eval import build_model
from scripts.refine_joint_fast_rgb import _q_delta
from scripts.refine_dynamic_all_fullscene_rgb import rotation_z_batch
from scripts.refine_vehicle_trajectory_rgb import interpolate_corrections
from scripts.render_final_global_runtime import combine, load_boxes, load_payload, projected_box_bounds
from scripts.delivery_runtime_helpers import render_command, save


class GaussianDeltas:
    """Full 128D deltas and bounded geometry; same parameterization as scene350."""
    def __init__(self, base, bounds, lrs, label):
        self.base, self.bounds = base, bounds
        self.raw = {k: torch.nn.Parameter(torch.zeros_like(getattr(base, field)))
                    for k, field in [('feature', 'colors'), ('xyz', 'means'),
                        ('scale', 'scales'), ('rotation', 'means'),
                        ('opacity', 'opacities'), ('time', 'times'), ('sigma', 'sigma')]}
        self.q = F.normalize(base.quats, dim=-1)
        self.o = torch.logit(base.opacities.clamp(1e-5, 1-1e-5))
        self.groups = [dict(name=label+'.'+key, params=[value], lr=lrs[key])
                       for key, value in self.raw.items()]

    def current(self):
        q = {k: torch.tanh(v)*self.bounds[k] for k, v in self.raw.items()}
        b = self.base
        return replace(b, colors=b.colors+q['feature'], means=b.means+q['xyz'],
            scales=b.scales*q['scale'].exp(),
            quats=F.normalize(quaternion_multiply(_q_delta(q['rotation']), self.q), dim=-1),
            opacities=(self.o+q['opacity']).sigmoid(), times=b.times+q['time'],
            sigma=b.sigma*q['sigma'].exp())

    def anchor(self):
        return torch.stack([torch.tanh(v).square().mean() for v in self.raw.values()]).mean()

    @torch.no_grad()
    def diagnostics(self):
        return {k: dict(delta_rms=float((torch.tanh(v)*self.bounds[k]).square().mean().sqrt()),
                        delta_absmax=float(torch.tanh(v).abs().max()*self.bounds[k]),
                        bound=self.bounds[k]) for k, v in self.raw.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'ckpt', 'payload-dir', 'out', 'audit-run'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--scene', default='017')
    p.add_argument('--steps', type=int, default=950)
    p.add_argument('--milestones', type=int, nargs='+', default=[350, 550, 750, 950])
    p.add_argument('--roi-weight', type=float, default=.25)
    p.add_argument('--pose-lr', type=float, default=.001)
    p.add_argument('--lr-decay-steps', type=int, nargs='*', default=[240, 700])
    p.add_argument('--lr-decay-factor', type=float, default=.4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.steps < 1 or any(s < 1 or s > args.steps for s in args.milestones):
        p.error('Invalid steps or milestones')
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    run = Path(args.audit_run).resolve(); source = Path(args.payload_dir).resolve()
    save(out/'options.json', vars(args))
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.setdefault('model', {})['pretrained_backbone'] = False
    if cfg['data'].get('layout') == 'argoverse_drivestudio':
        cfg['data']['rigid_classes'] = ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'TRUCK', 'BUS',
            'VEHICULAR_TRAILER', 'MOTORCYCLE', 'BICYCLE', 'BICYCLIST', 'MOTORCYCLIST']
    from scripts.train import build_dataset
    dataset = build_dataset(cfg, 'val')
    scene = next(s for s in dataset.scenes if s['name'] == args.scene)
    frames = list(scene['frames']); slots = {int(f): i for i, f in enumerate(frames)}
    times = dataset._normalised_times(frames)
    world_from = np.linalg.inv(dataset._ego_pose(scene, frames[0]))
    h, w = dataset.resolution
    static = load_payload(source/'final_static_128d.pt', device)
    dynamic = load_payload(source/'final_dynamic_128d.pt', device)
    meta = torch.load(source/'global_dynamic_meta.pt', map_location='cpu', weights_only=True)
    boxes = load_boxes(meta['boxes'], device)
    evaluation_boxes = load_boxes(meta.get('evaluation_boxes', meta['boxes']), device)
    # Snapshot-compatible builder; depth/backbone are not scene-optimized.
    model = build_model(cfg, device, args.ckpt).eval().requires_grad_(False)
    model.refiner.requires_grad_(True)
    def make(base, dynamic_branch=False):
        bounds = dict(feature=1. if dynamic_branch else .5, xyz=.3 if dynamic_branch else .02,
            scale=.7 if dynamic_branch else .25, rotation=math.radians(20 if dynamic_branch else 5),
            opacity=2. if dynamic_branch else .5, time=.05, sigma=.25)
        lrs = dict(feature=.006 if dynamic_branch else .03, xyz=.004 if dynamic_branch else .02,
            scale=.004 if dynamic_branch else .02, rotation=.004 if dynamic_branch else .02,
            opacity=.004 if dynamic_branch else .01, time=.004 if dynamic_branch else .01,
            sigma=.004 if dynamic_branch else .01)
        return GaussianDeltas(base, bounds, lrs, 'dynamic' if dynamic_branch else 'static')
    sd, dd = make(static), make(dynamic, True)
    pose = torch.nn.Parameter(torch.zeros(boxes.num_instances, math.ceil((len(frames)-1)/4)+1, 4, device=device))
    movable = boxes.valid.clone()
    movable[torch.arange(boxes.num_instances, device=device), boxes.canonical_index()] = False
    adjacent = boxes.valid[:, 1:] & boxes.valid[:, :-1]
    def correction():
        return interpolate_corrections(torch.tanh(pose), len(frames))*movable[..., None]
    def current_boxes():
        c = correction()
        return InstanceBoxes(centers=boxes.centers+c[..., :3]*.5, dims=boxes.dims,
            rots=boxes.rots@rotation_z_batch(c[..., 3]*math.radians(15)).transpose(-1, -2),
            times=boxes.times, valid=boxes.valid)
    groups = sd.groups+dd.groups+[
        dict(name='refiner', params=list(model.refiner.parameters()), lr=.0002),
        dict(name='vehicle_pose', params=[pose], lr=args.pose_lr)]
    optimizer = torch.optim.AdamW(groups, weight_decay=0, fused=True)
    params = [v for g in groups for v in g['params']]
    batches, masks, rois = [], [], []
    counts = np.zeros(evaluation_boxes.num_instances, dtype=np.int64)
    for slot, frame in enumerate(frames):
        batch = dataset._collect(scene, [int(frame)], slots, times, dataset.resolution, world_from, lidar=False)
        batches.append(batch)
        index = torch.full((len(dataset.cameras),), slot, dtype=torch.long, device=device)
        k = batch['intrinsics'].to(device); w2c = torch.linalg.inv(batch['cam_to_world'].to(device))
        masks.append(box_dynamic_masks(evaluation_boxes, index, k, w2c, h, w).cpu())
        bounds, valid = projected_box_bounds(evaluation_boxes, index, k, w2c, h, w, strict_visibility=True)
        rr = []
        for camera, actor in np.argwhere(valid.cpu().numpy()):
            x0, y0, x1, y1 = map(int, bounds[camera, actor].tolist())
            if (x1-x0+1)*(y1-y0+1) < 4: continue
            rr.append((int(camera), int(actor), max(0,x0-4), max(0,y0-4), min(w,x1+5), min(h,y1+5)))
            counts[actor] += 1
        rois.append(rr)
    active = max(1, int((counts>0).sum()))
    save(out/'parameter_contract.json', dict(groups=[dict(name=g['name'], lr=g['lr'],
        elements=sum(v.numel() for v in g['params'])) for g in groups],
        static_bounds=sd.bounds, dynamic_bounds=dd.bounds, actor_visible_counts=counts.tolist(),
        active_actors=active, strict_roi_count=int(counts.sum()), frozen=['depth and image/fusion networks',
        'Gaussian identity/count and cached source RGB', 'original evaluation boxes and calibration',
        'canonical reference pose'], optimizer_restarts=0, global_loss='mean per-camera log RGB MSE',
        roi_loss='per-actor inverse-observation-count log ROI MSE; averaged over epoch', roi_weight=args.roi_weight))
    audit_seconds = 0.; checkpoints = []; first_gradients = None
    def audit(step):
        nonlocal audit_seconds
        at = time.perf_counter()
        cp = out/f'checkpoint-{step:04d}'; cp.mkdir()
        payload = cp/'runtime-payload-refined'; payload.mkdir()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            cs, cd, cb = sd.current(), dd.current(), current_boxes()
            save_gaussian_payload(payload/'final_static_128d.pt', cs)
            save_gaussian_payload(payload/'final_dynamic_128d.pt', cd)
            torch.save(dict(meta, boxes=boxes_payload(cb), evaluation_boxes=meta.get('evaluation_boxes', meta['boxes'])),
                       payload/'global_dynamic_meta.pt')
            torch.save({'refiner': {k: v.detach().cpu() for k, v in model.refiner.state_dict().items()}}, cp/'refiner_state.pt')
            stats = dict(static=sd.diagnostics(), dynamic=dd.diagnostics(),
                pose_translation_mean_m=float((cb.centers-boxes.centers).norm(dim=-1)[boxes.valid].mean()),
                pose_translation_max_m=float((cb.centers-boxes.centers).norm(dim=-1).max()),
                pose_yaw_absmax_deg=float(correction()[..., 3].abs().max()*15))
            # Reloaded box graph must give the identical posed Gaussian geometry.
            reloaded = load_boxes(torch.load(payload/'global_dynamic_meta.pt', map_location='cpu', weights_only=True)['boxes'], device)
            maximum = 0.
            for slot in (0, len(frames)//2, len(frames)-1):
                a = model.pose_gaussians(cd, cb, slot, times[slot].to(device))
                b = model.pose_gaussians(cd, reloaded, slot, times[slot].to(device))
                maximum = max(maximum, *[float((a[k]-b[k]).abs().max()) for k in ('means','quats','opacities')])
            if maximum > 1e-6: raise RuntimeError('Pose round-trip mismatch')
            stats['pose_export_maxabs'] = maximum
            save(cp/'parameter_changes.json', stats)
            del cs, cd, cb, a, b, reloaded
        gc.collect(); torch.cuda.empty_cache()
        render = cp/'runtime-render-source'
        with (cp/'render.log').open('w') as log:
            subprocess.run(render_command(run, payload, render, cp/'refiner_state.pt'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        m = json.loads((render/'manifest.json').read_text())
        audit_seconds += time.perf_counter()-at
        timer = json.loads((run/'pipeline_timing.json').read_text())
        elapsed = time.perf_counter()-timer['start']
        row = dict(step=step, global_psnr=m['mean_psnr_source_frames_all_cameras'],
            dynamic_psnr=m['mean_psnr_dynamic_bbox_source_frames_all_cameras'],
            non_sky_psnr=m['mean_psnr_non_sky_source_frames_all_cameras'],
            payload=str(payload), render=str(render), refiner_state=str(cp/'refiner_state.pt'),
            incremental_elapsed_seconds=elapsed, historical_cumulative_seconds=timer['upstream_prefix_seconds']+elapsed,
            optimization_and_setup_seconds=time.perf_counter()-started-audit_seconds,
            checkpoint_audit_seconds=time.perf_counter()-at, cumulative_audit_seconds=audit_seconds,
            optimizer_max_step=int(max(float(s['step']) for s in optimizer.state.values())),
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
        checkpoints.append(row); save(out/'checkpoints.json', checkpoints)
        print('CHECKPOINT', json.dumps(row), flush=True)
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        if step in args.lr_decay_steps:
            for g in optimizer.param_groups: g['lr'] *= args.lr_decay_factor
        epoch, oi = divmod(step, len(frames))
        order = torch.randperm(len(frames), generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        slot = order[oi]; batch = batches[slot]
        optimizer.zero_grad(set_to_none=True)
        targets = TargetViews(intrinsics=batch['intrinsics'].to(device), cam_to_world=batch['cam_to_world'].to(device),
            times=times[slot:slot+1].to(device).expand(len(dataset.cameras)),
            frame_index=torch.full((len(dataset.cameras),), slot, dtype=torch.long, device=device))
        cs, cd, cb = sd.current(), dd.current(), current_boxes()
        pred = model.render(combine(cs, cd), cb, targets, height=h, width=w, render_depth=False,
                            static_source_rgb_blend=.75)['rgb']
        error = (pred-batch['images'].to(device)).square()
        mse_views = error.flatten(1).mean(1)
        global_loss = mse_views.clamp_min(1e-8).log().mean()
        roi_loss = error.sum()*0
        roi_values = []
        value = None
        for camera, actor, x0, y0, x1, y1 in rois[slot]:
            value = error[camera, :, y0:y1, x0:x1].mean().clamp_min(1e-8)
            roi_values.append(float(value.detach()))
            roi_loss = roi_loss+value.log()*len(frames)/active/max(1,int(counts[actor]))
        c = correction()
        smooth = (c[:,1:]-c[:,:-1])[adjacent].square().mean() if bool(adjacent.any()) else c.sum()*0
        anchor = (sd.anchor()+dd.anchor()+torch.tanh(pose).square().mean())/3
        loss = global_loss+args.roi_weight*roi_loss+.02*smooth+1e-4*anchor
        loss.backward()
        gradient_stats = {}
        if step == 0 or (step+1) in args.milestones:
            for g in optimizer.param_groups:
                values = [v.grad.detach().square().sum() for v in g['params'] if v.grad is not None]
                gradient_stats[g['name']] = float(torch.stack(values).sum().sqrt()) if values else None
            if first_gradients is None: first_gradients = gradient_stats
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        mask = masks[slot].to(device)
        row = dict(step=step+1, epoch=epoch+1, frame=int(frames[slot]),
            global_mse=float(mse_views.detach().mean()), bbox_union_mse=float((error.detach()*mask[:,None]).sum()/(mask.sum().clamp_min(1)*3)),
            vehicle_roi_mse=float(np.mean(roi_values)) if roi_values else None,
            loss=float(loss.detach()), global_log_loss=float(global_loss.detach()),
            actor_balanced_log_loss=float(roi_loss.detach()), grad_norm=float(norm),
            seconds=time.perf_counter()-started, audit_seconds_included=audit_seconds,
            lrs={g['name']: g['lr'] for g in optimizer.param_groups})
        if gradient_stats: row['gradient_norms'] = gradient_stats
        with (out/'loss_steps.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
        if step == 0 or (step+1)%20 == 0: print(json.dumps(row), flush=True)
        del targets, cs, cd, cb, pred, error, mse_views, global_loss, roi_loss, loss, c, smooth, anchor, value
        if step+1 in args.milestones: audit(step+1)
    save(out/'manifest.json', dict(status='completed', steps=args.steps, optimizer_restarts=0,
        first_gradient_norms=first_gradients, wall_seconds=time.perf_counter()-started,
        audit_seconds=audit_seconds, checkpoints=checkpoints, options=vars(args)))


if __name__ == '__main__':
    main()

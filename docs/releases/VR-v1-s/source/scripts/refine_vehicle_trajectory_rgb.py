"""Optional vehicle-shape/trajectory refinement, with static map and decoder fixed.

Pose corrections are interpolated on a temporal knot grid. Canonical reference
poses stay fixed, so saved canonical points and exported trajectories agree.
Original evaluation boxes never receive pose corrections.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pointforward.models import InstanceBoxes, TargetViews, box_dynamic_masks
from pointforward.models.rotation import quaternion_multiply
from scripts.deliver_continuous_scene import boxes_payload, save_gaussian_payload
from scripts.eval import build_model
from scripts.refine_dynamic_all_fullscene_rgb import rotation_z_batch, rotvec_to_quaternion
from scripts.refine_joint_fast_rgb import _load_state
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.render_final_global_runtime import projected_box_bounds
from scripts.train import build_dataset


def interpolate_corrections(knots, frame_count):
    # [actor,knot,channel] -> [actor,frame,channel], without changing units.
    return F.interpolate(knots.transpose(1,2), size=frame_count, mode='linear',
                         align_corners=True).transpose(1,2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','ckpt','payload-dir','refiner-state','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--scene',default='017')
    p.add_argument('--strategy',choices=('shape','joint','staged'),required=True)
    p.add_argument('--steps',type=int,default=624)
    p.add_argument('--start-step',type=int,default=0,
                   help='Global step index used for continuation and deterministic frame ordering.')
    p.add_argument('--resume-deltas',default=None,
                   help='Optional optimized_deltas.pt whose raw Gaussian and pose state is loaded before training.')
    p.add_argument('--warmup-steps',type=int,default=156)
    p.add_argument('--knot-stride',type=int,default=4)
    p.add_argument('--feature-lr',type=float,default=.015)
    p.add_argument('--geometry-lr',type=float,default=.01)
    p.add_argument('--opacity-lr',type=float,default=.01)
    p.add_argument('--pose-lr',type=float,default=.025)
    p.add_argument('--lr-multiplier',type=float,default=1.0,
                   help='Multiplier applied to all optimizer learning rates; useful for continuation.')
    p.add_argument('--lr-decay-start',type=int,default=None,
                   help='Global step at which the existing 0.4 LR decay begins; defaults to 75%% of this run.')
    p.add_argument('--max-xyz',type=float,default=.30)
    p.add_argument('--max-translation',type=float,default=.50)
    p.add_argument('--max-yaw-deg',type=float,default=15.)
    p.add_argument('--max-feature',type=float,default=1.)
    p.add_argument('--max-log-scale',type=float,default=.7)
    p.add_argument('--max-opacity-logit',type=float,default=2.)
    p.add_argument('--max-rotation-deg',type=float,default=20.)
    p.add_argument('--pose-smoothness',type=float,default=.02)
    p.add_argument('--anchor-weight',type=float,default=1e-4)
    p.add_argument('--global-weight',type=float,default=.05)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--actor-balanced',action='store_true',
                   help='Optional inverse-observation-count per-track log ROI objective.')
    p.add_argument('--pose-lr-warmup',type=int,default=0)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--audit-run',default=None,
                   help='Optional cold-run parent; evaluate/save milestones without restarting the optimizer.')
    args=p.parse_args()
    started=time.perf_counter()
    torch.manual_seed(args.seed)
    device=torch.device('cuda')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    source=Path(args.payload_dir).resolve()
    (out/'options.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    config=yaml.safe_load(Path(args.config).read_text())
    config.setdefault('model',{})['pretrained_backbone']=False
    if config.get('data',{}).get('layout')=='argoverse_drivestudio':
        config['data']['rigid_classes']=['REGULAR_VEHICLE','LARGE_VEHICLE','TRUCK','BUS',
            'VEHICULAR_TRAILER','MOTORCYCLE','BICYCLE','BICYCLIST','MOTORCYCLIST']
    dataset=build_dataset(config,'val')
    scene=next(s for s in dataset.scenes if s['name']==args.scene)
    frames=list(scene['frames']);slots={int(f):i for i,f in enumerate(frames)}
    times=dataset._normalised_times(frames)
    world_from=np.linalg.inv(dataset._ego_pose(scene,frames[0]))
    h,w=dataset.resolution
    static=load_payload(source/'final_static_128d.pt',device)
    dynamic=load_payload(source/'final_dynamic_128d.pt',device)
    meta=torch.load(source/'global_dynamic_meta.pt',map_location='cpu',weights_only=True)
    boxes=load_boxes(meta['boxes'],device)
    evaluation_boxes=load_boxes(meta.get('evaluation_boxes',meta['boxes']),device)
    model=build_model(config,device,args.ckpt).eval().requires_grad_(False)
    _load_state(model,args.refiner_state)
    raw={k:torch.nn.Parameter(torch.zeros_like(getattr(dynamic,f))) for k,f in (
        ('xyz','means'),('feature','colors'),('scale','scales'),('rotation','means'),
        ('opacity','opacities'))}
    nk=math.ceil((len(frames)-1)/args.knot_stride)+1
    pose=torch.nn.Parameter(torch.zeros(boxes.num_instances,nk,4,device=device),
                            requires_grad=args.strategy!='shape')
    optimizer=torch.optim.AdamW([
        {'params':[raw['feature']],'lr':args.feature_lr},
        {'params':[raw[k] for k in ('xyz','scale','rotation')],'lr':args.geometry_lr},
        {'params':[raw['opacity']],'lr':args.opacity_lr},
        {'params':[pose],'lr':args.pose_lr}],weight_decay=0,fused=True)
    initial_lrs=[g['lr'] for g in optimizer.param_groups]
    if args.resume_deltas:
        resume=torch.load(args.resume_deltas,map_location='cpu',weights_only=True)
        with torch.no_grad():
            for k,v in raw.items():
                v.copy_(resume['raw'][k].to(device=device,dtype=v.dtype))
            pose.copy_(resume['pose'].to(device=device,dtype=pose.dtype))
        del resume
    base_q=F.normalize(dynamic.quats,dim=-1)
    base_o=torch.logit(dynamic.opacities.clamp(1e-5,1-1e-5))
    movable=boxes.valid.clone()
    movable[torch.arange(boxes.num_instances,device=device),boxes.canonical_index()]=False
    adjacent=boxes.valid[:,1:] & boxes.valid[:,:-1]
    bounds={'xyz':args.max_xyz,'feature':args.max_feature,'scale':args.max_log_scale,
            'rotation':math.radians(args.max_rotation_deg),'opacity':args.max_opacity_logit}

    def state():
        q={k:torch.tanh(v)*bounds[k] for k,v in raw.items()}
        cd=replace(dynamic,means=dynamic.means+q['xyz'],colors=dynamic.colors+q['feature'],
            scales=dynamic.scales*q['scale'].exp(),opacities=(base_o+q['opacity']).sigmoid(),
            quats=F.normalize(quaternion_multiply(rotvec_to_quaternion(q['rotation']),base_q),dim=-1))
        correction=interpolate_corrections(torch.tanh(pose),len(frames))*movable[...,None]
        translation=correction[...,:3]*args.max_translation
        yaw=correction[...,3]*math.radians(args.max_yaw_deg)
        # rots is world-to-box: rotating the object around world Z right-multiplies Rz^T.
        cb=InstanceBoxes(centers=boxes.centers+translation,dims=boxes.dims,
            rots=boxes.rots@rotation_z_batch(yaw).transpose(-1,-2),
            times=boxes.times,valid=boxes.valid)
        return cd,cb,correction

    batches=[];masks=[];roi_cache=[]
    actor_counts=np.zeros(evaluation_boxes.num_instances,dtype=np.int64)
    for slot,frame in enumerate(frames):
        batch=dataset._collect(scene,[int(frame)],slots,times,dataset.resolution,world_from,lidar=False)
        batches.append(batch)
        index=torch.full((len(dataset.cameras),),slot,dtype=torch.long,device=device)
        mask=box_dynamic_masks(evaluation_boxes,index,batch['intrinsics'].to(device),
            torch.linalg.inv(batch['cam_to_world'].to(device)),h,w)
        masks.append(mask.cpu())
        rois=[]
        if args.actor_balanced:
            projected_bounds,visible=projected_box_bounds(evaluation_boxes,index,batch['intrinsics'].to(device),
                torch.linalg.inv(batch['cam_to_world'].to(device)),h,w)
            projected_bounds,visible=projected_bounds.cpu().numpy(),visible.cpu().numpy()
            for camera,track in np.argwhere(visible):
                x0,y0,x1,y1=map(int,projected_bounds[camera,track])
                if (x1-x0+1)*(y1-y0+1)<4:continue
                rois.append((int(camera),int(track),max(0,x0-4),max(0,y0-4),min(w,x1+5),min(h,y1+5)))
                actor_counts[track]+=1
        roi_cache.append(rois)
    active_actors=max(1,int((actor_counts>0).sum()))
    torch.cuda.reset_peak_memory_stats()
    records=[];seen=set();checkpoints=[];stop_reason='max_steps'
    audit_seconds=0.;previous_dynamic=None;small_gains=0

    def audit_checkpoint(completed):
        nonlocal audit_seconds,previous_dynamic,small_gains
        from scripts.run_two_stage_vehicle_cold import render_command,save
        audit_started=time.perf_counter()
        cp=out/f'checkpoint-{completed:04d}';cp.mkdir()
        refined=cp/'runtime-payload-refined';refined.mkdir()
        with torch.no_grad():
            cd,cb,correction=state()
            os.symlink((source/'final_static_128d.pt').resolve(),refined/'final_static_128d.pt')
            save_gaussian_payload(refined/'final_dynamic_128d.pt',cd)
            torch.save(dict(meta,boxes=boxes_payload(cb),evaluation_boxes=meta.get('evaluation_boxes',meta['boxes'])),
                       refined/'global_dynamic_meta.pt')
            torch.save(dict(raw={k:v.detach().cpu() for k,v in raw.items()},pose=pose.detach().cpu()),cp/'optimized_deltas.pt')
        render=cp/'runtime-render-source'
        run=Path(args.audit_run)
        command=render_command(run,refined,render,args.refiner_state)
        with (cp/'render.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads((render/'manifest.json').read_text())
        scene350=json.loads((run/'audit-scene350/manifest.json').read_text())
        global_psnr=manifest['mean_psnr_source_frames_all_cameras']
        dynamic_psnr=manifest['mean_psnr_dynamic_bbox_source_frames_all_cameras']
        if previous_dynamic is None:previous_dynamic=scene350['mean_psnr_dynamic_bbox_source_frames_all_cameras']
        small_gains=small_gains+1 if dynamic_psnr-previous_dynamic<.05 else 0
        previous_dynamic=dynamic_psnr
        row=dict(step=completed,payload=str(refined),render=str(render),global_psnr=global_psnr,
                 dynamic_psnr=dynamic_psnr,non_sky_psnr=manifest['mean_psnr_non_sky_source_frames_all_cameras'],
                 scene350_global_psnr=scene350['mean_psnr_source_frames_all_cameras'],
                 checkpoint_audit_seconds=time.perf_counter()-audit_started,
                 elapsed_from_cold_start_seconds=time.perf_counter()-json.loads((run/'pipeline_timing.json').read_text())['start'])
        audit_seconds+=row['checkpoint_audit_seconds']
        row['vehicle_stage_non_audit_seconds']=time.perf_counter()-started-audit_seconds
        checkpoints.append(row);save(out/'checkpoints.json',checkpoints)
        print('CHECKPOINT',json.dumps(row),flush=True)
        return small_gains>=2 or global_psnr<scene350['mean_psnr_source_frames_all_cameras']-.05
    lr_decay_start = (args.lr_decay_start if args.lr_decay_start is not None
                      else int(args.steps * .75))
    for local_step in range(args.steps):
        step=args.start_step+local_step
        epoch,oi=divmod(step,len(frames))
        order=torch.randperm(len(frames),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        slot=order[oi];frame=int(frames[slot]);seen.add(frame);batch=batches[slot]
        warm=args.strategy=='staged' and step<args.warmup_steps
        for k,v in raw.items():v.requires_grad_(not warm)
        for group,lr in zip(optimizer.param_groups,initial_lrs):
            group['lr']=lr*args.lr_multiplier*(.4 if step>=lr_decay_start else 1.)
        if args.pose_lr_warmup:
            optimizer.param_groups[-1]['lr']*=.25+.75*min(1.,(local_step+1)/args.pose_lr_warmup)
        optimizer.zero_grad(set_to_none=True)
        # Accumulate the derivative of exactly the all-camera pixel-weighted MSE,
        # while retaining only one camera rasterization graph at a time.
        mask_all=masks[slot].to(device)
        bbox_den=max(1,int(mask_all.sum())*3)
        global_den=len(dataset.cameras)*3*h*w
        sums=[];fulls=[];roi_mses={}
        with torch.no_grad():
            cd,cb,_=state();combined=combine(static,cd)
            for camera in range(len(dataset.cameras)):
                target=TargetViews(intrinsics=batch['intrinsics'][camera:camera+1].to(device),
                    cam_to_world=batch['cam_to_world'][camera:camera+1].to(device),
                    times=times[slot:slot+1].to(device),frame_index=torch.tensor([slot],device=device))
                pred=model.render(combined,cb,target,height=h,width=w,render_depth=False,
                                  static_source_rgb_blend=.75)['rgb']
                error=(pred-batch['images'][camera:camera+1].to(device)).square()
                sums.append(float((error*mask_all[camera][None,None]).sum()))
                fulls.append(float(error.sum()))
                if args.actor_balanced:
                    for ri,(view,track,x0,y0,x1,y1) in enumerate(roi_cache[slot]):
                        if view==camera:roi_mses[ri]=max(float(error[0,:,y0:y1,x0:x1].mean()),1e-8)
            bbox_mse=max(sum(sums)/bbox_den,1e-8);global_mse=max(sum(fulls)/global_den,1e-8)
            del cd,cb,combined,pred,error
        for camera in range(len(dataset.cameras)):
            cd,cb,_=state()
            target=TargetViews(intrinsics=batch['intrinsics'][camera:camera+1].to(device),
                cam_to_world=batch['cam_to_world'][camera:camera+1].to(device),
                times=times[slot:slot+1].to(device),frame_index=torch.tensor([slot],device=device))
            pred=model.render(combine(static,cd),cb,target,height=h,width=w,render_depth=False,
                              static_source_rgb_blend=.75)['rgb']
            error=(pred-batch['images'][camera:camera+1].to(device)).square()
            loss=(error*mask_all[camera][None,None]).sum()/(bbox_den*bbox_mse)
            if args.actor_balanced:
                loss=error.sum()*0
                for ri,(view,track,x0,y0,x1,y1) in enumerate(roi_cache[slot]):
                    if view==camera:
                        loss=loss+error[0,:,y0:y1,x0:x1].mean()/roi_mses[ri]/max(1,int(actor_counts[track]))*len(frames)/active_actors
            loss=loss+args.global_weight*error.sum()/(global_den*global_mse)
            loss.backward()
            del cd,cb,pred,error,loss
        _,_,correction=state()
        # Smooth the correction, rather than suppressing the true vehicle motion.
        smooth=(correction[:,1:]-correction[:,:-1])[adjacent].square().mean()
        anchor=sum(torch.tanh(v).square().mean() for v in raw.values())/len(raw)
        regularizer=args.pose_smoothness*smooth+args.anchor_weight*(anchor+torch.tanh(pose).square().mean())
        if regularizer.requires_grad:regularizer.backward()
        parameters=[v for v in (*raw.values(),pose) if v.grad is not None]
        if not all(bool(torch.isfinite(v.grad).all()) for v in parameters):
            raise RuntimeError('Nonfinite gradient; refusing to conceal an invalid pose experiment')
        norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
        optimizer.step()
        row=dict(step=step+1,epoch=epoch+1,frame=frame,phase='pose_warmup' if warm else args.strategy,
            bbox_mse=bbox_mse,global_mse=global_mse,
            loss=math.log(bbox_mse)+args.global_weight*math.log(global_mse),
            pose_smoothness=float(smooth.detach()),grad_norm=float(norm),seconds=time.perf_counter()-started)
        if args.actor_balanced:
            row['actor_balanced_log_loss']=sum(math.log(roi_mses[ri])/max(1,int(actor_counts[r[1]]))
                for ri,r in enumerate(roi_cache[slot]))*len(frames)/active_actors+args.global_weight*math.log(global_mse)
            row['vehicle_roi_mse']=sum(roi_mses.values())/len(roi_mses) if roi_mses else None
        row['audit_seconds_included']=audit_seconds
        records.append(row)
        with (out/'loss_steps.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if step==0 or (step+1)%20==0:print(json.dumps(row),flush=True)
        del correction,smooth,anchor,regularizer
        if args.checkpoint_every and (local_step+1)%args.checkpoint_every==0:
            if audit_checkpoint(local_step+1):
                stop_reason='checkpoint_plateau_or_global_drop';break
    with torch.no_grad():
        cd,cb,correction=state()
        refined=out/'runtime-payload-refined';refined.mkdir()
        os.symlink((source/'final_static_128d.pt').resolve(),refined/'final_static_128d.pt')
        save_gaussian_payload(refined/'final_dynamic_128d.pt',cd)
        exported=dict(meta,boxes=boxes_payload(cb),evaluation_boxes=meta.get('evaluation_boxes',meta['boxes']))
        torch.save(exported,refined/'global_dynamic_meta.pt')
        saved_boxes=load_boxes(exported['boxes'],device)
        maximum=0.
        for slot in (0,len(frames)//2,len(frames)-1):
            aa=model.pose_gaussians(cd,cb,slot,times[slot].to(device))
            bb=model.pose_gaussians(cd,saved_boxes,slot,times[slot].to(device))
            maximum=max(maximum,*[float((aa[k]-bb[k]).abs().max()) for k in ('means','quats','opacities')])
        if maximum>1e-6:raise RuntimeError(f'Pose export mismatch {maximum}')
        shift=(cb.centers-boxes.centers).norm(dim=-1)[boxes.valid]
        diagnostics=dict(pose_translation_mean_m=float(shift.mean()),pose_translation_max_m=float(shift.max()),
            pose_yaw_max_deg=float((correction[...,3]*args.max_yaw_deg).abs().max()),
            gaussian_xyz_rms_m=float((cd.means-dynamic.means).square().mean().sqrt()),
            gaussian_xyz_max_m=float((cd.means-dynamic.means).norm(dim=-1).max()),
            scale_multiplier_mean=float((cd.scales/dynamic.scales).mean()),export_pose_maxabs=maximum)
        torch.save(dict(raw={k:v.detach().cpu() for k,v in raw.items()},pose=pose.detach().cpu(),
                        pose_corrections_normalized=correction.cpu()),out/'optimized_deltas.pt')
    manifest=dict(status='completed',steps=len(records),stop_reason=stop_reason,strategy=args.strategy,seen_source_frames=sorted(seen),
        source_payload=str(source),refined_payload=str(refined),refiner_state=args.refiner_state,
        frozen=['static Gaussians','all networks including U-Net','canonical reference poses',
                'original evaluation boxes','camera calibration','Gaussian times/sigma'],
        loss='log original-bbox-union all-camera MSE +0.05 log global MSE + trajectory correction smoothness + delta anchor',
        diagnostics=diagnostics,wall_seconds=time.perf_counter()-started,
        audit_seconds=audit_seconds,checkpoints=checkpoints,actor_visible_roi_counts=actor_counts.tolist(),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,options=vars(args),records=records)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='records'}),flush=True)


if __name__=='__main__':main()

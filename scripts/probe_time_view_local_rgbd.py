#!/usr/bin/env python3
"""Round77 source/target-separated RGB-D local Gaussian probe.

This is deliberately a small causal probe.  Eight source RGBs build temporary
world-space local Gaussian banks; only the other two cameras at the same frame
are scored.  No evaluation/fine masks are read.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.models import Gaussians, render_features
from scripts.eval import build_model
from scripts.render_final_global_runtime import load_payload, load_boxes
from scripts.train import build_dataset
from scripts.audit_dynamic_depth_box_contract import prepare_dataset, ray_box_interval
from scripts.build_rgb_surface_proxy import world_points, aggregate_one_support, box_mask, box_may_hit_image

H, W = 576, 768
SOURCE = [(0,0),(34,1),(60,2),(101,1),(114,1),(132,0),(141,2),(146,0)]

def psnr(a,b,mask=None):
    e=(a.float()-b.float()).square()
    if mask is not None:
        m=mask.expand_as(e)
        if not m.any(): return None
        e=e[m]
    return float((-10*torch.log10(e.mean().clamp_min(1e-10))).cpu())

def payload(means, colors, scales, instance):
    n=len(means); q=np.zeros((n,4),np.float32); q[:,0]=1
    return Gaussians(
      means=torch.from_numpy(means.astype(np.float32)), quats=torch.from_numpy(q),
      scales=torch.from_numpy(np.repeat(scales[:,None],3,1).astype(np.float32)),
      opacities=torch.full((n,),.85), colors=torch.from_numpy(colors.astype(np.float32)),
      sigma=torch.full((n,),1e6), times=torch.zeros(n),
      instance_id=torch.from_numpy(instance.astype(np.int64)), extras={}
    )

def parse():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--ckpt',required=True)
    p.add_argument('--payload-dir',required=True); p.add_argument('--camera-refiners',nargs=3,required=True)
    p.add_argument('--sam-manifest',required=True); p.add_argument('--out',required=True); p.add_argument('--scene',default='017'); p.add_argument('--device',default='cuda')
    return p.parse_args()

@torch.inference_mode()
def main():
    a=parse(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); t0=time.perf_counter(); device=torch.device(a.device)
    cfg,ydata,metadata,world_from=prepare_dataset(Path(a.config),a.scene)
    frames=[int(x) for x in metadata['frames']]; fslot={f:i for i,f in enumerate(frames)}
    base_static=load_payload(Path(a.payload_dir)/'final_static_128d.pt',device); base_dyn=load_payload(Path(a.payload_dir)/'final_dynamic_128d.pt',device)
    full=__import__('scripts.render_final_global_runtime',fromlist=['combine']).combine(base_static,base_dyn)
    meta=torch.load(Path(a.payload_dir)/'global_dynamic_meta.pt',map_location='cpu',weights_only=True); boxes=load_boxes(meta['boxes'],device)
    track_ids=[int(x) for x in meta['track_ids']]; track_slot={x:i for i,x in enumerate(track_ids)}
    sm=json.loads(Path(a.sam_manifest).read_text()); records={}
    for x in sm['outputs']['mask_files']:
        k=(int(x['frame']),int(x['camera']))
        if k in SOURCE and x['single_view_reliable'] and not x['track_fallback_to_projected_bbox']:
            records.setdefault(k,[]).append(x)
    req=set(SOURCE)|{(f,c) for f,c in SOURCE for c in range(3)}
    # Collect all cameras at source frames, then run dense depth only once for these images.
    slot={f:i for i,f in enumerate(sorted({f for f,_ in req}))}; times=ydata._normalised_times(sorted(slot))
    batch=ydata._collect(metadata,sorted(slot),slot,times,(H,W),world_from,lidar=True)
    cams=list(ydata.cameras); cslot={int(c):i for i,c in enumerate(cams)}
    from pointforward.models import SceneInput
    model=build_model(cfg,device,str(Path(a.ckpt).resolve())).eval(); model_load=time.perf_counter()-t0
    keys=sorted(req); imgs=[]; intr=[]; poses=[]; sd=[]; smask=[]
    for f,c in keys:
        i=slot[f]*len(cams)+cslot[c]; imgs.append(batch['images'][i].numpy()); intr.append(batch['intrinsics'][i].numpy()); poses.append(batch['cam_to_world'][i].numpy()); sd.append(batch['sparse_depth'][i,0].numpy()); smask.append((batch['sparse_mask'][i,0].numpy()>0.5))
    pred={}; depth_t=time.perf_counter()
    for st in range(0,len(keys),8):
        kk=keys[st:st+8]; ix=[keys.index(k) for k in kk]; n=len(ix)
        scene=SceneInput(images=torch.from_numpy(np.stack([imgs[i] for i in ix])).to(device),intrinsics=torch.from_numpy(np.stack([intr[i] for i in ix])).float().to(device),cam_to_world=torch.from_numpy(np.stack([poses[i] for i in ix])).float().to(device),times=torch.zeros(n,device=device),frame_index=torch.arange(n,device=device),sparse_depth=torch.from_numpy(np.stack([sd[i] for i in ix]))[:,None].to(device),sparse_mask=torch.from_numpy(np.stack([smask[i] for i in ix]))[:,None].float().to(device))
        d=model.predict_depth(scene).float().cpu().numpy()
        for k,z in zip(kk,d): pred[k]=z
    # Build world-space source banks.
    dyn_by={}; bg_by={}; depth_pass=0; cand=0
    for key,items in records.items():
        f,c=key; ii=keys.index(key); datum={'image':imgs[ii].transpose(1,2,0),'intrinsic':intr[ii],'camera_to_world':poses[ii]}; depth=pred[key]
        dyn_chunks=[]; dyn_tid=[]; bg_chunks=[]; image_region=np.zeros((H,W),bool); foreground=np.zeros((H,W),bool); obbs=[]
        slotf=fslot[f]
        for item in items:
            tid=int(item['track_id']); gs=track_slot.get(tid)
            if gs is None or not bool(meta['boxes']['valid'][gs,slotf]): continue
            center=meta['boxes']['centers'][gs,slotf].numpy(); dims=meta['boxes']['dims'][gs,slotf].numpy(); rot=meta['boxes']['rots'][gs,slotf].numpy()
            from scripts.audit_dynamic_depth_box_contract import Box
            box=Box(center=center,dims=dims,rotation=rot); hit,near,far=ray_box_interval(box,datum['intrinsic'],datum['camera_to_world'],return_depth=True)
            mask=np.asarray(Image.open(item['output_path']).convert('L'))>0; image_region|=box_mask(item['bbox_xyxy']); foreground|=mask
            valid=mask&hit&np.isfinite(depth)&(depth>0)&(depth>=near-.1)&(depth<=far+.1); cand+=int(mask.sum()); depth_pass+=int(valid.sum())
            if valid.any():
                ys,xs=np.nonzero(valid); z=np.clip(depth[valid],near[valid],far[valid]); pts=world_points(ys,xs,z,datum['intrinsic'],datum['camera_to_world']); rgb=datum['image'][ys,xs].astype(np.float32); focal=math.sqrt(float(datum['intrinsic'][0,0]*datum['intrinsic'][1,1])); sc=np.clip(1.5*z/focal,.005,.04).astype(np.float32); ch=aggregate_one_support(pts,rgb,sc); dyn_chunks.append(ch); dyn_tid.append(np.full(len(ch[0]),gs,np.int32))
            obbs.append((hit,far))
        if dyn_chunks:
            kkeys=np.concatenate([x[0] for x in dyn_chunks]); mm=np.concatenate([x[1] for x in dyn_chunks]); cc=np.concatenate([x[2] for x in dyn_chunks]); ss=np.concatenate([x[3] for x in dyn_chunks]); tt=np.concatenate(dyn_tid); uniq,first,inv=np.unique(np.column_stack((tt,kkeys)),axis=0,return_index=True,return_inverse=True); cnt=np.bincount(inv); ms=np.column_stack([np.bincount(inv,weights=mm[:,j]) for j in range(3)]); cs=np.column_stack([np.bincount(inv,weights=cc[:,j]) for j in range(3)]); sc=np.bincount(inv,weights=ss); keep=cnt>=1; dyn_by[key]=payload(ms[keep]/cnt[keep,None],cs[keep]/cnt[keep,None],sc[keep]/cnt[keep],uniq[keep,0])
        allowed=image_region & ~foreground
        for hit,far in obbs: allowed &= (~hit)|(depth>far+.05)
        allowed &= np.isfinite(depth)&(depth>0)
        if allowed.any():
            ys,xs=np.nonzero(allowed); z=depth[allowed]; pts=world_points(ys,xs,z,datum['intrinsic'],datum['camera_to_world']); rgb=datum['image'][ys,xs].astype(np.float32); focal=math.sqrt(float(datum['intrinsic'][0,0]*datum['intrinsic'][1,1])); sc=np.clip(1.5*z/focal,.005,.04).astype(np.float32); ch=aggregate_one_support(pts,rgb,sc); bg_by[key]=payload(ch[1]/np.maximum(ch[4][:,None],1),ch[2]/np.maximum(ch[4][:,None],1),ch[3]/np.maximum(ch[4],1),np.full(len(ch[0]),-1,np.int64))
    refiners=[]
    for p in a.camera_refiners:
        r=__import__('copy').deepcopy(model.refiner).to(device).eval(); st=torch.load(p,map_location='cpu',weights_only=True); r.load_state_dict(st.get('refiner',st),strict=True); refiners.append(r)
    # Evaluate five arms on other cameras at each source frame.
    stats={arm:{'sse':0.,'n':0,'dynamic_sse':0.,'dynamic_n':0,'frames':0} for arm in ('baseline','suppress','foreground','background','joint')}; per=[]
    for skey in SOURCE:
        f,src=skey; dynp=dyn_by.get(skey); bgp=bg_by.get(skey); affected=set(dynp.instance_id.tolist()) if dynp is not None else set(); slotf=fslot[f]
        for cam in range(3):
            if cam==src: continue
            i=slot[f]*len(cams)+cslot[cam]; target=batch['images'][i].to(device).unsqueeze(0); intr_t=batch['intrinsics'][i].to(device).unsqueeze(0); pose_t=batch['cam_to_world'][i].to(device).unsqueeze(0); tv=torch.tensor([float(times[slot[f]])],device=device); posed=model.pose_gaussians(full,boxes,slotf,tv[0]);
            dm=full.instance_id>=0; suppress_mask=dm & torch.tensor([int(x) in affected for x in full.instance_id.tolist()],device=device)
            op=full.opacities.clone(); op[suppress_mask]=0; from dataclasses import replace
            sup=dict(posed); sup['opacities']=op
            base_feat,base_alpha,_=render_features(sup['means'],sup['quats'],sup['scales'],sup['opacities'],sup['colors'],intr_t,torch.linalg.inv(pose_t),H,W); B=refiners[cam](base_feat,base_alpha)
            owncol=torch.zeros((sup['means'].shape[0],1),device=device); owncol[~suppress_mask]= (full.instance_id[~suppress_mask]>=0).float().unsqueeze(1); _,Db,_=render_features(sup['means'],sup['quats'],sup['scales'],sup['opacities'],owncol,intr_t,torch.linalg.inv(pose_t),H,W)
            proxy_list=[]
            if dynp is not None: proxy_list.append(dynp)
            if bgp is not None: proxy_list.append(bgp)
            def render_arm(kind):
                plist=[]
                if kind in ('foreground','joint') and dynp is not None: plist.append(dynp)
                if kind in ('background','joint') and bgp is not None: plist.append(bgp)
                if not plist: return B,Db
                pm=torch.cat([x.means for x in plist]).to(device); pq=torch.cat([x.quats for x in plist]).to(device); ps=torch.cat([x.scales for x in plist]).to(device); po=torch.cat([x.opacities for x in plist]).to(device); pc=torch.cat([x.colors for x in plist]).to(device); pi=torch.cat([x.instance_id for x in plist]).to(device); pa,_,_=render_features(pm,pq,ps,po,pc,intr_t,torch.linalg.inv(pose_t),H,W); _,Ap,_=render_features(pm,pq,ps,po,torch.ones((len(pm),1),device=device),intr_t,torch.linalg.inv(pose_t),H,W); # joint visibility indicator
                jmeans=torch.cat([sup['means'],pm]); jq=torch.cat([sup['quats'],pq]); js=torch.cat([sup['scales'],ps]); jo=torch.cat([sup['opacities'],po]); jind=torch.cat([owncol, (pi>=0).float().view(-1,1)]); _,V,_=render_features(jmeans,jq,js,jo,jind,intr_t,torch.linalg.inv(pose_t),H,W); M=(Ap>=.25)&(V/(Ap.clamp_min(1e-8))>=.95); Y=pa+(1-Ap)*B; return torch.where(M,Y,B), torch.where(M,V,Db)
            for arm in ('baseline','suppress','foreground','background','joint'):
                if arm=='baseline': R,own=(refiners[cam](base_feat*0+0,base_alpha),Db) # replaced below
                if arm=='baseline':
                    pf,pa0,_=render_features(posed['means'],posed['quats'],posed['scales'],posed['opacities'],posed['colors'],intr_t,torch.linalg.inv(pose_t),H,W); R=refiners[cam](pf,pa0); owncol0=(full.instance_id>=0).float().view(-1,1); _,own,_=render_features(posed['means'],posed['quats'],posed['scales'],posed['opacities'],owncol0,intr_t,torch.linalg.inv(pose_t),H,W)
                elif arm=='suppress': R,own=B,Db
                else: R,own=render_arm(arm)
                mask=(own>=.05); stats[arm]['sse']+=float((R-target).square().sum()); stats[arm]['n']+=int(R.numel()); stats[arm]['dynamic_sse']+=float((R-target).square().masked_fill(~mask.expand_as(R),0).sum()); stats[arm]['dynamic_n']+=int(mask.sum())*3
            per.append({'source':skey,'target_camera':cam,'affected_tracks':len(affected)})
    report={'status':'completed','source':SOURCE,'source_count':len(SOURCE),'dynamic_input_pixels':cand,'depth_valid_pixels':depth_pass,'timing_seconds':{'total':time.perf_counter()-t0,'model_load':model_load},'banks':{'source_keys':list(dyn_by),'foreground_points':sum(x.means.numel()//3 for x in dyn_by.values()),'background_points':sum(x.means.numel()//3 for x in bg_by.values())},'metrics':{k:{**v,'global_psnr':(-10*math.log10(max(v['sse']/max(v['n'],1),1e-10))),'dynamic_support_psnr':(-10*math.log10(max(v['dynamic_sse']/max(v['dynamic_n'],1),1e-10)))} for k,v in stats.items()},'records':per}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

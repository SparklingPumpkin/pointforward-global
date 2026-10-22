"""RGB-mask shared effective-time calibration and geometry-only depth prompts.

No camera/background pose changes, SH transfer, per-track residual, or synthetic
depth. One shared dt gives one consistent world-space actor trajectory in all
cameras: E(f) inv(E(f+dt)) A(f+dt). Input membership uses original unpadded boxes.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import torch
import yaml
from tqdm import tqdm
from scripts.train import build_dataset
from scripts.build_track_once_dynamic import calibrated_intrinsics, track_view_candidates, SIGNS


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def transform(points, pose):
    return points @ pose[:3,:3].T + pose[:3,3]


def interp(poses, t):
    keys = sorted(poses)
    t = float(np.clip(t, keys[0], keys[-1]))
    hi = min(np.searchsorted(keys, t, side='right'), len(keys)-1)
    lo = max(0, hi-1)
    if lo == hi or t == keys[lo]:
        return poses[keys[lo]].copy()
    a,b = keys[lo],keys[hi]
    out = np.eye(4)
    out[:3,3] = poses[a][:3,3] + (poses[b][:3,3]-poses[a][:3,3])*(t-a)/(b-a)
    out[:3,:3] = Slerp([a,b],Rotation.from_matrix(np.stack([poses[a][:3,:3],poses[b][:3,:3]])))([t]).as_matrix()[0]
    return out


def pixels(cp, k, h, w):
    good = np.isfinite(cp).all(1) & (cp[:,2] > 1e-3)
    xy = np.zeros((len(cp),2),np.int64)
    xy[good] = np.rint((cp[good] @ k.T)[:,:2]/cp[good,2:]).astype(np.int64)
    good &= (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    return xy,good


def sparse(cp, k, h, w):
    xy, good = pixels(cp,k,h,w)
    flat = np.full(h*w,np.inf)
    np.minimum.at(flat,xy[good,1]*w+xy[good,0],cp[good,2])
    flat[~np.isfinite(flat)] = 0
    return flat.reshape(h,w).astype(np.float32)


class Scene:
    def __init__(self, args):
        self.out = Path(args.out)
        cfg = yaml.safe_load(Path(args.config).read_text())
        cfg['data'].pop('geometry_directory',None)
        cfg['pedestrians']['enabled'] = False
        self.ds = build_dataset(cfg,'val')
        self.meta = next(m for m in self.ds.scenes if m['name']==args.scene)
        self.ego = {f:self.ds._ego_pose(self.meta,f) for f in self.meta['frames']}
        self.actors = {tid:{f:p for f,(p,s) in track.items()} for tid,track in self.meta['instances'].items()}
        self.k = calibrated_intrinsics(self.ds,self.meta,self.meta['frames'][0])
        self.h,self.w = self.ds.resolution
        self.local_cache = {}

    def camera(self,f,c):
        return self.ego[f] @ self.meta['extrinsics'][c]

    def corrected_actor(self,tid,f,dt):
        return self.ego[f] @ np.linalg.inv(interp(self.ego,f+10*dt)) @ interp(self.actors[tid],f+10*dt)

    def world(self,f):
        return transform(self.ds.read_lidar(str(Path(self.meta['path'])/'lidar'/f'{f:03d}.bin')),self.ego[f])

    def local(self,o):
        key=(o['track_id'],o['frame'])
        if key not in self.local_cache:
            old,size=self.meta['instances'][key[0]][key[1]]
            local=transform(self.world(key[1]),np.linalg.inv(old))
            self.local_cache[key]=local[(np.abs(local)<=size*.5).all(1)]
        return self.local_cache[key]

    def prepare(self):
        (self.out/'inputs').mkdir(exist_ok=True)
        (self.out/'sparse').mkdir(exist_ok=True)
        world_from=np.linalg.inv(self.ego[self.meta['frames'][0]])
        observations=[]
        for tid in self.meta['instances']:
            for o in track_view_candidates(self.ds,self.meta,tid,set(self.meta['frames']),world_from,self.k):
                if o['bbox_area_fraction'] < .0005:
                    continue
                f,c=o['frame'],o['camera']
                o=dict(o,track_id=int(tid),key=f'{f:03d}_{c}_track{tid:03d}')
                o['oblique_quality']=float(abs(np.sin(2*o['view_angle_rad'])))
                observations.append(o)
        # Large cars, useful oblique view, spread over tracks and time. Keep
        # alternating per-track observations out of the calibration objective.
        ranked=sorted(observations,key=lambda o:o['bbox_area_fraction']*(.25+.75*o['oblique_quality']),reverse=True)
        selected=[]; pertrack={}
        for o in ranked:
            tid=o['track_id']; chosen=pertrack.setdefault(tid,[])
            if len(chosen)>=8 or len(pertrack)>12 and not chosen:
                continue
            if o['bbox_area_fraction'] < .002 or o['oblique_quality'] < .25:
                continue
            if any(abs(o['frame']-q['frame'])<5 for q in chosen):
                continue
            if len(self.local(o))<10:
                continue
            chosen.append(o); selected.append(o)
        fit_keys={}
        for tid,group in pertrack.items():
            for i,o in enumerate(sorted(group,key=lambda o:o['frame'])):
                fit_keys[o['key']]='fit' if i%2==0 else 'heldout'
        for o in observations:
            o['split']=fit_keys.get(o['key'],'filter-only')
        groups={}
        for o in observations:
            groups.setdefault((o['frame'],o['camera']),[]).append(o)
        for (f,c),group in tqdm(sorted(groups.items()),desc='Prepare RGB prompts'):
            image,k=self.ds._load_calibrated_image(self.meta,f'images/{f:03d}_{c}.jpg',self.ds.resolution,c)
            Image.fromarray((image.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(self.out/'inputs'/f'{f:03d}_{c}.png')
            for o in group:
                lo=np.array(o['bbox_xyxy'][:2]);hi=np.array(o['bbox_xyxy'][2:])
                pad=np.maximum((hi-lo)*.05,8)
                o['sam_prompt_xyxy']=np.r_[np.maximum(lo-pad,0),np.minimum(hi+pad,[self.w-1,self.h-1])].tolist()
                # Original target LiDAR median is a robust positive prompt.
                cp=transform(transform(self.local(o),self.actors[o['track_id']][f]),np.linalg.inv(self.camera(f,c)))
                xy,good=pixels(cp,k,self.h,self.w)
                o['sam_positive_xy']=(np.median(xy[good],axis=0) if good.any() else (lo+hi)*.5).tolist()
        write(self.out/'plan.json',dict(scene=self.meta['name'],observations=observations,
            fit_track_ids=sorted({o['track_id'] for o in selected}),frame_rate_hz_assumption=10,
            parameterization='one shared effective dt; no local residual; camera and background fixed',
            source='raw scene RGB, bbox, ego/camera calibration and LiDAR; no cached scene states'))

    def fit_apply(self):
        observations=json.loads((self.out/'plan.json').read_text())['observations']
        selected=[o for o in observations if o['split']!='filter-only']
        masks={o['key']:np.asarray(Image.open(self.out/'masks'/f"{o['key']}.png"))>0 for o in selected}
        def alignment(o,dt):
            mask=masks[o['key']]; small=cv2.resize(mask.astype(np.uint8),(self.w//4,self.h//4),interpolation=cv2.INTER_NEAREST)>0
            cp=transform(transform(self.local(o),self.corrected_actor(o['track_id'],o['frame'],dt)),np.linalg.inv(self.camera(o['frame'],o['camera'])))
            xy,good=pixels(cp,self.k[o['camera']],self.h,self.w)
            xy=xy[good]
            if len(xy)<3 or mask.sum()<150:
                return 0.,0.,0.
            hull=np.zeros(small.shape,np.uint8)
            cv2.fillConvexPoly(hull,cv2.convexHull((xy/4).astype(np.float32)).astype(np.int32),1)
            iou=float((small&(hull>0)).sum()/max(1,(small|(hull>0)).sum()))
            precision=float(mask[xy[:,1],xy[:,0]].mean())
            return .7*iou+.3*precision,iou,precision
        train=[o for o in selected if o['split']=='fit' and masks[o['key']].sum()>=150]
        if not train:
            raise RuntimeError('No usable scene-wide calibration observations')
        def score(dt):
            return float(np.average([alignment(o,dt)[0] for o in train],weights=[min(200,np.sqrt(masks[o['key']].sum())) for o in train]))
        grid=np.linspace(-.15,.15,61)
        scores=[score(dt) for dt in tqdm(grid,desc='Shared vehicle-only dt fit')]
        dt=float(grid[int(np.argmax(scores))])
        if max(scores)-score(0)<1e-4:
            dt=0.
        evaluations=[]
        for o in selected:
            before=alignment(o,0);after=alignment(o,dt)
            evaluations.append(dict(key=o['key'],track_id=o['track_id'],split=o['split'],score_before=before[0],score_after=after[0],iou_before=before[1],iou_after=after[1],mask_pixels=int(masks[o['key']].sum())))
        fit=dict(shared_effective_dt_seconds=dt,grid_seconds=grid.tolist(),grid_scores=scores,
            fit_track_ids=sorted({o['track_id'] for o in train}),fit_count=len(train),
            evaluations=evaluations,residual_local_xyz_m_by_track={},
            interpretation='Shared effective alignment assuming10Hz, not measured physical timestamps; one camera-independent trajectory per actor')
        for split in ('fit','heldout'):
            rows=[r for r in evaluations if r['split']==split]
            fit[split]={k:float(np.mean([r[k] for r in rows])) for k in ('score_before','score_after','iou_before','iou_after')} if rows else {}
        write(self.out/'corrections.json',fit)
        print(json.dumps(fit['fit']),json.dumps(fit['heldout']),'dt',dt,flush=True)
        corrected={tid:{f:(self.corrected_actor(tid,f,dt),size) for f,(pose,size) in track.items()} for tid,track in self.meta['instances'].items()}
        torch.save({self.meta['name']:corrected},self.out/'corrected_instances.pt')
        groups={}
        for o in observations:
            groups.setdefault((o['frame'],o['camera']),[]).append(o)
        audit=[]
        for f in tqdm(self.meta['frames'],desc='Geometry prompts'):
            world=self.world(f);moved=world.copy();owner=np.full(len(world),-1,np.int64)
            for tid,track in self.meta['instances'].items():
                if f not in track:
                    continue
                old,size=track[f]
                local=transform(world,np.linalg.inv(old))
                target=(np.abs(local)<=size*.5).all(1)&(owner<0)
                moved[target]=transform(local[target],corrected[tid][f][0]);owner[target]=tid
            for c in self.ds.cameras:
                camera=self.camera(f,c);k=self.k[c]
                cp=transform(moved,np.linalg.inv(camera));original=transform(world,np.linalg.inv(camera))
                xy,visible=pixels(cp,k,self.h,self.w)
                remove=np.zeros(len(cp),bool);support=np.zeros((self.h,self.w),np.uint8)
                conflict_count=spill_count=0
                for o in groups.get((f,c),[]):
                    tid=o['track_id'];ap,size=corrected[tid][f]
                    mask=np.asarray(Image.open(self.out/'masks'/f"{o['key']}.png"))>0
                    if mask.sum()<30:
                        continue
                    core=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
                    loose=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0
                    inside=np.zeros(len(cp),bool);near_mask=inside.copy()
                    inside[visible]=loose[xy[visible,1],xy[visible,0]]
                    near_mask[visible]=core[xy[visible,1],xy[visible,0]]
                    pose=np.linalg.inv(ap)@camera
                    rays=np.c_[xy,np.ones(len(cp))]@np.linalg.inv(k).T
                    directions=rays@pose[:3,:3].T
                    directions=np.where(np.abs(directions)<1e-10,1e-10,directions)
                    half=size*.5+np.array([.1,.05,.025])
                    a=(-half-pose[:3,3])/directions;b=(half-pose[:3,3])/directions
                    near=np.minimum(a,b).max(1).clip(0);far=np.maximum(a,b).min(1)
                    conflict=(owner<0)&visible&near_mask&(far>=near)&(cp[:,2]>=near-.1)
                    spill=(owner==tid)&visible&~inside
                    remove|=conflict|spill
                    conflict_count+=int(conflict.sum());spill_count+=int(spill.sum())
                    support[loose]=1
                # All moved actor supports belong to the allowed-change region.
                for cloud in (cp[owner>=0],original[owner>=0]):
                    uv,good=pixels(cloud,k,self.h,self.w); support[uv[good,1],uv[good,0]]=1
                support=cv2.dilate(support,np.ones((3,3),np.uint8))>0
                before=sparse(original,k,self.h,self.w);after=sparse(cp[~remove],k,self.h,self.w)
                assert np.array_equal(before[~support],after[~support]),'Unexpected background prompt change'
                np.savez_compressed(self.out/'sparse'/f'{f:03d}_{c}.npz',depth=after)
                audit.append(dict(frame=f,camera=c,assigned_vehicle_points=int((owner>=0).sum()),competing_background_removed=conflict_count,
                    actor_spill_removed=spill_count,removed_points=int(remove.sum()),
                    changed_prompt_pixels=int((before!=after).sum()),outside_actor_support_unchanged=True))
                if len(audit)<=3 or f in (55,100) and c==0:
                    img=np.asarray(Image.open(self.out/'inputs'/f'{f:03d}_{c}.png')).copy() if (self.out/'inputs'/f'{f:03d}_{c}.png').exists() else None
                    if img is not None:
                        idx=np.flatnonzero(visible&remove);img[xy[idx,1],xy[idx,0]]=[255,0,0]
                        Image.fromarray(img).save(self.out/f'filter_overlay_{f:03d}_{c}.jpg')
        write(self.out/'layer_filter_audit.json',dict(observations=audit,totals={k:sum(r[k] for r in audit) for k in ('assigned_vehicle_points','competing_background_removed','actor_spill_removed','removed_points','changed_prompt_pixels')},all_background_outside_actor_support_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--scene',default='017');p.add_argument('--mode',choices=['prepare','fit-apply'],required=True)
    args=p.parse_args();scene=Scene(args)
    scene.prepare() if args.mode=='prepare' else scene.fit_apply()

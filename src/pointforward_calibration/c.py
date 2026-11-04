"""Frozen LIM-v1-r dependency. See provenance.json; adapter is in pipeline.py."""
from __future__ import annotations
import argparse, copy, hashlib, html, json, math, os, shutil, sys, time
from collections import OrderedDict, defaultdict, Counter
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt, map_coordinates, label
from scipy.optimize import minimize, linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation, Slerp
HERE = Path(__file__).resolve().parent
PROJECT = HERE
CFG=json.loads((HERE/'c_config.json').read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

class Poses:
    def __init__(self, ts, mats):
        self.ts = np.asarray(ts, float)
        self.mats = np.asarray(mats)
        self.slerp = Slerp(self.ts, Rotation.from_matrix(self.mats[:,:3,:3])) if len(ts)>1 else None

    def at(self, t):
        if t<self.ts[0]-1e-7 or t>self.ts[-1]+1e-7:
            return None
        t = float(np.clip(t,self.ts[0],self.ts[-1]))
        a = self.mats[0].copy()
        if self.slerp is not None:
            a[:3,:3] = self.slerp(t).as_matrix()
            a[:3,3] = [np.interp(t,self.ts,self.mats[:,i,3]) for i in range(3)]
        return a

class Pipeline:

    def stage(self, name, start):
        self.stages[name] = time.perf_counter()-start
        print(self.spec['key'],name,f'{self.stages[name]:.1f}s',flush=True)

    def camera(self,f,c):
        e = self.ego.at(self.camtimes[f,c])
        # Last group can lack exposure support. Entire exposure action is held raw.
        return (self.ds._ego_pose(self.meta,f) if e is None else e)@self.meta['extrinsics'][c]

    def image(self,f,c):
        return self.base.load_image(self.ds,self.meta,f,c)

    def world(self,f):
        if f not in self.cache:
            path=self.root/'lidar'/f'{f:03d}.bin'
            self.cache[f]=self.base.transform(self.ds.read_lidar(str(path)),self.ds._ego_pose(self.meta,f))
            self.input_files[str(path)] = dict(size=path.stat().st_size,sha256=self.draw.sha(path))
            if len(self.cache)>4:self.cache.popitem(last=False)
        return self.cache[f]


    def prepare(self):
        start=time.perf_counter()
        for v in self.selection['vehicles']:
            tid=v['track_id']
            for r in v['observations']:
                f,c=r['frame'],r['camera']
                actor,size=self.meta['instances'][tid][f]
                world=self.world(f)
                local=self.base.transform(world,np.linalg.inv(actor))
                half=np.maximum(size*self.cfg['expanded_box_factor']*.5,size*.5+self.cfg['minimum_expansion_margin_m'])
                ids=np.flatnonzero((abs(local)<=half).all(1))
                plane,info=self.groundmod.Calibration.ground(self,world,actor,size,np.random.default_rng(self.cfg['seed']+tid*10000+f))
                ground=(world[ids]@plane[0]+plane[1]<=self.cfg['ground']['remove_height_m']) if plane else np.zeros(len(ids),bool)
                keep=ids[~ground]
                rgb,k=self.image(f,c)
                camera=self.camera(f,c)
                target=self.actors[tid].at(self.camtimes[f,c])
                time_supported=target is not None and self.ego.at(self.camtimes[f,c]) is not None
                if target is None:target=actor
                o=dict(**r,track=tid,time=self.times[f],camera_time=self.camtimes[f,c],
                    actor=target,camera_pose=camera,size=size,K=k,
                    points=world[keep].copy(),ids=keep,ground_ids=ids[ground],ground_info=info,
                    membership_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
                    raw_candidate_points=len(ids),removed_ground_points=int(ground.sum()),
                    time_supported=time_supported,mask_accepted=False,mask_reasons=['not_measured'])
                indices=np.arange(len(keep))
                if len(indices)>self.cfg['fit']['max_points']:
                    indices=np.random.default_rng(self.cfg['seed']+tid*10000+f).choice(indices,self.cfg['fit']['max_points'],replace=False)
                o['sample']=o['points'][indices]
                self.obs.append(o)
                np.savez_compressed(self.out/f'{o["key"]}_fixed_candidates.npz',point_ids=keep,world=o['points'],removed_ground_ids=ids[ground])
        self.stage('fixed_candidates_and_ground',start)

    def measure(self):
        import torch
        start=time.perf_counter()
        torch.set_num_threads(2)
        os.environ['YOLO_CONFIG_DIR']=str(self.out)
        sys.path.append(str(self.maskmod.ULTRA_SITE))
        from ultralytics import YOLO
        sys.path.insert(0,str(self.sam_root))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        detector=YOLO(self.maskmod.CONTRACT['detector'])
        predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',self.maskmod.CONTRACT['sam'],device='cuda'))
        groups=defaultdict(list)
        for o in self.obs:groups[o['frame'],o['camera']].append(o)
        measurement=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for n,((f,c),rows) in enumerate(sorted(groups.items())):
                rgb,k=self.image(f,c)
                preds=self.maskmod.proposals(detector,predictor,rgb,self.device)
                # Save independent proposals before looking at any actor geometry.
                for d in preds:
                    Image.fromarray(d['mask'].astype(np.uint8)*255).save(self.out/f'rgb_{f:03d}_{c}_proposal{d["index"]}.png')
                proposal_meta=[{k:v for k,v in d.items() if k not in ('mask','detector_mask')} for d in preds]
                write(self.out/f'rgb_{f:03d}_{c}_proposals.json',proposal_meta)
                # Associate all annotated vehicles, not just selected ones, to avoid duplicate identity ownership.
                allactors=[]
                for tid,ann in self.meta['instances'].items():
                    if f not in ann:continue
                    a=self.actors[int(tid)].at(self.camtimes[f,c])
                    if a is None:a=ann[f][0]
                    allactors.append(dict(key=str(tid),actor=a,camera_pose=self.camera(f,c),size=ann[f][1],K=k))
                hashes=[d['sha256'] for d in preds]
                assigned=self.maskmod.assign(allactors,preds)
                assert hashes==[hashlib.sha256(d['mask'].tobytes()).hexdigest() for d in preds]
                for o in rows:
                    a=assigned[str(o['track'])]
                    o['mask_accepted']=a['accepted']
                    o['mask_reasons']=a['reasons']
                    o['association']=a
                    if a['accepted']:
                        mask=preds[a['detection_index']]['mask']
                        o['mask']=mask
                        o['distance']=distance_transform_edt(~mask).astype(np.float32)
                        o['mask_sha256']=hashlib.sha256(mask.tobytes()).hexdigest()
                        Image.fromarray(mask.astype(np.uint8)*255).save(self.out/f'{o["key"]}_mask.png')
                    measurement.append(dict(key=o['key'],track=o['track'],frame=f,camera=c,role=o['role'],
                        **a,proposal_hashes=hashes,geometry_does_not_generate_masks=True))
                if n%8==0:print(self.spec['key'],'RGB masks',n+1,'/',len(groups),flush=True)
        del predictor,detector
        torch.cuda.empty_cache()
        write(self.out/'measurements.json',measurement)
        self.stage('independent_RGB_masks',start)



    def pixels(self,points,o):
        cp=self.base.transform(points,np.linalg.inv(o['camera_pose']))
        z=cp[:,2]
        uv=(cp@o['K'].T)[:,:2]/np.maximum(z[:,None],.01)
        return uv,z

    def hull(self,actor,o):
        uv=self.base.box_uv(actor,o['camera_pose'],o['size'],o['K'])
        mask=np.zeros((576,768),np.uint8)
        if uv is not None:
            cv2.fillConvexPoly(mask,cv2.convexHull(uv.astype(np.float32)).astype(np.int32),1)
        return mask>0





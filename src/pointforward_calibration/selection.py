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
from . import base
draw=base.draw


def motion(annotations):
    frames = sorted(annotations)
    positions = np.array([annotations[f][0][:3,3] for f in frames])
    diameter = float(pdist(positions[:,:2]).max()) if len(positions) > 1 else 0.
    return dict(world_xy_diameter_m=diameter,
                world_xy_first_last_m=float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),
                max_adjacent_step_m=float(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1).max()) if len(positions)>1 else 0.,
                annotated_frame_range=[int(frames[0]),int(frames[-1])])

def select(ds, meta, cfg, scene):
    candidates = defaultdict(list)
    frames = meta['frames'][::cfg['frame_step']]
    ks = {c:base.load_image(ds,meta,frames[0],c)[1] for c in ds.cameras}
    sizes = {c:Image.open(Path(meta['path'])/'images'/f'{frames[0]:03d}_{c}.jpg').size for c in ds.cameras}
    movements = {int(t):motion(a) for t,a in meta['instances'].items()}
    for f in frames:
        ego = ds._ego_pose(meta,f)
        for c in ds.cameras:
            camera = ego @ meta['extrinsics'][c]
            inv = np.linalg.inv(camera)
            projected = []
            for tid, annotations in meta['instances'].items():
                if f not in annotations:
                    continue
                actor, size = annotations[f]
                depth = float(base.transform(actor[None,:3,3],inv)[0,2])
                if not .5 < depth < 100:
                    continue
                uv = base.box_uv(actor,camera,size,ks[c])
                if uv is None:
                    continue
                hull = cv2.convexHull(uv.astype(np.float32))
                lo,hi = uv.min(0),uv.max(0)
                screen_mask = np.zeros((144,192),np.uint8)
                cv2.fillConvexPoly(screen_mask,np.rint(hull[:,0,:]/4).astype(np.int32),1)
                if not screen_mask.any():
                    continue
                projected.append((depth,int(tid),actor,size,uv,lo,hi,screen_mask))
            foreground = np.zeros((144,192),bool)
            for depth,tid,actor,size,uv,lo,hi,mask in sorted(projected,key=lambda x:x[0]):
                occluded = float((foreground & (mask>0)).sum()/mask.sum())
                foreground |= mask>0
                if tid in scene.get('exclude_tracks',[]):
                    continue
                if [tid,int(f),int(c)] in scene.get('exclude_views',[]):
                    continue
                if any(tid==t and c==cam and start<=f<=end for t,cam,start,end in scene.get('exclude_frame_ranges',[])):
                    continue
                clipped = np.maximum(0,np.minimum(hi,[768,576])-np.maximum(lo,[0,0]))
                inframe = float(np.prod(clipped)/max(np.prod(hi-lo),1))
                # Area in the actual source image, before adapter crop/resize.
                source_uv = base.box_uv(actor,camera,size,meta['intrinsics'][c])
                width,height = sizes[c]
                rect = np.array([[0,0],[width,0],[width,height],[0,height]],np.float32)
                source_area,_ = cv2.intersectConvexConvex(cv2.convexHull(source_uv.astype(np.float32)),rect)
                fraction = float(source_area/(width*height))
                if inframe < cfg['minimum_in_frame_fraction'] or occluded > cfg['maximum_box_occlusion_fraction'] or fraction < cfg['minimum_source_area_fraction']:
                    continue
                candidates[tid].append(dict(track_id=tid,frame=int(f),camera=int(c),depth_m=depth,
                    area_px=float(np.prod(clipped)),source_hull_area_fraction=fraction,
                    estimated_box_occlusion_fraction=occluded,in_frame_fraction=inframe,
                    bbox_xyxy=[*lo.tolist(),*hi.tolist()]))
    ranked = []
    for tid,rows in candidates.items():
        # Favor genuinely large views; overlap penalty discourages hidden cars.
        for row in rows:
            row['view_score'] = row['source_hull_area_fraction']*(1-row['estimated_box_occlusion_fraction'])**2
        rep = max(rows,key=lambda r:r['view_score'])
        eligible = [r for r in rows if r['source_hull_area_fraction'] >= cfg['minimum_relative_view_area']*rep['source_hull_area_fraction']]
        chosen = [rep]
        while len(chosen)<cfg['observations_per_vehicle']:
            remaining = [r for r in eligible if min(abs(r['frame']-a['frame']) for a in chosen) >= cfg['minimum_frame_gap']]
            if not remaining:
                break
            # Maximize world/image movement among sufficiently large views.
            def view_distance(r):
                p=meta['instances'][tid][r['frame']][0][:2,3]
                distances=[]
                for a in chosen:
                    q=meta['instances'][tid][a['frame']][0][:2,3]
                    distances.append(np.linalg.norm(p-q)+.03*abs(r['frame']-a['frame']))
                return min(distances)*np.sqrt(r['view_score']/rep['view_score'])
            chosen.append(max(remaining,key=view_distance))
        m = movements[tid]
        moving = m['world_xy_diameter_m'] >= cfg['moving_threshold_m']
        score = (rep['view_score']/.05)**.65 * (min(max(m['world_xy_diameter_m'],1),80)/20)**.35
        # No mandatory camera/depth strata: large moving vehicles take priority.
        ranked.append(dict(track_id=tid, representative=rep, observations=sorted(chosen,key=lambda r:r['frame']),
                           motion=m, moving=moving, score=score))
    ranked.sort(key=lambda v:(v['moving'],v['score']),reverse=True)
    vehicles=ranked[:cfg.get('scene_vehicle_limits',{}).get(scene['key'],cfg['vehicles_per_scene'])]
    return dict(spec=scene,eligible_tracks=len(ranked),eligible_moving_tracks=sum(v['moving'] for v in ranked),
                sampling='Large original source-image cuboid hulls and large WORLD-XY trajectory diameter. Moving tracks first; score=area^0.65*motion^0.35 with box-overlap penalty. No depth/camera quotas. No LiDAR/RGB agreement ranking.',
                thresholds=cfg['selection_thresholds'],vehicles=vehicles,
                all_ranked_tracks=[{k:v[k] for k in ('track_id','moving','score','motion','representative')} for v in ranked])

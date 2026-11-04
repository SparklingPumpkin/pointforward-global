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
SIGNS=np.asarray([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],dtype=float)


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')

def transform(points, pose):
    return points@pose[:3,:3].T+pose[:3,3]

def interp(poses,t):
    keys=np.array(sorted(poses))
    t=float(np.clip(t,keys[0],keys[-1]))
    hi=min(np.searchsorted(keys,t,side='right'),len(keys)-1)
    lo=max(hi-1,0)
    if hi==lo or t==keys[lo]:
        return poses[int(keys[lo])].copy()
    a,b=int(keys[lo]),int(keys[hi])
    out=np.eye(4)
    out[:3,3]=poses[a][:3,3]+(poses[b][:3,3]-poses[a][:3,3])*(t-a)/(b-a)
    out[:3,:3]=Slerp([a,b],Rotation.from_matrix(np.stack([poses[a][:3,:3],poses[b][:3,:3]])))([t]).as_matrix()[0]
    return out

def project(cp,k):
    z=cp[:,2]
    valid=np.isfinite(cp).all(1)&(z>.1)
    uv=np.zeros((len(cp),2))
    uv[valid]=(cp[valid]@k.T)[:,:2]/z[valid,None]
    valid &= (uv[:,0]>=0)&(uv[:,0]<768)&(uv[:,1]>=0)&(uv[:,1]<576)
    return uv,valid

def box_uv(actor,camera,size,k):
    cp=transform(transform(SIGNS*size*.5,actor),np.linalg.inv(camera))
    if (cp[:,2]<=.1).any():
        return None
    return (cp@k.T)[:,:2]/cp[:,2:]

def draw_box(image,uv,color):
    if uv is None:
        return
    draw=ImageDraw.Draw(image)
    for i in range(8):
        for j in range(i+1,8):
            if np.count_nonzero(SIGNS[i]!=SIGNS[j])==1:
                draw.line((tuple(uv[i]),tuple(uv[j])),fill=color,width=2)

def load_image(ds,meta,f,c):
    from scripts.build_track_once_dynamic import calibrated_intrinsics
    if hasattr(ds,'_load_calibrated_image'):
        im,k=ds._load_calibrated_image(meta,f'images/{f:03d}_{c}.jpg',ds.resolution,c)
    else:
        im,_,_=ds._load_image(meta,f'images/{f:03d}_{c}.jpg',ds.resolution)
        k=calibrated_intrinsics(ds,meta,f)[c]
    return (im.permute(1,2,0).numpy()*255).round().astype(np.uint8),k

def evaluate(local,actor,camera,k,mask):
    """Fixed point membership; offscreen/behind points penalized, never dropped."""
    import cv2
    cp=transform(transform(local,actor),np.linalg.inv(camera))
    uv,valid=project(cp,k)
    xy=np.rint(uv).astype(int)
    xy[:,0]=np.clip(xy[:,0],0,767);xy[:,1]=np.clip(xy[:,1],0,575)
    hit=mask[xy[:,1],xy[:,0]]&valid
    dist=distance_transform_edt(~mask)
    outside=dist[xy[:,1],xy[:,0]].clip(0,60)
    outside[~valid]=60
    small=cv2.resize(mask.astype(np.uint8),(192,144),interpolation=cv2.INTER_NEAREST)>0
    hull=np.zeros((144,192),np.uint8)
    if valid.sum()>=3:
        cv2.fillConvexPoly(hull,cv2.convexHull((uv[valid]/4).astype(np.float32)).astype(int),1)
    iou=float((small&(hull>0)).sum()/max(1,(small|(hull>0)).sum()))
    precision=float(hit.mean()) if len(hit) else 0.
    return dict(precision=precision,outside_mean_px=float(outside.mean()) if len(outside) else 60.,
                visible_fraction=float(valid.mean()) if len(valid) else 0.,hull_iou=iou,
                old_score=.7*iou+.3*precision,npoints=len(local))

def overlay(rgb,local,actor,camera,size,k,mask=None,title=''):
    im=Image.fromarray(rgb.copy())
    draw_box(im,box_uv(actor,camera,size,k),(255,220,0))
    arr=np.asarray(im).copy()
    if mask is not None:
        arr[mask]=(arr[mask]*.7+np.array([0,180,80])*.3).round().astype(np.uint8)
    uv,ok=project(transform(transform(local,actor),np.linalg.inv(camera)),k)
    xy=np.rint(uv[ok]).astype(int)
    xy[:,0]=xy[:,0].clip(0,767);xy[:,1]=xy[:,1].clip(0,575)
    arr[xy[:,1],xy[:,0]]=[255,80,80]
    im=Image.fromarray(arr);draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,768,25),fill='black');draw.text((8,5),title,fill='white')
    return im

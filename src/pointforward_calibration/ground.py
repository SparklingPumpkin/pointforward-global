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
CFG=json.loads((HERE/'c_config.json').read_text())


class Calibration():
    def ground(self,world,actor,size,rng):
        cfg=CFG['ground']; center=actor[:3,3]; bottom=center[2]-size[2]*.5
        local=base.transform(world,np.linalg.inv(actor))
        near=(np.linalg.norm(world[:,:2]-center[:2],axis=1)<cfg['neighborhood_m'])
        low=(np.abs(world[:,2]-bottom)<cfg['height_search_m'])
        # Fit the local low road layer outside the target's expanded footprint.
        outside=(np.abs(local[:,:2])>size[:2]*.8).any(1)
        candidates=world[near&low&outside]
        if len(candidates)>6500:
            candidates=candidates[rng.choice(len(candidates),6500,replace=False)]
        best=None;best_count=0
        for _ in range(cfg['ransac_trials']):
            if len(candidates)<3:
                break
            a,b,c=candidates[rng.choice(len(candidates),3,replace=False)]
            n=np.cross(b-a,c-a);length=np.linalg.norm(n)
            if length<1e-6:
                continue
            n/=length
            if n[2]<0:
                n=-n
            if n[2]<np.cos(np.deg2rad(cfg['max_tilt_deg'])):
                continue
            d=-n@a;mask=np.abs(candidates@n+d)<cfg['inlier_distance_m']
            count=int(mask.sum())
            if count>best_count:
                best_count=count;best=(n,d,mask)
        if best is None or best_count<cfg['minimum_inliers']:
            return None,dict(valid=False,inliers=best_count,reason='insufficient local plane evidence; no ground removal')
        n,d,mask=best
        p=candidates[mask];origin=p.mean(0)
        _,_,vh=np.linalg.svd(p-origin,full_matrices=False)
        refined=vh[-1];refined*=1 if refined[2]>0 else -1
        if refined[2]>=np.cos(np.deg2rad(cfg['max_tilt_deg'])):
            n=refined;d=-n@origin
        residual=np.abs(p@n+d)
        return (n,d),dict(valid=True,normal=n.tolist(),d=float(d),inliers=best_count,
                             candidate_points=len(candidates),median_residual_m=float(np.median(residual)))



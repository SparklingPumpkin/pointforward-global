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
from .geometry import box_uv, transform, overlay, write
CONTRACT={}
ULTRA_SITE=HERE


def iou(a,b):
    return float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))

def hull_for(o, shift=None):
    a=np.array(o['actor'])
    if shift is not None: a[:3,3]+=shift
    uv=box_uv(a,np.array(o['camera_pose']),np.array(o['size']),np.array(o['K']))
    hull=np.zeros((576,768),np.uint8)
    if uv is not None:
        cv2.fillConvexPoly(hull,cv2.convexHull(uv.astype(np.float32)).astype(int),1)
    return hull>0

def mask_stats(mask,hull):
    yy,xx=np.where(mask)
    return dict(pixels=int(mask.sum()),outside_original_hull_fraction=float((mask & ~hull).sum()/max(1,mask.sum())),
                bounds_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None)

def infer_yolo(model,rgb,device):
    # Ultralytics numpy inputs are BGR, whereas the image/SAM API is RGB.
    result=model.predict(source=np.ascontiguousarray(rgb[:,:,::-1]),device=device,conf=.25,iou=.7,
                         imgsz=768,retina_masks=True,classes=[2,5,7],verbose=False,save=False)[0]
    detections=[]
    if result.masks is None:return detections
    for i,b in enumerate(result.boxes):
        m=result.masks.data[i].cpu().numpy()>0
        assert m.shape==(576,768),m.shape
        detections.append(dict(class_id=int(b.cls.item()),confidence=float(b.conf.item()),
                               box=b.xyxy[0].cpu().numpy().tolist(),mask=m))
    return detections

def proposals(model,predictor,rgb,device):
    normal=infer_yolo(model,rgb,device)
    flipped=infer_yolo(model,np.ascontiguousarray(rgb[:,::-1]),device)
    for d in flipped:d['mask']=d['mask'][:,::-1].copy()
    matrix=np.array([[iou(a['mask'],b['mask']) if a['class_id']==b['class_id'] else 0.
                      for b in flipped] for a in normal]).reshape(len(normal),len(flipped))
    if normal:predictor.set_image(rgb)
    output=[]
    for i,d in enumerate(normal):
        reasons=[]
        j=int(np.argmax(matrix[i])) if len(flipped) else None
        flip_iou=float(matrix[i,j]) if j is not None else 0.
        if d['confidence']<.5:reasons.append('normal_confidence')
        if j is None or flipped[j]['confidence']<.5:reasons.append('flip_confidence')
        if j is None or int(np.argmax(matrix[:,j]))!=i:reasons.append('not_mutual_flip_match')
        if flip_iou<.75:reasons.append('flip_iou')
        dist=distance_transform_edt(d['mask'])
        y,x=np.unravel_index(np.argmax(dist),dist.shape)
        masks,scores,_=predictor.predict(box=np.array(d['box']),point_coords=np.array([[x,y]]),
                                         point_labels=np.array([1]),multimask_output=True)
        agreements=[iou(m>0,d['mask']) for m in masks]
        k=int(np.argmax(agreements));mask=masks[k]>0
        if agreements[k]<.65:reasons.append('sam_detector_iou')
        if mask.sum()<64:reasons.append('mask_lt64pixels')
        components,n=label(mask);counts=np.bincount(components.ravel())[1:]
        largest=float(counts.max()/max(1,mask.sum())) if n else 0.
        if largest<.95:reasons.append('fragmented_mask')
        output.append(dict(index=i,class_id=d['class_id'],confidence=d['confidence'],box=d['box'],
            flip_confidence=flipped[j]['confidence'] if j is not None else 0.,flip_iou=flip_iou,
            sam_iou=agreements[k],sam_model_confidence=float(scores[k]),all_sam_detector_ious=agreements,
            component_fraction=largest,reasons=reasons,qualified=not reasons,
            sha256=hashlib.sha256(mask.tobytes()).hexdigest(),mask=mask,detector_mask=d['mask']))
    return output

def assign(observations, candidates, shift=None):
    eligible=[d for d in candidates if d['qualified']]
    hulls=[hull_for(o,shift) for o in observations]
    matrix=np.array([[iou(h,d['mask']) for d in eligible] for h in hulls]).reshape(len(hulls),len(eligible))
    result={o['key']:dict(accepted=False,reasons=['no_qualified_detection']) for o in observations}
    if not len(eligible):return result
    rows,cols=linear_sum_assignment(-matrix)
    for r,c in zip(rows,cols):
        row=np.sort(np.delete(matrix[r],c));col=np.sort(np.delete(matrix[:,c],r))
        rowmargin=float(matrix[r,c]-(row[-1] if len(row) else 0.))
        colmargin=float(matrix[r,c]-(col[-1] if len(col) else 0.))
        reasons=[]
        if matrix[r,c]<.2:reasons.append('hull_mask_iou')
        if rowmargin<.1:reasons.append('row_ambiguity')
        if colmargin<.1:reasons.append('column_ambiguity')
        result[observations[r]['key']]=dict(accepted=not reasons,reasons=reasons,detection_index=eligible[c]['index'],
            hull_mask_iou=float(matrix[r,c]),row_margin=rowmargin,column_margin=colmargin)
    for o in observations:
        if result[o['key']]['reasons']==['no_qualified_detection']:
            result[o['key']]['reasons']=['unassigned_one_to_one']
    return result

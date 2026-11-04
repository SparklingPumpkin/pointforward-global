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
from . import j
e=j.e
CFG=json.loads((HERE/'k_config.json').read_text())


def robust(x):
    return np.minimum(np.sqrt(1 + np.asarray(x)**2)-1, 10)

class Refiner(j.Completed):


    def extent_target(self, o):
        info = dict(enabled=False, reason='insufficient_support')
        if not o['mask_accepted'] or not o['fixed_supported'] or len(o['ids'])<CFG['extent_minimum_points']: return info
        yy,xx = np.where(o['mask'])
        if not len(xx) or xx.min()<=2 or xx.max()>=765 or yy.min()<=2 or yy.max()>=573:
            return dict(enabled=False,reason='mask_touches_image_boundary')
        center = o['target'][:3,3]
        probe = np.array([center, center+o['target'][:3,0]])
        uv,z = self.pixels(probe,o); axis=uv[1]-uv[0]
        side_ratio = np.linalg.norm(axis) / (o['K'][0,0]/max(z[0],.1))
        if side_ratio<CFG['extent_minimum_side_projection']:return dict(enabled=False,reason='not_side_view',side_projection=float(side_ratio))
        axis/=max(np.linalg.norm(axis),1e-8)
        pts,z = self.pixels(o['base_world'],o)
        visible=(z>.1)&(pts[:,0]>=0)&(pts[:,0]<768)&(pts[:,1]>=0)&(pts[:,1]<576)
        if visible.sum()<CFG['extent_minimum_points']:return info
        q=CFG['extent_quantiles']
        endpoints=np.quantile(np.c_[xx,yy]@axis,q)
        observed=np.quantile(pts[visible]@axis,q)
        ratio=float(np.diff(observed)[0]/max(np.diff(endpoints)[0],1))
        if not CFG['extent_minimum_length_ratio']<=ratio<=CFG['extent_maximum_length_ratio']:
            return dict(enabled=False,reason='partial_or_oversized_longitudinal_support',length_ratio=ratio)
        return dict(enabled=True,axis=axis.tolist(),mask_endpoints=endpoints.tolist(),length_ratio=ratio,
            side_projection=float(side_ratio),reason='supported_longitudinal_extents_only')

    def move(self,o,residual,allpoints=False):
        pts=o['base_world'] if allpoints else o['sample_world']
        return pts+o['target'][:3,:3]@np.asarray(residual)

    def image_cost(self,o,pts):
        uv,z=self.pixels(pts,o)
        distance=map_coordinates(o['distance'],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100.)
        distance[z<=.1]=100.
        outside=float(np.mean(robust(distance/CFG['outside_pixel_scale'])))
        edge=0.
        if o['extent']['enabled']:
            valid=z>.1
            if valid.sum():
                endpoints=np.quantile(uv[valid]@np.asarray(o['extent']['axis']),CFG['extent_quantiles'])
                edge=float(np.mean(robust((endpoints-np.asarray(o['extent']['mask_endpoints']))/CFG['outside_pixel_scale'])))
            else:edge=10.
        return dict(total=outside+CFG['extent_weight']*edge,outside=outside,longitudinal_edges=edge)

    def usable(self,o):
        return o['mask_accepted'] and o['ground_info']['valid'] and o['fixed_supported'] and len(o['ids'])>=CFG['minimum_points']

    def bbox_fit(self,rows,residual,initial):
        bounds=np.array(CFG['bbox_bounds_m'])
        def loss(b):
            values=[]
            for o in rows:
                points=self.move(o,residual)
                local=self.base.transform(points,np.linalg.inv(o['target']))
                d=np.linalg.norm(np.maximum(abs(local-b)-o['size']/2,0),axis=1)
                values.append(np.mean(robust(d/CFG['bbox_geometry_scale_m'])))
            return float(np.mean(values))+CFG['bbox_prior_weight']*np.sum(((b-initial)/bounds)**2)
        opt=minimize(loss,np.clip(initial,-bounds,bounds),method='Powell',bounds=list(zip(-bounds,bounds)),options=dict(maxiter=35,xtol=.002,ftol=1e-5))
        return opt.x.tolist()

    def fit(self):
        self.tracks={}
        bounds=np.array(CFG['point_residual_bounds_m'])
        for v in self.selection['vehicles']:
            tid=v['track_id']; rows=[o for o in self.rows if o['track']==tid and o['split']=='fit' and self.usable(o)]
            initial=np.array(self.frozen['source_box_offsets_m'].get(str(tid),[0,0,0]))
            if len(rows)<CFG['minimum_fit_views']:
                self.tracks[str(tid)]=dict(residual_m=[0.,0.,0.],bbox_center_m=initial.tolist(),fit_views=len(rows),apply=False,reasons=['insufficient_fit_views_or_heldout_track'])
                continue
            def loss(r):
                return float(np.mean([self.image_cost(o,self.move(o,r))['total'] for o in rows]))+CFG['point_prior_weight']*np.sum((r/bounds)**2)
            candidates=[np.zeros(3)]
            if self.use_residual:
                # Common longitudinal starts avoid a single flat inside-mask
                # basin; all tracks/scenes use the same declared starts.
                for start in [np.zeros(3),np.array([-.6,0,0]),np.array([.6,0,0])]:
                    opt=minimize(loss,start,method='Powell',bounds=list(zip(-bounds,bounds)),options=dict(maxiter=40,xtol=.002,ftol=1e-5))
                    candidates.append(opt.x)
            best=min(candidates,key=loss)
            bbox=self.bbox_fit(rows,best,initial) if self.use_residual else initial.tolist()
            gate=[o for o in self.rows if o['track']==tid and o['split']=='gate' and self.usable(o)]
            reasons=[]; comparisons=[]
            for o in gate:
                moved=self.move(o,best,True)
                before=self.score(o,o['prior_world'],o['prior_box'])
                after=self.score(o,moved,self.box(o,bbox))
                beforecost=self.image_cost(o,o['prior_world'])
                aftercost=self.image_cost(o,moved)
                comparisons.append(dict(key=o['key'],before=before,proposal=after,before_objective=beforecost,proposal_objective=aftercost))
            if not gate:reasons.append('no_gate_support')
            if comparisons:
                bc=np.mean([x['before_objective']['total'] for x in comparisons]); ac=np.mean([x['proposal_objective']['total'] for x in comparisons])
                if ac>=bc-1e-6:reasons.append('no_gate_image_objective_improvement')
                for metric,limit in [('point_mask_distance_px',CFG['gate_max_mean_outside_increase_px']),('outside3_fraction',CFG['gate_max_outside3_increase'])]:
                    if np.mean([x['proposal'][metric]-x['before'][metric] for x in comparisons])>limit:reasons.append('gate_regression_'+metric)
                if max(x['proposal']['point_mask_distance_px']-x['before']['point_mask_distance_px'] for x in comparisons)>CFG['gate_max_frame_outside_increase_px']:reasons.append('gate_single_frame_regression')
            if not self.use_residual:reasons.append('residual_disabled')
            self.tracks[str(tid)]=dict(residual_m=best.tolist(),bbox_center_m=bbox,fit_views=len(rows),extent_fit_views=sum(o['extent']['enabled'] for o in rows),
                fit_zero_objective=float(loss(np.zeros(3))),fit_final_objective=float(loss(best)),
                boundary_hit=bool((abs(best)>=bounds-.01).any()),apply=not reasons,reasons=reasons,gate=comparisons)
            print(self.key,'track',tid,json.dumps({k:x for k,x in self.tracks[str(tid)].items() if k!='gate'}),flush=True)

    def box(self,o,b):
        actor=o['target'].copy();actor[:3,3]+=actor[:3,:3]@np.array(b);return actor



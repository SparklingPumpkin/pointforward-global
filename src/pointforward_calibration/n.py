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
from . import m
m.k.CFG=dict(m.k.CFG,minimum_fit_views=1)


class AllFit(m.Pipeline):
    def prepare_selected(self):
        super().prepare_selected()
        self.mask_recovery=[]
        for o in self.rows:
            o['original_split']=o['split'];o['original_mask_accepted']=o['mask_accepted']
            o['split']='fit'
            info=dict(key=o['key'],original_split=o['original_split'],original_mask_accepted=o['original_mask_accepted'],mode='original_mask')
            if not o['mask_accepted']:
                path=self.source/f'rgb_{o["frame"]:03d}_{o["camera"]}_proposals.json'
                candidates=json.loads(path.read_text()) if path.exists() else []
                matches=[]
                for d in candidates:
                    if not d['qualified']:continue
                    mask=np.array(Image.open(self.source/f'rgb_{o["frame"]:03d}_{o["camera"]}_proposal{d["index"]}.png'))>0
                    score=self.maskmod.iou(self.hull(o['group_actor'],o),mask)
                    matches.append((score,d['index'],mask))
                if matches and max(x[0] for x in matches)>=.2:
                    score,index,mask=max(matches,key=lambda x:x[0])
                    o['mask']=mask;o['distance']=distance_transform_edt(~mask).astype(np.float32);o['mask_accepted']=True
                    info.update(mode='independent_best_overlap_no_exclusive_assignment',proposal_index=index,hull_mask_iou=score,
                        warning='Identity is not certified; duplicate/competing track assignments may share a mask')
                else:
                    info.update(mode='geometry_only_no_reliable_image_association',warning='Participates in geometry fitting; no invented RGB mask')
            info['point_count']=len(o['ids']);self.mask_recovery.append(info)
            # The selected actor fallback already supplied a valid orientation.
            # Missing interpolation must not erase a finite per-frame annotation.
            o['fixed_supported']=True
            o['extent']=self.extent_target(o)
        m.e.save(self.out/'all_fit_observations.json',dict(records=self.mask_recovery,
            no_heldout_tracks=True,no_observation_mask_application_gate=True,
            note='All existing selected observations are used in fitting, so all reported errors are in-sample diagnostics.'))

    def usable(self,o):
        return len(o['ids'])>0 and np.isfinite(o['base_world']).all()

    def image_cost(self,o,pts):
        if o['mask_accepted']:return super().image_cost(o,pts)
        # No reliable RGB measurement: still include the observation in a
        # bounded geometry fit rather than associating an unrelated car mask.
        local=self.base.transform(pts,np.linalg.inv(o['selected_box']))
        d=np.linalg.norm(np.maximum(abs(local)-o['size']/2,0),axis=1)
        value=float(np.mean(m.k.robust(d/m.k.CFG['bbox_geometry_scale_m'])))
        return dict(total=value,outside=0.,longitudinal_edges=0.,geometry_only=value)

    def fit(self):
        # Reuse the unchanged numerical fit; all rows now belong to fit.
        super().fit()
        for tid,t in self.tracks.items():
            t['reasons']=[x for x in t['reasons'] if x!='no_gate_support']
            t['apply']=t['fit_views']>=1
            t['all_fit_mode']=True;t['independent_validation']=False
        assert all(t['fit_views']>=1 and t['apply'] for t in self.tracks.values())



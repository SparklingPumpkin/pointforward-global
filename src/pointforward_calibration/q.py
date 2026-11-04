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
from . import n,m,k,j,e,c


class FreshFields(m.Pipeline):
    def prepare_selected(self):
        """Same M field setup without nonexistent old K cache dependencies."""
        self.frozen = dict(source_tau_s=0., cameras={}, source_box_offsets_m={})
        ld = self.frame_selection['lidar']['offset']
        bd = self.frame_selection['bbox']['offset']
        for o in self.rows:
            o['raw_world'] = o['arms'][0]['points']
            o['raw_ids'] = o['arms'][0]['ids']
            arm = o['arms'][ld]
            o['lidar_fallback'] = not arm['supported'] or len(arm.get('ids', [])) < k.CFG['minimum_points']
            if o['lidar_fallback']:
                arm = o['arms'][0]
            box = self.box_arm(o, bd)
            o['bbox_fallback'] = box is None
            if box is None:
                box = self.box_arm(o, 0)
            o['lidar_frame'] = arm['source_frame']
            o['bbox_frame'] = o['frame'] + (0 if o['bbox_fallback'] else bd)
            o['points'], o['ids'] = arm['points'], arm['ids']
            o['size'] = box[1].copy()
            o['selected_box'] = box[0].copy()
            o['target'] = self.actors[o['track']].at(o['camera_time'])
            o['fixed_supported'] = o['target'] is not None
            if o['target'] is None:
                o['target'] = o['group_actor'].copy()
            o['base_world'] = o['points'].copy()
            o['base_box'] = o['selected_box'].copy()
            idx = np.arange(len(o['ids']))
            if len(idx) > k.CFG['fit_sample_points']:
                idx = np.random.default_rng(e.CFG['seed']+o['track']*10000+o['frame']).choice(idx, k.CFG['fit_sample_points'], replace=False)
            o['sample_indices'], o['sample_world'] = idx, o['base_world'][idx]
            o['prior_world'], o['prior_box'] = o['base_world'], o['selected_box']
            o['extent'] = self.extent_target(o)

class Fresh(n.AllFit, FreshFields):

    def prepare_pairings(self):
        self.rows = self.obs
        for o in self.rows:
            o['group_actor'] = self.meta['instances'][o['track']][o['frame']][0].copy()
            o['split'] = 'gate' if o['role'] == 'validation_frame' else 'fit'
            o['arms'] = {}
            for delta in m.CFG['source_offsets']:
                g = o['frame'] + delta
                if g not in self.frames or g not in self.meta['instances'][o['track']]:
                    o['arms'][delta] = dict(supported=False, source_frame=g, reason='neighbor_frame_or_actor_unavailable')
                    continue
                cachekey = (o['track'], g)
                if cachekey not in self.sample_cache:
                    world = self.world(g)
                    actor, size = self.meta['instances'][o['track']][g]
                    local = self.base.transform(world, np.linalg.inv(actor))
                    half = np.maximum(size*self.cfg['expanded_box_factor']*.5, size*.5+self.cfg['minimum_expansion_margin_m'])
                    ground = o['ground_info']
                    good = np.ones(len(world), bool)
                    if ground['valid']:
                        good = world@np.array(ground['normal'])+ground['d'] > self.cfg['ground']['remove_height_m']
                    seeds = np.flatnonzero((abs(local) <= half).all(1) & good)
                    roi = np.flatnonzero((abs(local) <= half+np.array(j.CONFIG['completion']['search_extension_m'])).all(1) & good)
                    ids, info = j.complete_clusters(local[roi], roi, seeds, j.CONFIG['completion']) if ground['valid'] else (seeds, dict(status='ground_unreliable_no_completion'))
                    self.sample_cache[cachekey] = dict(points=world[ids].copy(), ids=ids, actor=actor.copy(), completion=info)
                arm = copy.copy(self.sample_cache[cachekey])
                arm.update(supported=True, source_frame=g, source_time=self.times[g], source_dt_s=self.times[g]-o['time'])
                o['arms'][delta] = arm
            # The delta-zero seeds are freshly completed with J, rather than
            # loaded from a previous scene-specific completed-cache file.
            o['points'], o['ids'] = o['arms'][0]['points'], o['arms'][0]['ids']
            o['target'] = self.actors[o['track']].at(o['camera_time'])
            o['fixed_supported'] = o['target'] is not None
            if o['target'] is None:
                o['target'] = o['group_actor'].copy()
            o['base_world'] = o['points']
        self.choose_frames()
        # N method calls FreshFields.prepare_selected through normal MRO,
        # then performs its unchanged rejected-mask recovery and all-fit setup.
        n.AllFit.prepare_selected(self)
        ld, bd = (self.frame_selection[x]['offset'] for x in ['lidar', 'bbox'])
        self.excluded = []
        retained = []
        for o in self.rows:
            o['exclusion_reasons'] = []
            if o['lidar_frame'] != o['frame']+ld:
                o['exclusion_reasons'].append('actual_lidar_frame_differs_from_selected_offset')
            if o['bbox_frame'] != o['frame']+bd:
                o['exclusion_reasons'].append('actual_bbox_frame_differs_from_selected_offset')
            (self.excluded if o['exclusion_reasons'] else retained).append(o)
        self.rows = retained
        def identity(o):
            return dict(key=o['key'],track=o['track'],rgb_frame=o['frame'],lidar_frame=o['lidar_frame'],bbox_frame=o['bbox_frame'],reasons=o['exclusion_reasons'])
        e.save(self.out/'source_consistency.json',dict(selected_offsets=dict(lidar=ld,bbox=bd),retained=[identity(o) for o in self.rows],excluded=[identity(o) for o in self.excluded]))
        print(self.key, 'PAIRINGS', len(self.rows), 'retained', len(self.excluded), 'excluded', flush=True)



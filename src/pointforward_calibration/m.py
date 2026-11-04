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
from . import l
k,j,e=l.k,l.j,l.e
CFG=json.loads((HERE/'m_config.json').read_text())


def track_mean(rows, value):
    """Equal weight per vehicle, then equal weight per observation."""
    tracks = sorted({o['track'] for o in rows})
    return float(np.mean([np.mean([value(o) for o in rows if o['track'] == tid]) for tid in tracks])) if tracks else None

class Pipeline(l.Pairing):
    def box_arm(self, o, delta):
        return self.meta['instances'][o['track']].get(o['frame'] + delta)

    def choose_frames(self):
        self.frame_selection = {}
        frame_records = []
        for stream in ['lidar', 'bbox']:
            # Independent support populations: a bbox test does not require a
            # usable adjacent LiDAR crop, and vice versa.
            def supported(o):
                if not o['mask_accepted']: return False
                if stream == 'lidar':
                    return o['ground_info']['valid'] and all(a['supported'] and len(a['ids']) >= k.CFG['minimum_points'] for a in o['arms'].values())
                return all(self.box_arm(o, d) is not None for d in CFG['source_offsets'])

            rows = [o for o in self.rows if supported(o)]
            def value(o, d):
                if stream == 'lidar':
                    return self.score(o, o['arms'][d]['points'], o['group_actor'])['point_mask_distance_px']
                actor, _ = self.box_arm(o, d)
                return 1. - self.maskmod.iou(self.hull(actor, o), o['mask'])

            for o in rows:
                frame_records.append(dict(key=o['key'],track=o['track'],camera=o['camera'],split=o['split'],stream=stream,
                    scores={str(d):value(o,d) for d in CFG['source_offsets']}))

            profiles = {}
            for split in ['fit', 'gate']:
                population = [o for o in rows if o['split'] == split]
                profiles[split] = dict(observations=len(population), tracks=len({o['track'] for o in population}),
                    keys=[o['key'] for o in population], scores={str(d): track_mean(population, lambda o: value(o, d)) for d in CFG['source_offsets']})
            fit, gate = profiles['fit'], profiles['gate']; best = 0; reasons = []
            if fit['observations'] < CFG['minimum_selection_views']:
                reasons.append('insufficient_fit_observations')
            else:
                best = min(CFG['source_offsets'], key=lambda d: (fit['scores'][str(d)], abs(d)))
                gain = fit['scores']['0'] - fit['scores'][str(best)]
                threshold = (max(CFG['lidar_minimum_improvement_px'], CFG['lidar_minimum_relative_improvement']*fit['scores']['0'])
                             if stream == 'lidar' else CFG['bbox_minimum_iou_improvement'])
                if best != 0 and gain < threshold: reasons.append('fit_gain_below_threshold')
                if best != 0:
                    if gate['observations'] < CFG['minimum_selection_views']: reasons.append('insufficient_gate_observations')
                    else:
                        limit = CFG['gate_lidar_max_regression_px'] if stream == 'lidar' else CFG['gate_bbox_max_iou_regression']
                        if gate['scores'][str(best)] > gate['scores']['0'] + limit: reasons.append('gate_regression')
            chosen = best if not reasons else 0
            self.frame_selection[stream] = dict(offset=chosen, fit_best=best, retain_zero_reasons=reasons,
                metric='track_balanced_mean_point_mask_distance_px' if stream == 'lidar' else 'track_balanced_one_minus_bbox_hull_mask_iou',
                profiles=profiles, confidence='multiple_fit_tracks' if fit['tracks'] >= 2 else 'single_track_or_insufficient')
        print(self.key, 'FRAME SELECTION', json.dumps(self.frame_selection), flush=True)
        e.save(self.out/'frame_selection.json', self.frame_selection)
        e.save(self.out/'frame_test_metrics.json',dict(records=frame_records,note='Only fit/gate contribute to selection; test and heldout are evaluation only. Scores use fixed RGB f.'))


    def bbox_fit(self, rows, residual, initial):
        # Same K geometry objective and bounds; bbox reference is now the
        # independently selected annotation, not an actor clock transport.
        bounds = np.array(k.CFG['bbox_bounds_m'])
        def loss(b):
            values = []
            for o in rows:
                local = self.base.transform(self.move(o, residual), np.linalg.inv(o['selected_box']))
                d = np.linalg.norm(np.maximum(abs(local-b)-o['size']/2, 0), axis=1)
                values.append(np.mean(k.robust(d/k.CFG['bbox_geometry_scale_m'])))
            return float(np.mean(values))+k.CFG['bbox_prior_weight']*np.sum(((b-initial)/bounds)**2)
        opt = minimize(loss, initial, method='Powell', bounds=list(zip(-bounds,bounds)), options=dict(maxiter=35,xtol=.002,ftol=1e-5))
        return opt.x.tolist()

    def box(self, o, b):
        actor = o['selected_box'].copy(); actor[:3,3] += actor[:3,:3]@np.asarray(b); return actor




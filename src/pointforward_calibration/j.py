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
from . import e
CONFIG=json.loads((HERE/'j_config.json').read_text())


def complete_clusters(local, roi_ids, seed_ids, cfg):
    """Return original seeds + adjacent components; never prune the seed set."""
    seed_mask = np.isin(roi_ids, seed_ids)
    if not len(seed_ids) or not seed_mask.any():
        return seed_ids.copy(), dict(status='no_seed_support', added=0)
    seeds = local[seed_mask]
    spacing = 0.
    if len(seeds) > 1:
        nn = cKDTree(seeds).query(seeds, k=2)[0][:, 1]
        nn = nn[nn > 1e-5]
        if len(nn):
            spacing = float(np.quantile(nn, .75))
    radius = float(np.clip(spacing * cfg['spacing_multiplier'], *cfg['cluster_radius_bounds_m']))
    bridge = float(np.clip(radius * cfg['bridge_radius_multiplier'], *cfg['bridge_radius_bounds_m']))
    pairs = cKDTree(local).query_pairs(radius, output_type='ndarray')
    ii = np.r_[pairs[:, 0], pairs[:, 1]]
    jj = np.r_[pairs[:, 1], pairs[:, 0]]
    graph = coo_matrix((np.ones(len(ii), np.uint8), (ii, jj)), shape=(len(local), len(local))).tocsr()
    ncomp, labels = connected_components(graph, directed=False)
    count = np.bincount(labels, minlength=ncomp)
    seed_count = np.bincount(labels[seed_mask], minlength=ncomp)
    near = cKDTree(seeds).query(local)[0]
    gap = np.full(ncomp, np.inf)
    np.minimum.at(gap, labels, near)
    eligible = count >= cfg['minimum_component_points']
    linked = eligible & ((seed_count > 0) | (gap <= bridge))
    # Only directly adjacent components bridge to ORIGINAL seeds. New points
    # cannot recursively bridge to ever more disconnected clusters.
    selected = np.union1d(seed_ids, roi_ids[linked[labels]])
    added = np.setdiff1d(selected, seed_ids)
    return selected, dict(status='completed' if len(added) else 'no_adjacent_component',
        added=len(added), radius_m=radius, bridge_m=bridge, seed_spacing_p75_m=spacing,
        components=ncomp, accepted_components=int(linked.sum()),
        touching_seed_components=int((linked & (seed_count > 0)).sum()),
        adjacent_disjoint_components=int((linked & (seed_count == 0)).sum()),
        search_roi_points=len(local), selected_reaches_search_boundary=False)

class Completed(e.Comparison):



    def dot_layer(self, rgb, points, o, actor, mask=None, additions=False):
        # Natural RGB brightness; display-only nearest 2px zbuffer. Never used
        # to choose fitting/scoring points. Magenta shows newly recovered IDs.
        arr = rgb.copy()
        uv, z = self.pixels(points, o)
        xy = np.rint(uv).astype(int)
        ok = (z > .1) & (xy[:, 0] >= 0) & (xy[:, 0] < 768) & (xy[:, 1] >= 0) & (xy[:, 1] < 576)
        vis = np.flatnonzero(ok)
        cells = (xy[vis, 1] // 2) * 384 + xy[vis, 0] // 2
        order = np.lexsort((z[vis], cells)); ordered = vis[order]; cs = cells[order]
        ok[:] = False
        if len(ordered): ok[ordered[np.r_[True, cs[1:] != cs[:-1]]]] = True
        q = self.base.transform(points, np.linalg.inv(actor))
        inside = (abs(q) <= o['size'] * .5).all(1)
        groups = [(~inside, (0, 235, 255)), (inside, (255, 155, 0))]
        if additions:
            new = ~np.isin(o['ids'], o['seed_ids'])
            groups = [(~new, (255, 155, 0)), (new, (255, 30, 220))]
        for membership, color in groups:
            dots = np.zeros((576, 768), np.uint8)
            pix = xy[ok & membership]; dots[pix[:, 1], pix[:, 0]] = 255
            halo = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
            body = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
            arr[halo] = 0; arr[body] = color
        if mask is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(arr, contours, -1, (100, 255, 80), 1)
        return self.draw.box_layer(arr, self.base.box_uv(actor, o['camera_pose'], o['size'], o['K']))


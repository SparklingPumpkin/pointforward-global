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
from .geometry import SIGNS, transform, box_uv
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
YELLOW=(255,220,0)
ORANGE=(255,110,35)
CYAN=(0,225,255)


def fonts(size):
    return (ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size),
            ImageFont.truetype(FONT, size))

def text(draw, position, message, size):
    latin, cjk = fonts(size)
    x, y = position
    for char in message:
        font = latin if ord(char) < 128 else cjk
        draw.text((x, y), char, font=font, fill='white')
        x += font.getlength(char)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def box_layer(rgb, uv):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1:
                draw.line([tuple(uv[i]), tuple(uv[j])], fill=YELLOW, width=2)
    return np.asarray(image).copy()

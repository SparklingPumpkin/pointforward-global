"""Depth-guided sky repair. True means sky; input arrays are never modified."""
import numpy as np

RULE = dict(local_window_px=31, background_depth_percentile=75, maximum_foreground_background_depth_ratio=.6, minimum_inverse_depth_gap_per_m=.0005, minimum_component_pixels=4, maximum_valid_depth_m=10000)

def repair(z,sky):
    from scipy.ndimage import percentile_filter,label
    valid=np.isfinite(z)&(z>0)&(z<=RULE['maximum_valid_depth_m'])
    fallback=float(np.nanmedian(z[valid&sky])) if (valid&sky).any() else 1000.
    filled=np.where(valid,z,fallback)
    background=percentile_filter(filled,percentile=RULE['background_depth_percentile'],size=RULE['local_window_px'],mode='reflect')
    candidate=sky&valid&(z<RULE['maximum_foreground_background_depth_ratio']*background)&((1/filled-1/background)>RULE['minimum_inverse_depth_gap_per_m'])
    labs,_=label(candidate,structure=np.ones((3,3)));sizes=np.bincount(labs.ravel());keep=sizes>=RULE['minimum_component_pixels'];keep[0]=False
    removed=keep[labs]
    return sky&~removed,removed

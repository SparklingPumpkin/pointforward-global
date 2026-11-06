"""Metric scene sizing and optional radial mapping; independent of reconstruction."""
import numpy as np
import torch


def measure_scene(point_chunks, camera_positions, scene_multiplier=1.5, sky_multiplier=2.):
    """Union AABB midpoint and exact maximum distance R, in common metric space.

    point_chunks: iterable of (N,3) arrays, e.g. world-coordinate LiDAR sweeps.
    camera_positions: (V,3) array in the same coordinate frame.
    Arrays are retained for the exact second radius pass, as in the research code.
    """
    if not np.isfinite([scene_multiplier, sky_multiplier]).all() or not 1 <= scene_multiplier < sky_multiplier:
        raise ValueError('Require finite 1 <= scene multiplier < sky multiplier')
    cameras = np.asarray(camera_positions, dtype=np.float64)
    if cameras.ndim != 2 or cameras.shape[1] != 3 or not len(cameras) or not np.isfinite(cameras).all():
        raise ValueError('Expected finite nonempty camera positions (N,3)')
    chunks = [cameras]
    count = 0
    for points in point_chunks:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite LiDAR points (N,3)')
        if len(points):
            chunks.append(points)
            count += len(points)
    if count == 0:
        raise ValueError('Metric sizing requires measured points, not only camera centers')
    lower = np.min([points.min(axis=0) for points in chunks], axis=0)
    upper = np.max([points.max(axis=0) for points in chunks], axis=0)
    center = (lower+upper)/2
    radius = max(float(np.linalg.norm(points-center, axis=1).max()) for points in chunks)
    if radius <= 0:
        raise ValueError('Degenerate zero-radius scene')
    scene_radius, sky_radius = scene_multiplier*radius, sky_multiplier*radius
    return dict(center=center.tolist(), measured_radius_m=radius, measured_diameter_m=2*radius,
        scene_boundary_multiplier=scene_multiplier, scene_boundary_radius_m=scene_radius,
        scene_boundary_diameter_m=2*scene_radius, sky_multiplier=sky_multiplier,
        sky_radius_m=sky_radius, sky_diameter_m=2*sky_radius,
        compression_upper_radius_m=scene_radius+.2*(sky_radius-scene_radius),
        bounds_m=[lower.tolist(), upper.tolist()], lidar_point_count=count, camera_count=len(cameras),
        definition='Union AABB midpoint and exact maximum distance R of raw measured points and camera centers in common metric space',
        radial_mapping='Optional downstream mapping only; preprocessing does not modify any Gaussian payload')

def compress(points, center, lower, upper, maximum):
    offsets = points-center
    radius = torch.linalg.vector_norm(offsets, dim=-1)
    selected = radius > lower
    factor = min(1., (upper-lower)/(maximum-lower)) if maximum > lower else 1.
    result = points.clone()
    mapped = lower+(radius[selected]-lower)*factor
    result[selected] = center+offsets[selected]*(mapped/radius[selected])[:, None]
    return result

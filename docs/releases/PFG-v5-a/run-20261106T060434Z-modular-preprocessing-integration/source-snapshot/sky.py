"""Measured finite sky: shared two-pass appearance, explicit unobserved fill."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class SkyView:
    """One source view; directions are unit world rays (H,W,3)."""
    frame: int
    camera: int
    rgb: np.ndarray
    sky_mask: np.ndarray
    ego_mask: np.ndarray
    camera_position: torch.Tensor
    directions: torch.Tensor


def build_sky(views_factory, sizing, pixel_stride=2, settings=None, coverage=(512, 256)):
    """Build reusable sky state from a repeatable source-view iterable factory.

    Two source passes are required. No novel-view RGB, training or renderer is
    used. Unobserved directions are explicitly marked as extrapolated.
    """
    from scipy.ndimage import binary_erosion
    settings = settings or SkySettings()
    if pixel_stride < 1 or settings.longitude_cells < 1 or settings.latitude_cells < 1:
        raise ValueError('Positive sampling stride and grid sizes required')
    geometry = dict(center=sizing['center'], radius_m=sizing['sky_radius_m'],
        diameter_m=2*sizing['sky_radius_m'], scene_scale_m=sizing['measured_diameter_m'],
        diameter_to_scene_scale=sizing['sky_multiplier'])
    center = np.asarray(geometry['center'], dtype=np.float64)
    radius = float(geometry['radius_m'])
    accumulator = DirectionAccumulator(settings)
    initial_color = None
    records, first_ids = [], []
    for pass_index in range(2):
        if pass_index:
            keys, colors = accumulator.result()
            initial_color = np.full_like(accumulator.rgb_sum, np.nan)
            initial_color[keys] = colors
            accumulator = DirectionAccumulator(settings)
        ids = []
        for view in views_factory():
            rgb = np.asarray(view.rgb)
            sky = np.asarray(view.sky_mask, dtype=bool)
            ego = np.asarray(view.ego_mask, dtype=bool)
            if rgb.shape != (*sky.shape, 3) or ego.shape != sky.shape or tuple(view.directions.shape) != rgb.shape:
                raise ValueError('Source RGB/mask/ray resolution mismatch')
            if np.linalg.norm(np.asarray(view.camera_position)-center)+settings.translation_budget_m >= radius:
                raise ValueError('Cameras plus translation budget exceed measured sky shell')
            directions = shell_directions(view.camera_position[None], view.directions[None], geometry)[0].numpy()
            mask = binary_erosion(sky & ~ego, iterations=settings.boundary_erosion) if settings.boundary_erosion else sky & ~ego
            sampled = np.zeros_like(mask)
            sampled[::pixel_stride, ::pixel_stride] = True
            mask &= sampled
            rays, colors = directions[mask], rgb[mask]
            weights = None
            if initial_color is not None:
                reference = initial_color[direction_keys(rays, settings)]
                residual = np.linalg.norm(colors-reference, axis=-1)
                weights = np.minimum(1., .08/np.maximum(residual, 1e-8))
                if not np.isfinite(weights).all():
                    raise ValueError('Second-pass sky source observations changed')
            accumulator.add(rays, colors, weights)
            ids.append((int(view.frame), int(view.camera)))
            if pass_index == 0:
                records.append(dict(frame=int(view.frame), camera=int(view.camera), sky_pixels=int(sky.sum()),
                    ego_pixels=int(ego.sum()), used_observations=int(mask.sum())))
        if not pass_index:
            first_ids = ids
        elif ids != first_ids:
            raise ValueError('Source identity/order must match across both sky passes')
    keys, colors = accumulator.result()
    if not len(keys):
        raise ValueError('No valid sky observations')
    directions = key_directions(keys, settings)
    latitude_cosine = np.sqrt(np.maximum(0, 1-directions[:, 2]**2))
    tangent_scale = radius*.7*np.maximum(2*np.pi/settings.longitude_cells*latitude_cosine,
                                        np.pi/settings.latitude_cells)
    scales = np.stack((tangent_scale, tangent_scale, tangent_scale*.01), axis=-1)
    quats = np.stack((1+directions[:, 2], -directions[:, 1], directions[:, 0], np.zeros(len(keys))), axis=-1)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    colors = np.clip(colors, 1e-4, 1-1e-4)
    settings_dict = asdict(settings)
    settings_dict.pop('minimum_radius_m')
    settings_dict.pop('parallax_budget_px')
    settings_dict['diameter_to_scene_scale'] = sizing['sky_multiplier']
    metadata = dict(geometry, gaussian_count=len(keys), settings=settings_dict,
        representation='Fixed finite sphere, separately alpha-composited behind scene',
        appearance='Two-pass Huber-weighted source RGB, no target lookup', geometry_trainable=False,
        synthetic_depth_prompts=0, pixel_stride=pixel_stride)
    state = dict(means=torch.from_numpy(center+radius*directions).float(), scales=torch.from_numpy(scales).float(),
        quats=torch.from_numpy(quats).float(), opacities=torch.full((len(keys),), .995),
        rgb_logits=torch.from_numpy(np.log(colors/(1-colors))).float(), keys=torch.from_numpy(keys), metadata=metadata)
    state = complete_sphere(state, *coverage)
    return state, records

@dataclass(frozen=True)
class SkySettings:
    longitude_cells: int = 2048
    latitude_cells: int = 1024
    boundary_erosion: int = 4
    translation_budget_m: float = 3.0
    parallax_budget_px: float = 0.25
    minimum_radius_m: float = 10000.0

def direction_keys(directions, settings=SkySettings()):
    directions = np.asarray(directions, dtype=np.float64)
    norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    if not np.isfinite(directions).all() or np.any(norm <= 0):
        raise ValueError('Sky directions must be finite and nonzero')
    directions = directions / norm
    longitude = np.arctan2(directions[..., 1], directions[..., 0])
    latitude = np.arcsin(np.clip(directions[..., 2], -1, 1))
    column = np.floor((longitude + np.pi) / (2 * np.pi) * settings.longitude_cells).astype(np.int64)
    row = np.floor((latitude + np.pi / 2) / np.pi * settings.latitude_cells).astype(np.int64)
    column %= settings.longitude_cells
    row = np.clip(row, 0, settings.latitude_cells - 1)
    return row * settings.longitude_cells + column

def key_directions(keys, settings=SkySettings()):
    keys = np.asarray(keys, dtype=np.int64)
    longitude = ((keys % settings.longitude_cells) + .5) * (2 * np.pi / settings.longitude_cells) - np.pi
    latitude = ((keys // settings.longitude_cells) + .5) * (np.pi / settings.latitude_cells) - np.pi / 2
    return np.stack((np.cos(latitude) * np.cos(longitude),
                     np.cos(latitude) * np.sin(longitude), np.sin(latitude)), axis=-1)

class DirectionAccumulator:
    def __init__(self, settings=SkySettings()):
        self.settings = settings
        count = settings.longitude_cells * settings.latitude_cells
        self.weight = np.zeros(count, dtype=np.float64)
        self.rgb_sum = np.zeros((count, 3), dtype=np.float64)

    def add(self, directions, colors, weights=None):
        keys = direction_keys(directions, self.settings).reshape(-1)
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(keys) != len(colors) or not np.isfinite(colors).all():
            raise ValueError('Sky color observations must match directions and be finite')
        weights = np.ones(len(keys)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(keys) or np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError('Invalid sky observation weights')
        np.add.at(self.weight, keys, weights)
        np.add.at(self.rgb_sum, keys, colors * weights[:, None])

    def result(self):
        keys = np.flatnonzero(self.weight > 0)
        return keys, self.rgb_sum[keys] / self.weight[keys, None]

def shell_directions(camera_positions, directions, geometry):
    center = directions.new_tensor(geometry['center'])
    offset = camera_positions[:,None,None,:] - center
    radius = float(geometry['radius_m'])
    if bool((torch.linalg.vector_norm(offset,dim=-1)>=radius).any()):
        raise ValueError('Camera outside finite sky sphere')
    rays = torch.nn.functional.normalize(directions,dim=-1)
    projection = (offset*rays).sum(-1,keepdim=True)
    discriminant = projection.square()+radius*radius-offset.square().sum(-1,keepdim=True)
    travel = -projection+discriminant.sqrt()
    radial = (offset+travel*rays)/radius
    if not torch.isfinite(radial).all() or bool((travel<=0).any()):
        raise ValueError('Invalid ray/sphere intersection')
    return torch.nn.functional.normalize(radial,dim=-1)

def complete_sphere(state, columns=512, rows=256):
    settings = state['metadata']['settings']
    native_columns, native_rows = settings['longitude_cells'], settings['latitude_cells']
    longitude = (np.arange(columns) + .5) * (2 * np.pi / columns) - np.pi
    latitude = (np.arange(rows) + .5) * (np.pi / rows) - np.pi / 2
    longitude, latitude = np.meshgrid(longitude, latitude)
    directions = np.column_stack((np.cos(latitude).ravel() * np.cos(longitude).ravel(),
                                  np.cos(latitude).ravel() * np.sin(longitude).ravel(), np.sin(latitude).ravel()))
    center = np.asarray(state['metadata']['center'])
    radius = state['metadata']['radius_m']
    observed_directions = (state['means'].numpy().astype(np.float64) - center) / radius
    distances, neighbors = cKDTree(observed_directions).query(directions, k=1)
    missing = distances > (2 * np.pi / native_columns * .75)
    directions = directions[missing]
    neighbors = neighbors[missing]
    angular = np.maximum(2 * np.pi / columns * np.sqrt(1 - directions[:, 2] ** 2), np.pi / rows)
    tangent = radius * .9 * angular
    scales = np.column_stack((tangent, tangent, tangent * .01))
    quats = np.column_stack((1 + directions[:, 2], -directions[:, 1], directions[:, 0], np.zeros(len(directions))))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    column = np.floor((np.arctan2(directions[:, 1], directions[:, 0]) + np.pi) / (2 * np.pi) * native_columns).astype(np.int64) % native_columns
    row = np.clip(np.floor((np.arcsin(directions[:, 2]) + np.pi / 2) / np.pi * native_rows).astype(np.int64), 0, native_rows - 1)
    extra = dict(means=torch.from_numpy(center + radius * directions).float(),
                 scales=torch.from_numpy(scales).float(), quats=torch.from_numpy(quats).float(),
                 opacities=torch.full((len(directions),), .999), keys=torch.from_numpy(row * native_columns + column),
                 rgb_logits=state['rgb_logits'][torch.from_numpy(neighbors)])
    result = {name: torch.cat((state[name], extra[name]), dim=0) for name in extra}
    observed_count = len(state['means'])
    result['observed'] = torch.cat((torch.ones(observed_count, dtype=torch.bool), torch.zeros(len(directions), dtype=torch.bool)))
    metadata = dict(state['metadata'])
    metadata.update(gaussian_count=len(result['means']), observed_gaussian_count=observed_count,
        extrapolated_gaussian_count=len(directions), coverage_grid=[columns, rows],
        coverage_policy='Full-sphere coarse grid where original high-resolution observations do not cover; nearest observed direction RGB extrapolation',
        extrapolation_note='Unobserved sky appearance is inferred; not new ground-truth observation or recovered building geometry',
        maximum_fill_nearest_angle_degrees=float(np.degrees(2 * np.arcsin(np.minimum(distances.max() / 2, 1)))))
    result['metadata'] = metadata
    for name, value in result.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite completed sky {name}')
    return result

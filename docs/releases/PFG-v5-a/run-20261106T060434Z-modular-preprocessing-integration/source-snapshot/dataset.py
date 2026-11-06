"""PointForward processed-dataset adapter; numerical APIs do not depend on it."""
from pathlib import Path
import sys
import types

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'docs/releases/VR-v1-s/source'


def configure_config(config, scene=None, geometry_dir=None):
    """Copy configuration and explicitly select any calibrated input."""
    import copy
    result = copy.deepcopy(config)
    if scene is not None:
        result['data']['train_scenes'] = result['data']['val_scenes'] = [scene]
        result['data'].pop('split_file', None)
    if geometry_dir is not None:
        result['data']['geometry_directory'] = str(Path(geometry_dir).resolve())
    return result


class PointForwardViews:
    def __init__(self, config, frames=None, mask_dir=None):
        # Only the dataset CLI uses the bundled frozen implementation. Array
        # APIs do not change another project's module import search path.
        for path in reversed([SOURCE, PROJECT/'src', SOURCE/'InfiniDepth']):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        if config['data'].get('geometry_directory'):
            from pointforward_calibration.consumer import install
            install()
        from scripts.train import build_dataset
        self.dataset = build_dataset(config, 'val')
        if len(self.dataset.scenes) != 1:
            raise ValueError('Exactly one scene per preprocessing invocation')
        self.scene = self.dataset.scenes[0]
        all_frames = list(self.scene['frames'])
        self.frames = list(frames) if frames is not None else all_frames
        if not self.frames or len(set(self.frames)) != len(self.frames) or not set(self.frames) <= set(all_frames):
            raise ValueError('Frames must be unique supported source frames')
        self.slots = {frame: index for index, frame in enumerate(all_frames)}
        self.times = self.dataset._normalised_times(all_frames)
        self.origin = np.linalg.inv(self.dataset._ego_pose(self.scene, all_frames[0]))
        self.origin_frame = all_frames[0]
        self.mask_dir = Path(mask_dir).resolve() if mask_dir else None
        self.config = config

    def collect(self, frame, lidar=False):
        batch = self.dataset._collect(self.scene, [frame], self.slots, self.times,
            self.dataset.resolution, self.origin, lidar=lidar)
        if 'ego_mask' not in batch:
            # Frozen adapters omit this key when ego exclusion is disabled or
            # the dataset has no ego masks. Sky accumulation still needs an
            # explicit mask; use available dataset masks, otherwise empty.
            ego = []
            for camera in self.dataset.cameras:
                if self.scene.get('has_ego_mask', False):
                    relative = f'ego_masks/{frame:03d}_{camera}.png'
                    if not (Path(self.scene['path'])/relative).exists():
                        relative = f'ego_masks/{camera}.png'
                    value, _, _ = self.dataset._load_image(self.scene, relative, self.dataset.resolution, gray=True)
                else:
                    value = torch.zeros((1, *self.dataset.resolution))
                ego.append(value)
            batch['ego_mask'] = torch.stack(ego)
        if self.mask_dir:
            from PIL import Image
            masks = []
            for camera in self.dataset.cameras:
                with Image.open(self.mask_dir/f'{frame:03d}_{camera}.png') as image:
                    a = np.asarray(image).copy()
                if tuple(a.shape) != tuple(self.dataset.resolution):
                    raise ValueError('Repaired mask resolution mismatch')
                masks.append(torch.from_numpy((a > 127).astype('float32'))[None])
            batch['sky_mask'] = torch.stack(masks)
        return batch

    def sky_views(self):
        from pointforward.models.query import camera_rays
        from .sky import SkyView
        for frame in self.frames:
            batch = self.collect(frame)
            h, w = self.dataset.resolution
            _, directions = camera_rays(h, w, batch['intrinsics'], batch['cam_to_world'])
            for index, camera in enumerate(self.dataset.cameras):
                yield SkyView(frame, camera, batch['images'][index].permute(1, 2, 0).numpy(),
                    batch['sky_mask'][index, 0].numpy() > .5,
                    batch['ego_mask'][index, 0].numpy() > .5,
                    batch['cam_to_world'][index, :3, 3], directions[index])

    def measured_range(self, scene_multiplier=1.5, sky_multiplier=2.):
        from .range import measure_scene
        from .io import identity
        cameras = np.stack([self.dataset._cam_to_world(self.scene, frame, cam, self.origin)[:3, 3]
            for frame in self.frames for cam in self.dataset.cameras])
        inputs = []
        geometry = self.config['data'].get('geometry_directory')
        if geometry:
            for name in ('application_manifest.json', 'camera_poses.pt'):
                path = Path(geometry)/name
                if path.exists():
                    inputs.append(identity(path))
        def chunks():
            for frame in self.frames:
                path = Path(self.scene['path'])/'lidar'/f'{frame:03d}.bin'
                points = self.dataset.read_lidar(str(path))
                if points is None or not len(points):
                    raise ValueError('Missing/unsupported measured LiDAR: '+str(path))
                transform = self.origin@self.dataset._ego_pose(self.scene, frame)
                inputs.extend([identity(path), identity(Path(self.scene['path'])/'ego_pose'/f'{frame:03d}.txt')])
                yield points.astype(np.float64)@transform[:3, :3].T+transform[:3, 3]
        sizing = measure_scene(chunks(), cameras, scene_multiplier, sky_multiplier)
        sizing.update(scene=self.scene['name'], frames=self.frames, inputs=inputs,
            coordinate_frame='first-supported-frame ego meters', origin_frame=self.origin_frame,
            origin_world_to_ego=self.origin.tolist())
        return sizing


def build_depth(config, device='cuda:0'):
    """Fresh generic InfiniDepth, retaining positive uncapped reciprocal depth."""
    from pointforward.models.depth import FrozenInfiniDepth
    depth = config['depth']
    model = FrozenInfiniDepth(**{k: depth[k] for k in ('repo_root', 'checkpoint', 'variant', 'encoder')}).eval()
    def convert(self, pred):
        pred = pred.float()
        positive = torch.isfinite(pred) & (pred > 0)
        return torch.where(positive, torch.where(positive, pred, torch.ones_like(pred)).reciprocal(),
                           torch.full_like(pred, float('nan'))), pred
    model.model._to_depth_disparity = types.MethodType(convert, model.model)
    return model.to(device)

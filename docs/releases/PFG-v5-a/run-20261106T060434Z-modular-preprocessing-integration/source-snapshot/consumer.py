"""Opt-in consumption of exported masks; no sky-renderer or optimizer patch."""
from pathlib import Path
import types


def apply_masks(dataset, directory):
    """Attach pixel-aligned masks to just this dataset instance, not its class."""
    from PIL import Image
    import numpy as np
    import torch
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Missing repaired sky mask directory: '+str(directory))
    original = dataset._collect
    def collect(self, meta, frames, slot_of, norm_times, resolution, world_from, lidar):
        batch = original(meta, frames, slot_of, norm_times, resolution, world_from, lidar)
        masks = []
        for frame in frames:
            for camera in self.cameras:
                with Image.open(directory/f'{frame:03d}_{camera}.png') as image:
                    arr = np.asarray(image).copy()
                if tuple(arr.shape) != tuple(resolution):
                    raise ValueError('Repaired mask resolution mismatch; no silent resizing')
                masks.append(torch.from_numpy((arr > 127).astype('float32'))[None])
        updated = torch.stack(masks)
        if updated.shape != batch['sky_mask'].shape:
            raise ValueError('Repaired mask ordering mismatch')
        batch['sky_mask'] = updated
        return batch
    dataset._collect = types.MethodType(collect, dataset)
    return dataset


def install():
    """Enable config.data.sky_mask_directory for explicitly wrapped scripts."""
    from pointforward_calibration.consumer import install as install_calibration
    install_calibration()
    from scripts import train
    if getattr(train.build_dataset, '_preprocessing_masks', False):
        return
    original = train.build_dataset
    def build_dataset(config, split):
        dataset = original(config, split)
        path = config['data'].get('sky_mask_directory')
        return apply_masks(dataset, path) if path else dataset
    build_dataset._preprocessing_masks = True
    train.build_dataset = build_dataset

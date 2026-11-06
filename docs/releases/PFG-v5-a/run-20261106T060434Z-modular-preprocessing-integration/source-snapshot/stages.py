"""File-producing stages, with explicit optional input dependencies."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .io import empty_output, identity, save


def run_depth_mask(config, out, frames=None, geometry_dir=None, depth_dir=None,
                   device='cuda:0', workers=8):
    """New depth inference by default; existing depth is explicit cached input."""
    from .dataset import PointForwardViews, configure_config, build_depth
    from .depth_mask import RULE, repair
    out = empty_output(out)
    start = time.perf_counter()
    timer = dict(scope='stage', start=start, status='running',
        input_boundary='before_dataset_depth_model_loading', output_boundary='depth_masks_diagnostics_written',
        cached_depth_reused=bool(depth_dir))
    try:
        if workers < 1:
            raise ValueError('workers must be positive')
        cfg = configure_config(config, geometry_dir=geometry_dir)
        views = PointForwardViews(cfg, frames)
        for name in ('masks', 'depth-cache', 'diagnostics'):
            (out/name).mkdir()
        np.random.seed(0)
        torch.manual_seed(0)
        model = build_depth(cfg, device) if depth_dir is None else None
        representative = {views.frames[0], views.frames[len(views.frames)//2], views.frames[-1]}
        rows, pending = [], set()
        depth_inputs = []
        def process(frame, cam, z, sky, rgb):
            if z.shape != sky.shape:
                raise ValueError('Depth and sky mask resolution mismatch')
            repaired, removed = repair(z, sky)
            stem = f'{frame:03d}_{cam}'
            Image.fromarray(repaired.astype('uint8')*255).save(out/'masks'/f'{stem}.png')
            np.save(out/'depth-cache'/f'{stem}.npy', z)
            if frame in representative and cam == views.dataset.cameras[0]:
                before, after = rgb.copy(), rgb.copy()
                before[sky] = .55*before[sky]+.45*np.array([1, 0, 0])
                after[repaired] = .55*after[repaired]+.45*np.array([1, 0, 0])
                after[removed] = .5*after[removed]+.5*np.array([0, 1, 0])
                Image.fromarray(np.uint8(np.clip(np.concatenate([rgb, before, after], axis=1), 0, 1)*255)).save(
                    out/'diagnostics'/f'mask_full_f{frame:04d}.jpg', quality=95)
            return dict(frame=frame, camera=cam, original_sky=int(sky.sum()), removed=int(removed.sum()),
                new_sky=int(repaired.sum()), mask=identity(out/'masks'/f'{stem}.png'),
                depth=identity(out/'depth-cache'/f'{stem}.npy'))
        with torch.inference_mode(), ThreadPoolExecutor(max_workers=workers) as pool:
            for frame in views.frames:
                batch = views.collect(frame, lidar=depth_dir is None)
                if model is not None:
                    h, w = views.dataset.resolution
                    depths = model(batch['images'].to(device), h, w,
                        sparse_depth=batch['sparse_depth'].to(device), sparse_mask=batch['sparse_mask'].to(device)).cpu().numpy()
                else:
                    paths = [Path(depth_dir)/f'{frame:03d}_{cam}.npy' for cam in views.dataset.cameras]
                    depths = [np.load(path, allow_pickle=False) for path in paths]
                    depth_inputs.extend(identity(path) for path in paths)
                for index, cam in enumerate(views.dataset.cameras):
                    pending.add(pool.submit(process, frame, cam, depths[index].copy(),
                        batch['sky_mask'][index, 0].numpy() > .5,
                        batch['images'][index].permute(1, 2, 0).numpy()))
                while len(pending) >= 2*workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    rows.extend(f.result() for f in done)
                print('depth-mask', frame, flush=True)
            rows.extend(f.result() for f in pending)
        rows.sort(key=lambda row: (row['frame'], row['camera']))
        manifest = dict(module='depth-mask', status='completed', scene=views.scene['name'], frames=views.frames,
            cameras=list(views.dataset.cameras), resolution=list(views.dataset.resolution),
            coordinate_frame='dataset image pixels after source crop/resize', rule=RULE, records=rows,
            removed_total=sum(row['removed'] for row in rows),
            cached_depth_reused=bool(depth_dir), depth_inputs=depth_inputs,
            geometry_input=str(geometry_dir or cfg['data'].get('geometry_directory', '')),
            generic_depth_checkpoint=identity(cfg['depth']['checkpoint']) if model is not None else None,
            partial_export=frames is not None,
            depth_conversion='Positive disparity reciprocal without the old near/far clamp; invalid -> NaN')
        save(out/'mask_manifest.json', manifest)
        timer['status'] = 'completed'
        return manifest
    except BaseException as exc:
        timer['error'] = repr(exc)
        timer['status'] = 'failed'
        raise
    finally:
        timer['end'] = time.perf_counter()
        timer['seconds'] = timer['end']-start
        save(out/'preprocess_timing.json', timer)


def run_scene_sky(config, out, frames=None, geometry_dir=None, mask_dir=None,
                  scene_multiplier=1.5, sky_multiplier=2., pixel_stride=2):
    from .dataset import PointForwardViews, configure_config
    from .sky import build_sky
    out = empty_output(out)
    start = time.perf_counter()
    timer = dict(scope='stage', start=start, status='running', stages_seconds={},
        input_boundary='before_dataset_measurement', output_boundary='measured_range_sky_state_written')
    try:
        cfg = configure_config(config, geometry_dir=geometry_dir)
        views = PointForwardViews(cfg, frames, mask_dir)
        at = time.perf_counter()
        sizing = views.measured_range(scene_multiplier, sky_multiplier)
        save(out/'measured_range.json', sizing)
        timer['stages_seconds']['range_measurement'] = time.perf_counter()-at
        at = time.perf_counter()
        state, records = build_sky(views.sky_views, sizing, pixel_stride)
        state['metadata'].update(scene=views.scene['name'], frame_count=len(views.frames), frames=views.frames,
            cameras=list(views.dataset.cameras), resolution=list(views.dataset.resolution),
            origin_frame=views.origin_frame, coordinate_frame='first-supported-frame ego meters',
            mask_directory=str(mask_dir) if mask_dir else None, partial_export=frames is not None)
        torch.save(state, out/'shared_sky.pt')
        # Portable, pickle-free interchange for projects that do not use Torch.
        np.savez_compressed(out/'sky_arrays.npz', **{name: value.numpy() for name, value in state.items()
            if isinstance(value, torch.Tensor)})
        restored = torch.load(out/'shared_sky.pt', map_location='cpu', weights_only=True)
        if any(not torch.equal(value, restored[name]) for name, value in state.items() if isinstance(value, torch.Tensor)):
            raise ValueError('Sky tensor readback mismatch')
        timer['stages_seconds']['sky_build_and_export'] = time.perf_counter()-at
        manifest = dict(state['metadata'], module='scene-sky', status='completed', records=records,
            geometry_input=str(geometry_dir or cfg['data'].get('geometry_directory', '')),
            raw_masks_used=mask_dir is None, exact_tensor_readback=True,
            output_format='Torch tensor dictionary and identical numpy sky_arrays.npz; metadata in this manifest',
            gaussian_payload_modified=False)
        save(out/'sky_manifest.json', manifest)
        timer['status'] = 'completed'
        return manifest
    except BaseException as exc:
        timer.update(status='failed', error=repr(exc))
        raise
    finally:
        timer['end'] = time.perf_counter()
        timer['seconds'] = timer['end']-start
        save(out/'preprocess_timing.json', timer)

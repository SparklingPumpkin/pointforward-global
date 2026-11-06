"""Run any subset of the three independent preprocessing stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import MODULES
from .io import empty_output, identity, save

PROJECT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modules', nargs='+', choices=MODULES, default=['lim-r'])
    parser.add_argument('--scene', help='Override config scene selection; otherwise require one configured scene')
    parser.add_argument('--geometry-dir', type=Path, help='Explicit previously generated calibration; not implicit cold input')
    parser.add_argument('--mask-dir', type=Path, help='Explicit sky masks for scene-sky when depth-mask is not selected')
    parser.add_argument('--depth-dir', type=Path, help='Explicit cached .npy depth for depth-mask; default predicts fresh depth')
    parser.add_argument('--frames', type=int, nargs='+', help='Diagnostic partial export, explicitly marked; not a full scene result')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES for fresh depth and LIM-r subprocess')
    parser.add_argument('--sam-python', default=sys.executable)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--scene-multiplier', type=float, default=1.5)
    parser.add_argument('--sky-multiplier', type=float, default=2.)
    parser.add_argument('--pixel-stride', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    selected = [name for name in MODULES if name in args.modules]
    if len(selected) != len(args.modules):
        parser.error('Select each module at most once')
    if 'lim-r' in selected and args.geometry_dir:
        parser.error('Choose new lim-r or explicit existing geometry-dir, not both')
    if args.mask_dir and ('scene-sky' not in selected or 'depth-mask' in selected):
        parser.error('mask-dir is for independent scene-sky; selected depth-mask supplies new masks')
    if args.depth_dir and 'depth-mask' not in selected:
        parser.error('depth-dir requires depth-mask')
    if args.workers < 1 or args.pixel_stride < 1:
        parser.error('workers and pixel-stride must be positive')
    import math
    if not all(math.isfinite(x) for x in (args.scene_multiplier, args.sky_multiplier)) or not 1 <= args.scene_multiplier < args.sky_multiplier:
        parser.error('Require finite 1 <= scene multiplier < sky multiplier')
    out = args.out.resolve()
    geometry = out/'lim-r' if 'lim-r' in selected else args.geometry_dir
    mask = out/'depth-mask/masks' if 'depth-mask' in selected else args.mask_dir
    plan = dict(modules=selected, config=str(args.config.resolve()), out=str(out), scene=args.scene,
        geometry_directory=str(geometry.resolve()) if geometry else 'as explicitly configured, or raw input',
        sky_mask_directory=str(mask.resolve()) if mask else 'original dataset masks',
        cached_depth_directory=str(args.depth_dir.resolve()) if args.depth_dir else None,
        partial_frames=args.frames, gaussian_construction=False, optimization=False,
        order='lim-r, then depth-mask, then scene-sky; only selected modules run',
        outputs={name: str(out/name) for name in selected})
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    # Set visibility before any tensor module import.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import yaml
    out = empty_output(out)
    start = time.perf_counter()
    timer = dict(scope='stage', start=start, status='running', stages=[],
        input_boundary='before_config_dataset_and_model_reads', output_boundary='selected_preprocessing_outputs_written',
        reconstruction_included=False, explicit_reused_inputs=[])
    try:
        cfg = yaml.safe_load(args.config.read_text())
        scenes = cfg['data'].get('val_scenes', cfg['data'].get('train_scenes', []))
        scene = args.scene or (scenes[0] if len(scenes) == 1 else None)
        if scene is None:
            raise ValueError('Specify one scene via --scene or config val_scenes')
        from .dataset import configure_config
        cfg = configure_config(cfg, str(scene), args.geometry_dir)
        import copy
        raw_cfg = copy.deepcopy(cfg)
        if 'lim-r' in selected:
            # A fresh calibration must not consume the configuration's old calibration.
            raw_cfg['data'].pop('geometry_directory', None)
        input_config = out/'input_config.yaml'
        input_config.write_text(yaml.safe_dump(raw_cfg, sort_keys=False))
        sources = [identity(path) for path in sorted(Path(__file__).parent.glob('*.py'))]
        sources.append(identity(Path(__file__).parent/'provenance.json'))
        reused = [dict(kind='calibration', path=str(args.geometry_dir.resolve()))] if args.geometry_dir else []
        if not args.geometry_dir and 'lim-r' not in selected and cfg['data'].get('geometry_directory'):
            reused.append(dict(kind='calibration', path=cfg['data']['geometry_directory']))
        for name, path in [('depth', args.depth_dir), ('mask', args.mask_dir)]:
            if path:
                reused.append(dict(kind=name, path=str(path.resolve())))
        timer['explicit_reused_inputs'] = reused
        save(out/'input_manifest.json', dict(plan=plan, config=identity(args.config), code=sources,
            raw_data_root=cfg['data']['root'], scene=str(scene), explicit_reused_inputs=reused,
            meaning='Preprocessing only, no end-to-end reconstruction performance claim'))
        for name in selected:
            at = time.perf_counter()
            row = dict(module=name, start=at, status='running')
            timer['stages'].append(row)
            save(out/'preprocess_timing.json', timer)
            try:
                if name == 'lim-r':
                    cmd = [args.sam_python, str(PROJECT/'scripts/preprocess_lidar_bbox.py'), '--config', str(input_config),
                        '--out', str(out/name), '--scene', str(scene)]
                    if args.frames:
                        cmd += ['--apply-frames', *map(str, args.frames)]
                    row['command'] = cmd
                    with (out/'lim-r.log').open('w') as log:
                        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                    cfg['data']['geometry_directory'] = str(out/name)
                elif name == 'depth-mask':
                    from .stages import run_depth_mask
                    run_depth_mask(cfg, out/name, args.frames, geometry, args.depth_dir, workers=args.workers)
                    cfg['data']['sky_mask_directory'] = str(out/name/'masks')
                else:
                    from .stages import run_scene_sky
                    run_scene_sky(cfg, out/name, args.frames, geometry, mask,
                        args.scene_multiplier, args.sky_multiplier, args.pixel_stride)
                    cfg.setdefault('preprocessing', {})['scene_sky_state'] = str(out/name/'shared_sky.pt')
                row['status'] = 'completed'
            except BaseException:
                row['status'] = 'failed'
                raise
            finally:
                row['end'] = time.perf_counter()
                row['seconds'] = row['end']-at
        (out/'config_preprocessed.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        save(out/'preprocessing_manifest.json', dict(plan, status='completed', scene=str(scene),
            explicit_reused_inputs=reused, effective_config=str(out/'config_preprocessed.yaml'),
            partial_export=args.frames is not None,
            sky_consumption='Explicit separate background alpha compositing required; original PFG-v5 renderer is not silently changed'))
        timer['status'] = 'completed'
    except BaseException as exc:
        timer.update(status='failed', error=repr(exc))
        raise
    finally:
        timer['end'] = time.perf_counter()
        timer['seconds'] = timer['end']-start
        save(out/'preprocess_timing.json', timer)
    print(json.dumps(timer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

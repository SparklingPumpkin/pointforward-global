"""VR-v1-s runtime with selectable vehicle calibration and stopping settings.

Fresh never reads a delivered scene state. Replay is explicitly cached rendering.
The accepted historical pipeline remains in run_vr_v1_s_pipeline.py.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT/'configs/runtime-stop'
SOURCE = PROJECT/'docs/releases/VR-v1-s/source'
ASSETS = PROJECT/'artifacts/scene017_vr_v1_s'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['fresh', 'replay'], default='fresh')
    p.add_argument('--data-root', required=True, type=Path,
                   help='Processed RGB, LiDAR, masks, calibration, instances and SMPL annotations')
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--scene', default='017')
    p.add_argument('--checkpoint', type=Path,
                   default=PROJECT/'weights/pointforward_waymo_ds_paper_ckpt_020000.pt')
    p.add_argument('--gpu', default='0', help='Static, optimizer and renderer GPU index or UUID')
    p.add_argument('--vehicle-gpu', help='Second GPU for parallel FF; same GPU means sequential FF')
    p.add_argument('--sam-python', default=sys.executable,
                   help='Python with SAM2 dependencies; defaults to current interpreter')
    p.add_argument('--vehicle-calibration', choices=['lim-r', 'lim-r-both', 'bbox-expand', 'adaptive', 'both'],
                   default=None, help='Fresh default: accepted LIM-v1-r. adaptive/both retain the legacy algorithm; replay uses frozen adaptive assets')
    p.add_argument('--bbox-front-back-margin', type=float, default=.5,
                   help='Meters added at EACH longitudinal end in bbox-expand/both (default .5)')
    p.add_argument('--bbox-left-right-margin', type=float, default=.3,
                   help='Meters added at EACH lateral side in bbox-expand/both (default .3)')
    p.add_argument('--use-pinned-extension', action='store_true',
                   help='Only with the documented matching torch/CUDA ABI; otherwise use normal gsplat')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.vehicle_calibration is None:
        args.vehicle_calibration = 'adaptive' if args.mode == 'replay' else 'lim-r'
    if any(not math.isfinite(v) or v < 0 for v in
           (args.bbox_front_back_margin, args.bbox_left_right_margin)):
        p.error('bbox margins must be finite and nonnegative')
    if args.mode == 'replay' and args.scene != '017':
        p.error('Delivered scene state is scene017 only')
    if args.mode == 'replay' and (args.vehicle_calibration != 'adaptive' or
            args.bbox_front_back_margin != .5 or args.bbox_left_right_margin != .3):
        p.error('Calibration options are fresh-only; replay uses the frozen adaptive scene state')
    out = args.output_root.resolve()
    if not args.dry_run and out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty; no implicit resume or scene-state reuse')
    replacements = {'__SOURCE__': str(SOURCE), '__OUTPUT_ROOT__': str(out),
                    '__PROJECT_ROOT__': str(PROJECT), '__DATA_ROOT__': str(args.data_root.resolve()),
                    '__CHECKPOINT__': str(args.checkpoint.resolve()), '__PYTHON__': sys.executable,
                    '__SAM_PYTHON__': args.sam_python}

    def resolve(value):
        for key, target in replacements.items():
            value = value.replace(key, target)
        return value

    recipe = json.loads((RELEASE/'pipeline_recipe.json').read_text())
    commands = {k: [resolve(v) for v in row] for k, row in recipe.items()}
    lim_r = args.vehicle_calibration in ('lim-r', 'lim-r-both')
    adaptive = args.vehicle_calibration in ('adaptive', 'both') or lim_r
    expand = args.vehicle_calibration in ('bbox-expand', 'both', 'lim-r-both')
    geometry_stages = ('geometry_prepare', 'rgb_masks', 'geometry_fit_filter')
    for name in ('geometry_prepare', 'geometry_fit_filter'):
        # The original geometry recipe omitted --scene and defaulted to017.
        commands[name] += ['--scene', args.scene]
    for row in commands.values():
        if '--scene' in row:
            row[row.index('--scene')+1] = args.scene
    commands['rgb_masks'] += ['--sam-root', str(SOURCE/'SAM2'),
                             '--checkpoint', str(PROJECT/'weights/sam2.1_hiera_tiny.pt')]
    if lim_r:
        for name in geometry_stages:
            del commands[name]
        geometry_stages = ('lim_r_preprocessing',)
        commands['lim_r_preprocessing'] = [args.sam_python, str(PROJECT/'scripts/preprocess_lidar_bbox.py'),
            '--config', str(out/'config_build.yaml'), '--out', str(out/'geometry-input'),
            '--scene', args.scene]
        # New runtime adapter is explicitly selected for new calibrated inputs.
        # Historical source snapshots and replay execution remain immutable.
        for name, command in list(commands.items()):
            if name != 'lim_r_preprocessing':
                commands[name] = [command[0], str(PROJECT/'scripts/run_calibrated_stage.py'), *command[1:]]
    vehicle = commands['vehicle_build']
    vehicle[vehicle.index('--decode-box-length-margin')+1] = str(2*args.bbox_front_back_margin if expand else 0.)
    vehicle[vehicle.index('--decode-box-width-margin')+1] = str(2*args.bbox_left_right_margin if expand else 0.)
    if not adaptive:
        for name in geometry_stages:
            del commands[name]
    calibration = dict(mode=args.vehicle_calibration, adaptive_bbox_lidar=adaptive,
        algorithm='LIM-v1-r' if lim_r else 'legacy-adaptive' if adaptive else 'bbox-expand',
        bbox_expanded=expand, front_back_each_m=args.bbox_front_back_margin if expand else 0.,
        left_right_each_m=args.bbox_left_right_margin if expand else 0.,
        length_total_added_m=2*args.bbox_front_back_margin if expand else 0.,
        width_total_added_m=2*args.bbox_left_right_margin if expand else 0.,
        height_added_m=0., application='vehicle feedforward local decode/query boxes',
        order='adaptive calibration/filtering first, then reconstruction box expansion' if adaptive and expand else args.vehicle_calibration,
        evaluation_boxes_expanded=False)
    render = out/'runtime-render-source'
    vehicle_gpu = args.vehicle_gpu or args.gpu
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, commands=commands,
            fresh_parallel_ff=args.gpu != vehicle_gpu, source=str(SOURCE),
            replay_payload=str(ASSETS/'runtime-payload-refined'),
            checkpoint=str(args.checkpoint.resolve()),
            final_checkpoint='fresh: read actual joint-all/checkpoints.json; replay: delivered1670',
            vehicle_calibration=calibration,
            scene_state_reused=args.mode == 'replay'), ensure_ascii=False, indent=2))
        return
    import yaml
    out.mkdir(parents=True, exist_ok=True)
    # Start before materializing config and before any raw input is read.
    timer = dict(run_id=out.name, scope='end_to_end' if args.mode == 'fresh' else 'cached_incremental',
        start=time.perf_counter(), start_epoch_seconds=time.time(), status='running', stages=[],
        input_boundary='before_raw_input_preprocessing' if args.mode == 'fresh' else 'load_delivered_scene_state',
        output_boundary='final_ply_videos_metrics_written',
        historical_scene_state_reused=[] if args.mode == 'fresh' else [str(ASSETS)],
        resources=dict(static_gpu=args.gpu, vehicle_gpu=vehicle_gpu))
    lock = threading.Lock()

    def flush():
        save(out/'pipeline_timing.json', timer)

    def call(name, command, gpu=None):
        env = dict(os.environ, PYTHONPATH=str(SOURCE), HF_HUB_OFFLINE='1',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   CUDA_VISIBLE_DEVICES=gpu or args.gpu)
        env.pop('POINTFORWARD_USE_PINNED_GSPLAT', None)
        if args.use_pinned_extension and name != 'rgb_masks':
            env['POINTFORWARD_USE_PINNED_GSPLAT'] = '1'
        row = dict(name=name, command=command, start=time.perf_counter(), status='running',
                   gpu=env['CUDA_VISIBLE_DEVICES'])
        with lock:
            timer['stages'].append(row)
            flush()
        print('[start]', name, flush=True)
        with (out/(name+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            row.update(end=time.perf_counter(), returncode=result.returncode,
                       status='completed' if result.returncode == 0 else 'failed')
            row['seconds'] = row['end']-row['start']
            flush()
        if result.returncode:
            raise RuntimeError(name+' failed: '+str(out/(name+'.log')))
        print('[done]', name, round(row['seconds'], 3), 's', flush=True)

    flush()
    try:
        cfg = yaml.safe_load(resolve((RELEASE/'config.template.yaml').read_text()))
        cfg['data']['train_scenes'] = cfg['data']['val_scenes'] = [args.scene]
        if not adaptive:
            cfg['data'].pop('geometry_directory', None)
        if args.mode == 'replay':
            cfg['data']['geometry_directory'] = str(ASSETS/'geometry-input')
            cfg['pedestrians']['tracks_file'] = str(ASSETS/'pedestrian_tracks.pt')
        for name in ('config_build.yaml', 'config_render.yaml'):
            (out/name).write_text(yaml.safe_dump(cfg, sort_keys=False))
        save(out/'delivery_options.json', dict(release='PFG-v5' if lim_r else 'VR-v1-s', mode=args.mode, scene=args.scene,
            checkpoint=str(args.checkpoint.resolve()), data_root=str(args.data_root.resolve()),
            source=str(SOURCE), scene_state_reused=args.mode == 'replay', commands=commands,
            pinned_extension=args.use_pinned_extension, vehicle_calibration=calibration))
        if args.mode == 'fresh':
            if adaptive:
                for name in geometry_stages:
                    call(name, commands[name])
            if args.gpu != vehicle_gpu:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    static = pool.submit(call, 'static_build', commands['static_build'], args.gpu)
                    vehicle = pool.submit(call, 'vehicle_build', commands['vehicle_build'], vehicle_gpu)
                    static.result()
                    vehicle.result()
            else:
                call('static_build', commands['static_build'])
                call('vehicle_build', commands['vehicle_build'])
            static_file = out/'static-build/runtime-payload/final_static_128d.pt'
            if not static_file.is_file() or static_file.stat().st_size < 1000000:
                raise RuntimeError('Fresh static producer missing')
            call('pedestrian_build', commands['pedestrian_build'], vehicle_gpu)
            # This link is strictly inside this fresh run, never to a historical payload.
            (out/'runtime-payload/final_static_128d.pt').symlink_to(static_file)
            call('feature_projection', commands['feature_projection'])
            call('joint', commands['joint'])
            timer['optimization_end'] = time.perf_counter()
            timer['raw_to_optimization_seconds'] = timer['optimization_end']-timer['start']
            cp = json.loads((out/'joint-all/checkpoints.json').read_text())[-1]
            payload, refiner = Path(cp['payload']), Path(cp['refiner_state'])
            save(out/'optimization_timing.json', dict(scope='end_to_end', run_id=out.name,
                input_boundary=timer['input_boundary'], output_boundary='optimization_checkpoint_write_complete',
                start=timer['start'], end=timer['optimization_end'],
                seconds=timer['raw_to_optimization_seconds'], historical_scene_state_reused=[],
                outputs=[str(payload), str(refiner)], source_timer='pipeline_timing.json'))
        else:
            payload = ASSETS/'runtime-payload-refined'
            refiner = PROJECT/'weights/refiner_scene017_vr_v1_s.pt'
            cp = dict(step=1670)

        def replace_option(command, option, value):
            command[command.index(option)+1] = str(value)

        render_command = list(commands['render_full'])
        replace_option(render_command, '--payload-dir', payload)
        replace_option(render_command, '--refiner-state', refiner)
        call('render_full', render_command)
        ply_command = list(commands['ply_export'])
        replace_option(ply_command, '--payload-dir', payload)
        replace_option(ply_command, '--pedestrian-tracks', cfg['pedestrians']['tracks_file'])
        call('ply_export', ply_command)
        outputs = [out/'final_global_gaussians_frame000_preview.ply']
        for name in ('final_global_runtime_predicted_front3.mp4',
                     'final_global_runtime_gt_vs_predicted_front3.mp4'):
            shutil.copy2(render/name, out/name)
            outputs.append(out/name)
        m = json.loads((render/'manifest.json').read_text())
        save(out/'result_summary.json', dict(global_psnr=m['mean_psnr_source_frames_all_cameras'],
            vehicle_psnr=m['mean_psnr_dynamic_bbox_source_frames_all_cameras'], step=cp['step'],
            cold_measured=args.mode == 'fresh', scene_state_reused=args.mode == 'replay',
            runtime_color_path='32D Gaussian features + feature_projection.pt + adapted U-Net; PLY preview only'))
        for path in outputs:
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        timer.update(status='completed', outputs=list(map(str, outputs)))
    except BaseException as exc:
        timer.update(status='failed', error=repr(exc))
        raise
    finally:
        timer.update(end=time.perf_counter(), end_epoch_seconds=time.time())
        timer['total_delivery_seconds'] = timer['end']-timer['start']
        flush()
    print(json.dumps(timer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

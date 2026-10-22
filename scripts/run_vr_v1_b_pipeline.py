"""VR-v1-b: portable raw-input joint950 pipeline or delivered-state replay.

Fresh mode rejects existing output and never loads a delivered scene state.
Replay is explicitly cached rendering, not a cold reconstruction measurement.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT/'docs/releases/VR-v1-b'
SOURCE = RELEASE/'source'
ASSETS = PROJECT/'artifacts/scene017_vr_v1_b'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fresh', 'replay'], default='fresh')
    parser.add_argument('--data-root', required=True, type=Path,
        help='Preprocessed dataset training directory: RGB, calibration, poses, instances, LiDAR, masks')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--scene', default='017')
    parser.add_argument('--checkpoint', type=Path, default=PROJECT/'weights/pointforward_waymo_ds_paper_ckpt_020000.pt')
    parser.add_argument('--gpu', help='CUDA device index or UUID; defaults to inherited CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dry-run', action='store_true', help='Print the exact commands without creating outputs')
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.mode == 'replay' and args.scene != '017':
        parser.error('The delivered payload is scene017 only; use fresh for another scene')
    if not args.dry_run and output.exists() and any(output.iterdir()):
        parser.error('Output must be empty; existing scene results are never silently resumed')
    ckpt = args.checkpoint.resolve()
    def py(name, *values):
        return [sys.executable, str(SOURCE/'scripts'/name), *map(str, values)]
    static = output/'static-build'
    raw = output/'runtime-payload'
    joint = output/'joint-all'
    build_cfg = output/'config_build.yaml'
    commands = []
    if args.mode == 'fresh':
        commands += [('static_build', py('deliver_continuous_scene.py', '--config', build_cfg,
            '--ckpt', ckpt, '--out', static, '--scene', args.scene, '--seed', 0,
            '--reconstruction-mode', 'global-gaussians', '--clip-order', 'reverse',
            '--window-length', 20, '--context-frame-stride', 5, '--window-step', 20,
            '--global-voxel-size', .01, '--global-depth-threshold', 1.,
            '--global-growth-policy', 'three-clip-mutable-frozen-position',
            '--global-dynamic-policy', 'persistent-tracks', '--global-dynamic-forward-refresh',
            '--global-update-distance-policy', 'closest-smoothstep', '--global-update-alpha', .5,
            '--global-update-near-distance', 20., '--global-update-far-distance', 80.,
            '--global-update-min-improvement', 0., '--cache-overlap-image-networks',
            '--gaussian-scale-postprocess', 'isotropic-trace', '--isotropic-max-scale', .005,
            '--skip-frame-render', '--skip-ply', '--save-runtime-payload', static/'runtime-payload',
            '--global-query-origin', 'clip', '--global-static-offset-gain', 0.)),
          ('track_once_build', py('build_track_once_dynamic.py', '--config', build_cfg,
            '--ckpt', ckpt, '--static-payload', static/'runtime-payload/final_static_128d.pt',
            '--out', raw, '--scene', args.scene, '--seed', 0, '--view-selection', 'diverse-large',
            '--views-per-track', 12, '--view-angle-weight', .65,
            '--gaussian-scale-postprocess', 'isotropic-trace', '--isotropic-max-scale', .05,
            '--query-origin', 'scene', '--decode-box-scale', 1.,
            '--decode-box-length-margin', 1., '--decode-box-width-margin', .6)),
          ('joint_all950', py('refine_unified_scene_rgb.py', '--config', build_cfg, '--ckpt', ckpt,
            '--payload-dir', raw, '--out', joint, '--audit-run', output, '--scene', args.scene,
            '--steps', 950, '--milestones', 350, 550, 750, 950, '--roi-weight', .25,
            '--pose-lr', .001, '--lr-decay-steps', 240, 700, '--lr-decay-factor', .4))]
        payload = joint/'checkpoint-0950/runtime-payload-refined'
        render = joint/'checkpoint-0950/runtime-render-source'
    else:
        payload = ASSETS/'runtime-payload-refined'
        render = output/'runtime-render-source'
        commands += [('replay', py('render_final_global_runtime.py', '--config', output/'config_render.yaml',
            '--ckpt', ckpt, '--payload-dir', payload, '--out', render, '--scene', args.scene,
            '--device', 'cuda', '--camera-batch-size', 1, '--fps', 10,
            '--dynamic-bbox-metrics', '--per-track-bbox-metrics', '--static-source-rgb-blend', .75,
            '--refiner-state', PROJECT/'weights/refiner_scene017_vr_v1_b.pt'))]
    ply = output/'final_global_gaussians_frame000_preview.ply'
    commands += [('ply_export', py('export_runtime_payload_ply.py', '--payload-dir', payload,
                                 '--out', ply, '--frame', 0))]
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join([str(SOURCE), str(PROJECT/'src'), str(PROJECT)]),
        HF_HOME=str(PROJECT/'weights/huggingface'), HF_HUB_OFFLINE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, commands=commands, source=str(SOURCE),
            pythonpath=env['PYTHONPATH']), ensure_ascii=False, indent=2))
        return
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    for name in ('config_build', 'config_render'):
        template = (RELEASE/(name+'.template.yaml')).read_text()
        template = template.replace('__PROJECT_ROOT__', str(PROJECT)).replace('__DATA_ROOT__', str(args.data_root.resolve()))
        cfg = yaml.safe_load(template)
        cfg['data']['train_scenes'] = cfg['data']['val_scenes'] = [args.scene]
        (output/(name+'.yaml')).write_text(yaml.safe_dump(cfg, sort_keys=False))
    save(output/'delivery_options.json', dict(release='VR-v1-b', mode=args.mode, scene=args.scene,
        checkpoint=str(ckpt), data_root=str(args.data_root.resolve()), commands=commands,
        source=str(SOURCE), scene_state_reused=args.mode == 'replay'))
    timer = dict(scope='end_to_end' if args.mode == 'fresh' else 'cached_incremental',
        start=time.perf_counter(), start_epoch_seconds=time.time(), upstream_prefix_seconds=0,
        input_boundary='before_raw_rgb_read' if args.mode == 'fresh' else 'load_delivered_scene_state',
        output_boundary='ply_and_video_write_complete', status='running', stages=[])
    try:
        for name, command in commands:
            at = time.perf_counter()
            row = dict(name=name, start=at, command=command)
            timer['stages'].append(row)
            save(output/'pipeline_timing.json', timer)
            print('[start]', name, flush=True)
            with (output/(name+'.log')).open('w') as log:
                result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
            row.update(end=time.perf_counter(), returncode=result.returncode)
            row['seconds'] = row['end']-at
            if result.returncode:
                raise RuntimeError(name+' failed; see '+str(output/(name+'.log')))
            print('[done]', name, round(row['seconds'], 3), 's', flush=True)
        outputs = [ply]
        for name in ('final_global_runtime_predicted_front3.mp4', 'final_global_runtime_gt_vs_predicted_front3.mp4'):
            dest = output/name
            shutil.copy2(render/name, dest)
            outputs.append(dest)
        for path in outputs:
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        timer.update(status='completed', outputs=list(map(str, outputs)))
    except BaseException as exc:
        timer.update(status='failed', error=repr(exc))
        raise
    finally:
        timer.update(end=time.perf_counter(), end_epoch_seconds=time.time())
        timer['seconds'] = timer['end']-timer['start']
        save(output/'pipeline_timing.json', timer)
    print(json.dumps(timer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

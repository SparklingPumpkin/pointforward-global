"""Audited CE-v1-h feedforward experiments with optional refiner adaptation.

Prepare/register each run before executing it. Cached runs explicitly reuse the
track-once dynamic control; cold runs reconstruct every scene-specific asset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = Path('/mnt/fjn/FJN-Agent/skills/fjn-project-workflow/scripts/project_workflow.py')
SOURCE = ROOT / 'experiments/argoverse-inference/cold-scene-delivery/CE-v1-h/run-20261010T150637Z-cold-refiner2-stride6'
PYTHON = '/root/miniconda3/envs/drivestudio/bin/python'


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def workflow(*arguments: str):
    with Path('/tmp/pointforward-feedforward-workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(['python3', str(WORKFLOW), arguments[0],
            '--project-root', str(ROOT), *arguments[1:]], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr + result.stdout)
    output = result.stdout.strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def prepare(args) -> Path:
    run_id = workflow('suggest-run-id', '--variant', args.variant)
    group = args.group.rstrip('/') + '/' if args.group else ''
    relative = f'experiments/argoverse-inference/cold-scene-delivery/{group}{args.candidate}/{run_id}'
    decision = workflow('plan-layout', '--path', relative, '--level', 'run',
        '--parent-directory-id', args.parent_directory, '--scope-id', args.scope,
        '--candidate-id', args.candidate,
        '--candidate-id', args.candidate, '--action', 'new_child',
        '--purpose', '独立前馈运行及可选本run refiner适配和正式视频',
        '--reason', '同一候选参数/控制臂，不新增版本号',
        '--alternative', '覆盖旧CE-v1-h产物会损失对照证据', '--record')
    decision_id = decision.get('decision', decision).get('decision_id')
    directory = workflow('register-directory', '--decision-id', decision_id, '--create-directory')
    directory_id = directory.get('directory', directory).get('directory_id')
    run = ROOT / relative
    for name in ('config_build.yaml', 'config_render.yaml'):
        shutil.copyfile(SOURCE / name, run / name)
    options = dict(mode=args.mode, gpu=args.gpu, alpha=args.alpha, delta=args.delta,
        candidate=args.candidate, variant=args.variant, growth=args.growth,
        source_view=args.source_view, distance_policy=args.distance_policy, refresh_fields=args.refresh_fields,
        window_length=args.window_length, context_frame_stride=args.context_frame_stride,
        window_step=args.window_step, voxel=args.voxel, near=args.near, far=args.far,
        cold_static_only=args.cold_static_only, query_origin=args.query_origin,
        static_scale_cap=args.static_scale_cap, static_scale_postprocess=args.static_scale_postprocess,
        static_offset_gain=args.static_offset_gain,
        static_fusion_depth_tolerance=args.static_fusion_depth_tolerance,
        static_view_bank=args.static_view_bank, static_view_bank_alpha=args.static_view_bank_alpha,
        static_view_bank_depth_tolerance=args.static_view_bank_depth_tolerance,
        static_voxel_representative=args.static_voxel_representative,
        dynamic_query_origin=args.dynamic_query_origin,
        cpu_threads=args.cpu_threads, dynamic_gpu=args.dynamic_gpu,
        static_source_rgb_blend=args.static_source_rgb_blend)
    options['static_source_pixel_radius'] = args.static_source_pixel_radius
    options['static_source_min_scale'] = args.static_source_min_scale
    options['dynamic_views_per_track'] = args.dynamic_views_per_track
    options['refiner_epochs'] = args.refiner_epochs
    options['refiner_frame_stride'] = args.refiner_frame_stride
    options['render_unadapted_control'] = args.render_unadapted_control
    options['source_snapshot_from'] = args.source_snapshot_from
    for name in ('joint_group', 'joint_epochs', 'joint_frame_stride', 'joint_feature_groups',
                 'joint_feature_lr', 'joint_geometry_lr', 'joint_refiner_lr', 'joint_opacity_lr',
                 'joint_max_xyz_delta', 'joint_max_log_scale_delta', 'joint_max_rotation_deg',
                 'joint_max_opacity_logit_delta', 'joint_max_feature_delta', 'joint_anchor_weight',
                 'cold_budget_seconds', 'joint_delivery_reserve_seconds'):
        options[name] = getattr(args, name)
    save(run / 'options.json', options)
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    workflow('register-run', '--candidate-id', args.candidate, '--run-id', run.name,
        '--variant', args.variant, '--directory-id', directory_id,
        '--config-identity', f'{relative}/options.json::前馈和显式可选适配配置',
        '--code-identity', f'scripts/benchmark_feedforward_updates.py sha256:{code}',
        '--data-identity', 'Argoverse processed/training/017 RGB+LiDAR+calibration+instances+masks; 156 frames cameras012')
    snapshot_relative = relative + '/source-snapshot'
    snapshot_plan = workflow('plan-layout', '--path', snapshot_relative, '--level', 'artifact',
        '--parent-directory-id', directory_id, '--scope-id', args.scope,
        '--candidate-id', args.candidate,
        '--candidate-id', args.candidate, '--action', 'new_child',
        '--purpose', '本run推理源码快照，隔离并行研究对共享实现的修改',
        '--reason', '运行时读取快照，固定真实执行代码',
        '--alternative', '直接执行共享源码可能在另一会话修改时串版本', '--record')
    workflow('register-directory', '--decision-id', snapshot_plan.get('decision', snapshot_plan)['decision_id'],
        '--create-directory')
    snapshot = run / 'source-snapshot'
    snapshot_source = Path(args.source_snapshot_from).resolve() if args.source_snapshot_from else ROOT
    for name in ('pointforward', 'scripts'):
        shutil.copytree(snapshot_source / name, snapshot / name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Preserve the selected numerical implementation, but execute the current
    # orchestration with its audited optional adaptation and timing boundaries.
    shutil.copyfile(Path(__file__), snapshot / 'scripts' / Path(__file__).name)
    options['code_snapshot'] = str(snapshot)
    save(run / 'options.json', options)
    return run


def execute(run: Path) -> None:
    args = json.loads((run / 'options.json').read_text())
    code_root = Path(args.get('code_snapshot', ROOT))
    ckpt = ROOT / 'ckpt/ckpt_020000_waymo_ds_paper_original.pt'
    payload = run / 'runtime-payload'
    static_dir = run / 'static-build'
    render = run / 'runtime-render-source'
    ply = run / 'final_global_gaussians_frame000_preview.ply'
    if payload.exists() or static_dir.exists() or render.exists():
        raise FileExistsError('Run outputs must be fresh')
    environment = dict(os.environ, PYTHONPATH=str(code_root), CUDA_VISIBLE_DEVICES=args['gpu'],
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
    if args.get('cpu_threads', 0) > 0:
        # Bound CPU pools in shared multi-GPU experiments; no numerical rule,
        # model weight or selected view changes. Record the setting per run.
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment[name] = str(args['cpu_threads'])
    stages = []
    if args['mode'] != 'cached-control':
        build = [PYTHON, str(code_root/'scripts/deliver_continuous_scene.py'),
            '--config', str(run/'config_build.yaml'), '--ckpt', str(ckpt),
            '--out', str(static_dir), '--scene', '017', '--seed', '0',
            '--reconstruction-mode', 'global-gaussians', '--clip-order', 'reverse',
            '--window-length', str(args.get('window_length', 20)),
            '--context-frame-stride', str(args.get('context_frame_stride', 5)),
            '--window-step', str(args['window_step']), '--global-voxel-size', str(args['voxel']),
            '--global-depth-threshold', '1.0', '--global-growth-policy', args['growth'],
            '--global-dynamic-policy', 'persistent-tracks', '--global-dynamic-forward-refresh',
            '--global-update-distance-policy', args.get('distance_policy', 'closest-smoothstep'),
            '--global-update-alpha', str(args['alpha']),
            '--global-update-near-distance', str(args['near']),
            '--global-update-far-distance', str(args['far']),
            '--global-update-min-improvement', str(args['delta']),
            '--cache-overlap-image-networks', '--gaussian-scale-postprocess',
            args.get('static_scale_postprocess', 'isotropic-trace'),
            '--isotropic-max-scale', str(args.get('static_scale_cap', 0.05)), '--skip-frame-render', '--skip-ply',
            '--save-runtime-payload', str(static_dir/'runtime-payload')]
        if args['source_view'] != 'min-depth-error':
            build += ['--global-static-source-view', args['source_view']]
        if args.get('query_origin', 'scene') != 'scene':
            build += ['--global-query-origin', args['query_origin']]
        if args['refresh_fields'] != 'all':
            build += ['--global-static-refresh-fields', args['refresh_fields']]
        if args.get('static_offset_gain', 1.0) != 1.0:
            build += ['--global-static-offset-gain', str(args['static_offset_gain'])]
        if args.get('static_source_pixel_radius', 0.0) > 0:
            build += ['--global-static-source-pixel-radius', str(args['static_source_pixel_radius']),
                      '--global-static-source-min-scale', str(args['static_source_min_scale'])]
        if args.get('static_fusion_depth_tolerance', 0.0) > 0:
            build += ['--global-static-fusion-depth-tolerance', str(args['static_fusion_depth_tolerance'])]
        if args.get('static_view_bank', 'disabled') != 'disabled':
            build += ['--global-static-view-bank', args['static_view_bank'],
                      '--global-static-view-bank-alpha', str(args['static_view_bank_alpha']),
                      '--global-static-view-bank-depth-tolerance', str(args['static_view_bank_depth_tolerance'])]
        if args.get('static_voxel_representative', 'first') != 'first':
            build += ['--global-static-voxel-representative', args['static_voxel_representative']]
        stages.append(('static_build', build))
    if args['mode'] == 'cold':
        stages.append(('track_once_build', [PYTHON, str(code_root/'scripts/build_track_once_dynamic.py'),
            '--config', str(run/'config_build.yaml'), '--ckpt', str(ckpt),
            '--static-payload', str(static_dir/'runtime-payload/final_static_128d.pt'),
            '--out', str(payload), '--scene', '017', '--seed', '0',
            '--view-selection', 'diverse-large', '--views-per-track', str(args['dynamic_views_per_track']),
            '--view-angle-weight', '0.65', '--gaussian-scale-postprocess', 'isotropic-trace',
            '--isotropic-max-scale', '0.05',
            '--query-origin', args.get('dynamic_query_origin', 'scene')]))
        if args.get('dynamic_gpu') is not None:
            stages[-1][1].append('--allow-pending-static-payload')
    stages += [
        ('ply_export', [PYTHON, str(code_root/'scripts/export_runtime_payload_ply.py'),
            '--payload-dir', str(payload), '--out', str(ply), '--frame', '0']),
        ('video_render', [PYTHON, str(code_root/'scripts/render_final_global_runtime.py'),
            '--config', str(run/'config_render.yaml'), '--ckpt', str(ckpt),
            '--payload-dir', str(payload), '--out', str(render), '--scene', '017',
            '--device', 'cuda', '--camera-batch-size', '3', '--fps', '10',
            '--dynamic-bbox-metrics'])]
    if args.get('static_source_rgb_blend', 0.0) > 0:
        stages[-1][1].extend(['--static-source-rgb-blend', str(args['static_source_rgb_blend'])])
    control_command = list(stages[-1][1])
    refiner_epochs = args.get('refiner_epochs', 0)
    refiner_state = run / 'adaptation/refiner_state.pt' if refiner_epochs else None
    joint_group = args.get('joint_group')
    joint_out = run / 'joint-adaptation'
    if joint_group:
        stages.insert(2, ('joint_adaptation', [PYTHON, str(code_root/'scripts/refine_joint_fullscene_rgb.py'),
            '--config', str(run/'config_build.yaml'), '--ckpt', str(ckpt),
            '--payload-dir', str(payload), '--out', str(joint_out), '--group', str(joint_group),
            '--epochs', str(args.get('joint_epochs', 1)), '--frame-stride', str(args.get('joint_frame_stride', 6)),
            '--groups', str(args.get('joint_feature_groups', 4)), '--feature-lr', str(args.get('joint_feature_lr', 0.01)),
            '--geometry-lr', str(args.get('joint_geometry_lr', 0.001)), '--refiner-lr', str(args.get('joint_refiner_lr', 0.0002)),
            '--opacity-lr', str(args.get('joint_opacity_lr', 0.001)), '--max-xyz-delta', str(args.get('joint_max_xyz_delta', 0.02)),
            '--max-log-scale-delta', str(args.get('joint_max_log_scale_delta', 0.25)), '--max-rotation-deg', str(args.get('joint_max_rotation_deg', 5.0)),
            '--max-opacity-logit-delta', str(args.get('joint_max_opacity_logit_delta', 0.5)), '--max-feature-delta', str(args.get('joint_max_feature_delta', 0.5)),
            '--anchor-weight', str(args.get('joint_anchor_weight', 1e-4)), '--static-source-rgb-blend', str(args.get('static_source_rgb_blend', 0.0))]))
        refined_payload = joint_out / 'runtime-payload-refined'
        # stages are now static, track, joint, ply, render.
        stages[3][1][stages[3][1].index('--payload-dir') + 1] = str(refined_payload)
        stages[4][1][stages[4][1].index('--payload-dir') + 1] = str(refined_payload)
        stages[4][1].extend(['--refiner-state', str(joint_out / 'refiner_state.pt')])
        refiner_state = joint_out / 'refiner_state.pt'
    elif refiner_epochs:
        stages.insert(-1, ('adaptation', [PYTHON, str(code_root/'scripts/refine_cached_grouped_rgb.py'),
            '--config', str(run/'config_build.yaml'), '--ckpt', str(ckpt),
            '--payload-dir', str(payload), '--out', str(run/'adaptation'),
            '--feature-epochs', '0', '--refiner-epochs', str(refiner_epochs),
            '--frame-stride', str(args.get('refiner_frame_stride', 6))]))
        stages[-1][1].extend(['--refiner-state', str(refiner_state)])
    cold = args['mode'] == 'cold'
    source_timer = json.loads((SOURCE/'pipeline_timing.json').read_text())
    reused = [] if cold else [str(SOURCE/'runtime-payload/final_dynamic_128d.pt'),
        str(SOURCE/'runtime-payload/global_dynamic_meta.pt')]
    if args['mode'] == 'cached-control':
        reused.append(str(SOURCE/'runtime-payload/final_static_128d.pt'))
    code_paths = {Path(cmd[1]) for _, cmd in stages} | {Path(__file__).resolve()}
    code_paths.update((code_root/'pointforward').rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    save(run/'input_manifest.json', dict(raw_input='/mnt/data/dataset/argoverse/processed/training/017',
        raw_components=['RGB','LiDAR','intrinsics','extrinsics','ego_pose','instances','masks'],
        generic_pretrained=[str(ckpt),'/mnt/data/why/InfiniDepth/checkpoints/depth/infinidepth_depthsensor.ckpt'],
        scene_state_reused_from_outside_run=reused, current_code_sha256=hashes,
        scene_specific_optimization=bool(refiner_epochs or joint_group), refiner_state=str(refiner_state) if refiner_state else None,
        notes=[f'Refiner epochs={refiner_epochs}; joint_group={joint_group}; all scene adaptation is generated inside the timed run.',
            ('Joint loss uses the same source-RGB blend as final rendering.' if joint_group else
             'CE-v1-h adaptation loss uses unblended UNet output; rendering retains the requested source-RGB blend.'),
            'Static-build legacy dynamic map is discarded; final dynamic map is independent track-once.',
            'No scene caches cold; generic compiled kernels and OS cache are not flushed.']))
    workflow('set-run-status', '--run-id', run.name, '--status', 'running')
    start = time.perf_counter()
    journal = dict(run_id=run.name, status='running', scope='end_to_end' if cold else 'cached_incremental',
        input_boundary='first_rgb_read' if cold else 'cached_control_assets',
        output_boundary='ply_and_video_write_complete', start=start,
        start_epoch_seconds=time.time(), stages=[], outputs=[],
        scene_state_reused_from_outside_run=reused, refiner_state=str(refiner_state) if refiner_state else None,
        timer_starts_before_all_stage_subprocess_initialization=True)
    save(run/'pipeline_timing.json', journal)
    try:
        if not cold:
            payload.mkdir()
            static_source = SOURCE/'runtime-payload/final_static_128d.pt' if args['mode']=='cached-control' else static_dir/'runtime-payload/final_static_128d.pt'
            (payload/'final_static_128d.pt').symlink_to(static_source)
            for name in ('final_dynamic_128d.pt','global_dynamic_meta.pt'):
                (payload/name).symlink_to(SOURCE/'runtime-payload'/name)
        sequential_stages = stages
        if cold and args.get('dynamic_gpu') is not None:
            # Track-once only links the static path; it never consumes its
            # tensors. Join both producers before export/render. The complete
            # cold timer measures their overlapping wall time once.
            parallel_records = []
            for name, command in stages[:2]:
                record = dict(name=name, command=command, start=time.perf_counter(),
                    status='running', log=str(run/(name+'.log')),
                    gpu=args['dynamic_gpu'] if name=='track_once_build' else args['gpu'])
                parallel_records.append(record)
                journal['stages'].append(record)
                print(f'START {name} parallel gpu={record["gpu"]}', flush=True)
            journal['parallel_producers'] = True
            save(run/'pipeline_timing.json', journal)

            def execute_producer(record):
                producer_env = dict(environment, CUDA_VISIBLE_DEVICES=record['gpu'])
                with Path(record['log']).open('w') as log:
                    result = subprocess.run(record['command'], cwd=code_root,
                        env=producer_env, stdout=log, stderr=subprocess.STDOUT)
                end = time.perf_counter()
                return dict(end=end, seconds=end-record['start'], returncode=result.returncode,
                    status='completed' if result.returncode==0 else 'failed')

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(execute_producer, record): record
                           for record in parallel_records}
                for future in as_completed(futures):
                    record = futures[future]
                    record.update(future.result())
                    save(run/'pipeline_timing.json', journal)
                    print(f'END {record["name"]} seconds={record["seconds"]:.3f} '
                          f'returncode={record["returncode"]}', flush=True)
            if any(record['returncode'] for record in parallel_records):
                raise RuntimeError('Parallel producer failed; inspect both stage logs')
            for name in ('final_static_128d.pt', 'final_dynamic_128d.pt', 'global_dynamic_meta.pt'):
                if not (payload/name).is_file():
                    raise FileNotFoundError(payload/name)
            sequential_stages = stages[2:]
        for name, command in sequential_stages:
            if name == 'joint_adaptation' and args.get('cold_budget_seconds'):
                command += ['--deadline-monotonic', str(start + args['cold_budget_seconds'] - args.get('joint_delivery_reserve_seconds', 300.0))]
            record = dict(name=name, command=command, start=time.perf_counter(), status='running',
                log=str(run/(name+'.log')))
            journal['stages'].append(record)
            save(run/'pipeline_timing.json', journal)
            print(f'START {name} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            with Path(record['log']).open('w') as log:
                result = subprocess.run(command, cwd=code_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            record.update(end=time.perf_counter(), returncode=result.returncode,
                status='completed' if result.returncode==0 else 'failed')
            record['seconds'] = record['end']-record['start']
            save(run/'pipeline_timing.json', journal)
            print(f'END {name} seconds={record["seconds"]:.3f} returncode={result.returncode}', flush=True)
            if result.returncode:
                raise RuntimeError(f'{name} failed: {record["log"]}')
        outputs = [ply,render/'final_global_runtime_predicted_front3.mp4',render/'final_global_runtime_gt_vs_predicted_front3.mp4']
        for output in outputs:
            if not output.is_file() or output.stat().st_size==0:
                raise FileNotFoundError(output)
            with output.open('rb') as file:
                os.fsync(file.fileno())
        end = time.perf_counter()
        journal.update(status='completed',end=end,end_epoch_seconds=time.time(),seconds=end-start,
            outputs=[str(p) for p in outputs])
        if cold and (joint_group or not refiner_epochs):
            budget = args.get('cold_budget_seconds') or (1200 if joint_group else 900)
            journal['max_seconds'] = budget
            journal['within_budget'] = end-start<=budget
        elif not cold:
            upstream = sum(s['seconds'] for s in source_timer['stages'] if s['name']=='track_once_build' or
                (args['mode']=='cached-control' and s['name']=='static_build'))
            journal.update(historical_upstream_seconds=upstream,
                reconstructed_cold_seconds_estimate=end-start+upstream,
                cold_time_status='historical_dependency_sum_estimate_not_single_cold_measurement')
        save(run/'pipeline_timing.json', journal)
        metrics = json.loads((render/'manifest.json').read_text())
        print(json.dumps({k:metrics.get(k) for k in ('mean_psnr_source_frames_all_cameras',
            'mean_psnr_non_sky_source_frames_all_cameras','mean_psnr_dynamic_bbox_source_frames_all_cameras')},indent=2),flush=True)
        if args.get('render_unadapted_control', False):
            control_dir = run / 'runtime-render-unadapted-control'
            control_command[control_command.index('--out')+1] = str(control_dir)
            control_start = time.perf_counter()
            print('START unadapted control AFTER final delivery timer', flush=True)
            with (run/'unadapted_control.log').open('w') as log:
                result = subprocess.run(control_command, cwd=code_root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
            control_end = time.perf_counter()
            save(run/'control_timing.json', dict(scope='diagnostic_after_delivery',
                start=control_start, end=control_end, seconds=control_end-control_start,
                command=control_command, returncode=result.returncode,
                included_in_cold_delivery_seconds=False))
            if result.returncode:
                raise RuntimeError('Unadapted diagnostic failed; final delivery timer is already complete')
            baseline = json.loads((control_dir/'manifest.json').read_text())
            keys = ('mean_psnr_source_frames_all_cameras', 'mean_psnr_non_sky_source_frames_all_cameras',
                'mean_psnr_dynamic_bbox_source_frames_all_cameras')
            save(run/'quality_comparison.json', dict(
                unadapted={k:baseline.get(k) for k in keys}, adapted={k:metrics.get(k) for k in keys},
                gain_db={k:metrics[k]-baseline[k] for k in keys},
                same_fresh_payload=True, cold_delivery_seconds=end-start,
                additional_diagnostic_seconds=control_end-control_start))
        workflow('set-run-status','--run-id',run.name,'--status','completed',
            '--summary',f'{args["mode"]} refiner_epochs={refiner_epochs} joint_group={joint_group}; total={end-start:.3f}s; PSNR={metrics.get("mean_psnr_source_frames_all_cameras")}',
            '--evidence',f'{run.relative_to(ROOT)}/pipeline_timing.json::真实执行阶段和输入复用范围',
            '--evidence',f'{run.relative_to(ROOT)}/runtime-render-source/manifest.json::完整156帧正式运行时指标及适配输入')
    except BaseException as error:
        journal.update(status='failed',end=time.perf_counter(),error=repr(error))
        journal['seconds']=journal['end']-start
        save(run/'pipeline_timing.json',journal)
        workflow('set-run-status','--run-id',run.name,'--status','failed',
            '--summary',repr(error),'--evidence',f'{run.relative_to(ROOT)}/pipeline_timing.json::失败阶段和原始计时')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--candidate', default='CE-v1-j')
    parser.add_argument('--group', default='')
    parser.add_argument('--scope', default='cold-scene-delivery')
    parser.add_argument('--parent-directory', default='dir-0468')
    parser.add_argument('--variant', default='forward-update')
    parser.add_argument('--mode', choices=('cached-control','cached-probe','cold'), default='cached-probe')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cpu-threads', type=int, default=0,
        help='Optional CPU pool limit; zero preserves inherited environment.')
    parser.add_argument('--dynamic-gpu', default=None,
        help='Optional separate GPU for a cold parallel track-once producer; both producers are joined before PLY/video.')
    parser.add_argument('--refiner-epochs', type=int, default=0,
        help='Optional CE-v1-h frozen-Gaussian UNet adaptation; zero preserves pure feedforward.')
    parser.add_argument('--refiner-frame-stride', type=int, default=6)
    parser.add_argument('--render-unadapted-control', action='store_true',
        help='Render the same fresh payload without adaptation after the delivery timer ends.')
    parser.add_argument('--source-snapshot-from', default=None,
        help='Optional existing numerical code snapshot; no scene payload or adapted weights are copied.')
    parser.add_argument('--joint-group', choices=('features', 'xyz-scale-features', 'all'), default=None)
    parser.add_argument('--joint-epochs', type=int, default=1)
    parser.add_argument('--joint-frame-stride', type=int, default=6)
    parser.add_argument('--joint-feature-groups', type=int, default=4)
    parser.add_argument('--joint-feature-lr', type=float, default=0.01)
    parser.add_argument('--joint-geometry-lr', type=float, default=0.001)
    parser.add_argument('--joint-refiner-lr', type=float, default=0.0002)
    parser.add_argument('--joint-opacity-lr', type=float, default=0.001)
    parser.add_argument('--joint-max-xyz-delta', type=float, default=0.02)
    parser.add_argument('--joint-max-log-scale-delta', type=float, default=0.25)
    parser.add_argument('--joint-max-rotation-deg', type=float, default=5.0)
    parser.add_argument('--joint-max-opacity-logit-delta', type=float, default=0.5)
    parser.add_argument('--joint-max-feature-delta', type=float, default=0.5)
    parser.add_argument('--joint-anchor-weight', type=float, default=1e-4)
    parser.add_argument('--cold-budget-seconds', type=float, default=None)
    parser.add_argument('--joint-delivery-reserve-seconds', type=float, default=300.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.)
    parser.add_argument('--near', type=float, default=20.)
    parser.add_argument('--far', type=float, default=80.)
    # CE-v1-p/x matched controls: frozen positions, only the two most recent
    # clips mutable. Keep the preregistered control as the safe default so a
    # missing CLI flag cannot silently change the experiment population.
    parser.add_argument('--growth', default='two-clip-mutable-frozen-position')
    parser.add_argument('--source-view', default='min-depth-error')
    parser.add_argument('--distance-policy', choices=('uniform','smoothstep','closest-smoothstep'), default='closest-smoothstep')
    parser.add_argument('--refresh-fields', default='all')
    parser.add_argument('--query-origin', choices=('scene', 'clip', 'clip-rigid'), default='clip')
    parser.add_argument('--static-scale-cap', type=float, default=0.05)
    parser.add_argument('--static-scale-postprocess', choices=('isotropic-trace','isotropic-volume','axis-cap'), default='isotropic-trace')
    parser.add_argument('--static-offset-gain', type=float, default=1.0)
    parser.add_argument('--static-fusion-depth-tolerance', type=float, default=0.0)
    parser.add_argument('--static-view-bank', choices=('disabled','nearest4'), default='disabled')
    parser.add_argument('--static-source-rgb-blend', type=float, default=0.0)
    parser.add_argument('--static-source-pixel-radius', type=float, default=0.0)
    parser.add_argument('--static-source-min-scale', type=float, default=0.005)
    parser.add_argument('--static-view-bank-alpha', type=float, default=1.0)
    parser.add_argument('--static-view-bank-depth-tolerance', type=float, default=1.0)
    parser.add_argument('--static-voxel-representative', choices=('first','closest'), default='first')
    parser.add_argument('--dynamic-query-origin', choices=('scene','first-ego','canonical-centre'), default='scene')
    parser.add_argument('--dynamic-views-per-track', type=int, choices=(8, 10, 12), default=12)
    parser.add_argument('--window-length', type=int, default=20)
    parser.add_argument('--context-frame-stride', type=int, default=5)
    parser.add_argument('--window-step', type=int, default=20)
    parser.add_argument('--voxel', type=float, default=0.05)
    parser.add_argument('--cold-static-only', action='store_true')
    args=parser.parse_args()
    if args.refiner_epochs < 0 or args.refiner_frame_stride < 1:
        parser.error('Refiner epochs must be nonnegative and stride positive')
    if args.joint_epochs < 1 or args.joint_frame_stride < 1:
        parser.error('Joint epochs and frame stride must be positive')
    if args.prepare and args.joint_group and (args.mode != 'cold' or args.refiner_epochs):
        parser.error('Joint adaptation requires a fresh cold run and cannot also use sequential refiner epochs')
    if args.prepare and args.refiner_epochs and args.mode != 'cold':
        parser.error('This optional adaptation experiment requires a fresh cold run')
    if not 0.0 <= args.static_source_rgb_blend <= 1.0:
        parser.error('--static-source-rgb-blend must be in [0, 1]')
    if args.prepare and (args.static_source_pixel_radius < 0 or (
        args.static_source_pixel_radius > 0 and (
            args.static_scale_postprocess != 'isotropic-trace'
            or not 0 < args.static_source_min_scale <= args.static_scale_cap
        )
    )):
        parser.error('Source-pixel radius requires non-negative pixels and trace spheres with 0<minimum<=cap')
    if args.prepare and args.dynamic_gpu is not None and args.mode != 'cold':
        parser.error('--dynamic-gpu requires a cold run; cached runs cannot verify parallel cold delivery.')
    if args.prepare and not (
        args.context_frame_stride > 0
        and args.window_length >= 3 * args.context_frame_stride + 1
        and 0 < args.window_step <= args.window_length
    ):
        parser.error('Window must cover four context timestamps and step must be within window length.')
    if args.prepare and args.dynamic_query_origin != 'scene' and args.mode != 'cold':
        parser.error('A changed dynamic query origin requires --mode cold to reconstruct the actual dynamic assets.')
    if args.prepare:
        print(prepare(args))
    elif args.run:
        execute(args.run.resolve())
    else:
        parser.error('Use --prepare or --run')


if __name__=='__main__':
    main()

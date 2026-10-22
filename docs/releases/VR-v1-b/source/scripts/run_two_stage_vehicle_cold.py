"""Optional cold experiment: joint scene350, then one continuous vehicle optimizer.

Only source/configuration files may be copied from previous experiments. Every
scene tensor and adapted network is produced after the common timer starts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

from scripts.run_vehicle_trajectory_experiments import ROOT, PYTHON, workflow, directory, save

ROOT = Path(os.environ.get('POINTFORWARD_PROJECT_ROOT', str(ROOT)))

SCOPE = 'feedforward-refiner-delivery'
FAMILY = 'experiments/argoverse-inference/cold-scene-delivery/vehicle-trajectory'
CONFIG_SOURCE = ROOT/'experiments/argoverse-inference/cold-scene-delivery/vehicle-factorial/FFR-v1-t/run-20261019T031920Z-optimization-only-cold'
CODE_SOURCE = ROOT/'experiments/argoverse-inference/cold-scene-delivery/vehicle-trajectory/FFR-v1-y/run-20261019T061420Z-vehicle-joint-low-pose-continuation/source-snapshot'


def prepare():
    candidate = workflow('suggest-candidate-id', '--scope-id', SCOPE)
    relative = FAMILY+'/'+candidate
    did = directory(relative, 'candidate', 'dir-0836', '冷启动全局与车辆两阶段联合优化，省去独立ROI阶段')
    contract = dict(schema_version=1, criteria=[
        dict(id='cold', criterion='从原始scene017 RGB、LiDAR、标定、实例和mask与通用预训练权重重新建图；不复用历史场景payload、适配U-Net、预测深度或优化状态；记录连续RGB到PLY和完整视频墙钟。'),
        dict(id='schedule', criterion='全场景350步优化后，冻结静态Gaussian与U-Net；车辆XYZ、scale、rotation、opacity、128D feature、轨迹平移和yaw在同一优化器中连续联合更新，最多600步，省去旧200步ROI前置阶段。'),
        dict(id='audit', criterion='前馈、全局350步以及车辆200/400/600步检查点使用完整156帧三相机原始bbox统一评估；保存逐步loss、检查点耗时和视频，选择动态PSNR与耗时的折中点；如触发预登记早停则说明缺失的后续检查点。'),
        dict(id='review', criterion='亲自检查车辆可视化，报告与旧r/t/w/y的质量和时间差异，不将preview PLY宣称为独立复现128D与U-Net渲染质量。')])
    save(ROOT/relative/'acceptance_contract.json', contract)
    workflow('create-candidate', '--scope-id', SCOPE, '--slug', 'cold-two-stage-vehicle-joint',
             '--hypothesis', '从新鲜全场景350步直接进入车辆全属性与轨迹联合优化，可减少旧ROI前置训练并找到质量时间折中',
             '--change-type', 'code_architecture', '--directory-id', did,
             '--acceptance-contract', relative+'/acceptance_contract.json')
    variant = 'cold-scene350-vehicle-joint600'
    rid = workflow('suggest-run-id', '--variant', variant)
    rr = relative+'/'+rid
    run_did = directory(rr, 'run', did, '原始输入冷启动及逐检查点质量时间审计', candidate)
    directory(rr+'/source-snapshot', 'artifact', run_did, '固定原数值源码与两阶段可选执行入口', candidate)
    run = ROOT/rr
    snap = run/'source-snapshot'
    for name in ('scripts', 'pointforward'):
        shutil.copytree(CODE_SOURCE/name, snap/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('run_two_stage_vehicle_cold.py', 'refine_vehicle_trajectory_rgb.py'):
        shutil.copyfile(ROOT/'scripts'/name, snap/'scripts'/name)
    for name in ('config_build.yaml', 'config_render.yaml'):
        shutil.copyfile(CONFIG_SOURCE/name, run/name)
    template = json.loads((CONFIG_SOURCE/'pipeline_timing.json').read_text())['stages']
    commands = {}
    for row in template:
        if row['name'] in ('static_build', 'track_once_build', 'joint_adaptation'):
            commands[row['name']] = [v.replace(str(CONFIG_SOURCE), str(run)) for v in row['command']]
    options = dict(candidate=candidate, run_id=rid, scene='017', commands=commands,
        source_code_only=str(CODE_SOURCE), source_config_only=str(CONFIG_SOURCE),
        global_steps=350, vehicle_steps=600, checkpoints=[200, 400, 600],
        pose_lr=.001, feature_lr=.006, geometry_lr=.004, opacity_lr=.004,
        pose_warmup_steps=50, pose_warmup_initial=.25, actor_balanced=True,
        early_stop='two consecutive checkpoint gains <0.05dB dynamic, or global drops >0.05dB below scene350',
        best_rule='earliest vehicle checkpoint within0.05dB of best dynamic PSNR, with global>=scene350-0.05dB')
    save(run/'options.json', options)
    workflow('register-run', '--candidate-id', candidate, '--run-id', rid, '--variant', variant,
             '--directory-id', run_did, '--config-identity', rr+'/options.json::两阶段参数与检查点规则',
             '--code-identity', rr+'/source-snapshot::独立冻结数值源码，原默认分支不变',
             '--data-identity', 'Argoverse training017 RGB+LiDAR+标定+实例+masks;156帧三相机;无历史场景缓存')
    print(json.dumps(dict(candidate=candidate, run=str(run)), ensure_ascii=False))


def render_command(run, payload, out, refiner=None):
    snap = run/'source-snapshot'
    cmd = [PYTHON, str(snap/'scripts/render_final_global_runtime.py'), '--config', str(run/'config_render.yaml'),
           '--ckpt', str(ROOT/'ckpt/ckpt_020000_waymo_ds_paper_original.pt'), '--payload-dir', str(payload),
           '--out', str(out), '--scene', '017', '--device', 'cuda', '--camera-batch-size', '1', '--fps', '10',
           '--dynamic-bbox-metrics', '--per-track-bbox-metrics', '--static-source-rgb-blend', '.75']
    if refiner: cmd += ['--refiner-state', str(refiner)]
    return cmd


def prepare_retry(previous):
    previous=previous.resolve();old=json.loads((previous/'options.json').read_text())
    state=json.loads((ROOT/'docs/project-control/state.json').read_text())
    candidate=old['candidate'];relative=str(previous.parent.relative_to(ROOT))
    parent=next(k for k,v in state['directories'].items() if v['path']==relative)
    variant='cold-scene350-vehicle-joint600-retry'
    rid=workflow('suggest-run-id','--variant',variant);rr=relative+'/'+rid
    did=directory(rr,'run',parent,'修复actor审计变量重名后从原始输入完整重跑，不读取失败run产物',candidate)
    directory(rr+'/source-snapshot','artifact',did,'固定修复后的可选两阶段入口',candidate)
    run=ROOT/rr;snap=run/'source-snapshot'
    for name in ('scripts','pointforward'):
        shutil.copytree(previous/'source-snapshot'/name,snap/name,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run_two_stage_vehicle_cold.py','refine_vehicle_trajectory_rgb.py'):
        shutil.copyfile(ROOT/'scripts'/name,snap/'scripts'/name)
    for name in ('config_build.yaml','config_render.yaml'):shutil.copyfile(previous/name,run/name)
    old['run_id']=rid
    old['commands']={k:[s.replace(str(previous),str(run)) for s in cmd] for k,cmd in old['commands'].items()}
    old['retry_reason']='actor-balanced ROI precompute accidentally shadowed Gaussian bounds; zero vehicle optimizer steps ran; all scene assets regenerated in this retry'
    old['excluded_failed_run']=str(previous)
    save(run/'options.json',old)
    workflow('register-run','--candidate-id',candidate,'--run-id',rid,'--variant',variant,
        '--directory-id',did,'--config-identity',rr+'/options.json::同一数值配置，修复变量重名后冷启动重试',
        '--code-identity',rr+'/source-snapshot::修复后的独立源码快照',
        '--data-identity','Argoverse training017原始输入，所有场景状态在本次计时内生成；不复用失败run')
    print(json.dumps(dict(candidate=candidate,run=str(run))))


def training_worker(run):
    # Both stages run in the same Python process. State hand-off uses the new
    # run's payload; static optimizer memory is released before vehicle training.
    import torch
    o = json.loads((run/'options.json').read_text())
    audit = []
    def call(name, command):
        t = time.perf_counter()
        print('WORKER START', name, flush=True)
        sys.argv = command[1:]
        runpy.run_path(command[1], run_name='__main__')
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        audit.append(dict(name=name, start=t, end=time.perf_counter(), command=command))
        audit[-1]['seconds'] = audit[-1]['end']-t
        save(run/'training_stage_timing.json', dict(stages=audit))
        print('WORKER END', name, audit[-1]['seconds'], flush=True)
    call('joint_adaptation', o['commands']['joint_adaptation'])
    refiner = run/'joint-adaptation/refiner_state.pt'
    t = time.perf_counter()
    command = render_command(run, run/'joint-adaptation/runtime-payload-refined', run/'audit-scene350', refiner)
    subprocess.run(command, check=True)
    audit.append(dict(name='audit_scene350', start=t, end=time.perf_counter(), command=command))
    audit[-1]['seconds'] = audit[-1]['end']-t
    save(run/'training_stage_timing.json', dict(stages=audit))
    cmd = [PYTHON, str(run/'source-snapshot/scripts/refine_vehicle_trajectory_rgb.py'),
           '--config', str(run/'config_build.yaml'), '--ckpt', str(ROOT/'ckpt/ckpt_020000_waymo_ds_paper_original.pt'),
           '--payload-dir', str(run/'joint-adaptation/runtime-payload-refined'), '--refiner-state', str(refiner),
           '--out', str(run/'vehicle-joint'), '--strategy', 'joint', '--steps', '600',
           '--feature-lr', '.006', '--geometry-lr', '.004', '--opacity-lr', '.004', '--pose-lr', '.001',
           '--lr-decay-start', '1000000000', '--pose-lr-warmup', '50', '--actor-balanced',
           '--checkpoint-every', '200', '--audit-run', str(run)]
    call('vehicle_joint', cmd)


def execute(run, gpu):
    o = json.loads((run/'options.json').read_text())
    if (run/'pipeline_timing.json').exists(): raise FileExistsError('Cold run must be new')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(run/'source-snapshot'),
               POINTFORWARD_PROJECT_ROOT=str(ROOT),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    hashes = {str(p.relative_to(run/'source-snapshot')):hashlib.sha256(p.read_bytes()).hexdigest()
              for base in ('scripts', 'pointforward') for p in (run/'source-snapshot'/base).rglob('*.py')}
    save(run/'input_manifest.json', dict(raw_input='/mnt/data/dataset/argoverse/processed/training/017',
        raw_components=['RGB','LiDAR','intrinsics','extrinsics','ego_pose','instances','masks'],
        generic_pretrained=[str(ROOT/'ckpt/ckpt_020000_waymo_ds_paper_original.pt'),
                            '/mnt/data/why/InfiniDepth/checkpoints/depth/infinidepth_depthsensor.ckpt'],
        scene_state_reused_from_outside_run=[], code_sha256=hashes, gpu_uuid=gpu,
        source_code_only=o['source_code_only'], source_config_only=o['source_config_only'],
        notes=['Scene caches never reused. Generic compiled kernels and OS page cache not flushed.',
               'All audit renders, checkpoint saves and stage transitions included in continuous timer.']))
    workflow('set-run-status', '--run-id', run.name, '--status', 'running')
    start = time.perf_counter()
    journal = dict(run_id=run.name, status='running', scope='end_to_end', input_boundary='first_rgb_read',
        output_boundary='ply_and_video_write_complete', start=start, start_epoch_seconds=time.time(),
        stages=[], scene_state_reused_from_outside_run=[], audit_overhead_included=True)
    save(run/'pipeline_timing.json', journal)
    def stage(name, command):
        row=dict(name=name, command=command, start=time.perf_counter(), status='running')
        journal['stages'].append(row); save(run/'pipeline_timing.json', journal)
        print('START', name, flush=True)
        with (run/(name+'.log')).open('w') as log:
            p=subprocess.run(command, cwd=run/'source-snapshot', env=env, stdout=log, stderr=subprocess.STDOUT)
        row.update(end=time.perf_counter(), returncode=p.returncode)
        row.update(seconds=row['end']-row['start'], status='completed' if p.returncode==0 else 'failed')
        save(run/'pipeline_timing.json', journal)
        print('END', name, row['seconds'], flush=True)
        if p.returncode: raise RuntimeError(name+' failed; inspect log')
    try:
        for name in ('static_build','track_once_build'): stage(name, o['commands'][name])
        stage('audit_feedforward', render_command(run, run/'runtime-payload', run/'audit-feedforward'))
        stage('two_stage_training', [PYTHON, str(run/'source-snapshot/scripts/run_two_stage_vehicle_cold.py'),
                                    '--worker', str(run)])
        checkpoints=json.loads((run/'vehicle-joint/checkpoints.json').read_text())
        eligible=[r for r in checkpoints if r['global_psnr']>=checkpoints[0]['scene350_global_psnr']-.05]
        if not eligible: raise RuntimeError('No vehicle checkpoint preserves global quality')
        best=max(r['dynamic_psnr'] for r in eligible)
        selected=next(r for r in eligible if r['dynamic_psnr']>=best-.05)
        save(run/'selected_checkpoint.json', selected)
        payload=Path(selected['payload'])
        # Reuse the measured selected checkpoint's *new-run* video without
        # recomputing it. Copy/flush is inside the cold end boundary.
        for name in ('final_global_runtime_predicted_front3.mp4', 'final_global_runtime_gt_vs_predicted_front3.mp4'):
            shutil.copy2(Path(selected['render'])/name, run/name)
        stage('ply_export', [PYTHON, str(run/'source-snapshot/scripts/export_runtime_payload_ply.py'),
            '--payload-dir', str(payload), '--out', str(run/'final_global_gaussians_frame000_preview.ply'), '--frame','0'])
        outputs=[run/'final_global_gaussians_frame000_preview.ply',
                 run/'final_global_runtime_predicted_front3.mp4',run/'final_global_runtime_gt_vs_predicted_front3.mp4']
        for path in outputs:
            with path.open('rb') as stream: os.fsync(stream.fileno())
        journal.update(status='completed', end=time.perf_counter(), end_epoch_seconds=time.time(),
                       outputs=[str(p) for p in outputs], selected_checkpoint=selected)
        journal['seconds']=journal['end']-start
        save(run/'pipeline_timing.json', journal)
        workflow('set-run-status', '--run-id', run.name, '--status', 'completed',
                 '--summary', '冷启动两阶段实验完成，全部检查点审计成本包含在连续墙钟内',
                 '--evidence', str(run.relative_to(ROOT))+'/pipeline_timing.json::完整冷启动实际边界与阶段耗时',
                 '--evidence', str(run.relative_to(ROOT))+'/vehicle-joint/checkpoints.json::同一run检查点PSNR时间')
    except BaseException as e:
        journal.update(status='failed', end=time.perf_counter(), error=repr(e))
        journal['seconds']=journal['end']-start; save(run/'pipeline_timing.json', journal)
        workflow('set-run-status','--run-id',run.name,'--status','failed','--summary',repr(e),
                 '--evidence',str(run.relative_to(ROOT))+'/pipeline_timing.json::失败原因及已计时间')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--run',type=Path)
    p.add_argument('--worker',type=Path); p.add_argument('--gpu')
    p.add_argument('--prepare-retry',type=Path)
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.prepare_retry: prepare_retry(a.prepare_retry)
    elif a.worker: training_worker(a.worker.resolve())
    else: execute(a.run.resolve(),a.gpu)

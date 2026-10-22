"""Register and execute the uninterrupted 350+600-step joint experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.run_vehicle_trajectory_experiments import ROOT, PYTHON, workflow, save

SCOPE = 'vehicle-roi-visibility'
FAMILY = 'experiments/argoverse-inference/cold-scene-delivery/vehicle-roi-visibility'


def directory(path, level, parent, purpose, candidate=None):
    args = ['--path', path, '--level', level, '--parent-directory-id', parent,
        '--scope-id', SCOPE, '--action', 'new_child', '--purpose', purpose,
        '--reason', '在strict ROI正确性基础上独立比较全程联合与分阶段优化',
        '--alternative', '覆盖VR-v1-a会丢失分阶段对照', '--record']
    if candidate: args += ['--candidate-id', candidate]
    d = workflow('plan-layout', *args).get('decision')
    return workflow('register-directory', '--decision-id', d['decision_id'], '--create-directory')['directory']['directory_id']


def prepare(source):
    source = source.resolve()
    assert (source/'runtime-payload/final_dynamic_128d.pt').is_file()
    candidate = workflow('suggest-candidate-id', '--scope-id', SCOPE)
    relative = FAMILY+'/'+candidate
    did = directory(relative, 'candidate', 'dir-0864', '前馈后950步不冻结不重置的全参数联合训练')
    contract = dict(schema_version=1, criteria=[
        dict(id='initialization', criterion='仅复用FFR-v1-z原始前馈载荷及通用预训练权重，不读取其scene350或车辆优化结果、适配U-Net；独立保存源码与参数。'),
        dict(id='joint', criterion='950步内静态和动态Gaussian的XYZ、scale、rotation、opacity、完整128D feature、time/sigma，以及U-Net和车辆轨迹平移/yaw始终属于同一优化器；350步处不冻结、不重置，检查点只评估；保留canonical参考位姿和原始评估框。'),
        dict(id='evaluation', criterion='350、550、750、950步均完成156帧三相机runtime指标与预测/GT对照视频，记录全局、非天空、动态bbox PSNR、逐步正值MSE、loss和参数变化；交付950步PLY预览与完整运行时资产。'),
        dict(id='audit', criterion='亲自检查车辆可视化并比较旧scene350、VR-v1-a和y；记录本次缓存后连续实际增量墙钟及加回原始前馈前缀的历史账目，不能宣称一次RGB冷启动实测；如质量不提升也如实报告。')])
    save(ROOT/relative/'acceptance_contract.json', contract)
    workflow('create-candidate', '--scope-id', SCOPE, '--slug', 'unified-all950',
        '--hypothesis', '前馈后始终联合优化全场景、车辆和U-Net可减少冻结造成的动态误差，并提高同950步质量',
        '--change-type', 'code_architecture', '--directory-id', did,
        '--acceptance-contract', relative+'/acceptance_contract.json')
    variant = 'feedforward-unified-all950'
    rid = workflow('suggest-run-id', '--variant', variant); rr = relative+'/'+rid
    run_did = directory(rr, 'run', did, '950步连续联合与完整检查点交付审计', candidate)
    directory(rr+'/source-snapshot', 'artifact', run_did, '保留z数值模型与独立全场景联合优化入口', candidate)
    run = ROOT/rr; snap = run/'source-snapshot'
    for name in ('scripts', 'pointforward'):
        shutil.copytree(source/'source-snapshot'/name, snap/name, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('refine_unified_scene_rgb.py', 'run_unified_scene_experiment.py', 'render_final_global_runtime.py'):
        shutil.copyfile(ROOT/'scripts'/name, snap/'scripts'/name)
    for name in ('config_build.yaml', 'config_render.yaml'):
        shutil.copyfile(source/name, run/name)
    os.symlink(source/'audit-feedforward', run/'audit-feedforward', target_is_directory=True)
    save(run/'options.json', dict(candidate=candidate, source_run=str(source),
        source_payload=str(source/'runtime-payload'), snapshot=str(snap), steps=950,
        milestones=[350,550,750,950], roi_weight=.25, pose_lr=.001, lr_decay_steps=[240,700],
        lr_decay_factor=.4, initialization='raw feedforward + generic checkpoint',
        selected_step=950, scope='cached_incremental', scene='017'))
    workflow('register-run', '--candidate-id', candidate, '--run-id', rid, '--variant', variant,
        '--directory-id', run_did, '--config-identity', rr+'/options.json::950步参数、目标和评估点',
        '--code-identity', rr+'/source-snapshot::z数值模型与全程联合优化入口',
        '--data-identity', str(source/'runtime-payload')+'::只复用原始前馈Gaussian；重新读取156帧RGB；通用U-Net初始化')
    print(json.dumps(dict(candidate=candidate,run=str(run)),ensure_ascii=False))


def execute(run, gpu):
    run = run.resolve(); o = json.loads((run/'options.json').read_text())
    source, snap = Path(o['source_run']), Path(o['snapshot'])
    if (run/'pipeline_timing.json').exists(): raise FileExistsError('Register a new run')
    parent = json.loads((source/'pipeline_timing.json').read_text())
    prefix = next(r['end'] for r in parent['stages'] if r['name']=='audit_feedforward')-parent['start']
    hashes = {str(p.relative_to(snap)): hashlib.sha256(p.read_bytes()).hexdigest()
              for name in ('scripts','pointforward') for p in (snap/name).rglob('*.py')}
    save(run/'input_manifest.json', dict(source_run=str(source), source_payload=o['source_payload'],
        upstream_timer=str(source/'pipeline_timing.json'), upstream_prefix_seconds=prefix,
        upstream_prefix_boundary='end of raw feedforward source video audit',
        reused_scene_state='raw feedforward only', adapted_network_reused=False,
        code_sha256=hashes, gpu_uuid=gpu, cold_start_measured=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(snap),
        POINTFORWARD_PROJECT_ROOT=str(ROOT), PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    ckpt = ROOT/'ckpt/ckpt_020000_waymo_ds_paper_original.pt'
    cmd = [PYTHON,str(snap/'scripts/refine_unified_scene_rgb.py'), '--config',str(run/'config_build.yaml'),
        '--ckpt',str(ckpt), '--payload-dir',o['source_payload'], '--out',str(run/'joint-all'),
        '--audit-run',str(run), '--scene',o['scene'], '--steps',str(o['steps']),
        '--milestones', *map(str,o['milestones']), '--roi-weight',str(o['roi_weight']),
        '--pose-lr',str(o['pose_lr']), '--lr-decay-steps',*map(str,o['lr_decay_steps']),
        '--lr-decay-factor',str(o['lr_decay_factor'])]
    workflow('set-run-status', '--run-id',run.name, '--status','running')
    started = time.perf_counter()
    timer = dict(run_id=run.name, scope='cached_incremental', status='running', start=started,
        start_epoch_seconds=time.time(), input_boundary='load_raw_feedforward_state',
        output_boundary='ply_and_video_write_complete', upstream_prefix_seconds=prefix, stages=[])
    save(run/'pipeline_timing.json',timer)
    def call(name,command):
        at=time.perf_counter();row=dict(name=name,command=command,start=at)
        timer['stages'].append(row);save(run/'pipeline_timing.json',timer)
        print('START',name,flush=True)
        with (run/(name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=snap,env=env,stdout=log,stderr=subprocess.STDOUT)
        row.update(end=time.perf_counter(),returncode=result.returncode);row['seconds']=row['end']-at
        save(run/'pipeline_timing.json',timer)
        if result.returncode:raise RuntimeError(name+' failed')
        print('END',name,row['seconds'],flush=True)
    try:
        call('joint_all950',cmd)
        cps=json.loads((run/'joint-all/checkpoints.json').read_text())
        selected=next(c for c in cps if c['step']==o['selected_step'])
        save(run/'selected_checkpoint.json',selected)
        for name in ('final_global_runtime_predicted_front3.mp4','final_global_runtime_gt_vs_predicted_front3.mp4'):
            shutil.copy2(Path(selected['render'])/name,run/name)
        ply=run/'final_global_gaussians_frame000_preview.ply'
        call('ply_export',[PYTHON,str(snap/'scripts/export_runtime_payload_ply.py'),
            '--payload-dir',selected['payload'],'--out',str(ply),'--frame','0'])
        outputs=[ply,run/'final_global_runtime_predicted_front3.mp4',run/'final_global_runtime_gt_vs_predicted_front3.mp4']
        for path in outputs:
            with path.open('rb') as stream:os.fsync(stream.fileno())
        timer.update(end=time.perf_counter(),end_epoch_seconds=time.time(),status='completed',outputs=list(map(str,outputs)))
        timer['seconds']=timer['end']-started;timer['historical_sum_seconds']=prefix+timer['seconds']
        save(run/'pipeline_timing.json',timer)
        workflow('set-run-status','--run-id',run.name,'--status','completed',
            '--summary','原始前馈起点950步全程联合与四检查点视频及PLY完成，历史累计非单次冷启动',
            '--evidence',str(run.relative_to(ROOT))+'/pipeline_timing.json::实际增量与上游前馈前缀',
            '--evidence',str(run.relative_to(ROOT))+'/joint-all/checkpoints.json::完整950步及156帧三相机检查点')
    except BaseException as exc:
        timer.update(end=time.perf_counter(),status='failed',error=repr(exc));timer['seconds']=timer['end']-started
        save(run/'pipeline_timing.json',timer)
        workflow('set-run-status','--run-id',run.name,'--status','failed','--summary',repr(exc),
            '--evidence',str(run.relative_to(ROOT))+'/pipeline_timing.json::失败阶段与真实已耗时间')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',type=Path);p.add_argument('--run',type=Path);p.add_argument('--gpu')
    a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    else:execute(a.run,a.gpu)

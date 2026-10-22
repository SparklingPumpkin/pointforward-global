"""Register and deliver isolated cached vehicle trajectory/shape experiments."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT=Path(__file__).resolve().parents[1]
WORKFLOW=Path('/mnt/fjn/FJN-Agent/skills/fjn-project-workflow/scripts/project_workflow.py')
PYTHON='/root/miniconda3/envs/drivestudio/bin/python'
FAMILY='experiments/argoverse-inference/cold-scene-delivery/vehicle-trajectory'
BASE=ROOT/'experiments/argoverse-inference/cold-scene-delivery/vehicle-factorial/FFR-v1-t/run-20261019T031920Z-optimization-only-cold'
SCOPE='feedforward-refiner-delivery'


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def workflow(action,*args):
    with open('/tmp/pointforward-feedforward-workflow.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=subprocess.run(['python3',str(WORKFLOW),action,'--project-root',str(ROOT),*args],
            capture_output=True,text=True,check=True)
    try:return json.loads(result.stdout)
    except json.JSONDecodeError:return result.stdout.strip()


def directory(path,level,parent,purpose,candidate=None):
    args=['--path',path,'--level',level,'--parent-directory-id',parent,'--scope-id',SCOPE,
        '--action','new_group' if level=='experiment_family' else 'new_child','--purpose',purpose,
        '--reason','独立保存轨迹与车辆形状对照，不覆盖冻结的历史实验',
        '--alternative','混入旧run会破坏复现和因素归因','--record']
    if candidate:args+=['--candidate-id',candidate]
    if level=='experiment_family':args+=['--override-reason',
        '本轮是车辆轨迹与形状机制研究，需与既有冷启动剪枝及前馈因子实验分组；不移动其他活动研究']
    d=workflow('plan-layout',*args);d=d.get('decision',d)
    r=workflow('register-directory','--decision-id',d['decision_id'],'--create-directory')
    return r.get('directory',r)['directory_id']


def prepare():
    family_id=directory(FAMILY,'experiment_family','dir-0447','复用t场景的车辆轨迹和大范围形状修正；不宣称冷启动')
    rows=[]
    for strategy in ('shape','joint','staged'):
        candidate=workflow('suggest-candidate-id','--scope-id',SCOPE)
        relative=FAMILY+'/'+candidate
        did=directory(relative,'candidate',family_id,'独立检验'+strategy+'车辆修正假设')
        contract={'schema_version':1,'criteria':[
            {'id':'isolation','criterion':f'从同一FFR-v1-t最终状态开始，策略{strategy}；车辆Gaussian持续优化并放宽XYZ到每轴30cm；静态Gaussian、U-Net、标定和原始评估框固定，记录所有实际优化参数。'},
            {'id':'delivery','criterion':'保存可重放的轨迹和Gaussian状态、PLY预览、完整156帧三相机预测与GT对照视频；区分缓存后增量耗时与历史上游耗时，不宣称新冷启动实测。'},
            {'id':'quality','criterion':'比较三组和FFR-v1-t的全局、非天空及原始动态bbox PSNR，亲自检查近车清晰度和轨迹抖动；如未大幅改善则如实报告局限。'}]}
        save(ROOT/relative/'acceptance_contract.json',contract)
        workflow('create-candidate','--scope-id',SCOPE,'--slug','vehicle-'+strategy,
            '--hypothesis',f'{strategy}轨迹/形状策略可在放宽Gaussian位移后减少车辆模糊',
            '--change-type','code_architecture','--directory-id',did,
            '--acceptance-contract',relative+'/acceptance_contract.json')
        variant='vehicle-'+strategy
        run_id=workflow('suggest-run-id','--variant',variant);run_relative=relative+'/'+run_id
        rid=directory(run_relative,'run',did,'固定步数车辆轨迹形状对照与正式交付',candidate)
        run=ROOT/run_relative
        directory(run_relative+'/source-snapshot','artifact',rid,'固定t数值源码及局部轨迹优化代码',candidate)
        snapshot=run/'source-snapshot'
        for name in ('pointforward','scripts'):
            shutil.copytree(BASE/'source-snapshot'/name,snapshot/name,dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('refine_vehicle_trajectory_rgb.py','run_vehicle_trajectory_experiments.py'):
            shutil.copyfile(ROOT/'scripts'/name,snapshot/'scripts'/name)
        # Only guard the unselected sqrt(0) quaternion candidates. Forward values
        # are identical, and this prevents NaN gradients once poses are trainable.
        rotation=snapshot/'pointforward/models/rotation.py'
        text=rotation.read_text()
        old='    ).clamp_min(0.0).sqrt()'
        if text.count(old)!=1:raise ValueError('Unexpected baseline rotation implementation')
        text=text.replace('    q_abs = torch.stack(','    q_squared = torch.stack(',1).replace(old,
            '    ).clamp_min(0.0)\n    q_abs = torch.zeros_like(q_squared)\n'
            '    positive = q_squared > 0\n    q_abs[positive] = q_squared[positive].sqrt()',1)
        rotation.write_text(text)
        for name in ('config_build.yaml','config_render.yaml'):shutil.copyfile(BASE/name,run/name)
        options=dict(candidate=candidate,strategy=strategy,steps=624,warmup_steps=156,
            source_run=str(BASE),source_payload=str(BASE/'vehicle-roi-adaptation/runtime-payload-refined'),
            refiner_state=str(BASE/'joint-adaptation/refiner_state.pt'),snapshot=str(snapshot),
            max_xyz=.3,max_translation=.5,max_yaw_deg=15.,knot_stride=4)
        save(run/'options.json',options)
        workflow('register-run','--candidate-id',candidate,'--run-id',run_id,'--variant',variant,
            '--directory-id',rid,'--config-identity',run_relative+'/options.json::轨迹和Gaussian参数',
            '--code-identity',run_relative+'/source-snapshot::t数值源码与独立轨迹优化入口，四元数反向稳定修复',
            '--data-identity',str(BASE)+'::计时外复用t场景Gaussian及适配U-Net；scene017完整156帧RGB')
        rows.append(dict(candidate=candidate,run=str(run),strategy=strategy))
    save(ROOT/FAMILY/'runs.json',rows)
    print(json.dumps(rows,indent=2))


def execute(run,gpu):
    run=run.resolve();o=json.loads((run/'options.json').read_text())
    snap=Path(o['snapshot']);ckpt=ROOT/'ckpt/ckpt_020000_waymo_ds_paper_original.pt'
    env=dict(os.environ,PYTHONPATH=str(snap),CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    payload=run/'adaptation/runtime-payload-refined';render=run/'runtime-render-source'
    commands=[('adaptation',[PYTHON,str(snap/'scripts/refine_vehicle_trajectory_rgb.py'),
        '--config',str(run/'config_build.yaml'),'--ckpt',str(ckpt),'--payload-dir',o['source_payload'],
        '--refiner-state',o['refiner_state'],'--out',str(run/'adaptation'),'--strategy',o['strategy'],
        '--steps',str(o['steps']),'--warmup-steps',str(o['warmup_steps']),
        '--pose-lr',str(o.get('pose_lr',.025)),
        '--pose-smoothness',str(o.get('pose_smoothness',.02))]),
        ('ply_export',[PYTHON,str(snap/'scripts/export_runtime_payload_ply.py'),'--payload-dir',str(payload),
            '--out',str(run/'final_global_gaussians_frame000_preview.ply'),'--frame','0']),
        ('video_render',[PYTHON,str(snap/'scripts/render_final_global_runtime.py'),
            '--config',str(run/'config_render.yaml'),'--ckpt',str(ckpt),'--payload-dir',str(payload),
            '--refiner-state',o['refiner_state'],'--out',str(render),'--scene','017','--device','cuda',
            '--camera-batch-size','1','--fps','10','--dynamic-bbox-metrics','--per-track-bbox-metrics',
            '--static-source-rgb-blend','0.75'])]
    hashes={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest()
        for base in ('pointforward','scripts') for p in (snap/base).rglob('*.py')}
    save(run/'input_manifest.json',dict(source_run=o['source_run'],source_payload=o['source_payload'],
        source_refiner=o['refiner_state'],scene_state_reused=True,code_sha256=hashes,gpu_uuid=gpu,
        timer_scope='cached_incremental',upstream_timer=str(BASE/'pipeline_timing.json')))
    workflow('set-run-status','--run-id',run.name,'--status','running')
    start=time.perf_counter();journal=dict(status='running',scope='cached_incremental',start=start,
        start_epoch_seconds=time.time(),stages=[],source_run=o['source_run'])
    save(run/'pipeline_timing.json',journal)
    try:
        for name,cmd in commands:
            stage=dict(name=name,start=time.perf_counter(),command=cmd,status='running')
            journal['stages'].append(stage);save(run/'pipeline_timing.json',journal)
            print('START',name,flush=True)
            with (run/(name+'.log')).open('w') as log:
                r=subprocess.run(cmd,cwd=snap,env=env,stdout=log,stderr=subprocess.STDOUT)
            stage.update(end=time.perf_counter(),returncode=r.returncode)
            stage.update(seconds=stage['end']-stage['start'],status='completed' if r.returncode==0 else 'failed')
            save(run/'pipeline_timing.json',journal)
            print('END',name,stage['seconds'],r.returncode,flush=True)
            if r.returncode:raise RuntimeError(f'{name} failed, see log')
        for path in (run/'final_global_gaussians_frame000_preview.ply',
            render/'final_global_runtime_predicted_front3.mp4',render/'final_global_runtime_gt_vs_predicted_front3.mp4'):
            if not path.is_file() or not path.stat().st_size:raise FileNotFoundError(path)
        journal.update(status='completed',end=time.perf_counter(),end_epoch_seconds=time.time())
        journal['seconds']=journal['end']-start
        upstream=json.loads((BASE/'pipeline_timing.json').read_text())['seconds']
        journal.update(historical_upstream_seconds=upstream,historical_sum_seconds=upstream+journal['seconds'],
            cold_time_status='not_measured_this_run; dependency sum is historical accounting only')
        save(run/'pipeline_timing.json',journal)
        workflow('set-run-status','--run-id',run.name,'--status','completed',
            '--summary',o['strategy']+'车辆轨迹形状实验完成；缓存后增量交付，非冷启动',
            '--evidence',str(run.relative_to(ROOT))+'/runtime-render-source/manifest.json::同一run完整156帧指标',
            '--evidence',str(run.relative_to(ROOT))+'/pipeline_timing.json::增量阶段实际计时及历史上游账目')
    except BaseException as e:
        journal.update(status='failed',error=repr(e),end=time.perf_counter());journal['seconds']=journal['end']-start
        save(run/'pipeline_timing.json',journal)
        workflow('set-run-status','--run-id',run.name,'--status','failed','--summary',repr(e),
            '--evidence',str(run.relative_to(ROOT))+'/pipeline_timing.json::失败原因与计时')
        raise


def prepare_low_pose(source_run):
    old=source_run.resolve();o=json.loads((old/'options.json').read_text())
    state=json.loads((ROOT/'docs/project-control/state.json').read_text())
    # Resolve the already registered candidate directory without guessing IDs.
    directories=state['directories']
    entries=directories.values() if isinstance(directories,dict) else directories
    relative=str(old.parent.relative_to(ROOT))
    parent=next(d['directory_id'] for d in entries if d['path']==relative)
    variant='vehicle-joint-low-pose'
    run_id=workflow('suggest-run-id','--variant',variant)
    run_relative=relative+'/'+run_id
    did=directory(run_relative,'run',parent,'相同联合假设的低轨迹学习率参数对照',o['candidate'])
    directory(run_relative+'/source-snapshot','artifact',did,'固定低学习率实验执行源码',o['candidate'])
    run=ROOT/run_relative;snapshot=run/'source-snapshot'
    for name in ('scripts','pointforward'):
        shutil.copytree(old/'source-snapshot'/name,snapshot/name,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(Path(__file__),snapshot/'scripts'/Path(__file__).name)
    for name in ('config_build.yaml','config_render.yaml'):shutil.copyfile(old/name,run/name)
    o.update(snapshot=str(snapshot),pose_lr=.0025,pose_smoothness=.02)
    save(run/'options.json',o)
    workflow('register-run','--candidate-id',o['candidate'],'--run-id',run_id,'--variant',variant,
        '--directory-id',did,'--config-identity',run_relative+'/options.json::仅轨迹学习率从0.025降为0.0025',
        '--code-identity',run_relative+'/source-snapshot::同一车辆轨迹形状联合优化快照',
        '--data-identity',str(BASE)+'::复用同一t最终场景；全部156帧原始RGB')
    print(json.dumps(dict(candidate=o['candidate'],run=str(run),strategy='joint-low-pose')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--run',type=Path);p.add_argument('--gpu')
    p.add_argument('--prepare-low-pose',type=Path);args=p.parse_args()
    if args.prepare:prepare()
    elif args.prepare_low_pose:prepare_low_pose(args.prepare_low_pose)
    else:execute(args.run,args.gpu)

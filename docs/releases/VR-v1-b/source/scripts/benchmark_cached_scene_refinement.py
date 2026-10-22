"""Run a clearly labelled cached-input mechanism probe, never cold acceptance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--payload-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--feature-epochs', type=int, required=True)
    p.add_argument('--refiner-epochs', type=int, default=6)
    p.add_argument('--frame-stride', type=int, default=1)
    a = p.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    if (out/'pipeline_timing.json').exists():
        raise FileExistsError('New run required for every probe')
    out.mkdir(parents=True, exist_ok=True)
    opt = out/'adaptation'
    render = out/'runtime-render-source'
    payload = opt/'runtime-payload-refined' if a.feature_epochs else Path(a.payload_dir).resolve()
    env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=a.gpu,
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8')
    commands = [
        ('adaptation', [sys.executable, str(root/'scripts/refine_cached_grouped_rgb.py'),
            '--config',a.config,'--ckpt',a.ckpt,'--payload-dir',a.payload_dir,
            '--out',str(opt),'--feature-epochs',str(a.feature_epochs),
            '--refiner-epochs',str(a.refiner_epochs),'--frame-stride',str(a.frame_stride),'--parity-check']),
        ('render', [sys.executable,str(root/'scripts/render_final_global_runtime.py'),
            '--config',a.config,'--ckpt',a.ckpt,'--payload-dir',str(payload),
            '--refiner-state',str(opt/'refiner_state.pt'),'--out',str(render),
            '--scene','017','--device','cuda','--camera-batch-size','3',
            '--fps','10','--dynamic-bbox-metrics']),
    ]
    start = time.perf_counter()
    journal = {'run_id':out.name,'scope':'cached_incremental','status':'running',
        'input_boundary':'unoptimized_feedforward_payload_read',
        'output_boundary':'adaptation_and_source_video_complete','start':start,
        'start_epoch_seconds':time.time(),'scene_state_reused_from_outside_run':[a.payload_dir],
        'raw_rgb_also_read':True,'configuration':vars(a),'stages':[],'cold_goal_met':False}
    def save():
        (out/'pipeline_timing.json').write_text(json.dumps(journal,indent=2)+'\n')
    save()
    try:
        for name, command in commands:
            rec={'name':name,'command':command,'start':time.perf_counter(),'status':'running'}
            journal['stages'].append(rec)
            save()
            print(f'START {name}',flush=True)
            with (out/(name+'.log')).open('w') as log:
                process=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            rec.update(end=time.perf_counter(),returncode=process.returncode,
                status='completed' if process.returncode==0 else 'failed')
            rec['seconds']=rec['end']-rec['start']
            save()
            print(f'END {name}: {rec["seconds"]:.3f}s, exit={process.returncode}',flush=True)
            if process.returncode:
                raise RuntimeError(f'{name} failed')
        journal.update(status='completed',end=time.perf_counter(),end_epoch_seconds=time.time())
        journal['seconds']=journal['end']-start
    except BaseException as exc:
        journal.update(status='failed',error=repr(exc),end=time.perf_counter())
        journal['seconds']=journal['end']-start
        save()
        raise
    save()


if __name__ == '__main__':
    main()

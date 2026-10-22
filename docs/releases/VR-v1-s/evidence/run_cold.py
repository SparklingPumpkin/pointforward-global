"""One raw scene replay, no fitted-scene ancestors. Independent FF producers."""
import argparse,json,os,subprocess,time,shutil,threading
from pathlib import Path
import yaml
RUN=Path(__file__).resolve().parent;SNAP=RUN/'source-snapshot'
PY='/root/miniconda3/envs/drivestudio/bin/python';SAM='/root/miniconda3/envs/sam/bin/python'
BASE=RUN.parents[1]/'VR-v1-l/run-20261021T165140Z-argo017-geometry-input-correction'
def main():
 p=argparse.ArgumentParser();p.add_argument('--static-gpu',default='0');p.add_argument('--vehicle-gpu',default='5');p.add_argument('--loss',default='union');p.add_argument('--steps',default=1650,type=int);a=p.parse_args()
 if (RUN/'pipeline_timing.json').exists():raise RuntimeError('Fresh run required')
 timer=dict(run_id=RUN.name,status='running',scope='end_to_end',start=time.perf_counter(),start_epoch_seconds=time.time(),stages=[],input_boundary='before_raw_input_preprocessing',output_boundary='final_ply_videos_metrics_written',historical_scene_state_reused=[],generic_offline_weights=['PointForward020000','SMPLhumanhead','SAM'],resources=dict(static_gpu=a.static_gpu,vehicle_gpu=a.vehicle_gpu))
 lock=threading.Lock()
 def save(): (RUN/'pipeline_timing.json').write_text(json.dumps(timer,indent=2))
 save()
 # Rewrite all scene-specific references before the first raw read, inside timer.
 cfg=yaml.safe_load((RUN/'config_build.yaml').read_text());cfg['data']['geometry_directory']=str(RUN/'geometry-input');cfg['pedestrians']['tracks_file']=str(RUN/'pedestrian_tracks.pt');(RUN/'config_build.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 uuid={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in [a.static_gpu,a.vehicle_gpu]}
 def call(name,cmd,gpu):
  row=dict(name=name,start=time.perf_counter(),command=list(map(str,cmd)),status='running',gpu_uuid=uuid[gpu]);
  with lock:timer['stages'].append(row);save()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid[gpu],PYTHONPATH=str(SNAP),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
  with (RUN/(name+'.log')).open('w') as log:r=subprocess.run(row['command'],cwd=SNAP,env=env,stdout=log,stderr=subprocess.STDOUT)
  with lock:row.update(end=time.perf_counter(),seconds=time.perf_counter()-row['start'],returncode=r.returncode,status='completed' if r.returncode==0 else 'failed');save()
  print(name,row['seconds'],r.returncode,flush=True)
  if r.returncode:raise RuntimeError(name)
 old=json.loads((BASE/'pipeline_timing.json').read_text())
 def command(name):
  cmd=next(s['command'] for s in old['stages'] if s['name']==name)
  return [x.replace(str(BASE),str(RUN)) for x in cmd]
 try:
  for name in ['geometry_prepare','rgb_masks','geometry_fit_filter']:call(name,command(name),a.static_gpu)
  import concurrent.futures
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   f=pool.submit(call,'static_build',command('static_build'),a.static_gpu)
   vehicle=command('vehicle_build')+['--allow-pending-static-payload']
   g=pool.submit(call,'vehicle_build',vehicle,a.vehicle_gpu)
   f.result();g.result()
  static=RUN/'static-build/runtime-payload/final_static_128d.pt'
  if not static.is_file() or static.stat().st_size<1000000:raise RuntimeError('Static producer missing')
  call('pedestrian_build',command('pedestrian_build'),a.vehicle_gpu)
  (RUN/'runtime-payload/final_static_128d.pt').symlink_to(static)
  # Projection loads the actual run-local static and dynamic payloads; failures count.
  call('feature_projection',[PY,SNAP/'scripts/feature_projection.py','--source',RUN/'runtime-payload','--out',RUN/'runtime-payload-compressed','--rank','32'],a.static_gpu)
  common=['--config',RUN/'config_build.yaml','--ckpt','/mnt/data/fjn/projects/PointForward/ckpt/ckpt_020000_waymo_ds_paper_original.pt','--scene','017']
  call('joint',[PY,SNAP/'scripts/refine_unified_scene_rgb.py',*common,'--payload-dir',RUN/'runtime-payload-compressed','--out',RUN/'joint-all','--audit-run',RUN,'--steps',str(a.steps),'--milestones',str(a.steps),'--roi-weight','1','--vehicle-loss',a.loss,'--freeze-static-after','100000','--lr-decay-steps','600','1200','--lr-decay-factor','.5','--defer-audit-render','--cold-stop-at','1480'],a.static_gpu)
  timer['optimization_end']=time.perf_counter();timer['raw_to_optimization_seconds']=timer['optimization_end']-timer['start'];save()
  cp=json.loads((RUN/'joint-all/checkpoints.json').read_text())[-1];dest=RUN/'runtime-render-source'
  (RUN/'optimization_timing.json').write_text(json.dumps(dict(run_id=RUN.name,scope='end_to_end',status='completed',input_boundary='before_raw_input_preprocessing',output_boundary='optimization_checkpoint_write_complete',start=timer['start'],end=timer['optimization_end'],seconds=timer['raw_to_optimization_seconds'],historical_scene_state_reused=[],outputs=[cp['payload'],cp['refiner_state']],source_timer='pipeline_timing.json'),indent=2))
  call('render_full',[PY,SNAP/'scripts/render_final_global_runtime.py',*common,'--payload-dir',cp['payload'],'--out',dest,'--device','cuda','--camera-batch-size','1','--fps','10','--dynamic-bbox-metrics','--per-track-bbox-metrics','--static-source-rgb-blend','.75','--refiner-state',cp['refiner_state']],a.static_gpu)
  call('ply_export',[PY,SNAP/'scripts/export_runtime_payload_ply.py','--payload-dir',cp['payload'],'--out',RUN/'final_global_gaussians_frame000_preview.ply','--frame','0','--pedestrian-tracks',RUN/'pedestrian_tracks.pt','--smpl-model','/mnt/data/output/pointforward/smpl_models/SMPL_NEUTRAL.npz'],a.static_gpu)
  for n in ['final_global_runtime_predicted_front3.mp4','final_global_runtime_gt_vs_predicted_front3.mp4']:shutil.copy2(dest/n,RUN/n)
  m=json.loads((dest/'manifest.json').read_text());result=dict(global_psnr=m['mean_psnr_source_frames_all_cameras'],vehicle_psnr=m['mean_psnr_dynamic_bbox_source_frames_all_cameras'],raw_to_optimization_seconds=timer['raw_to_optimization_seconds'],cold_measured=True,video=str(RUN/'final_global_runtime_gt_vs_predicted_front3.mp4'),render=str(dest),step=cp['step'],runtime_color_path='rank32 PCA Gaussian features + folded/adapted U-Net; PLY preview only')
  (RUN/'result_summary.json').write_text(json.dumps(result,indent=2));timer.update(status='completed',end=time.perf_counter(),total_delivery_seconds=time.perf_counter()-timer['start']);save()
 except BaseException as e:
  timer.update(status='failed',error=repr(e),end=time.perf_counter());save();raise
if __name__=='__main__':main()

"""Freeze VR-v1-s assets and register independent delivery acceptance."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

WORKFLOW='/mnt/fjn/FJN-Agent/skills/fjn-project-workflow/scripts/project_workflow.py'

def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument('--source-run',type=Path,required=True);a.add_argument('--target',type=Path,default=Path(__file__).resolve().parents[1]);args=a.parse_args()
 src=args.source_run.resolve();target=args.target.resolve();release=target/'docs/releases/VR-v1-s';assets=target/'artifacts/scene017_vr_v1_s'
 def wf(action,*v):
  s=subprocess.check_output(['python',WORKFLOW,action,'--project-root',str(target),*map(str,v)],text=True)
  try:return json.loads(s)
  except ValueError:return s.strip()
 def directory(path,level,parent,purpose,candidate=None):
  state=json.loads((target/'docs/project-control/state.json').read_text())
  old=next((k for k,v in state['directories'].items() if v['path']==path),None)
  if old:return old
  v=['--path',path,'--level',level,'--parent-directory-id',parent,'--scope-id','project','--action','new_child','--purpose',purpose,'--reason','用户指定将当前VR-v1-s固化为当前交付主线','--alternative','保留历史冻结证据，仅覆盖当前入口和根说明','--override-reason','用户授权在既有包内固化新主线，历史文件不删除','--record']
  if candidate:v+=['--candidate-id',candidate]
  d=wf('plan-layout',*v)['decision']['decision_id']
  return wf('register-directory','--decision-id',d,'--create-directory')['directory']['directory_id']
 candidate=wf('suggest-candidate-id','--scope-id','project');base='docs/releases/'+candidate
 cd=directory(base,'candidate','dir-0013','VR-v1-s独立交付验收')
 contract=dict(schema_version=1,criteria=[dict(id='assets',criterion='完整固化VR-v1-s第1670步32D静态动态Gaussian、PCA基、轨迹、SMPL状态、配套U-Net、通用权重、源码、PLY预览及视频；源与交付资产SHA256一致。'),dict(id='pipeline',criterion='独立交付源码、依赖和可移植参数支持原始处理后RGB/LiDAR到几何校正、并行前馈、SMPL、32D投影、全三相机联合优化及视频和PLY；fresh不复用历史场景状态；提供固定资产replay。'),dict(id='replay',criterion='包内源码与资产重渲染完整156帧三相机；全局、车辆bbox和非天空PSNR相比源VR-v1-s误差各不超过0.0001dB，PLY一致并人工查看结果。'),dict(id='honesty',criterion='准确保留源记录的各阶段边界、完整交付开销以及研究验收失败结论；本次固化仅验证交付与重渲染，不伪称重新冷启动训练或单PLY复现RGB指标。')])
 save(target/base/'acceptance_contract.json',contract)
 wf('create-candidate','--scope-id','project','--slug','compact32-union-delivery','--hypothesis','用户选定VR-v1-s可独立固化并精确重渲染；交付验收独立于未通过的研究29dB目标','--change-type','code_architecture','--directory-id',cd,'--acceptance-contract',base+'/acceptance_contract.json')
 rid=wf('suggest-run-id','--variant','compact-delivery-validation');rr=base+'/'+rid
 rd=directory(rr,'run',cd,'完整独立重渲染与交付校验',candidate);directory(rr+'/replay','artifact',rd,'包内重渲染输出',candidate)
 releaseid=directory('docs/releases/VR-v1-s','archive','dir-0013','VR-v1-s不可变版本源码与证据')
 sourceid=directory('docs/releases/VR-v1-s/source','archive',releaseid,'VR-v1-s数值源码及便携依赖')
 directory('docs/releases/VR-v1-s/source/SAM2','archive',sourceid,'SAM2.1 Tiny源码及许可')
 directory('docs/releases/VR-v1-s/source/InfiniDepth','archive',sourceid,'本次原始深度模型源码')
 directory('docs/releases/VR-v1-s/evidence','archive',releaseid,'原始计时指标与研究失败验收记录')
 directory('artifacts/scene017_vr_v1_s','archive','dir-0022','当前VR-v1-s运行时及预览资产')
 copies={}
 def copy(s,d):
  d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d);copies[str(d.relative_to(target))]=str(s)
 ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','*.so')
 for name in ('pointforward','scripts','pedestrian_reference'):
  shutil.copytree(src/'source-snapshot'/name,release/'source'/name,dirs_exist_ok=True,ignore=ignore)
 copy(src/'source-snapshot/sitecustomize.py',release/'source/sitecustomize.py')
 copy(src/'source-snapshot/gsplat_cuda.so',release/'source/gsplat_cuda.so')
 sam=Path('/mnt/data/fjn/projects/tools/sam2');depth=Path('/mnt/data/why/InfiniDepth')
 shutil.copytree(sam/'sam2',release/'source/SAM2/sam2',dirs_exist_ok=True,ignore=ignore)
 for n in ('LICENSE','README.md'):copy(sam/n,release/'source/SAM2'/n)
 shutil.copytree(depth/'InfiniDepth',release/'source/InfiniDepth/InfiniDepth',dirs_exist_ok=True,ignore=ignore)
 for n in ('LICENSE','requirements.txt'):copy(depth/n,release/'source/InfiniDepth'/n)
 cp=src/'joint-all/checkpoint-1670'
 for sub,origin in [('runtime-payload',src/'runtime-payload'),('runtime-payload-compressed',src/'runtime-payload-compressed'),('runtime-payload-refined',cp/'runtime-payload-refined')]:
  for f in sorted(origin.iterdir()):
   if f.is_file():copy(f,assets/sub/f.name)
 for f in sorted((src/'geometry-input').rglob('*')):
  if f.is_file():copy(f,assets/'geometry-input'/f.relative_to(src/'geometry-input'))
 copy(src/'pedestrian_tracks.pt',assets/'pedestrian_tracks.pt')
 copy(cp/'refiner_state.pt',target/'weights/refiner_scene017_vr_v1_s.pt')
 for source,name in [(sam/'checkpoints/sam2.1_hiera_tiny.pt','sam2.1_hiera_tiny.pt'),(Path('/mnt/fjn/projects/Pointforward-pedestrian/ckpt/ckpt_002000.pt'),'pedestrian_head_ckpt_002000.pt'),(Path('/mnt/data/output/pointforward/smpl_models/SMPL_NEUTRAL.npz'),'SMPL_NEUTRAL.npz')]:copy(source,target/'weights'/name)
 for n in ('final_global_gaussians_frame000_preview.ply','final_global_runtime_predicted_front3.mp4','final_global_runtime_gt_vs_predicted_front3.mp4'):copy(src/n,assets/n)
 for n in ('pipeline_timing.json','optimization_timing.json','result_summary.json','acceptance_report.json','input_provenance.md','report.md','run_cold.py'):
  copy(src/n,release/'evidence'/n)
 for n in ('options.json','checkpoints.json','manifest.json','parameter_contract.json','loss_steps.jsonl'):
  copy(src/'joint-all'/n,release/'evidence'/('joint_'+n))
 copy(cp/'parameter_changes.json',release/'evidence/parameter_changes.json')
 copy(src/'runtime-render-source/manifest.json',release/'evidence/original_render_manifest.json')
 copy(src.parent/'acceptance_contract.json',release/'evidence/research_acceptance_contract.json')
 rounds=src.parents[5]/'docs/improvement-loop/rounds/round-104'
 # Resolve via the known project root rather than version-number ancestry.
 project=next(p for p in src.parents if (p/'docs/improvement-loop/rounds/round-104').is_dir())
 for n in ('loss_psnr_trends.png','vehicle_crops.png'):copy(project/'docs/improvement-loop/rounds/round-104'/n,assets/n)
 import yaml
 cfg=yaml.safe_load((src/'config_build.yaml').read_text());cfg['data']['root']='__DATA_ROOT__';cfg['data']['geometry_directory']='__OUTPUT_ROOT__/geometry-input';cfg['model']['pretrained_backbone']=False
 cfg['depth']['repo_root']='__PROJECT_ROOT__/docs/releases/VR-v1-s/source/InfiniDepth';cfg['depth']['checkpoint']='__PROJECT_ROOT__/weights/infinidepth_depthsensor.ckpt'
 cfg['pedestrians'].update(smpl_model_path='__PROJECT_ROOT__/weights/SMPL_NEUTRAL.npz',checkpoint='__PROJECT_ROOT__/weights/pedestrian_head_ckpt_002000.pt',tracks_file='__OUTPUT_ROOT__/pedestrian_tracks.pt')
 (release/'config.template.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 timer=json.loads((src/'pipeline_timing.json').read_text());recipe={}
 for stage in timer['stages']:
  cmd=stage['command'];out=[]
  for i,x in enumerate(cmd):
   x=x.replace(str(src/'source-snapshot'),'__SOURCE__').replace(str(src),'__OUTPUT_ROOT__').replace('/mnt/data/fjn/projects/PointForward/ckpt/ckpt_020000_waymo_ds_paper_original.pt','__CHECKPOINT__').replace('/mnt/data/output/pointforward/smpl_models/SMPL_NEUTRAL.npz','__PROJECT_ROOT__/weights/SMPL_NEUTRAL.npz')
   if i==0:x='__SAM_PYTHON__' if stage['name']=='rgb_masks' else '__PYTHON__'
   out.append(x)
  recipe[stage['name']]=out
 save(release/'pipeline_recipe.json',recipe)
 save(target/base/'preparation.json',dict(candidate=candidate,run_id=rid,validation_dir=rr,release='docs/releases/VR-v1-s',source_run=str(src),copy_sources=copies))
 wf('register-run','--candidate-id',candidate,'--run-id',rid,'--variant','compact-delivery-validation','--directory-id',rd,'--config-identity','docs/releases/VR-v1-s/pipeline_recipe.json;rank32;full3cam;union1;1700max;raw1480stop','--code-identity','docs/releases/VR-v1-s/source;portable orchestration only','--data-identity','delivered scene017 state; original156frame3cam dataset for independent replay')
 wf('set-run-status','--run-id',rid,'--status','running')
 print(json.dumps(dict(candidate=candidate,run_id=rid,validation_dir=rr,release=str(release)),ensure_ascii=False))

if __name__=='__main__':main()

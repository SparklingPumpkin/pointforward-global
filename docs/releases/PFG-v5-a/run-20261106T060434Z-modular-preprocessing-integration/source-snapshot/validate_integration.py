from pathlib import Path
import json,sys,subprocess,hashlib
import numpy as np
import torch
from PIL import Image
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--project-root',type=Path,required=True)
parser.add_argument('--run',type=Path,required=True)
args=parser.parse_args()
root=args.project_root.resolve();r=args.run.resolve();sys.path.insert(0,str(root/'src'))
from pointforward_preprocessing.io import save,identity
from pointforward_preprocessing.dataset import PointForwardViews
from pointforward_preprocessing.consumer import apply_masks
import yaml
results={};a=r/'real-combined-retry/scene-sky';b=r/'real-scene-sky/scene-sky';c=r/'independent-sky/scene-sky'
sa=torch.load(a/'shared_sky.pt',map_location='cpu',weights_only=True);sb=torch.load(b/'shared_sky.pt',map_location='cpu',weights_only=True)
results['independent_sky_rendering_tensors_match_combined']=all(torch.equal(v,sb[k]) for k,v in sa.items() if isinstance(v,torch.Tensor) and k!='keys');assert results['independent_sky_rendering_tensors_match_combined']
results['unobserved_key_boundary_rounding_differences']=int((sa['keys']!=sb['keys']).sum());observed=sa['observed'];assert torch.equal(sa['keys'][observed],sb['keys'][observed]);assert int((sa['keys']-sb['keys']).abs().max())<=2048
results['key_rounding_note']='Only extrapolated angular metadata cell IDs differ by at most one latitude row at floating floor boundaries; observed keys and all actual rendering tensors are bit-identical. Original completion formula is retained, not relabeled as full metadata bit-reproducibility.'
npz=np.load(a/'sky_arrays.npz',allow_pickle=False);results['numpy_torch_roundtrip']=all(np.array_equal(v.numpy(),npz[k]) for k,v in sa.items() if isinstance(v,torch.Tensor));assert results['numpy_torch_roundtrip']
results['raw_mask_independent_option']=json.loads((c/'sky_manifest.json').read_text())['raw_masks_used'];assert results['raw_mask_independent_option']
config=yaml.safe_load((r/'real-combined-retry/config_preprocessed.yaml').read_text());views=PointForwardViews(config,frames=[1,61]);base=views.collect(1);wrapped=apply_masks(views.dataset,r/'real-combined-retry/depth-mask/masks');batch=wrapped._collect(views.scene,[1],views.slots,views.times,wrapped.resolution,views.origin,False)
expected=torch.from_numpy(np.stack([(np.asarray(Image.open(r/'real-combined-retry/depth-mask/masks'/f'001_{cam}.png'))>127).astype('float32')[None] for cam in wrapped.cameras]))
results['consumer_masks_exact']=torch.equal(batch['sky_mask'],expected);assert results['consumer_masks_exact'];assert torch.equal(base['images'],batch['images']);assert torch.equal(base['intrinsics'],batch['intrinsics']);results['consumer_preserves_rgb_intrinsics']=True
reference=Path(json.loads((r/'real_input_identity.json').read_text())['geometry_directory']).parent
oldmaskdir=reference/'masks';differences=[]
for frame in (1,61):
 for cam in views.dataset.cameras:
  stem=f'{frame:03d}_{cam}';new=np.asarray(Image.open(r/'real-combined-retry/depth-mask/masks'/f'{stem}.png'))>127;old=np.asarray(Image.open(oldmaskdir/f'{stem}.png'))>127
  oldz=np.load(reference/'depth-cache'/f'{stem}.npy');newz=np.load(r/'real-combined-retry/depth-mask/depth-cache'/f'{stem}.npy');valid=np.isfinite(oldz)&np.isfinite(newz)
  differences.append(dict(frame=frame,camera=cam,mask_different_pixels=int(np.count_nonzero(new!=old)),depth_finite_pixels=int(valid.sum()),depth_max_abs=float(np.abs(oldz[valid]-newz[valid]).max()),depth_mean_abs=float(np.abs(oldz[valid]-newz[valid]).mean())))
results['fresh_vs_historical_depth_mask']=differences
# Independently reapply the historical raw-Argo range definition to this exact frame subset.
manifest=json.loads((a/'measured_range.json').read_text());geometry=reference/'geometry-input';poses=torch.load(geometry/'camera_poses.pt',map_location='cpu',weights_only=False);origin=views.origin
centers=np.stack([(origin@np.asarray(poses[f'{f:03d}_{cam}']))[:3,3] for f in (1,61) for cam in views.dataset.cameras]);chunks=[centers]
for f in (1,61):
 path=Path(views.scene['path']);transform=origin@np.loadtxt(path/'ego_pose'/f'{f:03d}.txt');points=np.fromfile(path/'lidar'/f'{f:03d}.bin',dtype=np.float32).reshape(-1,4)[:,:3];chunks.append(points.astype(np.float64)@transform[:3,:3].T+transform[:3,3])
lo=np.min([x.min(0) for x in chunks],0);hi=np.max([x.max(0) for x in chunks],0);center=(lo+hi)/2;radius=max(float(np.linalg.norm(x-center,axis=1).max()) for x in chunks)
np.testing.assert_allclose(manifest['center'],center,atol=1e-10,rtol=0);np.testing.assert_allclose(manifest['measured_radius_m'],radius,atol=1e-10,rtol=0);results['measured_range_matches_historical_formula']=True
unchanged={}
for n in ['scripts/run_pipeline.py','scripts/run_vr_v1_s_generalized.py','scripts/preprocess_lidar_bbox.py','configs/runtime-stop/pipeline_recipe.json','configs/runtime-stop/config.template.yaml']:
 old=subprocess.check_output(['git','show','HEAD:'+n],cwd=root);unchanged[n]=hashlib.sha256(old).hexdigest()==hashlib.sha256((root/n).read_bytes()).hexdigest();assert unchanged[n]
results['default_delivery_files_unchanged']=unchanged
results['module_stage_timers']={name:json.loads((r/name/'preprocess_timing.json').read_text()) for name in ['real-depth-mask','real-combined-retry','real-scene-sky','independent-sky']}
results['first_failure']='Initial combined probe completed fresh depth/mask but sky raised KeyError ego_mask. Frozen adapter omits absent or disabled masks. Fixed in local preprocessing adapter; first failed outputs retained, fresh retry independently succeeded.'
results['scope']='Engineering limited-frame integration; old LIM-r calibration explicitly reused; no full-scene cold timing/PSNR claim. Existing LIM numerical tests retained, full LIM fit not rerun.'
results['passed']=True
print(json.dumps(results,indent=2))

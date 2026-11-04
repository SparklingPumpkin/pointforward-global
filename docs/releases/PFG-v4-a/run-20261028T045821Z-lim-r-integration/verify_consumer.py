from pathlib import Path
import json
import numpy as np
import torch
import yaml
from pointforward_calibration.consumer import install
install()
from scripts.train import build_dataset
run=Path(__file__).resolve().parent
root=next(p for p in run.parents if (p/'src/pointforward_calibration').is_dir())
records=[]
for label in ['argo','ncore']:
 cfg=yaml.safe_load((run/(label+'_config.yaml')).read_text());cfg['pedestrians']['enabled']=False
 directory=run/(label+'-fresh')/'reapplied-final';cfg['data']['geometry_directory']=str(directory)
 ds=build_dataset(cfg,'val');meta=ds.scenes[0]
 manifest=json.loads((directory/'application_manifest.json').read_text())
 assert meta['frames']==manifest['frames']
 poses=torch.load(directory/'corrected_instances.pt',weights_only=False)
 cameras=torch.load(directory/'camera_poses.pt',weights_only=False)
 for tid,ann in poses[meta['name']].items():
  for f,(pose,size) in ann.items():np.testing.assert_array_equal(meta['instances'][tid][f][0],pose)
 views=[]
 for f in meta['frames']:
  for cam in ds.cameras:
   expected=np.load(directory/'sparse'/f'{f:03d}_{cam}.npz')['depth']
   depth,mask=ds._lidar_depth(meta,f,cam,np.eye(3),ds.resolution)
   np.testing.assert_array_equal(depth.numpy()[0],expected)
   np.testing.assert_array_equal(mask.numpy()[0],expected>0)
   np.testing.assert_allclose(ds._cam_to_world(meta,f,cam,np.eye(4)),cameras[f'{f:03d}_{cam}'])
   views.append(dict(frame=f,camera=cam,nonzero=int((expected>0).sum())))
 records.append(dict(dataset=label,scene=meta['name'],frames=meta['frames'],views=views,
   corrected_poses_loaded=True,per_camera_depth_exact=True,per_camera_exposure_pose_exact=True))
print(json.dumps(records,ensure_ascii=False,indent=2))
(run/'consumer_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')

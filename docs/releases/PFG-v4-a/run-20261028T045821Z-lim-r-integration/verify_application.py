"""Reapply the final export implementation to this run's fresh fits (cached stage)."""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import yaml
from pointforward_calibration.pipeline import FreshScene,scene_export,save

RUN=Path(__file__).resolve().parent
PROJECT=next(p for p in RUN.parents if (p/'src/pointforward_calibration').is_dir())
SOURCE=PROJECT/'docs/releases/VR-v1-s/source'
for label in ['argo','ncore']:
    before=RUN/(label+'-fresh')
    calibration=json.loads((before/'calibration.json').read_text())
    cfg=yaml.safe_load((RUN/(label+'_config.yaml')).read_text())
    scene=cfg['data']['val_scenes'][0]
    dest=before/'reapplied-final';dest.mkdir(exist_ok=True)
    a=FreshScene(cfg,scene,dest,'0',before/'selection.json',PROJECT/'weights/yolo11n-seg.pt',
                 PROJECT/'weights/sam2.1_hiera_tiny.pt',SOURCE/'SAM2')
    a.frame_selection=json.loads((before/'frame_selection.json').read_text())
    recovery={o['key']:o for o in json.loads((before/'all_fit_observations.json').read_text())['records']}
    a.rows=[]
    for rec in json.loads((before/'evaluation.json').read_text())['records']:
        with np.load(before/(rec['key']+'_result.npz')) as d:
            o=dict(rec,ids=d['pool_point_ids'].copy(),old_world=d['old_world'].copy(),
                   old_box=d['old_actor'].copy(),old_score=rec['old'])
        if o['mask_accepted']:
            path=before/(o['key']+'_mask.png');row=recovery[o['key']]
            if row['mode']=='independent_best_overlap_no_exclusive_assignment':
                path=before/f'rgb_{o["frame"]:03d}_{o["camera"]}_proposal{row["proposal_index"]}.png'
            o['mask']=np.array(Image.open(path))>0
            o['distance']=distance_transform_edt(~o['mask']).astype(np.float32)
        a.rows.append(o)
    frames=json.loads((before/'application_manifest.json').read_text())['frames']
    start=time.perf_counter()
    result=scene_export(a,calibration['clocks'],calibration['spatial'],dest,frames)
    save(dest/'cached_validation_timer.json',dict(start=start,end=time.perf_counter(),scope='cached_incremental',
        source_calibration=str(before/'calibration.json'),scene_cache_reused=True,application=result))
    print(label,result,flush=True)

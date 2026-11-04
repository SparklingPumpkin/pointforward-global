#!/usr/bin/env python3
"""Run accepted LIM-v1-r preprocessing from raw processed-dataset inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'docs/releases/VR-v1-s/source'
sys.path.insert(0,str(SOURCE))
sys.path.insert(1,str(PROJECT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scene',default='017')
    p.add_argument('--device',default='0',help='Logical device within CUDA_VISIBLE_DEVICES')
    p.add_argument('--selection',type=Path,help='Optional preselected geometry-only vehicle list')
    p.add_argument('--detector',type=Path,default=PROJECT/'weights/yolo11n-seg.pt')
    p.add_argument('--sam-checkpoint',type=Path,default=PROJECT/'weights/sam2.1_hiera_tiny.pt')
    p.add_argument('--sam-root',type=Path,default=SOURCE/'SAM2')
    p.add_argument('--apply-frames',type=int,nargs='+',help='Diagnostic partial export only; never used by default reconstruction')
    args=p.parse_args()
    import yaml
    from pointforward_calibration.pipeline import FreshScene,accepted_fit,scene_export,save,PACKAGE
    for path in [args.config,args.detector,args.sam_checkpoint]:
        if not path.is_file():p.error('Required local input missing: '+str(path))
    if args.out.exists() and any(args.out.iterdir()):p.error('Output must be empty; implicit historical scene reuse is disabled')
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();stages={};status='running'
    save(args.out/'input_manifest.json',dict(scene=args.scene,config=str(args.config.resolve()),
        selection=str(args.selection.resolve()) if args.selection else 'fresh geometry-only selection',
        source_scene_cache_reused=False,algorithm='LIM-v1-r',
        code_files=[dict(path=str(x.resolve()),sha256=hashlib.sha256(x.read_bytes()).hexdigest())
                    for x in sorted(PACKAGE.glob('*.py'))]+[
                    dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())],
        generic_weights=[dict(path=str(x.resolve()),sha256=hashlib.sha256(x.read_bytes()).hexdigest())
                         for x in [args.detector,args.sam_checkpoint]],
        source_code_provenance=json.loads((PACKAGE/'provenance.json').read_text())))
    for path in PACKAGE.glob('*_config.json'):shutil.copy2(path,args.out/path.name)
    shutil.copy2(args.config,args.out/'input_config.yaml')
    try:
        cfg=yaml.safe_load(args.config.read_text())
        if tuple(cfg['data'].get('resolution',[576,768]))!=(576,768):
            p.error('Accepted LIM-v1-r image/measurement resolution is fixed at 576x768')
        a=FreshScene(cfg,args.scene,args.out,args.device,args.selection,args.detector,args.sam_checkpoint,args.sam_root)
        stages['initialization']=time.perf_counter()-start
        t=time.perf_counter();clocks,space=accepted_fit(a,args.out);stages['fit_and_diagnostics']=time.perf_counter()-t
        t=time.perf_counter();application=scene_export(a,clocks,space,args.out,args.apply_frames);stages['full_scene_export']=time.perf_counter()-t
        save(args.out/'raw_input_identity.json',dict(files=a.input_files,timestamp_source=a.timestamp_source,
             static_world_coordinates_unchanged=True,rolling_shutter='not corrected'))
        save(args.out/'result_summary.json',dict(algorithm='LIM-v1-r',application=application,
             diagnostic_only_partial_export=args.apply_frames is not None,
             shared_lidar_seconds=clocks['lidar']['shared_s'],shared_bbox_seconds=clocks['bbox']['shared_s']))
        status='completed'
    finally:
        save(args.out/'preprocess_timing.json',dict(status=status,start=start,end=time.perf_counter(),
             stages_seconds=stages,scope='stage',input_boundary='before_fresh_dataset_and_models',
             output_boundary='diagnostics_corrected_instances_and_sparse_depth_written',scene_cache_reused=False))


if __name__=='__main__':main()

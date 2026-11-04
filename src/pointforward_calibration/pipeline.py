"""Fresh-input adapter and full-scene export for the accepted LIM-v1-r solver.

Historical numerical functions live in r.py and its frozen dependencies. This
module only supplies raw inputs and converts the result to the reconstruction
loader's geometry contract. It never imports the research repository.
"""
from __future__ import annotations
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch

from . import base, c, ground, masks, q, r, selection

PACKAGE = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def zbuffer(points, camera, intrinsic, resolution=(576, 768)):
    """Nearest positive camera-z; same projection for static and dynamic inputs."""
    h, w = resolution
    cp = base.transform(points, np.linalg.inv(camera))
    valid = np.isfinite(cp).all(1) & (cp[:, 2] > .1)
    uv = np.full((len(cp), 2), -1.)
    uv[valid] = (cp[valid] @ intrinsic.T)[:, :2] / cp[valid, 2:]
    xy = np.rint(uv).astype(np.int64)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    flat = np.full(h*w, np.inf)
    np.minimum.at(flat, xy[valid, 1]*w+xy[valid, 0], cp[valid, 2])
    flat[~np.isfinite(flat)] = 0.
    return flat.reshape(h, w).astype(np.float32)


class FreshScene(q.Fresh):
    """Use the established dataset API with portable paths and fresh selection."""
    def __init__(self, config, scene, out, device='0', selection_file=None,
                 detector=None, sam_checkpoint=None, sam_root=None):
        from scripts.train import build_dataset
        cfg = copy.deepcopy(config)
        cfg['data'].pop('geometry_directory', None)
        cfg.setdefault('pedestrians', {})['enabled'] = False
        self.ds = build_dataset(cfg, 'val')
        self.meta = next(m for m in self.ds.scenes if m['name'] == scene)
        self.cfg = json.loads((PACKAGE/'c_config.json').read_text())
        self.spec = dict(key=scene, scene=scene, root=cfg['data']['root'],
                         source='argoverse' if cfg['data']['layout']=='argoverse_drivestudio' else 'waymo')
        self.out = self.source = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.key, self.device = scene, device
        self.started = time.perf_counter()
        self.stages, self.input_files, self.images = {}, {}, {}
        self.cache = OrderedDict()
        self.base, self.draw = base, base.draw
        self.groundmod, self.maskmod, self.selectmod = ground, masks, selection
        self.sam_root = Path(sam_root)
        self.maskmod.CONTRACT = dict(detector=str(detector), sam=str(sam_checkpoint))
        self.root = Path(self.meta['path'])
        self.frames = sorted(self.meta['frames'])
        path = self.root/'infos/timestep_to_timestamps.json'
        self.timestamps = json.loads(path.read_text()) if path.exists() else None
        self.timestamp_source = str(path) if path.exists() else 'nominal 10Hz; per-camera exposure unavailable'
        origin = int(self.timestamps[f'{self.frames[0]:06d}']['ego']) if self.timestamps else 0
        self.times = {f: (int(self.timestamps[f'{f:06d}']['ego'])-origin)*1e-6
                      if self.timestamps else f*.1 for f in self.frames}
        # Missing per-camera timestamps in a present manifest are an error;
        # silently treating them as synchronous would change accepted semantics.
        self.camtimes = {(f,cam): (int(self.timestamps[f'{f:06d}']['camera'][str(cam)])-origin)*1e-6
                         if self.timestamps else self.times[f]
                         for f in self.frames for cam in self.ds.cameras}
        self.ego = c.Poses([self.times[f] for f in self.frames],
                           [self.ds._ego_pose(self.meta,f) for f in self.frames])
        self.actors = {int(t): c.Poses([self.times[f] for f in sorted(a)],
                       [a[f][0] for f in sorted(a)]) for t,a in self.meta['instances'].items() if a}
        self.obs, self.sources, self.sample_cache = [], {}, {}
        self.use_residual = True
        if selection_file:
            self.selection = json.loads(Path(selection_file).read_text())
            self.selection['spec'] = self.spec
        else:
            self.select()
        save(self.out/'selection.json', self.selection)

    def select(self):
        # The accepted selector uses only original geometry, source-image area,
        # trajectory motion and visibility. Mask agreement never picks vehicles.
        result = selection.select(self.ds, self.meta, self.cfg['selection'], self.spec)
        ranked = result['vehicles']
        for v in ranked:
            v['pipeline_score'] = v['score']*(self.cfg['selection']['side_camera_multiplier']
                                           if v['representative']['camera'] != 0 else 1.)
        ranked.sort(key=lambda v:(v['moving'],v['pipeline_score']), reverse=True)
        chosen = ranked[:self.cfg['selection']['deliver_vehicles']]
        for v in chosen:
            v['split'] = 'development_track'
            for i, row in enumerate(v['observations']):
                row['role'] = 'validation_frame' if i%3 == 1 else 'fit'
                row['key'] = f't{v["track_id"]:03d}_f{row["frame"]:03d}_c{row["camera"]}'
        self.selection = dict(vehicles=chosen, eligible_tracks=len(ranked), spec=self.spec,
            frozen_before_masks_and_fit=True, selection_uses_agreement=False,
            split='fit/gate used only for integer frame choice; every retained observation fits continuous clocks')

    def prepare_pairings(self):
        # Accepted Q path: real neighboring sweeps, ground removal, bounded
        # completion, independent integer frame tests, then uniform-source rule.
        super().prepare_pairings()
        # Q excludes unsupported observations, but its selection still lists
        # them. Keep that original evidence and render only retained entries.
        save(self.out/'selection_before_source_exclusion.json', self.selection)
        keys = {o['key'] for o in self.rows}
        for v in self.selection['vehicles']:
            v['observations'] = [o for o in v['observations'] if o['key'] in keys]
        unsupported = [v['track_id'] for v in self.selection['vehicles'] if not v['observations']]
        save(self.out/'unsupported_tracks.json', dict(tracks=unsupported,
            reason='No uniform real source observation; no current-frame fallback'))
        self.selection['vehicles'] = [v for v in self.selection['vehicles'] if v['observations']]
        save(self.out/'selection.json', self.selection)


def accepted_fit(adapter, out):
    """Q preceding reference is computed fresh, then unchanged R fits once."""
    a = adapter
    a.prepare()
    a.measure()
    a.prepare_pairings()
    if not a.rows:
        raise RuntimeError('No source-consistent calibration observations; see source_consistency.json')
    a.fit()  # Supplies R's original previous-good/reference protection.
    save(out/'reference_calibration.json', dict(tracks=a.tracks, frame_selection=a.frame_selection))
    save(out/'rgb_measurement_contract.json', a.maskmod.CONTRACT)
    recovery = {x['key']: x for x in a.mask_recovery}
    for o in a.rows:
        t = a.tracks[str(o['track'])]
        o['pool'] = o['base_world'].copy()
        o['old_world'] = a.move(o,t['residual_m'],True) if a.usable(o) and t['apply'] else o['base_world'].copy()
        o['old_box'] = a.box(o,t['bbox_center_m']) if a.usable(o) and t['apply'] else o['selected_box'].copy()
        o.update(batch=a.key, scene=a.key, lidar_time=a.times[o['lidar_frame']],
                 bbox_time=a.times[o['bbox_frame']], image_supported=o['mask_accepted'],
                 mask_identity_note='independent RGB measurement; '+recovery[o['key']]['mode'])
        if o['image_supported']:
            proposals = json.loads((out/f'rgb_{o["frame"]:03d}_{o["camera"]}_proposals.json').read_text())
            sha = hashlib.sha256(o['mask'].tobytes()).hexdigest()
            match = next((d for d in proposals if d.get('sha256')==sha),None)
            if match and match['class_id']==2 and o['size'][0]>7.5 and o['size'][2]>2.4:
                o['image_supported'] = False
                o['mask_identity_note'] = 'large vehicle vs sedan mask mismatch; geometry-only fit retained'
        o['old_score'] = a.score(o,o['old_world'],o['old_box'])
        o['selected_score'] = a.score(o,o['pool'],o['selected_box'])
    r.export_ownership(a,a.rows,a.selection,out)
    r.owned_rows(a,a.rows,out)
    lidar, lp = r.fit_hierarchy(a,a.rows,'lidar')
    for o in a.rows:
        tau = lidar['tracks'][str(o['track'])]['total_s']
        o['time_world'] = r.warp(a,o,tau,o['pool'][o['owned']])
        o['time_points'] = r.warp(a,o,tau,o['fit_points'])
    bbox, bp = r.fit_hierarchy(a,a.rows,'bbox')
    clocks = dict(lidar=lidar,bbox=bbox)
    for o in a.rows:
        o['time_box'] = r.warp(a,o,bbox['tracks'][str(o['track'])]['total_s'],box=True)
    space = r.small_spatial(a,a.rows)
    save(out/'time_profiles.json', dict(lidar=lp,bbox=bp))
    r.finish_batch(a,a.rows,a.selection,out,clocks,space)
    return clocks, space


def source_frames(frames, lidar_offset, bbox_offset, scan_exists):
    """Uniform offsets, including both endpoints; no per-frame fallback."""
    available = set(frames)
    kept, excluded = [], []
    for f in frames:
        reasons = []
        if f+lidar_offset not in available or not scan_exists(f+lidar_offset):
            reasons.append('actual_selected_lidar_source_missing')
        if f+bbox_offset not in available:
            reasons.append('actual_selected_bbox_source_missing')
        (excluded if reasons else kept).append(dict(frame=f,reasons=reasons) if reasons else f)
    return kept, excluded


def track_time(clocks, stream, tid):
    return clocks[stream]['tracks'].get(str(tid),{}).get('total_s',clocks[stream]['shared_s'])


def scene_export(a, clocks, space, out, only_frames=None):
    """Apply fitted scene clocks to all vehicles; unsampled residuals are zero.

    Static world coordinates come from RGB frame f's real scan, unchanged.
    Owned dynamic points are replaced by real scan f+offset and transported
    separately for each camera exposure. Ambiguous/rejected points stay raw.
    """
    import tqdm
    ld, bd = (a.frame_selection[s]['offset'] for s in ['lidar','bbox'])
    frames, excluded = source_frames(a.frames,ld,bd,
        lambda f:(a.root/'lidar'/f'{f:03d}.bin').exists())
    if only_frames is not None:
        requested=set(only_frames)
        frames=[f for f in frames if f in requested]
    if not frames:
        raise RuntimeError('No output frames with uniform actual source offsets')
    corrected, bbox_audit = {}, []
    for tid,ann in a.meta['instances'].items():
        corrected[tid]={}
        bias=np.array(space.get(str(tid),{}).get('bbox_bias_m',[0.,0.,0.]))
        for f in frames:
            if f not in ann:continue
            g=f+bd
            if g not in ann:
                bbox_audit.append(dict(track=int(tid),frame=f,reason='selected_bbox_source_annotation_missing'))
                continue
            src,_=r.pose(a,tid,a.times[g]+track_time(clocks,'bbox',tid))
            target,_=r.pose(a,tid,a.times[f])
            if src is None or target is None:
                bbox_audit.append(dict(track=int(tid),frame=f,reason='trajectory_outside_bounded_support'))
                continue
            box=target@np.linalg.inv(src)@ann[g][0]
            box[:3,3]+=box[:3,:3]@bias
            corrected[tid][f]=(box,ann[g][1].copy())
    torch.save({a.meta['name']:corrected},out/'corrected_instances.pt')
    sparse=out/'sparse';sparse.mkdir(exist_ok=True)
    camera_poses={};audit=[]
    observed={(o['track'],o['frame'],o['camera']):o for o in a.rows}
    pool_cache=OrderedDict()
    ownership_cache=OrderedDict()

    def assign_ownership(row):
        # Geometry-only ownership has no camera dependence. Share the exact
        # result across cameras and adjacent RGB frames using the same scan.
        key=(row['track'],row['lidar_frame'])
        if not row['image_supported'] and key in ownership_cache:
            row['owned']=ownership_cache[key]
        else:
            r.ownership(a,row)
            if not row['image_supported']:
                ownership_cache[key]=row['owned'].copy()
                if len(ownership_cache)>256:ownership_cache.popitem(last=False)
    for f in tqdm.tqdm(frames,desc='LIM-r full-scene prompts'):
        current=a.world(f).copy();g=f+ld;world=a.world(g).copy()
        # Fresh completion is the same Q point-pool constructor; recompute
        # per source scan and retain raw IDs in each application audit.
        pools={};current_pools={}
        for tid,ann in a.meta['instances'].items():
            if f not in ann or g not in ann:continue
            for sf,raw,dest in [(g,world,pools),(f,current,current_pools)]:
                cachekey=(tid,sf)
                if cachekey in pool_cache:
                    dest[tid]=pool_cache[cachekey]
                    continue
                actor,size=ann[sf]
                local=base.transform(raw,np.linalg.inv(actor))
                _,info=ground.Calibration.ground(a,raw,actor,size,np.random.default_rng(a.cfg['seed']+tid*10000+sf))
                good=np.ones(len(raw),bool)
                if info['valid']:good=raw@np.array(info['normal'])+info['d']>a.cfg['ground']['remove_height_m']
                half=np.maximum(size*a.cfg['expanded_box_factor']*.5,size*.5+a.cfg['minimum_expansion_margin_m'])
                seeds=np.flatnonzero((abs(local)<=half).all(1)&good)
                roi=np.flatnonzero((abs(local)<=half+np.array(q.j.CONFIG['completion']['search_extension_m'])).all(1)&good)
                ids,_=q.j.complete_clusters(local[roi],roi,seeds,q.j.CONFIG['completion']) if info['valid'] else (seeds,{})
                dest[tid]=(ids,raw[ids],size)
                pool_cache[cachekey]=dest[tid]
                if len(pool_cache)>256:pool_cache.popitem(last=False)
        for cam in a.ds.cameras:
            camera=a.camera(f,cam);camera_poses[f'{f:03d}_{cam}']=camera
            rgb,k=a.image(f,cam);keep=np.ones(len(current),bool);pieces=[];per_track=[]
            # Exclusive point claims: each real source ID is inserted at most
            # once, nearest normalized box-center wins. No duplicated overlap.
            claims={}
            for tid,(ids,points,size) in pools.items():
                anchor=a.meta['instances'][tid][g][0]
                loc=base.transform(points,np.linalg.inv(anchor))
                for pid,d in zip(ids,np.linalg.norm(loc/np.maximum(size/2,.01),axis=1)):
                    if int(pid) not in claims or d<claims[int(pid)][0]:claims[int(pid)]=(float(d),tid)
            for tid,(ids,points,size) in pools.items():
                template=observed.get((tid,f,cam))
                boxsource=a.meta['instances'][tid].get(f+bd)
                if boxsource is None or f not in corrected[tid]:continue
                o=dict(track=tid,frame=f,camera=cam,camera_time=a.camtimes[f,cam],
                    lidar_time=a.times[g],bbox_time=a.times[f+bd],lidar_frame=g,bbox_frame=f+bd,
                    camera_pose=camera,K=k,size=size,pool=points,selected_box=boxsource[0],
                    old_world=points,old_box=boxsource[0],image_supported=False,
                    mask_identity_note='full-scene geometry-only ownership, no new mask claim',
                    old_score=dict(point_mask_distance_px=100.,containment=0.))
                if template:
                    for key in ['mask','distance','image_supported','mask_identity_note','old_score','old_box','old_world']:
                        if key in template:o[key]=template[key]
                    # Current full-scene population may differ from selected
                    # reference population; protection must use identical IDs.
                    if not np.array_equal(ids,template['ids']):
                        o['old_world']=points;o['old_score']=dict(point_mask_distance_px=100.,containment=0.)
                assign_ownership(o)
                own=o['owned'] & np.array([claims[int(pid)][1]==tid for pid in ids],dtype=bool)
                moved=r.warp(a,o,track_time(clocks,'lidar',tid),points[own])
                target,gap=r.pose(a,tid,o['camera_time'])
                if moved is None or target is None:
                    per_track.append(dict(track=int(tid),status='unsupported_dynamic_transport_no_fallback'))
                    continue
                moved+=target[:3,:3]@np.array(space.get(str(tid),{}).get('point_bias_m',[0.,0.,0.]))
                pieces.append(moved)
                # Only owned current-scan points are removed. Rejected candidates
                # remain unchanged; there is no global/static time warp.
                ci,cp,cs=current_pools[tid]
                co=dict(o,pool=cp,size=cs,lidar_time=a.times[f],lidar_frame=f,
                        old_world=cp,old_score=dict(point_mask_distance_px=100.,containment=0.))
                if f==g:
                    keep[ci[o['owned']]]=False
                    removed=int(o['owned'].sum())
                else:
                    assign_ownership(co);keep[ci[co['owned']]]=False
                    removed=int(co['owned'].sum())
                per_track.append(dict(track=int(tid),source_frame=g,source_points=int(own.sum()),
                    removed_current_dynamic_points=removed,target_extrapolation_s=gap,
                    track_fitted=str(tid) in clocks['lidar']['tracks']))
            background=current[keep]
            joined=np.concatenate([background]+pieces) if pieces else background
            depth=zbuffer(joined,camera,k,a.ds.resolution)
            np.savez_compressed(sparse/f'{f:03d}_{cam}.npz',depth=depth)
            # Persist IDs sufficient to reconstruct which untouched source points
            # and which dynamic scan participated in this per-camera projection.
            np.savez_compressed(sparse/f'{f:03d}_{cam}_support.npz',raw_background_ids=np.flatnonzero(keep))
            audit.append(dict(frame=f,camera=int(cam),lidar_frame=g,bbox_frame=f+bd,
                raw_static_points=int(keep.sum()),static_coordinates_unchanged=True,
                tracks=per_track,nonzero_depth_pixels=int((depth>0).sum())))
    torch.save(camera_poses,out/'camera_poses.pt')
    save(out/'application_manifest.json',dict(schema_version=1,algorithm='LIM-v1-r',scene=a.key,
        frames=frames,cameras=list(a.ds.cameras),resolution=list(a.ds.resolution),
        frame_offsets=dict(lidar=ld,bbox=bd),excluded_frames=excluded,
        excluded_track_frames=bbox_audit,records=audit,
        source_data_read_only=True,static_world_coordinates_unchanged=True,
        camera_policy='original calibration at each recorded exposure; no fitted camera shift',
        bbox_pose_convention='corrected common trajectory at group times; interpolated by existing model',
        unselected_tracks='scene-shared clocks, zero per-track residual and space bias; geometry-only ownership',
        diagnostics='automatic in-sample masks; no independent physical timing guarantee',rolling_shutter='not corrected'))
    return dict(frames=len(frames),camera_views=len(audit),excluded_frames=len(excluded))

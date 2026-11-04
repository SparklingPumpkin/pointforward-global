"""Frozen LIM-v1-r dependency. See provenance.json; adapter is in pipeline.py."""
from __future__ import annotations
import argparse, copy, hashlib, html, json, math, os, shutil, sys, time
from collections import OrderedDict, defaultdict, Counter
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt, map_coordinates, label
from scipy.optimize import minimize, linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation, Slerp
HERE = Path(__file__).resolve().parent
PROJECT = HERE
CFG=json.loads((HERE/'r_config.json').read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def distance(points, o, adapter):
    uv,z=adapter.pixels(points,o)
    d=map_coordinates(o['distance'],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100.)
    d[z<=.1]=100.
    return np.minimum(d,100.)

def pose(adapter, tid, t):
    """True interpolation, or explicitly audited bounded SE(3) extrapolation.

    A missing source scan is NEVER synthesized. Endpoint trajectory continuation
    uses its two nearest annotations; it is not current-frame source fallback.
    """
    trajectory=adapter.actors[tid]
    result=trajectory.at(t)
    if result is not None:return result,0.
    ts,mats=trajectory.ts,trajectory.mats
    gap=max(ts[0]-t,t-ts[-1],0.)
    if gap>CFG['time']['maximum_pose_extrapolation_s']+1e-7:return None,gap
    if len(ts)==1:return mats[0].copy(),gap
    i,jj=(0,1) if t<ts[0] else (-2,-1)
    alpha=(t-ts[i])/(ts[jj]-ts[i])
    out=mats[i].copy()
    out[:3,3]=mats[i,:3,3]+alpha*(mats[jj,:3,3]-mats[i,:3,3])
    rot=Rotation.from_matrix(mats[i,:3,:3].T@mats[jj,:3,:3]).as_rotvec()
    out[:3,:3]=mats[i,:3,:3]@Rotation.from_rotvec(alpha*rot).as_matrix()
    return out,gap

def warp(adapter, o, tau, points=None, box=False):
    source_t=o['bbox_time'] if box else o['lidar_time']
    source,eg=pose(adapter,o['track'],source_t+tau)
    target,tg=pose(adapter,o['track'],o['camera_time'])
    if source is None or target is None:return None
    transform=target@np.linalg.inv(source)
    if box:return transform@o['selected_box']
    return adapter.base.transform(o['pool'] if points is None else points,transform)

def sample_indices(o):
    idx=np.flatnonzero(o['owned'])
    if len(idx)>CFG['time']['sample_points']:
        idx=np.random.default_rng(CFG['seed']+o['track']*10000+o['frame']).choice(idx,CFG['time']['sample_points'],replace=False)
    return idx

def ownership(adapter, o):
    cfg=CFG['ownership'];points=o['pool'];n=len(points)
    geom_best=np.full(n,np.inf);joint_best=np.full(n,np.inf);best_mask=np.full(n,np.inf)
    image=o['image_supported']
    for tau in np.arange(-cfg['time_tube_bound_s'],cfg['time_tube_bound_s']+1e-7,cfg['time_tube_step_s']):
        source,_=pose(adapter,o['track'],o['lidar_time']+float(tau))
        if source is None:continue
        local=adapter.base.transform(points,np.linalg.inv(source))
        excess=np.linalg.norm(np.maximum(abs(local)-o['size']/2-np.array(cfg['box_margin_m']),0),axis=1)
        geom_best=np.minimum(geom_best,excess)
        if image:
            moved=warp(adapter,o,float(tau))
            if moved is None:continue
            d=distance(moved,o,adapter)
            joint=(excess<=cfg['geometry_excess_m'])&(d<=cfg['mask_margin_px'])
            joint_best=np.minimum(joint_best,np.where(joint,d,np.inf))
            best_mask=np.minimum(best_mask,d)
    feasible=np.isfinite(joint_best) if image else geom_best<=cfg['geometry_excess_m']
    # Components are evaluated independently: roof/front/rear need not form one
    # dominant cluster. Unrelated nearby components do not auto-inherit seeds.
    anchor=adapter.meta['instances'][o['track']][o['lidar_frame']][0]
    local=adapter.base.transform(points,np.linalg.inv(anchor))
    if n:
        cells,inv=np.unique(np.floor(local/cfg['voxel_m']).astype(np.int32),axis=0,return_inverse=True)
        pairs=cKDTree(cells*cfg['voxel_m']).query_pairs(cfg['component_radius_m'],output_type='ndarray')
        ii=np.r_[pairs[:,0],pairs[:,1]];jj=np.r_[pairs[:,1],pairs[:,0]]
        graph=coo_matrix((np.ones(len(ii),np.uint8),(ii,jj)),shape=(len(cells),len(cells))).tocsr()
        _,labels=connected_components(graph,directed=False);labels=labels[inv]
        good_counts=np.bincount(labels,weights=feasible.astype(int))
        owned=feasible&(good_counts[labels]>=cfg['minimum_component_points'])
    else:owned=np.zeros(0,bool)
    trusted=np.zeros(n,bool)
    old_good=image and o['old_score'].get('point_mask_distance_px',100)<=cfg['protect_previous_good_mean_px'] and o['old_score']['containment']>=cfg['protect_previous_good_containment']
    sparse_reference=cfg.get('pointwise_reference_protection',False) and n<=cfg.get('pointwise_maximum_sparse_pool_points',128)
    if image and (old_good or sparse_reference):
        old_local=adapter.base.transform(o['old_world'],np.linalg.inv(o['old_box']))
        factor=cfg.get('pointwise_reference_box_factor',1.) if sparse_reference else 1.
        trusted=(distance(o['old_world'],o,adapter)<=cfg['trusted_reference_distance_px'])&(abs(old_local)<=o['size']*factor/2+cfg['trusted_reference_box_margin_m']).all(1)
        # Conservative regression protection, openly identified as an old-fit
        # proxy. It is NOT a claim of human-verified point ownership.
        owned|=trusted
    unresolved=False
    if not owned.any() and n:
        owned=geom_best<=max(.3,float(np.min(geom_best))+.02)
        unresolved=True
    o['owned']=owned;o['weights']=np.where(owned,1. if image else .25,0.).astype(np.float32)
    o['ownership']=dict(candidate_points=n,retained_points=int(owned.sum()),removed_points=int((~owned).sum()),fraction=float(owned.mean()) if n else 0.,image_supported=image,mask_identity_note=o['mask_identity_note'],previous_good_reference=old_good,trusted_reference_points=int(trusted.sum()),trusted_reference_retained=int((trusted&owned).sum()),unresolved_geometry_fallback=unresolved,point_truth='RGB/3D/trajectory feasibility proxy, not certified LiDAR instance labels')
    o['trusted']=trusted
    return o['ownership']

def draw_sheet(adapter,o,actions,labels,caption):
    rgb,_=adapter.image(o['frame'],o['camera']);arrays=[rgb]+[adapter.dot_layer(rgb,pts,o,box,o.get('mask')) for pts,box in actions]
    pools=[adapter.base.box_uv(o['selected_box'],o['camera_pose'],o['size'],o['K'])]
    for pts,box in actions:
        uv,z=adapter.pixels(pts,o);valid=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<768)&(uv[:,1]>=0)&(uv[:,1]<576);pools.append(uv[valid])
    if o['mask_accepted']:
        yy,xx=np.where(o['mask']);pools.append(np.c_[xx,yy])
    pool=np.concatenate(pools);lo,hi=pool.min(0),pool.max(0);pad=np.maximum((hi-lo)*.08,[18,18]);crop=tuple(np.r_[np.maximum(0,lo-pad),np.minimum([768,576],hi+pad)].astype(int))
    w=440;im=Image.new('RGB',(w*len(arrays),400),'#171717');dr=ImageDraw.Draw(im)
    if not o['image_supported']:caption+=' | RGB ASSOCIATION UNVERIFIED: geometry fit'
    adapter.draw.text(dr,(8,2),f'{o["batch"]} ID{o["track"]} f{o["frame"]} c{o["camera"]} | '+caption,17)
    for i,(arr,label) in enumerate(zip(arrays,['GT RGB']+labels)):
        adapter.draw.text(dr,(i*w+4,32),label,17)
        im.paste(ImageOps.pad(Image.fromarray(arr).crop(crop),(w,310),method=Image.Resampling.LANCZOS,color='#171717'),(i*w,58))
    adapter.draw.text(dr,(8,374),'Green: automatic RGB proxy | Yellow: bbox | Orange/cyan: candidate inside/outside bbox | rejected points shown separately',14)
    return im

def stack(path,images):
    if not images:return
    out=Image.new('RGB',(images[0].width,sum(im.height for im in images)),'#171717');y=0
    for im in images:out.paste(im,(0,y));y+=im.height
    out.save(path,quality=95)

def export_ownership(adapter,rows,selection,out):
    overview=[];lookup={}
    for o in rows:
        info=ownership(adapter,o)
        np.savez_compressed(out/f'{o["key"]}_ownership.npz',point_ids=o['ids'],pool=o['pool'],owned=o['owned'],trusted=o['trusted'],weights=o['weights'])
        lookup[o['key']]=info
    for v in selection['vehicles']:
        rs=[o for o in rows if o['track']==v['track_id']]
        rep=max(rs,key=lambda o:o['source_hull_area_fraction'])
        o=rep;keep=o['owned'];info=o['ownership']
        actions=[(o['pool'],o['selected_box']),(o['pool'][keep],o['selected_box']),(o['pool'][~keep],o['selected_box'])]
        im=draw_sheet(adapter,o,actions,['Previous selected pool','Owned BEFORE time fit','Rejected points (still archived)'],f'kept {keep.sum()}/{len(keep)}; protected {info["trusted_reference_retained"]}/{info["trusted_reference_points"]}')
        im.save(out/f'{o["key"]}_ownership.jpg',quality=95);overview.append(im)
        # Explicit first/last views ensure a representative cannot conceal loss
        # at another time, especially the originally problematic ID185.
        for endpoint in [rs[0],rs[-1]]:
            if endpoint['key']==o['key']:continue
            k=endpoint['owned']
            draw_sheet(adapter,endpoint,[(endpoint['pool'],endpoint['selected_box']),(endpoint['pool'][k],endpoint['selected_box']),(endpoint['pool'][~k],endpoint['selected_box'])],['Previous selected pool','Owned BEFORE time fit','Rejected points (archived)'],f'kept {k.sum()}/{len(k)}').save(out/f'{endpoint["key"]}_ownership.jpg',quality=95)
    stack(out/'ownership_overview.jpg',overview)
    save(out/'ownership_audit.json',dict(records=lookup,previous_good_proxy_preserved=all(v['trusted_reference_points']==v['trusted_reference_retained'] for v in lookup.values()),raw_candidate_data_preserved=True,no_human_instance_truth=True))

def owned_rows(adapter,rows,out):
    info=json.loads((out/'ownership_audit.json').read_text())['records']
    for o in rows:
        with np.load(out/f'{o["key"]}_ownership.npz') as d:
            assert np.array_equal(d['point_ids'],o['ids']) and np.array_equal(d['pool'],o['pool'])
            o['owned']=d['owned'].copy();o['trusted']=d['trusted'].copy();o['weights']=d['weights'].copy()
        o['ownership']=info[o['key']]
        o['sample_indices']=sample_indices(o)
        o['fit_points']=o['pool'][o['sample_indices']]
        o['target'],_=pose(adapter,o['track'],o['camera_time'])
        o['good_reference']=o['image_supported'] and min(o['old_score'].get('point_mask_distance_px',100),o['selected_score'].get('point_mask_distance_px',100))<=1.
        if o['image_supported']:
            baseline=min(float(distance(o['old_world'][o['sample_indices']],o,adapter).mean()),float(distance(o['fit_points'],o,adapter).mean()))
            o['good_reference_outside']=baseline

def speed(adapter,tid):
    a=adapter.actors[tid]
    return float(np.median(np.linalg.norm(np.diff(a.mats[:,:3,3],axis=0),axis=1)/np.diff(a.ts))) if len(a.ts)>1 else 0.

def image_cost(adapter,o,points):
    if not o['image_supported']:
        local=adapter.base.transform(points,np.linalg.inv(o['selected_box']))
        d=np.linalg.norm(np.maximum(abs(local)-o['size']/2,0),axis=1)
        return float(np.mean(np.sqrt(1+(d/.2)**2)-1)),0.
    d=distance(points,o,adapter)
    value=float(np.mean(np.minimum(np.sqrt(1+(d/CFG['time']['outside_pixel_scale'])**2)-1,10)))
    if o.get('extent',{}).get('enabled',False):
        uv,z=adapter.pixels(points,o)
        if (z>.1).sum()>=20:
            ends=np.quantile(uv[z>.1]@np.array(o['extent']['axis']),[.02,.98]);target=np.array(o['extent']['mask_endpoints'])
            value+=CFG['time']['extent_weight']*float(np.mean(np.sqrt(1+((ends-target)/10)**2)-1))
    outside=float(d.mean())
    if o['good_reference']:
        excess=max(outside-o['good_reference_outside']-CFG['time']['good_frame_regression_tolerance_px'],0.)
        value+=CFG['time']['good_frame_regression_weight']*excess**2
    return value,outside

def fit_hierarchy(adapter,rows,stream):
    tids=sorted({o['track'] for o in rows});cfg=CFG['time'];bound=cfg['scene_bound_s']+cfg['track_residual_bound_s']
    grid=np.arange(-bound,bound+1e-7,cfg['profile_step_s']);profiles={};profile_grids={};weights=[]
    for tid in tids:
        rs=[o for o in rows if o['track']==tid];costs=[]
        def profile_cost(tau):
            vals=[]
            for o in rs:
                if not len(o['fit_points']):continue
                if stream=='lidar':
                    pts=warp(adapter,o,float(tau),o['fit_points'])
                    vals.append(image_cost(adapter,o,pts)[0] if pts is not None else 1000.)
                else:
                    box=warp(adapter,o,float(tau),box=True)
                    if box is None:vals.append(1000.);continue
                    local=adapter.base.transform(o['time_points'],np.linalg.inv(box));d=np.linalg.norm(np.maximum(abs(local)-o['size']/2,0),axis=1)
                    cost=float(np.mean(np.sqrt(1+(d/.2)**2)-1))
                    if o['image_supported']:
                        cost+=.2*(1-adapter.maskmod.iou(adapter.hull(box,o),o['mask']))
                    vals.append(cost)
            return float(np.mean(vals)) if vals else 1000.
        costs=[profile_cost(tau) for tau in grid]
        center=grid[np.argmin(costs)]
        fine=np.arange(max(-bound,center-cfg['local_profile_radius_s']),min(bound,center+cfg['local_profile_radius_s'])+1e-8,cfg['local_profile_step_s'])
        combined=sorted([(float(t),float(v)) for t,v in zip(grid,costs)]+[(float(t),profile_cost(t)) for t in fine if np.min(abs(grid-t))>1e-8])
        profile_grids[tid]=np.array([t for t,v in combined]);profiles[tid]=np.array([v for t,v in combined])
        weights.append(1. if speed(adapter,tid)>=cfg['static_speed_threshold_mps'] and any(o['image_supported'] for o in rs) else .1)
        print(adapter.key,stream,'profile',tid,'best',round(float(profile_grids[tid][np.argmin(profiles[tid])]),4),flush=True)
    w=np.asarray(weights);w/=w.sum();curves=[PchipInterpolator(profile_grids[t],profiles[t]) for t in tids];derivs=[curve.derivative() for curve in curves]
    def objective(x):
        total=x[0]+x[1:]
        return float(sum(wi*curve(v) for wi,curve,v in zip(w,curves,total))+cfg['track_prior_weight']*np.mean((x[1:]/cfg['track_residual_sigma_s'])**2)+cfg['scene_prior_weight']*(x[0]/cfg['scene_bound_s'])**2)
    def jac(x):
        d=np.array([wi*curve(x[0]+di) for wi,curve,di in zip(w,derivs,x[1:])]);return np.r_[d.sum()+2*cfg['scene_prior_weight']*x[0]/cfg['scene_bound_s']**2,d+2*cfg['track_prior_weight']*x[1:]/(len(tids)*cfg['track_residual_sigma_s']**2)]
    bounds=[(-cfg['scene_bound_s'],cfg['scene_bound_s'])]+[(-cfg['track_residual_bound_s'],cfg['track_residual_bound_s']) for _ in tids]
    constraint=dict(type='eq',fun=lambda x:float(w@x[1:]),jac=lambda x:np.r_[0.,w])
    candidates=[]
    for start in [-.1,0.,.1]:
        opt=minimize(objective,np.r_[start,np.zeros(len(tids))],jac=jac,method='SLSQP',bounds=bounds,constraints=[constraint],options=dict(maxiter=160,ftol=1e-9))
        if abs(w@opt.x[1:])<1e-6:candidates.append(opt)
    opt=min(candidates,key=lambda z:z.fun)
    result=dict(shared_s=float(opt.x[0]),tracks={str(t):dict(residual_s=float(opt.x[i+1]),total_s=float(opt.x[0]+opt.x[i+1]),speed_mps=speed(adapter,t),temporal_confidence='moving' if speed(adapter,t)>=cfg['static_speed_threshold_mps'] else 'weak_stationary',profile_min_s=float(profile_grids[t][np.argmin(profiles[t])])) for i,t in enumerate(tids)},weighted_residual_mean_s=float(w@opt.x[1:]),success=bool(opt.success),objective=float(opt.fun),weights={str(t):float(wi) for t,wi in zip(tids,w)})
    raw=dict(grid_s_by_track={str(t):profile_grids[t].tolist() for t in tids},costs={str(t):profiles[t].tolist() for t in tids})
    return result,raw

def small_spatial(adapter,rows):
    result={};cfg=CFG['space'];bounds=np.asarray(cfg['point_bounds_m'])
    for tid in sorted({o['track'] for o in rows}):
        rs=[o for o in rows if o['track']==tid];images=[o for o in rs if o['image_supported'] and len(o['time_points'])]
        r=np.zeros(3);applied=False;reasons=[]
        if images:
            def loss(v):
                return float(np.mean([image_cost(adapter,o,o['time_points']+o['target'][:3,:3]@v)[0] for o in images]))+cfg['prior_weight']*np.sum((v/bounds)**2)
            opt=minimize(loss,np.zeros(3),method='Powell',bounds=list(zip(-bounds,bounds)),options=dict(maxiter=25,xtol=.001,ftol=1e-5))
            before=np.array([distance(o['time_points'],o,adapter).mean() for o in images]);after=np.array([distance(o['time_points']+o['target'][:3,:3]@opt.x,o,adapter).mean() for o in images])
            gain=float(before.mean()-after.mean())
            if gain<max(cfg['minimum_image_gain_px'],cfg['minimum_relative_gain']*before.mean()):reasons.append('insufficient_small_space_gain')
            if any(a>b+cfg['maximum_good_frame_regression_px'] for o,a,b in zip(images,after,before) if o['good_reference']):reasons.append('good_frame_regression')
            if not reasons:r=opt.x;applied=True
        else:reasons.append('no_reliable_image_evidence_for_optional_space')
        for o in rs:
            o['final_points']=o['time_points']+o['target'][:3,:3]@r
            o['final_world']=o['time_world']+o['target'][:3,:3]@r
        # Small bbox-center correction is assessed independently after clocks.
        bb=np.asarray(cfg['bbox_bounds_m'])
        def box_loss(v):
            values=[]
            for o in rs:
                box=o['time_box'].copy();box[:3,3]+=box[:3,:3]@v
                local=adapter.base.transform(o['final_points'],np.linalg.inv(box));d=np.linalg.norm(np.maximum(abs(local)-o['size']/2,0),axis=1)
                value=float(np.mean(np.sqrt(1+(d/.2)**2)-1)) if len(d) else 0.
                if o['image_supported']:value+=.2*(1-adapter.maskmod.iou(adapter.hull(box,o),o['mask']))
                values.append(value)
            return float(np.mean(values))+cfg['prior_weight']*np.sum((v/bb)**2)
        optb=minimize(box_loss,np.zeros(3),method='Powell',bounds=list(zip(-bb,bb)),options=dict(maxiter=25,xtol=.001,ftol=1e-5))
        b=np.zeros(3);gains=[];iou_drops=[]
        for o in rs:
            proposal=o['time_box'].copy();proposal[:3,3]+=proposal[:3,:3]@optb.x
            old=adapter.score(o,o['final_points'],o['time_box']);new=adapter.score(o,o['final_points'],proposal)
            gains.append(new['containment']-old['containment'])
            if o['image_supported']:iou_drops.append(old['hull_mask_iou']-new['hull_mask_iou'])
        box_apply=np.mean(gains)>=cfg['minimum_containment_gain'] and (not iou_drops or max(iou_drops)<=cfg['maximum_bbox_iou_drop'])
        if box_apply:b=optb.x
        for o in rs:
            o['final_box']=o['time_box'].copy();o['final_box'][:3,3]+=o['final_box'][:3,:3]@b
        result[str(tid)]=dict(point_bias_m=r.tolist(),point_applied=applied,point_reasons=reasons,bbox_bias_m=b.tolist(),bbox_applied=bool(box_apply))
    return result

def finish_batch(adapter,rows,selection,out,clocks,space):
    records=[];lookup={}
    for o in rows:
        k=o['owned'];record=dict(key=o['key'],track=o['track'],frame=o['frame'],camera=o['camera'],ownership=o['ownership'],mask_accepted=o['mask_accepted'],image_supported=o['image_supported'],lidar_frame=o['lidar_frame'],bbox_frame=o['bbox_frame'],old=o['old_score'],selected=o['selected_score'],owned_before=adapter.score(o,o['pool'][k],o['selected_box']),time=adapter.score(o,o['time_world'],o['time_box']),final=adapter.score(o,o['final_world'],o['final_box']),space=space[str(o['track'])])
        # Full-pool counterfactual separates actual motion improvement from
        # improved scores caused by membership changes.
        tau=clocks['lidar']['tracks'][str(o['track'])]['total_s']
        full=warp(adapter,o,tau)
        record['time_on_original_pool']=adapter.score(o,full,o['time_box'])
        record['old_on_owned']=adapter.score(o,o['old_world'][k],o['old_box'])
        record['trajectory_support']={}
        for stream in ['lidar','bbox']:
            dt=clocks[stream]['tracks'][str(o['track'])]['total_s']
            source_t=o['lidar_time'] if stream=='lidar' else o['bbox_time']
            _,source_gap=pose(adapter,o['track'],source_t+dt)
            _,target_gap=pose(adapter,o['track'],o['camera_time'])
            record['trajectory_support'][stream]=dict(source_s=float(source_t),clock_s=dt,camera_s=float(o['camera_time']),source_extrapolation_s=source_gap,target_extrapolation_s=target_gap)
        np.savez_compressed(out/f'{o["key"]}_result.npz',point_ids=o['ids'][k],world=o['final_world'],actor=o['final_box'],size=o['size'],pool_point_ids=o['ids'],pool_world=o['pool'],owned=k,old_world=o['old_world'],old_actor=o['old_box'],time_world=o['time_world'],time_actor=o['time_box'],rgb_frame=o['frame'],lidar_frame=o['lidar_frame'],bbox_frame=o['bbox_frame'])
        records.append(record);lookup[o['key']]=record
    overviews=[];worst=[];details=[]
    def sheet(o):
        acts=[(o['old_world'],o['old_box']),(o['pool'][o['owned']],o['selected_box']),(o['time_world'],o['time_box']),(o['final_world'],o['final_box'])]
        tau=clocks['lidar']['tracks'][str(o['track'])]['total_s']
        return draw_sheet(adapter,o,acts,['Previous P/Q final','Ownership BEFORE time fit','Shared + per-car TIME','Final (optional tiny space)'],f'LiDAR clock {tau*1000:.1f}ms; kept {o["owned"].sum()}/{len(o["pool"])}')
    for v in selection['vehicles']:
        rs=[o for o in rows if o['track']==v['track_id']];rep=max(rs,key=lambda o:o['source_hull_area_fraction']);im=sheet(rep);im.save(out/f'{rep["key"]}_comparison.jpg',quality=95);overviews.append(im)
        measured=[o for o in rs if o['image_supported']]
        if measured:worst.append(sheet(max(measured,key=lambda o:lookup[o['key']]['final'].get('point_mask_distance_px',0))))
        for o in {x['key']:x for x in [rep,rs[0],rs[-1]]}.values():
            fn=f'{o["key"]}_comparison.jpg';sheet(o).save(out/fn,quality=95);details.append((o['key'],fn))
    stack(out/'overview.jpg',overviews);stack(out/'worst_overview.jpg',worst)
    measured=[r for r in records if r['image_supported'] and r['final']['points']]
    metrics=['point_mask_distance_px','point_mask_inside','outside3_fraction','hull_mask_iou','containment']
    summary=dict(observations=len(records),tracks=len(selection['vehicles']),image_scored=len(measured),stages={stage:{metric:float(np.mean([r[stage][metric] for r in measured])) for metric in metrics} if measured else {} for stage in ['old','selected','owned_before','time','final','time_on_original_pool','old_on_owned']},candidate_points=sum(r['ownership']['candidate_points'] for r in records),owned_points=sum(r['ownership']['retained_points'] for r in records))
    save(out/'evaluation.json',dict(summary=summary,records=records,note='All observations participate; automatic in-sample masks. Membership changes reported explicitly; old/owned means have different point populations.'))
    save(out/'calibration.json',dict(clocks=clocks,spatial=space,configuration=CFG,temporal_action='A_i(t_camera) @ inv(A_i(t_actual_source + scene_tau + track_delta)); independent LiDAR and bbox clocks',static_background='untouched',rolling_shutter='not corrected'))
    (out/'index.html').write_text('<meta charset="utf-8"><style>body{background:#171717;color:#eee}a{color:#8cf}img{max-width:100%}</style><h1>'+out.name+'</h1><p>GT / previous P-Q / ownership before time / hierarchical time / final optional tiny space.</p><p><a href="ownership_overview.jpg">Ownership: retained AND rejected points</a> | <a href="worst_overview.jpg">Worst final views</a></p>'+''.join(f'<p>{key}</p><a href="{fn}"><img src="{fn}"></a>' for key,fn in details))
    return summary

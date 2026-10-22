"""Decode identical vehicle queries under controlled evidence interventions.

All arms reuse the same in-run depth/backbone/query tensors. No optimization.
Source-only retains V=12; sigma-mediated effects remain part of intervention.
Saved variants are stage diagnostics, never a cold end-to-end delivery claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts import build_track_once_dynamic as build
from scripts.deliver_continuous_scene import concatenate_gaussians, save_gaussian_payload
from pointforward.models.fusion import project_points
from pointforward.models.backbone import sample_features
from pointforward.models.pointforward import Gaussians, TargetViews


def stats(values):
    values = values.detach().float()
    return {"mean": float(values.mean()), "p10_p50_p90": torch.quantile(
        values, values.new_tensor([.1, .5, .9])).cpu().tolist()}


def clone_with(g, **fields):
    return Gaussians(**{k: fields.get(k, getattr(g, k)) for k in (
        "means", "quats", "scales", "opacities", "colors", "sigma", "times",
        "instance_id", "extras")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--modes", default="baseline,source-only,anchor,depth-gate,zero-depth")
    opts, remaining = parser.parse_known_args()
    out = Path(opts.diagnostic_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    modes = opts.modes.split(",")
    original_builder = build.build_model
    original_rebase = build.rebase_dynamic_to_global_tracks
    original_postprocess = build.postprocess_gaussian_scales
    pending = {}
    global_parts = {mode: [] for mode in modes}
    records = []
    scale_options = {}
    track_counter = 0

    def patched_postprocess(scales, **kwargs):
        scale_options.update(kwargs)
        return original_postprocess(scales, **kwargs)

    def patched_builder(*args, **kwargs):
        model = original_builder(*args, **kwargs)
        decode = model.decode
        fusion_forward = model.fusion.forward

        @torch.inference_mode()
        def probe_decode(scene, depth, sampled, **decode_kwargs):
            nonlocal track_counter
            track_counter += 1
            base = sampled["base_xyz"]
            points, valid = model._pose_points(base, sampled["instance_id"],
                                               scene.boxes, scene.frame_index)
            uv, z, front = project_points(points, scene.intrinsics,
                                         torch.linalg.inv(scene.cam_to_world))
            h, w = scene.image_size
            inbounds = (uv[..., 0]>=0)&(uv[..., 0]<w)&(uv[..., 1]>=0)&(uv[..., 1]<h)
            depth_at = sample_features(depth[:, None], uv, h, w)[..., 0]
            error = (z-depth_at).abs()
            source = sampled["view_id"]
            source_mask = torch.arange(scene.num_views, device=base.device)[:,None]==source[None]
            geometric_valid = valid & front & inbounds
            rec = {"track_index": track_counter, "queries": int(len(base)),
                   "source_view_counts": torch.bincount(source, minlength=scene.num_views).cpu().tolist(),
                   "frustum_view_count": stats(geometric_valid.sum(0)),
                   "depth_consistent_view_count_1m": stats((geometric_valid & (error<=1)).sum(0)),
                   "source_depth_error": stats(error[source, torch.arange(len(base), device=base.device)]),
                   "modes": {}}
            diagnostics = {}
            for mode in modes:
                def intervention(**fusion_kwargs):
                    validity = fusion_kwargs["point_valid"]
                    if mode == "source-only":
                        fusion_kwargs["point_valid"] = validity & source_mask
                    elif mode in ("depth-gate", "depth-gate-half"):
                        threshold = .5 if mode.endswith("half") else 1.
                        fusion_kwargs["point_valid"] = validity & ((error<=threshold)|source_mask)
                    return fusion_forward(**fusion_kwargs)
                model.fusion.forward = intervention
                kws = dict(decode_kwargs)
                if mode == "anchor":
                    kws["source_anchor_dynamic_fq"] = True
                if mode == "zero-depth":
                    kws["neutralize_dynamic_depth_cue"] = True
                try:
                    g = decode(scene, depth, sampled, **kws)
                finally:
                    model.fusion.forward = fusion_forward
                pending[mode] = g
                diagnostics[mode] = g
                rec["modes"][mode] = {"opacity":stats(g.opacities),
                    "low_opacity_fraction":float((g.opacities<.1).float().mean()),
                    "query_displacement_m":stats((g.means-base).norm(dim=-1)),
                    "scale":stats(g.scales.amax(-1)), "sigma":stats(g.sigma)}
            diagnostics["baseline-source-opacity"] = clone_with(
                pending["baseline"], opacities=pending.get("source-only", pending["baseline"]).opacities)
            # Render isolated actor alpha and RGB in 3 selected views. The RGB
            # diagnostic has no background, so its PSNR is NOT acceptance PSNR.
            views = sorted(set([0, scene.num_views//2, scene.num_views-1]))
            panels = []
            for view in views:
                gt = scene.images[view].permute(1,2,0).cpu().numpy()
                query_uv = uv[view].cpu().numpy()
                ok = geometric_valid[view].cpu().numpy()
                observed = query_uv[ok]
                if len(observed):
                    lo=np.maximum(np.quantile(observed,.01,axis=0)-12,0).astype(int)
                    hi=np.minimum(np.quantile(observed,.99,axis=0)+12,[w-1,h-1]).astype(int)
                else:
                    lo,hi=np.array([0,0]),np.array([w-1,h-1])
                if np.any(hi<=lo):
                    lo,hi=np.array([0,0]),np.array([w-1,h-1])
                x0,y0=lo; x1,y1=hi
                def tile(array, label):
                    im=Image.fromarray(np.clip(array*255,0,255).astype(np.uint8))
                    im=im.crop((int(x0),int(y0),int(x1)+1,int(y1)+1)).resize((280,200))
                    canvas=Image.new("RGB",(280,224),(24,24,24));canvas.paste(im,(0,24))
                    ImageDraw.Draw(canvas).text((4,5),label,fill="white")
                    return canvas
                row=[tile(gt,f"GT view {view}")]
                target=TargetViews(scene.intrinsics[view:view+1], scene.cam_to_world[view:view+1],
                                   scene.times[view:view+1],scene.frame_index[view:view+1])
                for mode,g in diagnostics.items():
                    scales,_=original_postprocess(g.scales,mode="isotropic-trace",isotropic_max_scale=.05)
                    render_g=clone_with(g,scales=scales,sigma=torch.full_like(g.sigma,1e6),times=torch.zeros_like(g.times))
                    rendered=model.render(render_g,scene.boxes,target,h,w)
                    rgb=rendered["rgb"][0].permute(1,2,0).cpu().numpy()
                    alpha=rendered["alpha"][0].squeeze().cpu().numpy()
                    row.append(tile(rgb,mode))
                    row.append(tile(np.repeat(alpha[...,None],3,axis=-1),mode+" alpha"))
                    rec["modes"].setdefault(mode,{})[f"alpha_mean_view{view}"]=float(alpha[int(y0):int(y1)+1,int(x0):int(x1)+1].mean())
                canvas=Image.new("RGB",(280*len(row),224))
                for j,im in enumerate(row):canvas.paste(im,(280*j,0))
                panels.append(canvas)
            sheet=Image.new("RGB",(panels[0].width,224*len(panels)))
            for j,im in enumerate(panels):sheet.paste(im,(0,224*j))
            sheet.save(out/f"track_{track_counter:02d}_fixed_queries.jpg",quality=94)
            rec["visual"]=str(out/f"track_{track_counter:02d}_fixed_queries.jpg")
            records.append(rec)
            (out/"diagnostics.json").write_text(json.dumps(records,indent=2)+"\n")
            return pending["baseline"]
        model.decode=probe_decode
        return model

    def patched_rebase(dynamic, *args, **kwargs):
        for mode,g in pending.items():
            scales,_=original_postprocess(g.scales,**scale_options)
            g=clone_with(g,scales=scales)
            global_parts[mode].append(original_rebase(g,*args,**kwargs))
        pending.clear()
        return original_rebase(dynamic,*args,**kwargs)

    build.build_model=patched_builder
    build.rebase_dynamic_to_global_tracks=patched_rebase
    build.postprocess_gaussian_scales=patched_postprocess
    sys.argv=["build_track_once_dynamic.py"]+remaining
    start=time.perf_counter()
    build.main()
    payload_out=Path(remaining[remaining.index("--out")+1]).resolve()
    for mode,parts in global_parts.items():
        path=out/mode
        path.mkdir()
        save_gaussian_payload(path/"final_dynamic_128d.pt",concatenate_gaussians(parts))
        for name in ("final_static_128d.pt","global_dynamic_meta.pt"):
            os.symlink((payload_out/name).resolve(),path/name)
    (out/"timing.json").write_text(json.dumps({"scope":"cached_incremental",
        "start":start,"end":time.perf_counter(),"optimization":False,"modes":modes},indent=2)+"\n")


if __name__=="__main__":
    main()

"""Joint RGB adaptation of Gaussian attributes and the PointForward refiner.

This is an explicit experiment tool.  It keeps the original checkpoint and
the input payload immutable, and writes ordinary runtime payloads plus a
refiner overlay.  ``group`` selects the trainable Gaussian fields:

* features: static/dynamic 128-D feature groups + refiner;
* xyz-scale-features: those fields plus means and scales + refiner;
* all: means, scales, quaternions, opacity, features, temporal sigma/time
  and the refiner. Instance identity and observed source RGB are fixed.

Features use grouped deltas to keep the full scene within GPU memory.  The
loss log records every tenth step and every epoch so later experiments can
adjust learning rates and bounds from measured trends.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pointforward.models import TargetViews
from pointforward.models.rotation import quaternion_multiply
from scripts.deliver_continuous_scene import save_gaussian_payload
from scripts.eval import build_model
from scripts.render_final_global_runtime import combine, load_boxes, load_payload
from scripts.train import build_dataset


def _load_state(model, path: str | None) -> None:
    if not path:
        return
    payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=True)
    model.refiner.load_state_dict(payload.get("refiner", payload), strict=True)


def _q_delta(rotvec: torch.Tensor) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / angle
    half = angle * 0.5
    return F.normalize(torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1), dim=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--payload-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scene", default="017")
    p.add_argument("--device", default="cuda")
    p.add_argument("--group", choices=("features", "xyz-scale-features", "all"), required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=6)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--feature-lr", type=float, default=0.01)
    p.add_argument("--geometry-lr", type=float, default=0.001)
    p.add_argument("--refiner-lr", type=float, default=0.0002)
    p.add_argument("--opacity-lr", type=float, default=0.001)
    p.add_argument("--max-xyz-delta", type=float, default=0.02)
    p.add_argument("--max-log-scale-delta", type=float, default=0.25)
    p.add_argument("--max-rotation-deg", type=float, default=5.0)
    p.add_argument("--max-opacity-logit-delta", type=float, default=0.5)
    p.add_argument("--max-feature-delta", type=float, default=0.5)
    p.add_argument("--anchor-weight", type=float, default=1e-4)
    p.add_argument("--static-source-rgb-blend", type=float, default=0.75)
    p.add_argument("--refiner-state-in", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deadline-monotonic", type=float, default=None,
                   help="Stop training at a step boundary to reserve delivery time; same system monotonic clock as the cold supervisor.")
    args = p.parse_args()
    if 128 % args.groups or args.frame_stride < 1:
        p.error("groups must divide 128 and frame-stride must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source = Path(args.payload_dir).resolve()
    out = Path(args.out).resolve()
    refined = out / "runtime-payload-refined"
    refined.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("data", {}).get("layout") == "argoverse_drivestudio":
        config["data"]["rigid_classes"] = [
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS",
            "VEHICULAR_TRAILER", "MOTORCYCLE", "BICYCLE", "BICYCLIST",
            "MOTORCYCLIST",
        ]
    dataset = build_dataset(config, "val")
    meta_scene = next(x for x in dataset.scenes if x["name"] == args.scene)
    all_frames = list(meta_scene["frames"])
    frames = all_frames[::args.frame_stride]
    slots = {int(f): i for i, f in enumerate(all_frames)}
    times = dataset._normalised_times(all_frames)
    world_from = np.linalg.inv(dataset._ego_pose(meta_scene, all_frames[0]))
    height, width = dataset.resolution

    static = load_payload(source / "final_static_128d.pt", device)
    dynamic = load_payload(source / "final_dynamic_128d.pt", device)
    meta = torch.load(source / "global_dynamic_meta.pt", map_location="cpu", weights_only=True)
    boxes = load_boxes(meta["boxes"], device)
    model = build_model(config, device, args.ckpt).eval()
    _load_state(model, args.refiner_state_in)
    model.refiner.requires_grad_(True)

    # Deltas are bounded with tanh, so an unlucky RGB gradient cannot create
    # an unbounded Gaussian or an invalid rotation.
    train_geometry = args.group in ("xyz-scale-features", "all")
    train_all = args.group == "all"
    color_groups = args.groups
    cpg = 128 // color_groups
    params: list[dict] = []
    raw_sf = torch.nn.Parameter(torch.zeros((static.count, color_groups), device=device))
    raw_df = torch.nn.Parameter(torch.zeros((dynamic.count, color_groups), device=device))
    params.append({"params": [raw_sf, raw_df], "lr": args.feature_lr})
    raw_sm = raw_dm = raw_ss = raw_ds = None
    if train_geometry:
        raw_sm = torch.nn.Parameter(torch.zeros_like(static.means))
        raw_dm = torch.nn.Parameter(torch.zeros_like(dynamic.means))
        raw_ss = torch.nn.Parameter(torch.zeros_like(static.scales))
        raw_ds = torch.nn.Parameter(torch.zeros_like(dynamic.scales))
        params.append({"params": [raw_sm, raw_dm, raw_ss, raw_ds], "lr": args.geometry_lr})
    raw_sq = raw_dq = raw_so = raw_do = None
    raw_st = raw_dt = raw_sg = raw_dg = None
    if train_all:
        raw_sq = torch.nn.Parameter(torch.zeros_like(static.means))
        raw_dq = torch.nn.Parameter(torch.zeros_like(dynamic.means))
        raw_so = torch.nn.Parameter(torch.zeros_like(static.opacities))
        raw_do = torch.nn.Parameter(torch.zeros_like(dynamic.opacities))
        raw_st = torch.nn.Parameter(torch.zeros_like(static.times))
        raw_dt = torch.nn.Parameter(torch.zeros_like(dynamic.times))
        raw_sg = torch.nn.Parameter(torch.zeros_like(static.sigma))
        raw_dg = torch.nn.Parameter(torch.zeros_like(dynamic.sigma))
        params += [{"params": [raw_sq, raw_dq], "lr": args.geometry_lr},
                   {"params": [raw_so, raw_do, raw_st, raw_dt, raw_sg, raw_dg], "lr": args.opacity_lr}]
    trainable = [x for g in params for x in g["params"]]
    trainable += list(model.refiner.parameters())
    optimizer = torch.optim.AdamW(params + [{"params": list(model.refiner.parameters()), "lr": args.refiner_lr}], weight_decay=0.0)
    base_so = torch.logit(static.opacities.clamp(1e-5, 1 - 1e-5))
    base_do = torch.logit(dynamic.opacities.clamp(1e-5, 1 - 1e-5))
    base_sq = F.normalize(static.quats.detach(), dim=-1)
    base_dq = F.normalize(dynamic.quats.detach(), dim=-1)
    channel_index = torch.arange(128, device=device) // cpg
    max_rot = math.radians(args.max_rotation_deg)
    records: list[dict] = []
    started = time.perf_counter()

    def current():
        sf = static.colors + torch.tanh(raw_sf)[:, channel_index] * args.max_feature_delta
        df = dynamic.colors + torch.tanh(raw_df)[:, channel_index] * args.max_feature_delta
        sm, dm = static.means, dynamic.means
        ss, ds = static.scales, dynamic.scales
        sq, dq = base_sq, base_dq
        so, do = static.opacities, dynamic.opacities
        st, dt = static.times, dynamic.times
        sg, dg = static.sigma, dynamic.sigma
        if train_geometry:
            sm = sm + torch.tanh(raw_sm) * args.max_xyz_delta
            dm = dm + torch.tanh(raw_dm) * args.max_xyz_delta
            ss = ss * torch.exp(torch.tanh(raw_ss) * args.max_log_scale_delta)
            ds = ds * torch.exp(torch.tanh(raw_ds) * args.max_log_scale_delta)
        if train_all:
            sq = F.normalize(quaternion_multiply(_q_delta(torch.tanh(raw_sq) * max_rot), base_sq), dim=-1)
            dq = F.normalize(quaternion_multiply(_q_delta(torch.tanh(raw_dq) * max_rot), base_dq), dim=-1)
            so = torch.sigmoid(base_so + torch.tanh(raw_so) * args.max_opacity_logit_delta)
            do = torch.sigmoid(base_do + torch.tanh(raw_do) * args.max_opacity_logit_delta)
            st = st + torch.tanh(raw_st) * 0.05
            dt = dt + torch.tanh(raw_dt) * 0.05
            sg = sg * torch.exp(torch.tanh(raw_sg) * 0.25)
            dg = dg * torch.exp(torch.tanh(raw_dg) * 0.25)
        return replace(static, means=sm, scales=ss, quats=sq, opacities=so, colors=sf, sigma=sg, times=st), replace(dynamic, means=dm, scales=ds, quats=dq, opacities=do, colors=df, sigma=dg, times=dt)

    step = 0
    stop_reason = "epoch_limit"
    step_rows = out / "loss_steps.jsonl"
    for epoch in range(args.epochs):
        order = torch.randperm(len(frames), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        epoch_losses = []
        for oi in order:
            if args.deadline_monotonic is not None and time.perf_counter() >= args.deadline_monotonic:
                stop_reason = "delivery_time_reserve"
                break
            frame = int(frames[oi]); slot = slots[frame]
            batch = dataset._collect(meta_scene, [frame], slots, times, dataset.resolution, world_from, lidar=False)
            targets = TargetViews(intrinsics=batch["intrinsics"].to(device), cam_to_world=batch["cam_to_world"].to(device), times=times[slot:slot+1].to(device).expand(len(dataset.cameras)), frame_index=torch.full((len(dataset.cameras),), slot, dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            cs, cd = current()
            rendered = model.render(combine(cs, cd), boxes, targets, height=height, width=width, render_depth=False, static_source_rgb_blend=args.static_source_rgb_blend)["rgb"]
            truth = batch["images"].to(device)
            mse_views = (rendered - truth).square().flatten(1).mean(1)
            data_loss = torch.log(mse_views.clamp_min(1e-8)).mean()
            anchor_terms = [(torch.tanh(raw_sf)).square().mean(), (torch.tanh(raw_df)).square().mean()]
            if train_geometry:
                anchor_terms += [(torch.tanh(raw_sm)).square().mean(), (torch.tanh(raw_dm)).square().mean(), (torch.tanh(raw_ss)).square().mean(), (torch.tanh(raw_ds)).square().mean()]
            if train_all:
                anchor_terms += [(torch.tanh(q)).square().mean() for q in (raw_sq, raw_dq, raw_so, raw_do, raw_st, raw_dt, raw_sg, raw_dg)]
            anchor = torch.stack(anchor_terms).mean()
            loss = data_loss + args.anchor_weight * anchor
            loss.backward()
            for q in trainable:
                if q.grad is not None:
                    torch.nan_to_num_(q.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            epoch_losses.extend(float(x) for x in mse_views.detach())
            row = {"step": step, "epoch": epoch + 1, "frame": frame, "rgb_mse": float(mse_views.detach().mean()),
                   "loss": float(loss.detach()), "seconds": time.perf_counter()-started,
                   "lr": {str(i): g["lr"] for i, g in enumerate(optimizer.param_groups)}}
            with step_rows.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if step == 1 or step % 10 == 0:
                row = {"step": step, "epoch": epoch + 1, "frame": frame, "rgb_mse": float(np.mean(epoch_losses[-len(dataset.cameras):])), "epoch_mean_mse_so_far": float(np.mean(epoch_losses)), "loss": float(loss.detach()), "anchor": float(anchor.detach()), "grad_norm": float(grad_norm), "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
                records.append(row); print(json.dumps(row), flush=True)
            del batch, targets, cs, cd, rendered, truth, mse_views, loss
        if epoch_losses:
            records.append({"epoch": epoch + 1, "epoch_mean_mse": float(np.mean(epoch_losses)), "epoch_psnr_proxy": float(-10 * np.log10(np.mean(epoch_losses))), "seconds": time.perf_counter() - started})
            print(json.dumps(records[-1]), flush=True)
        if stop_reason == "delivery_time_reserve":
            break

    with torch.no_grad():
        rs, rd = current()
        save_gaussian_payload(refined / "final_static_128d.pt", rs)
        save_gaussian_payload(refined / "final_dynamic_128d.pt", rd)
    os.symlink((source / "global_dynamic_meta.pt").resolve(), refined / "global_dynamic_meta.pt")
    torch.save({"refiner": {k: v.detach().cpu() for k, v in model.refiner.state_dict().items()}, "group": args.group}, out / "refiner_state.pt")
    diagnostics = {"static_feature_delta_rms": float((torch.tanh(raw_sf).detach() * args.max_feature_delta).square().mean().sqrt()),
                   "dynamic_feature_delta_rms": float((torch.tanh(raw_df).detach() * args.max_feature_delta).square().mean().sqrt())}
    if train_geometry:
        diagnostics.update(static_xyz_delta_rms=float((torch.tanh(raw_sm).detach()*args.max_xyz_delta).square().mean().sqrt()),
                           dynamic_xyz_delta_rms=float((torch.tanh(raw_dm).detach()*args.max_xyz_delta).square().mean().sqrt()),
                           static_log_scale_delta_rms=float((torch.tanh(raw_ss).detach()*args.max_log_scale_delta).square().mean().sqrt()))
    manifest = {"status": "completed", "group": args.group, "epochs": args.epochs, "steps": step, "stop_reason": stop_reason, "parameter_diagnostics": diagnostics, "frame_stride": args.frame_stride, "frames": [int(x) for x in frames], "source_payload": str(source), "refined_payload": str(refined), "refiner_state": str(out / "refiner_state.pt"), "trainable_parameter_count": sum(x.numel() for x in trainable), "wall_seconds": time.perf_counter() - started, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30, "records": records, "bounds": vars(args)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

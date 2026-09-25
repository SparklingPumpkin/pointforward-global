"""Render PointForward static Gaussians with DriveStudio dynamics and sky layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

from pointforward.models import TargetViews
from scripts.deliver_continuous_scene import (
    encode_video,
    horizontal_strip,
    interpolate_camera_pose,
    label,
    to_uint8,
    vertical_strip,
)
from scripts.eval import build_model
from scripts.render_final_global_runtime import load_payload, quaternion_slerp
from scripts.train import build_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_viewdirs(
    intrinsics: torch.Tensor,
    cam_to_world: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=intrinsics.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=intrinsics.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    camera_dirs = pixels @ torch.linalg.inv(intrinsics).T
    world_dirs = camera_dirs @ cam_to_world[:3, :3].T
    return torch.nn.functional.normalize(world_dirs, dim=-1)


def move_to_device(values: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in values.items()
    }


def setup_drivestudio(
    checkpoint: Path,
    repo_root: Path,
    data_root: Path,
    device: torch.device,
):
    sys.path.insert(0, str(repo_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
    cfg.data.data_root = str(data_root)
    cfg.data.pixel_source.load_semantic = False
    cfg.data.pixel_source.load_smpl = False
    # This hybrid delivery replaces vehicles only.  The external SMPL asset is
    # not part of the scene checkpoint and is intentionally not required.
    cfg.model.pop("SMPLNodes", None)
    cfg.trainer.render.packed = True
    dataset = DrivingDataset(data_cfg=cfg.data)
    # SceneGraph initialization expects these runtime fields, but the
    # standalone DrivingDataset constructor does not populate them.
    dataset.output_root = "/mnt/data/dataset/output/"
    dataset.project = "argoverse"
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        num_frames=dataset.pixel_source.num_frames,
        num_cameras=dataset.pixel_source.num_cams,
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.init_gaussians_from_dataset(dataset=dataset)
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()

    dynamic_names = [
        name for name in ("RigidNodes", "DeformableNodes", "SMPLNodes")
        if name in trainer.gaussian_classes
    ]
    trainer.gaussian_classes = {
        name: trainer.gaussian_classes[name] for name in dynamic_names
    }
    if "Sky" not in trainer.models or not dynamic_names:
        raise RuntimeError("DriveStudio checkpoint lacks dynamic nodes or Sky")
    return trainer, dataset, dynamic_names


def interpolate_model_pose(model, lower: int, alpha: float):
    """Temporarily interpolate a DriveStudio node's global pose in slot lower."""
    model.set_cur_frame(lower)
    if alpha <= 0.0 or lower + 1 >= model.num_frames:
        return None
    upper = lower + 1
    backup = {
        "trans": model.instances_trans.data[lower].clone(),
        "quats": model.instances_quats.data[lower].clone(),
        "valid": model.instances_fv[lower].clone(),
    }
    valid = model.instances_fv[lower] & model.instances_fv[upper]
    model.instances_trans.data[lower] = torch.lerp(
        model.instances_trans.data[lower], model.instances_trans.data[upper], alpha
    )
    model.instances_quats.data[lower] = quaternion_slerp(
        model.instances_quats.data[lower], model.instances_quats.data[upper], alpha
    )
    model.instances_fv[lower] = valid
    if hasattr(model, "smpl_qauts"):
        backup["smpl_qauts"] = model.smpl_qauts.data[lower].clone()
        model.smpl_qauts.data[lower] = quaternion_slerp(
            model.smpl_qauts.data[lower], model.smpl_qauts.data[upper], alpha
        )
    if hasattr(model, "normalized_timestamps"):
        backup["timestamp"] = model.normalized_timestamps[lower].clone()
        model.normalized_timestamps[lower] = torch.lerp(
            model.normalized_timestamps[lower],
            model.normalized_timestamps[upper],
            alpha,
        )
    return backup


def restore_model_pose(model, lower: int, backup) -> None:
    if backup is None:
        return
    model.instances_trans.data[lower] = backup["trans"]
    model.instances_quats.data[lower] = backup["quats"]
    model.instances_fv[lower] = backup["valid"]
    if "smpl_qauts" in backup:
        model.smpl_qauts.data[lower] = backup["smpl_qauts"]
    if "timestamp" in backup:
        model.normalized_timestamps[lower] = backup["timestamp"]


@torch.inference_mode()
def render_drivestudio_layers(
    trainer,
    dynamic_names: list[str],
    frame: int,
    alpha: float,
    camera: int,
    intrinsics: torch.Tensor,
    cam_to_world: torch.Tensor,
    height: int,
    width: int,
    num_cameras: int,
    render_sky: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    backups = {}
    try:
        for name in dynamic_names:
            backups[name] = interpolate_model_pose(trainer.models[name], frame, alpha)
        trainer.cur_frame = torch.tensor(frame, device=intrinsics.device)
        img_idx = torch.tensor(frame * num_cameras + camera, device=intrinsics.device)
        normed_time = torch.tensor(
            (frame + alpha) / max(1, trainer.models[dynamic_names[0]].num_frames - 1),
            device=intrinsics.device,
        )
        viewdirs = image_viewdirs(intrinsics, cam_to_world, height, width)
        image_infos = {
            "img_idx": img_idx.reshape(1),
            "normed_time": normed_time.reshape(1),
            "viewdirs": viewdirs,
        }
        camera_infos = {
            "camera_to_world": cam_to_world,
            "intrinsics": intrinsics,
            "height": torch.tensor(height, device=intrinsics.device),
            "width": torch.tensor(width, device=intrinsics.device),
        }
        processed_cam = trainer.process_camera(
            camera_infos=camera_infos, image_ids=img_idx, novel_view=True
        )
        gaussians = trainer.collect_gaussians(cam=processed_cam, image_ids=img_idx)
        outputs, _ = trainer.render_gaussians(
            gs=gaussians,
            cam=processed_cam,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
        )
        dynamic_rgb = trainer.affine_transformation(
            outputs["rgb_gaussians"], image_infos
        ).clamp(0.0, 1.0)
        sky = (
            trainer.models["Sky"](image_infos).clamp(0.0, 1.0)
            if render_sky else torch.zeros_like(dynamic_rgb)
        )
        return dynamic_rgb, outputs["opacity"], outputs["depth"], sky
    finally:
        for name in dynamic_names:
            restore_model_pose(trainer.models[name], frame, backups.get(name))


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction.float() - target.float()) ** 2).clamp_min(1e-10)
    return float((-10.0 * torch.log10(mse)).cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--static-payload", required=True)
    parser.add_argument("--drivestudio-checkpoint", required=True)
    parser.add_argument("--drivestudio-root", default="/mnt/fjn/projects/repos/51w-drivestudio")
    parser.add_argument("--drivestudio-data-root", default="/mnt/data/dataset/argoverse/processed/training")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--camera-interpolation-factor", type=int, default=6)
    parser.add_argument("--max-output-frames", type=int)
    parser.add_argument(
        "--disable-drivestudio-sky",
        action="store_true",
        help="Keep the PointForward/UNet RGB unchanged and omit DriveStudio EnvLight.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.out).resolve()
    predicted_dir = output_dir / "frames" / "predicted"
    comparison_dir = output_dir / "frames" / "comparison"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    checkpoint = Path(args.ckpt).resolve()
    static_path = Path(args.static_payload).resolve()
    ds_checkpoint = Path(args.drivestudio_checkpoint).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config["data"].get("layout") == "argoverse_drivestudio":
        from pointforward.data.argoverse import ARGOVERSE_RIGID_CLASSES
        config["data"]["rigid_classes"] = list(ARGOVERSE_RIGID_CLASSES)
    dataset = build_dataset(config, "val")
    metadata = next(item for item in dataset.scenes if item["name"] == args.scene)
    frames = list(metadata["frames"])
    all_slots = {frame: slot for slot, frame in enumerate(frames)}
    all_times = dataset._normalised_times(frames)
    world_from = np.linalg.inv(dataset._ego_pose(metadata, frames[0]))
    height, width = dataset.resolution

    static = load_payload(static_path, device)
    if bool((static.instance_id >= 0).any()):
        raise ValueError("Static payload unexpectedly contains dynamic Gaussians")
    model = build_model(config, device, str(checkpoint))
    ds_trainer, ds_dataset, dynamic_names = setup_drivestudio(
        ds_checkpoint,
        Path(args.drivestudio_root).resolve(),
        Path(args.drivestudio_data_root).resolve(),
        device,
    )

    samples = []
    for source_index, frame in enumerate(frames):
        if source_index + 1 < len(frames):
            samples.extend(
                (source_index, frame, step / args.camera_interpolation_factor)
                for step in range(args.camera_interpolation_factor)
            )
        else:
            samples.append((source_index, frame, 0.0))
    if args.max_output_frames is not None:
        samples = samples[:args.max_output_frames]

    started = time.perf_counter()
    frame_records = []
    cached_index = None
    target = next_target = None
    for output_index, (source_index, frame, alpha) in enumerate(samples):
        if cached_index != source_index:
            target = (
                next_target if cached_index is not None and source_index == cached_index + 1
                else dataset._collect(
                    metadata, [frame], all_slots, all_times,
                    dataset.resolution, world_from, lidar=False,
                )
            )
            next_target = (
                dataset._collect(
                    metadata, [frames[source_index + 1]], all_slots, all_times,
                    dataset.resolution, world_from, lidar=False,
                )
                if source_index + 1 < len(frames) else None
            )
            cached_index = source_index
        target_cam_to_world = (
            torch.stack([
                interpolate_camera_pose(current, following, alpha)
                for current, following in zip(
                    target["cam_to_world"], next_target["cam_to_world"]
                )
            ]) if alpha > 0.0 else target["cam_to_world"]
        )

        rendered_views = []
        layer_stats = []
        target_time = float(torch.lerp(
            all_times[source_index],
            all_times[min(source_index + 1, len(frames) - 1)],
            alpha,
        ))
        for camera_index, camera in enumerate(dataset.cameras):
            intrinsics = target["intrinsics"][camera_index].to(device)
            cam_to_world = target_cam_to_world[camera_index].to(device)
            views = TargetViews(
                intrinsics=intrinsics.unsqueeze(0),
                cam_to_world=cam_to_world.unsqueeze(0),
                times=torch.tensor([target_time], device=device),
                frame_index=torch.zeros(1, dtype=torch.long, device=device),
            )
            static_out = model.render(
                static, None, views, height=height, width=width, render_depth=True
            )
            static_rgb = static_out["rgb"][0].permute(1, 2, 0)
            static_alpha = static_out["alpha"][0].permute(1, 2, 0)
            static_depth = static_out["render_depth"][0].permute(1, 2, 0)
            dynamic_rgb, dynamic_alpha, dynamic_depth, sky = render_drivestudio_layers(
                ds_trainer, dynamic_names, frame, alpha, camera,
                intrinsics, cam_to_world, height, width,
                ds_dataset.pixel_source.num_cams,
                render_sky=not args.disable_drivestudio_sky,
            )
            if args.disable_drivestudio_sky:
                base = static_rgb
                sky_residual = torch.zeros_like(static_rgb)
            else:
                # EnvLight is the bottom layer.  It is added only through residual
                # static transparency; the UNet RGB itself is never masked/deleted.
                sky_residual = sky * (1.0 - static_alpha)
                base = (static_rgb + sky_residual).clamp(0.0, 1.0)
            front = (dynamic_depth <= static_depth + 0.05).to(dynamic_alpha.dtype)
            visible_alpha = dynamic_alpha * front
            hybrid = (dynamic_rgb * front + base * (1.0 - visible_alpha)).clamp(0.0, 1.0)
            rendered_views.append(hybrid.permute(2, 0, 1))
            layer_stats.append({
                "static_alpha_mean": float(static_alpha.mean().cpu()),
                "dynamic_alpha_mean": float(visible_alpha.mean().cpu()),
                "sky_residual_mean": float(sky_residual.mean().cpu()),
            })
        rendered = torch.stack(rendered_views)
        sample_label = (
            f"frame {frame:03d}" if alpha == 0.0
            else f"frame {frame:03d}->{frames[source_index + 1]:03d} | alpha {alpha:.3f}"
        )
        layer_label = "PF static + DS dynamic" + (
            "" if args.disable_drivestudio_sky else " + DS sky"
        )
        prediction_panels = [
            label(to_uint8(rendered[index]), f"{layer_label} | {sample_label} | camera {camera}")
            for index, camera in enumerate(dataset.cameras)
        ]
        gt_panels = [
            label(to_uint8(target["images"][index]), f"GT RGB held at frame {frame:03d} | camera {camera}")
            for index, camera in enumerate(dataset.cameras)
        ]
        prediction_strip = horizontal_strip(prediction_panels)
        comparison = vertical_strip((horizontal_strip(gt_panels), prediction_strip))
        Image.fromarray(prediction_strip).save(
            predicted_dir / f"frame_{output_index:04d}.jpg", quality=92
        )
        Image.fromarray(comparison).save(
            comparison_dir / f"frame_{output_index:04d}.jpg", quality=92
        )
        frame_records.append({
            "output_index": output_index,
            "frame": int(frame),
            "camera_interpolation_alpha": alpha,
            "psnr_all_cameras": psnr(rendered, target["images"].to(device)) if alpha == 0.0 else None,
            "layers": layer_stats,
        })
        if output_index % 25 == 0 or output_index == len(samples) - 1:
            print(json.dumps({
                "frame": int(frame), "index": output_index,
                "output_count": len(samples), "alpha": alpha,
            }), flush=True)

    predicted_video = output_dir / "final_global_runtime_predicted_front3_60fps.mp4"
    comparison_video = output_dir / "final_global_runtime_gt_vs_predicted_front3_60fps.mp4"
    encode_video(predicted_dir / "frame_%04d.jpg", predicted_video, args.fps)
    encode_video(comparison_dir / "frame_%04d.jpg", comparison_video, args.fps)
    source_psnr = [r["psnr_all_cameras"] for r in frame_records if r["psnr_all_cameras"] is not None]
    manifest = {
        "status": "completed",
        "scene": args.scene,
        "frame_count": len(frames),
        "output_frame_count": len(samples),
        "fps": args.fps,
        "camera_interpolation_factor": args.camera_interpolation_factor,
        "static_contract": "GD-v1-f final_static_128d only; all PointForward dynamic Gaussians removed",
        "dynamic_contract": f"DriveStudio {' + '.join(dynamic_names)} with interpolated global node poses",
        "sky_contract": (
            "DriveStudio Sky / EnvLight disabled; PointForward/UNet RGB retained unchanged"
            if args.disable_drivestudio_sky else
            "DriveStudio EnvLight is bottom layer; added through residual PointForward static alpha without deleting/masking UNet RGB"
        ),
        "occlusion_contract": "DriveStudio dynamic is composited over PointForward static only where dynamic expected depth is in front",
        "pointforward_checkpoint": str(checkpoint),
        "static_payload": str(static_path),
        "drivestudio_checkpoint": str(ds_checkpoint),
        "drivestudio_models": dynamic_names + ([] if args.disable_drivestudio_sky else ["Sky"]) + ["Affine", "CamPose"],
        "static_gaussian_count": static.count,
        "mean_psnr_source_frames_all_cameras": float(np.mean(source_psnr)),
        "predicted_video": str(predicted_video),
        "comparison_video": str(comparison_video),
        "sha256": {
            "static_payload": sha256(static_path),
            "drivestudio_checkpoint": sha256(ds_checkpoint),
            "predicted_video": sha256(predicted_video),
            "comparison_video": sha256(comparison_video),
        },
        "frame_records": frame_records,
        "wall_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "frame_records"}, indent=2))


if __name__ == "__main__":
    main()

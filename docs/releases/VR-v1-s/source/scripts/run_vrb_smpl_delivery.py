"""Cold Waymo102 VR-v1-b delivery with pose-completed global SMPL pedestrians."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent
PROJECT = Path("/mnt/data/fjn/projects/PointForward")
PYTHON = "/root/miniconda3/envs/drivestudio/bin/python"
CKPT = PROJECT / "ckpt/ckpt_020000_waymo_ds_paper_original.pt"
SMPL_MODEL = Path("/mnt/data/output/pointforward/smpl_models/SMPL_NEUTRAL.npz")
WORKFLOW = Path("/mnt/fjn/FJN-Agent/skills/fjn-project-workflow/scripts/project_workflow.py")
BASELINE = PROJECT / (
    "experiments/argoverse-inference/cold-scene-delivery/vehicle-roi-visibility/"
    "VR-v1-e/run-20261020T062255Z-cold-vrb-waymo102-joint900-psnr30"
)


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    timer_path = RUN / "pipeline_timing.json"
    if timer_path.exists() and not args.resume:
        raise FileExistsError("Use --resume for this same run")
    if args.resume:
        timer = json.loads(timer_path.read_text())
        timer["status"] = "running"
    else:
        timer = {
            "run_id": RUN.name,
            "scene": "102",
            "scope": "end_to_end",
            "status": "running",
            "start": time.perf_counter(),
            "start_epoch_seconds": time.time(),
            "input_boundary": "before_first_raw_scene_read",
            "output_boundary": "ply_full_videos_metrics_and_visualizations_written",
            "upstream_prefix_seconds": 0.0,
            "gpu": args.gpu,
            "historical_scene_state_reused": [],
            "stages": [],
        }
    save(timer_path, timer)
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )

    def write():
        save(timer_path, timer)

    def call(name: str, command: list[str]):
        completed = next(
            (row for row in timer["stages"] if row["name"] == name and row["status"] == "completed"),
            None,
        )
        if completed is not None:
            print(f"KEEP {name}: {completed['seconds']:.2f}s", flush=True)
            return
        row = {"name": name, "command": [str(value) for value in command],
               "status": "running", "start": time.perf_counter()}
        timer["stages"].append(row)
        write()
        print(f"START {name}", flush=True)
        with (RUN / f"{name}.log").open("w") as log:
            result = subprocess.run(row["command"], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        row.update(
            end=time.perf_counter(),
            returncode=result.returncode,
            status="completed" if result.returncode == 0 else "failed",
        )
        row["seconds"] = row["end"] - row["start"]
        write()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {RUN / (name + '.log')}")
        print(f"DONE {name}: {row['seconds']:.2f}s", flush=True)

    config = RUN / "config_build.yaml"
    static = RUN / "static-build"
    vehicle = RUN / "vehicle-payload"
    runtime = RUN / "runtime-payload"
    joint = RUN / "joint-all"
    common = ["--config", str(config), "--ckpt", str(CKPT), "--scene", "102"]
    try:
        call("static_build", [PYTHON, str(ROOT / "scripts/deliver_continuous_scene.py"), *common,
            "--out", str(static), "--seed", "0", "--reconstruction-mode", "global-gaussians",
            "--clip-order", "reverse", "--window-length", "20", "--context-frame-stride", "5",
            "--window-step", "20", "--global-voxel-size", "0.01", "--global-depth-threshold", "1.0",
            "--global-growth-policy", "three-clip-mutable-frozen-position",
            "--global-dynamic-policy", "persistent-tracks", "--global-dynamic-forward-refresh",
            "--global-update-distance-policy", "closest-smoothstep", "--global-update-alpha", "0.5",
            "--global-update-near-distance", "20.0", "--global-update-far-distance", "80.0",
            "--global-update-min-improvement", "0.0", "--cache-overlap-image-networks",
            "--gaussian-scale-postprocess", "isotropic-trace", "--isotropic-max-scale", "0.005",
            "--skip-frame-render", "--skip-ply", "--save-runtime-payload", str(static / "runtime-payload"),
            "--global-query-origin", "clip", "--global-static-offset-gain", "0.0"])
        call("vehicle_build", [PYTHON, str(ROOT / "scripts/build_track_once_dynamic.py"), *common,
            "--static-payload", str(static / "runtime-payload/final_static_128d.pt"),
            "--out", str(vehicle), "--seed", "0", "--view-selection", "diverse-large",
            "--views-per-track", "12", "--view-angle-weight", "0.65",
            "--gaussian-scale-postprocess", "isotropic-trace", "--isotropic-max-scale", "0.05",
            "--query-origin", "scene", "--decode-box-scale", "1.0",
            "--decode-box-length-margin", "1.0", "--decode-box-width-margin", "0.6"])
        call("pedestrian_build", [PYTHON, str(ROOT / "scripts/build_smpl_pedestrians.py"), *common,
            "--payload-dir", str(vehicle), "--out", str(runtime)])
        static_link = runtime / "final_static_128d.pt"
        if not static_link.exists():
            static_link.symlink_to(static / "runtime-payload/final_static_128d.pt")
        train_config = RUN / "config_train.yaml"
        call("joint_all950_fullres", [PYTHON, str(ROOT / "scripts/refine_unified_scene_rgb.py"),
            "--config", str(train_config), "--ckpt", str(CKPT), "--payload-dir", str(runtime),
            "--out", str(joint), "--audit-run", str(RUN), "--scene", "102", "--steps", "950",
            "--milestones", "350", "550", "750", "950", "--roi-weight", "0.25",
            "--pose-lr", "0.001", "--lr-decay-steps", "240", "700", "--lr-decay-factor", "0.4",
            "--single-camera-training", "--defer-audit-render"])
        checkpoints = json.loads((joint / "checkpoints.json").read_text())
        for checkpoint in checkpoints:
            step = int(checkpoint["step"])
            checkpoint_dir = joint / f"checkpoint-{step:04d}"
            render_dir = checkpoint_dir / "runtime-render-source"
            call(f"render_fullres_checkpoint_{step:04d}", [PYTHON, str(ROOT / "scripts/render_final_global_runtime.py"),
                "--config", str(RUN / "config_render.yaml"), "--ckpt", str(CKPT),
                "--payload-dir", str(checkpoint["payload"]), "--out", str(render_dir),
                "--scene", "102", "--device", "cuda", "--camera-batch-size", "1",
                "--fps", "10", "--dynamic-bbox-metrics", "--per-track-bbox-metrics",
                "--static-source-rgb-blend", ".75", "--refiner-state", str(checkpoint["refiner_state"])])
            metrics = json.loads((render_dir / "manifest.json").read_text())
            checkpoint.update({
                "global_psnr": metrics.get("mean_psnr_source_frames_all_cameras"),
                "dynamic_psnr": metrics.get("mean_psnr_dynamic_bbox_source_frames_all_cameras"),
                "non_sky_psnr": metrics.get("mean_psnr_non_sky_source_frames_all_cameras"),
                "pedestrian_psnr": metrics.get("mean_psnr_pedestrian_bbox_source_frames_all_cameras"),
                "render": str(render_dir),
            })
        save(joint / "checkpoints.json", checkpoints)
        selected = next(row for row in checkpoints if row["step"] == 950)
        save(RUN / "selected_checkpoint.json", selected)
        render = Path(selected["render"])
        for name in ("final_global_runtime_predicted_front3.mp4",
                     "final_global_runtime_gt_vs_predicted_front3.mp4"):
            shutil.copy2(render / name, RUN / name)
        call("ply_export", [PYTHON, str(ROOT / "scripts/export_runtime_payload_ply.py"),
            "--payload-dir", selected["payload"], "--out", str(RUN / "final_global_gaussians_frame000_preview.ply"),
            "--frame", "0", "--pedestrian-tracks", str(RUN / "pedestrian_tracks.pt"),
            "--smpl-model", str(SMPL_MODEL)])
        call("pedestrian_update_audit", [PYTHON, str(ROOT / "scripts/verify_pedestrian_update.py"),
            "--delivery", str(RUN), "--after", selected["payload"]])
        call("pedestrian_visualization", [PYTHON, str(ROOT / "scripts/visualize_pedestrian_pose_fill.py"),
            "--run", str(RUN), "--baseline-render", str(BASELINE / "joint-all/checkpoint-0900/runtime-render-source"),
            "--candidate-render", str(render), "--out", str(RUN / "visualizations")])
        metrics = json.loads((render / "manifest.json").read_text())
        baseline_metrics = json.loads((BASELINE / "joint-all/checkpoint-0900/runtime-render-source/manifest.json").read_text())
        pedestrians = json.loads((runtime / "pedestrian_manifest.json").read_text())
        loss_rows = [json.loads(line) for line in (joint / "loss_steps.jsonl").read_text().splitlines()]
        summary = {
            "status": "completed", "scene": "102", "selected_step": 950,
            "global_psnr": metrics["mean_psnr_source_frames_all_cameras"],
            "non_sky_psnr": metrics["mean_psnr_non_sky_source_frames_all_cameras"],
            "vehicle_bbox_psnr": metrics["mean_psnr_dynamic_bbox_source_frames_all_cameras"],
            "pedestrian_bbox_psnr": metrics["mean_psnr_pedestrian_bbox_source_frames_all_cameras"],
            "baseline_vr_v1_e_global_psnr": baseline_metrics["mean_psnr_source_frames_all_cameras"],
            "baseline_vr_v1_e_vehicle_bbox_psnr": baseline_metrics["mean_psnr_dynamic_bbox_source_frames_all_cameras"],
            "pose_completion": pedestrians["totals"],
            "loss_first": loss_rows[0], "loss_last": loss_rows[-1],
            "video": str(RUN / "final_global_runtime_predicted_front3.mp4"),
            "comparison_video": str(RUN / "final_global_runtime_gt_vs_predicted_front3.mp4"),
            "ply": str(RUN / "final_global_gaussians_frame000_preview.ply"),
        }
        save(RUN / "result_summary.json", summary)
        timer.update(status="completed", end=time.perf_counter(), end_epoch_seconds=time.time())
        timer["seconds"] = timer["end"] - timer["start"]
        timer["outputs"] = [summary["video"], summary["comparison_video"], summary["ply"]]
        write()
        subprocess.run([PYTHON, str(WORKFLOW), "set-run-status", "--project-root", str(PROJECT),
            "--run-id", RUN.name, "--status", "completed", "--summary",
            "Cold VR-v1-b Waymo102 plus pose-completed SMPL pedestrians completed",
            "--evidence", str((RUN / "result_summary.json").relative_to(PROJECT)) +
            "::Final quality, pose-completion statistics, and artifact paths",
            "--evidence", str(timer_path.relative_to(PROJECT)) +
            "::Measured raw-input to final delivery wall clock"], check=True)
    except BaseException as error:
        timer.update(status="failed", end=time.perf_counter(), error=repr(error))
        timer["seconds"] = timer["end"] - timer["start"]
        write()
        raise


if __name__ == "__main__":
    main()

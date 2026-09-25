"""Run the PFG-v1 global-Gaussian recipe as resumable named stages.

The scene017 contract deliberately preserves every accepted stage that led to
27.000626 dB.  It is also usable for another processed Argoverse scene, but
the target-domain refinements then learn a new scene-specific payload/refiner
and no numeric 27 dB guarantee is implied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    evidence: Path


def run(stage: Stage, env: dict[str, str], force: bool) -> dict:
    if stage.evidence.exists() and not force:
        print(f"[skip] {stage.name}: {stage.evidence}", flush=True)
        return {"stage": stage.name, "status": "skipped", "evidence": str(stage.evidence)}
    stage.evidence.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run ] {stage.name}\n       {' '.join(stage.command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(stage.command, check=True, env=env)
    if not stage.evidence.exists():
        raise FileNotFoundError(f"stage did not produce {stage.evidence}")
    return {
        "stage": stage.name,
        "status": "completed",
        "evidence": str(stage.evidence),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved YAML from prepare_config.py")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene", default="017")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--through-stage", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    scripts = project / "scripts"
    python = sys.executable
    config = Path(args.config).resolve()
    output = Path(args.output_root).resolve()
    ckpt = Path(args.checkpoint or project / "weights/pointforward_waymo_ds_paper_ckpt_020000.pt").resolve()
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(project / "src"), str(project), env.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    env.setdefault("HF_HOME", str(project / "weights/huggingface"))

    def py(script: str, *values: object) -> list[str]:
        return [python, str(scripts / script), *(str(value) for value in values)]

    map_dir = output / "01_global_static"
    map_payload = map_dir / "runtime-payload"
    track_payload = output / "02_track_once_payload"
    stages: list[Stage] = [
        Stage(
            "global-map",
            py(
                "deliver_continuous_scene.py", "--config", config, "--ckpt", ckpt,
                "--out", map_dir, "--scene", args.scene, "--reconstruction-mode", "global-gaussians",
                "--clip-order", "reverse", "--window-length", 8, "--window-step", 2,
                "--context-frame-stride", 2, "--global-voxel-size", 0.02,
                "--global-depth-threshold", 1.0, "--global-update-alpha", 0.5,
                "--global-update-distance-policy", "closest-smoothstep",
                "--global-update-near-distance", 20, "--global-update-far-distance", 80,
                "--global-update-min-improvement", 0,
                "--global-growth-policy", "three-clip-mutable-frozen-position",
                "--global-dynamic-policy", "persistent-tracks", "--global-dynamic-forward-refresh",
                "--gaussian-scale-postprocess", "isotropic-trace", "--isotropic-max-scale", 0.05,
                "--cache-overlap-image-networks", "--skip-ply", "--skip-frame-render",
                "--save-runtime-payload", map_payload,
            ),
            map_payload / "final_static_128d.pt",
        ),
        Stage(
            "track-once-dynamic",
            py(
                "build_track_once_dynamic.py", "--config", config, "--ckpt", ckpt,
                "--static-payload", map_payload / "final_static_128d.pt", "--out", track_payload,
                "--scene", args.scene, "--view-selection", "diverse-large",
                "--views-per-track", 12, "--view-angle-weight", 0.65,
                "--gaussian-scale-postprocess", "isotropic-trace", "--isotropic-max-scale", 0.05,
            ),
            track_payload / "track_once_manifest.json",
        ),
    ]

    payload = track_payload
    for index in range(1, 7):
        out = output / f"03_static_g4_e{index}"
        stages.append(Stage(
            f"static-g4-e{index}",
            py(
                "refine_static_features_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
                "--payload-dir", payload, "--out", out, "--scene", args.scene,
                "--epochs", 1, "--learning-rate", 0.01, "--anchor-weight", 0.0001,
                "--groups", 4, "--device", args.device,
            ),
            out / "manifest.json",
        ))
        payload = out / "runtime-payload-refined"

    out = output / "04_static_g8_e1"
    stages.append(Stage(
        "static-g8-e1",
        py(
            "refine_static_features_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--out", out, "--scene", args.scene,
            "--epochs", 1, "--learning-rate", 0.005, "--anchor-weight", 0.0001,
            "--groups", 8, "--device", args.device,
        ),
        out / "manifest.json",
    ))
    payload = out / "runtime-payload-refined"

    refiner_specs = [
        ("05_refiner_head", "head", 1, 1e-3, "mse"),
        ("06_refiner_up_head", "up-head", 1, 5e-4, "mse"),
        ("07_refiner_full_e1", "full", 1, 2e-4, "mse"),
        ("08_refiner_full_e3", "full", 3, 1e-4, "mse"),
        ("09_refiner_log_e3", "full", 3, 5e-5, "log-mse"),
        ("10_refiner_log_e6", "full", 6, 2.5e-5, "log-mse"),
        ("11_refiner_frame_log_e3", "full", 3, 5e-5, "frame-log-mse"),
    ]
    refiner: Path | None = None
    for name, scope, epochs, learning_rate, loss_mode in refiner_specs:
        out = output / name
        command = py(
            "refine_refiner_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--out", out, "--scene", args.scene,
            "--epochs", epochs, "--learning-rate", learning_rate,
            "--anchor-weight", 0.0001, "--scope", scope, "--loss-mode", loss_mode,
            "--device", args.device,
        )
        if refiner is not None:
            command += ["--refiner-state-in", str(refiner)]
        stages.append(Stage(name, command, out / "manifest.json"))
        refiner = out / "refiner_state.pt"

    out = output / "12_dynamic_feature_e1"
    stages.append(Stage(
        "dynamic-feature-e1",
        py(
            "refine_dynamic_features_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--out", out, "--scene", args.scene,
            "--epochs", 1, "--learning-rate", 0.002, "--anchor-weight", 0.0001,
            "--max-feature-delta", 0.25, "--refiner-state", refiner,
            "--device", args.device,
        ),
        out / "manifest.json",
    ))
    payload = out / "runtime-payload-refined"

    out = output / "13_static_xyz_5cm"
    stages.append(Stage(
        "static-xyz-5cm",
        py(
            "refine_static_xyz_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--refiner-state", refiner, "--out", out,
            "--scene", args.scene, "--epochs", 1, "--learning-rate", 0.02,
            "--max-displacement-m", 0.05, "--anchor-weight", 0.001, "--device", args.device,
        ),
        out / "manifest.json",
    ))
    payload = out / "runtime-payload-refined"

    out = output / "14_refiner_postxyz_e2"
    stages.append(Stage(
        "refiner-postxyz-e2",
        py(
            "refine_refiner_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--out", out, "--scene", args.scene,
            "--epochs", 2, "--learning-rate", 5e-5, "--anchor-weight", 0.0001,
            "--scope", "full", "--loss-mode", "frame-log-mse",
            "--refiner-state-in", refiner, "--device", args.device,
        ),
        out / "manifest.json",
    ))
    refiner = out / "refiner_state.pt"

    out = output / "15_static_xyz_2cm"
    stages.append(Stage(
        "static-xyz-2cm",
        py(
            "refine_static_xyz_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--refiner-state", refiner, "--out", out,
            "--scene", args.scene, "--epochs", 1, "--learning-rate", 0.01,
            "--max-displacement-m", 0.02, "--anchor-weight", 0.001, "--device", args.device,
        ),
        out / "manifest.json",
    ))
    payload = out / "runtime-payload-refined"

    out = output / "16_refiner_postxyz_e1"
    stages.append(Stage(
        "refiner-postxyz-e1",
        py(
            "refine_refiner_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--out", out, "--scene", args.scene,
            "--epochs", 1, "--learning-rate", 2.5e-5, "--anchor-weight", 0.0001,
            "--scope", "full", "--loss-mode", "frame-log-mse",
            "--refiner-state-in", refiner, "--device", args.device,
        ),
        out / "manifest.json",
    ))
    refiner = out / "refiner_state.pt"

    out = output / "17_static_xyz_1cm"
    stages.append(Stage(
        "static-xyz-1cm",
        py(
            "refine_static_xyz_fullscene_rgb.py", "--config", config, "--ckpt", ckpt,
            "--payload-dir", payload, "--refiner-state", refiner, "--out", out,
            "--scene", args.scene, "--epochs", 1, "--learning-rate", 0.005,
            "--max-displacement-m", 0.01, "--anchor-weight", 0.001, "--device", args.device,
        ),
        out / "manifest.json",
    ))
    payload = out / "runtime-payload-refined"

    metric_out = output / "18_metrics"
    stages.append(Stage(
        "metrics",
        py(
            "render_final_global_runtime.py", "--config", config, "--ckpt", ckpt,
            "--refiner-state", refiner, "--payload-dir", payload, "--out", metric_out,
            "--scene", args.scene, "--metrics-only", "--dynamic-bbox-metrics",
            "--camera-batch-size", 3, "--device", args.device,
        ),
        metric_out / "manifest.json",
    ))
    ply = output / "final_global_gaussians_frame000_preview.ply"
    stages.append(Stage(
        "export-ply",
        py("export_runtime_payload_ply.py", "--payload-dir", payload, "--out", ply, "--frame", 0),
        ply,
    ))
    if not args.skip_video:
        video_out = output / "19_video"
        stages.append(Stage(
            "render-video",
            py(
                "render_final_global_runtime.py", "--config", config, "--ckpt", ckpt,
                "--refiner-state", refiner, "--payload-dir", payload, "--out", video_out,
                "--scene", args.scene, "--fps", 60, "--camera-interpolation-factor", 6,
                "--interpolate-dynamics", "--camera-batch-size", 3, "--device", args.device,
            ),
            video_out / "manifest.json",
        ))

    names = [stage.name for stage in stages]
    if args.from_stage and args.from_stage not in names:
        parser.error(f"unknown --from-stage {args.from_stage}; choices: {', '.join(names)}")
    if args.through_stage and args.through_stage not in names:
        parser.error(f"unknown --through-stage {args.through_stage}; choices: {', '.join(names)}")
    first = names.index(args.from_stage) if args.from_stage else 0
    last = names.index(args.through_stage) + 1 if args.through_stage else len(stages)
    selected = stages[first:last]
    records = []
    for stage in selected:
        records.append(run(stage, env, args.force))
        (output / "pipeline_manifest.json").write_text(json.dumps({
            "status": "running", "project_version": "PFG-v1", "scene": args.scene,
            "config": str(config), "checkpoint": str(ckpt), "records": records,
            "final_payload": str(payload), "final_refiner": str(refiner),
        }, indent=2) + "\n")
    manifest = {
        "status": "completed", "project_version": "PFG-v1", "scene": args.scene,
        "config": str(config), "checkpoint": str(ckpt), "records": records,
        "final_payload": str(payload), "final_refiner": str(refiner),
        "ply": str(ply),
    }
    (output / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

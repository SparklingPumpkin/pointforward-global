#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_DIR CUDA_DEVICE QUERY_GATE(0|1) FUSION_GATE(0|1)" >&2
  exit 2
fi

ROOT=/mnt/data/fjn/projects/PointForward
RUN=$1
CUDA_DEVICE=$2
QUERY_GATE=$3
FUSION_GATE=$4
BASE="$ROOT/experiments/argoverse-inference/global-dynamic-consistency/gaussian-post-refinement/static-geometry-readaptation/SX-v1-a/run-20260928T140408Z-static-xyz-2cm/runtime-payload-refined"
BUILD_CONFIG="$ROOT/experiments/argoverse-inference/scene-transfer/AV-v1-a/run-20260911T071003Z-scene017-front3-full/config_argoverse017.yaml"
RENDER_CONFIG="$ROOT/experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-diagnostics/DG-v1-f/run-20260923T091123Z-scene017-diverse-large-12views/config_argoverse017_render.yaml"
CKPT="$ROOT/ckpt/ckpt_020000_waymo_ds_paper_original.pt"
MASK_DIR=${MASK_DIR_OVERRIDE:-"$ROOT/experiments/argoverse-inference/global-dynamic-consistency/dynamic-roi-refinement/rgb-foreground-gating/DR-v1-q/run-20260929T114723Z-rgb-mask-precompute"}
MASK_LAYOUT=${MASK_LAYOUT:-shared}
RF="$ROOT/experiments/argoverse-inference/global-dynamic-consistency/gaussian-post-refinement/refiner-adaptation/RF-v1-c/run-20260928T142858Z-geometry-payload-camera-e4"
PYTHON=/root/miniconda3/envs/drivestudio/bin/python

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gate_args=()
if [[ "$QUERY_GATE" == 1 ]]; then
  gate_args+=(--foreground-query-gate)
fi
if [[ "$FUSION_GATE" == 1 ]]; then
  gate_args+=(--foreground-fusion-gate)
fi
if [[ "$QUERY_GATE" == 1 || "$FUSION_GATE" == 1 ]]; then
  gate_args+=(--foreground-mask-dir "$MASK_DIR" --foreground-mask-layout "$MASK_LAYOUT")
fi

START=$(date +%s.%N)
"$PYTHON" -u "$ROOT/scripts/build_track_once_dynamic.py" \
  --config "$BUILD_CONFIG" --ckpt "$CKPT" \
  --static-payload "$BASE/final_static_128d.pt" \
  --out "$RUN/rebuilt-top10" --scene 017 --split val \
  --track-ids 47,26,0,41,29,18,34,19,24,7 \
  --view-selection diverse-large --view-angle-weight 0.65 --view-pass-index 0 \
  --views-per-track 12 --dynamic-queries-per-track 10000 \
  --gaussian-scale-postprocess isotropic-trace --isotropic-max-scale 0.05 \
  "${gate_args[@]}" 2>&1 | tee "$RUN/build.log"

"$PYTHON" "$ROOT/scripts/replace_dynamic_tracks.py" \
  --base "$BASE" --replacement "$RUN/rebuilt-top10" \
  --out "$RUN/runtime-payload-refined" 2>&1 | tee "$RUN/replace.log"

"$PYTHON" "$ROOT/scripts/export_runtime_payload_ply.py" \
  --payload-dir "$RUN/runtime-payload-refined" \
  --out "$RUN/final_global_gaussians_frame000_preview.ply" --frame 0 \
  2>&1 | tee "$RUN/export.log"
END=$(date +%s.%N)

python3 - "$START" "$END" "$RUN" "$QUERY_GATE" "$FUSION_GATE" <<'PY'
import json, sys
start, end, run, query_gate, fusion_gate = sys.argv[1:]
record = {
    "run_start_epoch_seconds": float(start),
    "run_end_epoch_seconds": float(end),
    "branch_wall_seconds": float(end) - float(start),
    "branch_wall_minutes": (float(end) - float(start)) / 60.0,
    "foreground_query_gate": query_gate == "1",
    "foreground_fusion_gate": fusion_gate == "1",
    "shared_rgb_mask_time_is_separate": True,
}
with open(run + "/pipeline_timing.json", "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
print(json.dumps(record, indent=2))
PY

"$PYTHON" "$ROOT/scripts/render_final_global_runtime.py" \
  --config "$RENDER_CONFIG" --ckpt "$CKPT" \
  --payload-dir "$RUN/runtime-payload-refined" \
  --camera-refiner-states \
    "$RF/camera0/refiner_state.pt" \
    "$RF/camera1/refiner_state.pt" \
    "$RF/camera2/refiner_state.pt" \
  --out "$RUN/runtime-render-source-metrics" --scene 017 --device cuda \
  --camera-batch-size 1 --metrics-only --dynamic-bbox-metrics \
  --per-track-bbox-metrics 2>&1 | tee "$RUN/eval.log"

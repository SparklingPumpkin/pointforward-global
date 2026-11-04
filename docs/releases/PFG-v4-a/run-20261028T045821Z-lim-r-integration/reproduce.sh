#!/usr/bin/env bash
set -euo pipefail
project_root="${1:?pointforward-global root}"
probe_python="${2:-python}"
cd "$project_root"
run=docs/releases/PFG-v4-a/run-20261028T045821Z-lim-r-integration
export PYTHONPATH="$project_root/docs/releases/VR-v1-s/source:$project_root/src"
"$probe_python" -m unittest discover -s tests -p test_lidar_bbox_preprocessing.py -v
# Fresh reruns require EMPTY output paths. Change /tmp paths if already used.
"$probe_python" scripts/preprocess_lidar_bbox.py --config "$run/argo_config.yaml" --selection "$run/argo_selection.json" --scene 017 --out /tmp/pfg-lim-r-argo-fresh --apply-frames 125
"$probe_python" scripts/preprocess_lidar_bbox.py --config "$run/ncore_config.yaml" --selection "$run/ncore_selection.json" --scene 250d2261-e28c-47a7-b317-cb340a6722c0 --out /tmp/pfg-lim-r-ncore-fresh --apply-frames 77

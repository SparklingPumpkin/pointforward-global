"""Portable orchestration adapters; the frozen optimizer equations are unchanged."""
import json
import os
from pathlib import Path
import sys


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def render_command(run, payload, out, refiner=None):
    recipe = json.loads((Path(run)/'delivery_options.json').read_text())
    command = [sys.executable, str(Path(__file__).with_name('render_final_global_runtime.py')),
        '--config', str(Path(run)/'config_render.yaml'), '--ckpt', recipe['checkpoint'],
        '--payload-dir', str(payload), '--out', str(out), '--scene', recipe['scene'],
        '--device', 'cuda', '--camera-batch-size', '1', '--fps', '10',
        '--dynamic-bbox-metrics', '--per-track-bbox-metrics', '--static-source-rgb-blend', '.75']
    if refiner:
        command += ['--refiner-state', str(refiner)]
    return command

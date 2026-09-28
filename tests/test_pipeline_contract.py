"""Static contract tests for the PFG-v2 orchestration entry points."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fast_g8_is_default_and_legacy_is_available():
    source = (ROOT / "scripts/run_current_pipeline.py").read_text()
    assert 'choices=("fast-g8", "legacy-alternating")' in source
    assert 'default="fast-g8"' in source
    assert '"--groups", 8' in source
    assert '"--camera-batch-size", 3' in source
    assert '"--empty-cache-every", 0' in source


def test_feature_refiner_supports_overlay_and_camera_batching():
    source = (ROOT / "scripts/refine_static_features_fullscene_rgb.py").read_text()
    assert 'parser.add_argument("--refiner-state"' in source
    assert 'parser.add_argument("--camera-batch-size"' in source
    assert '"--empty-cache-every"' in source
    assert "load_refiner_state(model" in source

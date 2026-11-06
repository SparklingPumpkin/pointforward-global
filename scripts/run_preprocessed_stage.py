"""Run a stage with explicit calibrated geometry and repaired mask consumption."""
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'docs/releases/VR-v1-s/source'), str(PROJECT/'src')]
from pointforward_preprocessing.consumer import install
install()

if len(sys.argv) < 2:
    raise SystemExit('Usage: run_preprocessed_stage.py STAGE.py [stage arguments]')
script = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')

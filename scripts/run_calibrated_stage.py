"""Execute a frozen stage with the explicit LIM-r dataset consumer enabled."""
from pathlib import Path
import runpy
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'docs/releases/VR-v1-s/source'))
sys.path.insert(1,str(PROJECT/'src'))
from pointforward_calibration.consumer import install
install()
script=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(script.parent))
runpy.run_path(str(script),run_name='__main__')

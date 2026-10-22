"""Use a pinned generic gsplat extension, without mutating shared JIT cache."""
import importlib.util
from pathlib import Path
import sys
import os
binary = Path(__file__).parent/'gsplat_cuda.so'
if binary.is_file() and os.environ.get('POINTFORWARD_USE_PINNED_GSPLAT') == '1':
    import torch
    spec = importlib.util.spec_from_file_location('gsplat_cuda', binary)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules['gsplat.csrc'] = mod

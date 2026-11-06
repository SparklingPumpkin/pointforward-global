"""Small, explicit input/output identities shared by preprocessing entry points."""
import hashlib
import json
from pathlib import Path


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    result = dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size <= 10_000_000:
        result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def empty_output(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output must be an empty directory: '+str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path

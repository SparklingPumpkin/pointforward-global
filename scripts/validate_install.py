"""Lightweight delivery validation; it never trains a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    manifest_path = project / "weights/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    for record in manifest["files"]:
        path = project / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {record['path']}")
        if not args.skip_hash and sha256(path) != record["sha256"]:
            failures.append(f"sha256 mismatch: {record['path']}")
    import torch
    import pointforward
    import gsplat
    summary = {
        "status": "failed" if failures else "passed",
        "project": str(project),
        "torch": torch.__version__,
        "pointforward": str(Path(pointforward.__file__).resolve()),
        "gsplat": getattr(gsplat, "__version__", "unknown"),
        "weight_count": len(manifest["files"]),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

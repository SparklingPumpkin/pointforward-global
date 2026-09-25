"""Resolve a portable PointForward YAML template for one machine."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="configs/argoverse_global.template.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    template = Path(args.template)
    if not template.is_absolute():
        template = project / template
    text = template.read_text()
    replacements = {
        "__PROJECT_ROOT__": str(project),
        "__DATA_ROOT__": str(Path(args.data_root).resolve()),
        "__OUTPUT_ROOT__": str(Path(args.output_root).resolve()),
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    unresolved = sorted({part.split("__", 1)[0] for part in text.split("__")[1::2]})
    if "__" in text:
        raise ValueError(f"unresolved placeholders remain: {unresolved}")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out)


if __name__ == "__main__":
    main()

"""Replace selected scene-global dynamic tracks without appending duplicates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


FIELDS = (
    "means", "quats", "scales", "opacities", "colors",
    "sigma", "times", "instance_id",
)


def link_exact(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(target)
        return
    os.symlink(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--use-replacement-meta",
        action="store_true",
        help="Use replacement global_dynamic_meta.pt (needed for new segment slots).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base).resolve()
    replacement_dir = Path(args.replacement).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = torch.load(
        base_dir / "final_dynamic_128d.pt", map_location="cpu", weights_only=False
    )
    replacement = torch.load(
        replacement_dir / "final_dynamic_128d.pt",
        map_location="cpu",
        weights_only=False,
    )
    replaced_slots = torch.unique(replacement["instance_id"], sorted=True)
    if replaced_slots.numel() == 0:
        raise RuntimeError("Replacement payload contains no tracks")
    keep = ~torch.isin(base["instance_id"], replaced_slots)
    common_extras = sorted(set(base["extras"]) & set(replacement["extras"]))
    merged = {
        field: torch.cat([base[field][keep], replacement[field]], dim=0)
        for field in FIELDS
    }
    merged["extras"] = {
        name: torch.cat(
            [base["extras"][name][keep], replacement["extras"][name]], dim=0
        )
        for name in common_extras
    }
    order = torch.argsort(merged["instance_id"], stable=True)
    for field in FIELDS:
        merged[field] = merged[field][order]
    for name in common_extras:
        merged["extras"][name] = merged["extras"][name][order]
    torch.save(merged, output_dir / "final_dynamic_128d.pt")
    link_exact(base_dir / "final_static_128d.pt", output_dir / "final_static_128d.pt")
    meta_source = replacement_dir if args.use_replacement_meta else base_dir
    link_exact(meta_source / "global_dynamic_meta.pt", output_dir / "global_dynamic_meta.pt")
    manifest = {
        "status": "completed",
        "base": str(base_dir),
        "replacement": str(replacement_dir),
        "meta_source": str(meta_source),
        "policy": "remove every base Gaussian in replacement global slots, then insert rebuilt tracks",
        "replaced_slots": [int(value) for value in replaced_slots.tolist()],
        "base_dynamic_count": int(base["means"].shape[0]),
        "removed_count": int((~keep).sum()),
        "replacement_count": int(replacement["means"].shape[0]),
        "final_dynamic_count": int(merged["means"].shape[0]),
    }
    (output_dir / "track_replacement_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

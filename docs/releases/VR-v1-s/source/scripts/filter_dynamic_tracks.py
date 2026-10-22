"""Filter scene-global dynamic payload tracks and remap instance slots."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--classification-meta", required=True)
    parser.add_argument("--static", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dynamic = torch.load(args.dynamic, map_location="cpu", weights_only=False)
    meta = torch.load(args.meta, map_location="cpu", weights_only=False)
    classification = torch.load(
        args.classification_meta, map_location="cpu", weights_only=False
    )
    excluded = set(
        int(value)
        for value in classification["stationary_track_reclassification"]["track_ids"]
    )
    old_track_ids = [int(value) for value in meta["track_ids"]]
    kept_old_slots = [
        slot for slot, track_id in enumerate(old_track_ids)
        if track_id not in excluded
    ]
    slot_map = {old: new for new, old in enumerate(kept_old_slots)}
    instance_id = dynamic["instance_id"].long()
    keep = torch.zeros_like(instance_id, dtype=torch.bool)
    for old_slot in kept_old_slots:
        keep |= instance_id == old_slot
    for key, value in list(dynamic.items()):
        if key == "extras":
            dynamic[key] = {
                name: tensor[keep] for name, tensor in value.items()
            }
        elif torch.is_tensor(value) and value.shape[:1] == keep.shape:
            dynamic[key] = value[keep]
    remapped = dynamic["instance_id"].clone()
    for old_slot, new_slot in slot_map.items():
        remapped[dynamic["instance_id"] == old_slot] = new_slot
    dynamic["instance_id"] = remapped

    meta["track_ids"] = [old_track_ids[slot] for slot in kept_old_slots]
    index = torch.tensor(kept_old_slots, dtype=torch.long)
    meta["boxes"] = {
        key: torch.index_select(value, 0, index)
        for key, value in meta["boxes"].items()
    }
    meta["excluded_stationary_track_ids"] = sorted(excluded)
    meta["source_track_ids"] = old_track_ids

    torch.save(dynamic, out / "final_dynamic_128d.pt")
    torch.save(meta, out / "global_dynamic_meta.pt")
    static_link = out / "final_static_128d.pt"
    static_link.symlink_to(Path(args.static).resolve())
    print({
        "source_tracks": len(old_track_ids),
        "kept_tracks": len(meta["track_ids"]),
        "excluded_tracks": sorted(excluded.intersection(old_track_ids)),
        "kept_gaussians": int(dynamic["means"].shape[0]),
    })


if __name__ == "__main__":
    main()

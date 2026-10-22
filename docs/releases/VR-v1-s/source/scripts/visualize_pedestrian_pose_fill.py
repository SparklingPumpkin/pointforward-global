"""Create same-region pedestrian crops for GT, VR-v1-e, and the candidate."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline-render", required=True)
    parser.add_argument("--candidate-render", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    baseline = Path(args.baseline_render)
    candidate = Path(args.candidate_render)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(
        (run / "runtime-payload/pedestrian_manifest.json").read_text()
    )["tracks"]

    def usable(track):
        return [
            view for view in track["selected_views"]
            if view["bbox_area_fraction"] >= 0.001
            and view["bbox_xyxy"][0] > 2 and view["bbox_xyxy"][1] > 26
            and view["bbox_xyxy"][2] < 765 and view["bbox_xyxy"][3] < 573
        ]

    ranked = sorted(
        (track for track in tracks if usable(track)),
        key=lambda track: max(view["bbox_area_fraction"] for view in usable(track)),
        reverse=True,
    )[:5]
    records = []
    for track in ranked:
        views = sorted(usable(track), key=lambda view: (view["frame"], view["camera"]))
        count = min(4, len(views))
        chosen = [views[round(i * (len(views) - 1) / max(1, count - 1))] for i in range(count)]
        canvas = Image.new("RGB", (count * 300, 3 * 300 + 36), "#161a20")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"Pedestrian {track['track_id']} | identical crops", fill="white")
        for column, view in enumerate(chosen):
            frame = int(view["frame"])
            camera = int(view["camera"])
            name = f"frame_{frame:04d}.jpg"
            comparison = Image.open(candidate / "frames/comparison" / name)
            sources = [
                comparison.crop((0, 0, comparison.width, 576)),
                Image.open(baseline / "frames/predicted" / name),
                Image.open(candidate / "frames/predicted" / name),
            ]
            labels = ["GT", "VR-v1-e no pedestrians", "VR-v1-g pose-filled SMPL"]
            x1, y1, x2, y2 = view["bbox_xyxy"]
            pad_x = max(12, (x2 - x1) * 0.3)
            pad_y = max(12, (y2 - y1) * 0.2)
            box = (
                max(0, int(x1 - pad_x)) + camera * 768,
                max(24, int(y1 - pad_y)),
                min(768, int(x2 + pad_x)) + camera * 768,
                min(576, int(y2 + pad_y)),
            )
            for row, (source, label) in enumerate(zip(sources, labels)):
                crop = ImageOps.contain(source.crop(box), (286, 255), Image.Resampling.LANCZOS)
                left = column * 300 + (300 - crop.width) // 2
                top = 36 + row * 300 + 35 + (255 - crop.height) // 2
                canvas.paste(crop, (left, top))
                draw.text((column * 300 + 6, 36 + row * 300 + 4), label, fill="white")
                draw.text((column * 300 + 6, 36 + row * 300 + 18),
                          f"frame {frame} / camera {camera}", fill="#afc8dc")
        path = out / f"pedestrian_{int(track['track_id']):03d}_comparison.jpg"
        canvas.save(path, quality=95)
        records.append({"track_id": int(track["track_id"]), "views": chosen, "image": str(path)})
    (out / "manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

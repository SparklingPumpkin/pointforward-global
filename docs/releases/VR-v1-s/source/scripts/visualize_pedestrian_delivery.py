"""Presentation crops only; formal metrics come from float runtime rendering."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delivery", required=True)
    p.add_argument("--reference", required=True)
    a = p.parse_args()
    root, ref = Path(a.delivery), Path(a.reference)
    out = root / "visualizations"
    out.mkdir(exist_ok=True)
    people = json.loads((root / "smpl/runtime-payload/pedestrian_manifest.json").read_text())["tracks"]
    # Favor large, fully in-frame people. Choose four separated views per track.
    def usable(t):
        return [v for v in t["selected_views"] if v["bbox_area_fraction"] >= .002
                and v["bbox_xyxy"][0] > 3 and v["bbox_xyxy"][1] > 27
                and v["bbox_xyxy"][2] < 764 and v["bbox_xyxy"][3] < 572]
    ranked = sorted((t for t in people if len(usable(t)) >= 4),
                    key=lambda t: max(v["bbox_area_fraction"] for v in usable(t)), reverse=True)[:5]
    rows = ["GT", "FFR-r control / 350 steps", "SMPL / 350 steps", "WM-i reference / 800 steps"]
    records = []
    for t in ranked:
        views = sorted(usable(t), key=lambda v: (v["frame"], v["camera"]))
        views = [views[round(i * (len(views)-1)/3)] for i in range(4)]
        canvas = Image.new("RGB", (4*300, 4*320+36), "#161a20")
        d = ImageDraw.Draw(canvas)
        d.text((10, 10), f"Pedestrian {t['track_id']} | each column uses the same crop | source views", fill="white")
        for col, v in enumerate(views):
            frame, cam = v["frame"], v["camera"]
            name = f"frame_{frame:04d}.jpg"
            comp = Image.open(root / "smpl/runtime-render-source/frames/comparison" / name)
            sources = [comp.crop((0, 0, comp.width, 576)),
                Image.open(root / "control/runtime-render-source/frames/predicted" / name),
                Image.open(root / "smpl/runtime-render-source/frames/predicted" / name),
                Image.open(ref / "frames/predicted" / name)]
            x1, y1, x2, y2 = v["bbox_xyxy"]
            px, py = max(12, (x2-x1)*.25), max(12, (y2-y1)*.15)
            box = (max(0, int(x1-px))+cam*768, max(24, int(y1-py)),
                   min(768, int(x2+px))+cam*768, min(576, int(y2+py)))
            for row, (source, label) in enumerate(zip(sources, rows)):
                crop = ImageOps.contain(source.crop(box), (290, 275), Image.Resampling.LANCZOS)
                x, y = col*300, 36+row*320
                canvas.paste(crop, (x+(300-crop.width)//2, y+35+(275-crop.height)//2))
                d.text((x+6, y+3), label, fill="white")
                d.text((x+6, y+17), f"frame {frame} / camera {cam}", fill="#afc8dc")
        path = out / f"pedestrian_{t['track_id']:03d}_comparison.jpg"
        canvas.save(path, quality=95)
        records.append({"track_id": t["track_id"], "views": views, "image": str(path)})
    overview = Image.new("RGB", (1200, len(records)*350), "#161a20")
    for i, r in enumerate(records):
        im = Image.open(r["image"])
        representative = max(range(len(r["views"])), key=lambda j: r["views"][j]["bbox_area_fraction"])
        # Representative view with four methods side by side for quick inspection.
        panel = Image.new("RGB", (1200, 350), "#161a20")
        d = ImageDraw.Draw(panel)
        d.text((8, 4), f"Pedestrian {r['track_id']} | representative selected view", fill="white")
        for row in range(4):
            x = representative * 300
            panel.paste(im.crop((x, 36+row*320, x+300, 36+(row+1)*320)), (row*300, 28))
        overview.paste(panel, (0, i*350))
    overview.save(out / "pedestrians_overview.jpg", quality=95)
    (out / "crop_manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

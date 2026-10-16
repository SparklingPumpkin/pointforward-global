"""Build compact visual/metric evidence without changing any runtime asset."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--baseline-render', required=True)
    a = p.parse_args()
    run = Path(a.run).resolve()
    rendered = run/'runtime-render-source'
    manifest = json.loads((rendered/'manifest.json').read_text())
    timer = json.loads((run/'pipeline_timing.json').read_text())
    keys = ['mean_psnr_source_frames_all_cameras',
        'mean_psnr_non_sky_source_frames_all_cameras',
        'mean_psnr_dynamic_bbox_source_frames_all_cameras']
    selected = [0, 38, 77, 116, 155]
    montage = Image.new('RGB', (1152, (144+28)*len(selected)), 'white')
    draw = ImageDraw.Draw(montage)
    baseline = Path(a.baseline_render)
    # Baseline is 60FPS; source anchors are every sixth output. Candidate is10FPS.
    for row, frame in enumerate(selected):
        y = row*(144+28)
        draw.text((8,y+6), f'frame {frame:03d} | left: raw feedforward | right: candidate',fill='black')
        left = Image.open(baseline/'frames/predicted'/f'frame_{frame*6:04d}.jpg').convert('RGB')
        right = Image.open(rendered/'frames/predicted'/f'frame_{frame:04d}.jpg').convert('RGB')
        montage.paste(left.resize((576,144)),(0,y+28))
        montage.paste(right.resize((576,144)),(576,y+28))
    montage.save(run/'keyframes_montage.jpg',quality=95)
    summary = {'scope':'cached_incremental', 'cold_goal_met':False,
        'seconds':timer['seconds'], 'metrics':{k:manifest.get(k) for k in keys},
        'video':str(rendered/'final_global_runtime_predicted_front3.mp4'),
        'comparison_video':str(rendered/'final_global_runtime_gt_vs_predicted_front3.mp4'),
        'montage':str(run/'keyframes_montage.jpg')}
    (run/'probe_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

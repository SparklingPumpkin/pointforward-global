"""Summarize no-adaptation feedforward runs and produce comparable keyframes."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    rows=[]
    for run in args.runs:
        render=run/'runtime-render-source'
        if not (render/'manifest.json').exists():
            continue
        d=json.loads((render/'manifest.json').read_text())
        timer=json.loads((run/'pipeline_timing.json').read_text())
        opts=json.loads((run/'options.json').read_text())
        assert d['refiner_state'] is None and d['rgb_residual_state'] is None
        keys=['mean_psnr_source_frames_all_cameras','mean_psnr_non_sky_source_frames_all_cameras',
            'mean_psnr_dynamic_bbox_source_frames_all_cameras']
        item=dict(run_id=run.name,run_path=str(run.resolve()),options=opts,
            psnr={k:d[k] for k in keys},seconds=timer['seconds'],scope=timer['scope'],
            reconstructed_cold_seconds_estimate=timer.get('reconstructed_cold_seconds_estimate'),
            video=str(render/'final_global_runtime_predicted_front3.mp4'),
            comparison_video=str(render/'final_global_runtime_gt_vs_predicted_front3.mp4'))
        static=run/'static-build/manifest.json'
        if static.exists():
            sd=json.loads(static.read_text())
            item['static_updates_per_clip']=[r.get('observed_old_count') for r in sd['windows']]
            item['static_points']=sd.get('runtime_payload',{}).get('static_count')
            item['context_schedule']=[r.get('context_frames') for r in sd['windows']]
        montage=Image.new('RGB',(1152,5*620),'white')
        draw=ImageDraw.Draw(montage)
        for index,frame in enumerate([0,38,77,116,155]):
            image=Image.open(render/'frames/comparison'/f'frame_{frame:04d}.jpg').convert('RGB')
            y=index*620
            draw.text((8,y+4),f'{run.name} frame {frame} | top GT, bottom predicted',fill='black')
            image.thumbnail((1152,590))
            montage.paste(image,(0,y+25))
        montage.save(run/'keyframes_montage.jpg',quality=95)
        item['montage']=str(run/'keyframes_montage.jpg')
        rows.append(item)
    args.out.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

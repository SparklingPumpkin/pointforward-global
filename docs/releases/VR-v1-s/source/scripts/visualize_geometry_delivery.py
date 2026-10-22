"""Contact sheets from actual RGB calibration masks and final runtime frames."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args()
    root=args.run;geo=root/'geometry-input'
    plan=json.loads((geo/'plan.json').read_text())['observations']
    selected=[o for o in plan if o['split']!='filter-only']
    tracks=sorted({o['track_id'] for o in selected})
    tiles=[]
    for tid in tracks:
        o=max([o for o in selected if o['track_id']==tid],key=lambda o:o['bbox_area_fraction'])
        image=Image.open(geo/'masks'/f"{o['key']}_overlay.jpg").convert('RGB')
        x0,y0,x1,y1=o['bbox_xyxy'];pad=25
        image=image.crop((max(0,x0-pad),max(0,y0-pad),min(image.width,x1+pad),min(image.height,y1+pad)))
        image.thumbnail((384,250));tile=Image.new('RGB',(384,280),'#202020');tile.paste(image,((384-image.width)//2,25))
        ImageDraw.Draw(tile).text((8,7),f"track {tid}  frame {o['frame']} cam{o['camera']}  {o['split']}",fill='white')
        tiles.append(tile)
    cols=3;rows=(len(tiles)+cols-1)//cols;sheet=Image.new('RGB',(384*cols,280*rows),'#202020')
    for i,t in enumerate(tiles):sheet.paste(t,((i%cols)*384,(i//cols)*280))
    sheet.save(root/'geometry_calibration_examples.jpg')
    video=root/'final_global_runtime_gt_vs_predicted_front3.mp4'
    if video.exists():
        cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[]
        for f in (0,count//3,2*count//3,count-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read()
            if ok:
                image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));image.thumbnail((1152,576));frames.append(image)
        cap.release()
        sheet=Image.new('RGB',(max(i.width for i in frames),sum(i.height for i in frames)))
        y=0
        for i in frames:sheet.paste(i,(0,y));y+=i.height
        sheet.save(root/'final_video_examples.jpg')


if __name__=='__main__':main()

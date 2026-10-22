"""SAM2 RGB masks grouped by source image to share image-encoder work."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sam-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();root=args.out
    repo=args.sam_root.resolve();sys.path.insert(0,str(repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(args.checkpoint.resolve()),device='cuda'))
    (root/'masks').mkdir(exist_ok=True)
    groups={}
    for o in json.loads((root/'plan.json').read_text())['observations']:
        groups.setdefault((o['frame'],o['camera']),[]).append(o)
    records=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for (f,c),group in tqdm(sorted(groups.items()),desc='RGB vehicle masks'):
            image=np.array(Image.open(root/'inputs'/f'{f:03d}_{c}.png').convert('RGB'))
            predictor.set_image(image)
            for o in group:
                masks,scores,_=predictor.predict(box=np.asarray(o['sam_prompt_xyxy']),point_coords=np.asarray([o['sam_positive_xy']]),point_labels=np.array([1]),multimask_output=True)
                index=int(np.argmax(scores));mask=masks[index]>0
                Image.fromarray(mask.astype(np.uint8)*255).save(root/'masks'/f"{o['key']}.png")
                records.append(dict(key=o['key'],pixels=int(mask.sum()),scores=scores.tolist(),chosen=index))
                if o['split']!='filter-only':
                    overlay=image.copy();overlay[mask]=(overlay[mask]*.5+np.array([0,255,100])*.5).round().astype(np.uint8)
                    Image.fromarray(overlay).save(root/'masks'/f"{o['key']}_overlay.jpg")
    (root/'masks/manifest.json').write_text(json.dumps(dict(model='SAM2.1 Hiera Tiny',records=records,source='RGB predicted masks, not GT; largest predicted confidence, no bbox clipping'),indent=2)+'\n')


if __name__=='__main__':
    main()

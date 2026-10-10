"""Diagnostic phone model comparison on exported phone-positive sample frames.

Does not approve models or tune thresholds on this evaluation video.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import torch
from ultralytics import YOLO

from audit_cvat_video import digest, iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    coco = json.loads((args.dataset/'coco.json').read_text(encoding='utf-8'))
    phone_id = next(c['id'] for c in coco['categories'] if c['name']=='Phone')
    labels = {}
    for a in coco['annotations']:
        if a['category_id'] == phone_id:
            x,y,w,h = a['bbox']
            labels.setdefault(a['image_id'],[]).append([x,y,x+w,y+h])
    report = {'scope':'Phone-positive sampled frames from ONE session; diagnostic only. No precision, false-positive rate, training, threshold calibration or model release approval.', 'confidence_threshold':.25,'iou_threshold':.5,'sample_frames':sorted(labels),'models':[]}
    for model_path, aliases in [('models/weights/yolo26n.pt',{'cell phone'}),('models/weights/yolo26m_phone.pt',{'phone'})]:
        model = YOLO(model_path)
        for tiled, image_size in ([(False,640),(False,1280),(True,640)] if 'yolo26n.pt' in model_path else [(False,640),(False,1280)]):
            matched = count = proposals = 0
            frame_results = []
            started = perf_counter()
            for image_meta in coco['images']:
                frame = image_meta['id']
                if frame not in labels:
                    continue
                image = cv2.imread(str(args.dataset/image_meta['file_name']))
                if image is None:
                    raise RuntimeError('Cannot read exported image')
                height,width = image.shape[:2]
                views = [(0,0,width,height)]
                if tiled:
                    tw,th = int(width*.56),int(height*.56)
                    views += [(0,0,tw,th),(width-tw,0,width,th),(0,height-th,tw,height),(width-tw,height-th,width,height)]
                candidates = []
                for x,y,x2,y2 in views:
                    result = model.predict(image[y:y2,x:x2],conf=.25,imgsz=image_size,device='cuda:0',verbose=False)[0]
                    for box in result.boxes:
                        if str(model.names[int(box.cls.item())]).lower() in aliases:
                            bx,by,bx2,by2 = box.xyxy[0].tolist()
                            candidates.append({'box':[bx+x,by+y,bx2+x,by2+y],'confidence':box.conf.item()})
                retained = []
                for c in sorted(candidates,key=lambda c:c['confidence'],reverse=True):
                    if all(iou(c['box'],b['box']) < .5 for b in retained):
                        retained.append(c)
                used = set()
                hits = 0
                for c in retained:
                    best = max(((iou(c['box'],b),i) for i,b in enumerate(labels[frame]) if i not in used),default=(0,-1))
                    if best[0] >= .5:
                        used.add(best[1]); hits += 1
                matched += hits; count += len(labels[frame]); proposals += len(retained)
                frame_results.append({'frame_index':frame,'labelled_boxes':len(labels[frame]),'matched_boxes':hits,'predictions':retained})
            report['models'].append({'checkpoint':model_path,'sha256':digest(Path(model_path)),'class_mapping':model.names,'mode':'full_frame_plus_four_tiles' if tiled else 'full_frame','image_size':image_size,'labelled_boxes':count,'matched_boxes':matched,'recall_at_iou_0_5':matched/count if count else None,'proposals':proposals,'elapsed_sec_including_warmup':perf_counter()-started,'per_frame':frame_results})
            print(model_path,'tiled',tiled,'imgsz',image_size,'matched',matched,'/',count,flush=True)
        del model
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()

"""Bootstrap a phone-only candidate from trusted positive CVAT crops.

Missing annotations never generate negative training frames. Validation reuses
training crops ONLY as a fit diagnostic; independent evaluation is unavailable.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

import cv2
import torch
import yaml
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from drishti.data.video_policy import bounded_video_source
from audit_cvat_video import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default='input/videos/VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4')
    parser.add_argument('--annotations', type=Path, default=Path('MANUAL_ANNOTATIONS(CVAT)/VIDEO_3_EXAM_HALL_PHONE_USE/annotations.xml'))
    parser.add_argument('--output', type=Path, default=Path('outputs/training/video03_phone_bootstrap_20261002'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--base-model', default='models/weights/yolo26n.pt')
    args = parser.parse_args()
    if args.epochs <= 0 or args.stride <= 0:
        parser.error('epochs and stride must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this training workflow')
    torch.set_num_threads(4)
    video, bound = bounded_video_source(args.video, 2200, 'VIDEO_03')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output/'dataset/images/train'
    labels = args.output/'dataset/labels/train'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    raw_hash = digest(args.annotations)
    root = ET.parse(args.annotations).getroot()
    rows = defaultdict(list)
    visible_count = 0
    for track in root.findall('track'):
        if track.get('label') != 'Phone':
            continue
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            if frame >= bound or box.get('outside') == '1':
                continue
            visible_count += 1
            if frame % args.stride:
                continue
            rows[frame].append({'track_id':int(track.get('id')), 'entity_id':'phone_01', 'bbox':[float(box.get(k)) for k in ('xtl','ytl','xbr','ybr')]})
    if not rows:
        raise RuntimeError('No explicit positive phone annotations')
    rng = random.Random(42)
    capture = cv2.VideoCapture(video)
    exported = []
    try:
        for frame in range(bound):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'Canonical video ended early at {frame}')
            if frame not in rows:
                continue
            height,width = image.shape[:2]
            for primary in rows[frame]:
                x,y,x2,y2 = primary['bbox']
                if not (0 <= x < x2 <= width and 0 <= y < y2 <= height):
                    raise ValueError(f'Invalid raw phone geometry at frame {frame}')
                size = min(384, width, height)
                cx = max(0, min(width-size, round((x+x2)/2-size/2+rng.uniform(-45,45))))
                cy = max(0, min(height-size, round((y+y2)/2-size/2+rng.uniform(-45,45))))
                # Keep labelled objects intact; never clip their semantic boxes.
                if not (cx <= x and cy <= y and x2 <= cx+size and y2 <= cy+size):
                    raise ValueError('Primary object would be clipped')
                included = [r for r in rows[frame] if cx <= r['bbox'][0] and cy <= r['bbox'][1] and r['bbox'][2] <= cx+size and r['bbox'][3] <= cy+size]
                name = f'frame_{frame:06d}_track_{primary["track_id"]}'
                if not cv2.imwrite(str(images/(name+'.jpg')),image[cy:cy+size,cx:cx+size]):
                    raise RuntimeError('Crop write failed')
                label_lines = []
                for r in included:
                    ax,ay,bx,by = r['bbox']
                    label_lines.append(f'0 {((ax+bx)/2-cx)/size:.8f} {((ay+by)/2-cy)/size:.8f} {(bx-ax)/size:.8f} {(by-ay)/size:.8f}')
                (labels/(name+'.txt')).write_text('\n'.join(label_lines)+'\n',encoding='utf-8')
                exported.append({'image':str(images/(name+'.jpg')),'frame_index':frame,'crop_xyxy':[cx,cy,cx+size,cy+size],'objects':included,'identity_source':'User-confirmed same physical phone across tracks 3 and 4'})
    finally:
        capture.release()
    data = {'path':str(args.output/'dataset'),'train':'images/train','val':'images/train','names':{0:'phone'}}
    yaml_path = args.output/'dataset/data.yaml'
    yaml_path.write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    (args.output/'dataset/samples.json').write_text(json.dumps(exported,indent=2),encoding='utf-8')
    manifest = {'status':'TRAINING','started_at':datetime.now(timezone.utc).isoformat(),'video':video,'video_sha256':digest(Path(video)),'max_frames':bound,'annotations_sha256':raw_hash,'base_model':args.base_model,'base_model_sha256':digest(Path(args.base_model)),'visible_phone_boxes':visible_count,'training_samples':len(exported),'frame_stride':args.stride,'physical_phone_entities':1,'classes':['phone'],'validation_policy':'TRAINING_FIT_ONLY: val equals train, no independent validation/test data','negative_policy':'Only explicit positive phone crops; unannotated frames excluded','release_approved':False,'epochs_requested':args.epochs,'training_config':{'imgsz':640,'batch':8,'freeze':10,'seed':42,'mosaic':0,'amp':False}}
    manifest['environment'] = {'ultralytics':ultralytics.__version__, 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)}
    def save():
        (args.output/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    try:
        model = YOLO(args.base_model)
        model.train(data=str(yaml_path),epochs=args.epochs,imgsz=640,batch=8,device=0,workers=0,project=str(args.output),name='fit',exist_ok=False,seed=42,deterministic=True,amp=False,freeze=10,mosaic=0,mixup=0,fliplr=0,flipud=0,scale=.2,translate=.15,cache=False,plots=False,val=False,patience=0,save=True)
        checkpoint = args.output/'fit/weights/last.pt'
        if not checkpoint.is_file():
            raise RuntimeError('Training produced no last.pt')
        manifest['status'] = 'COMPLETE'
        manifest['candidate_checkpoint'] = str(checkpoint)
        manifest['candidate_sha256'] = digest(checkpoint)
        manifest['training_fit_metrics'] = model.metrics.results_dict
        manifest['training_fit_metrics_checkpoint'] = str(args.output/'fit/weights/best.pt')
        manifest['training_fit_metrics_warning'] = 'Resubstitution on training crops; NOT held-out accuracy or full-video quality'
    except Exception as error:
        manifest['status'] = 'FAILED'
        manifest['error'] = str(error)
        raise
    finally:
        if digest(args.annotations) != raw_hash:
            manifest['status'] = 'FAILED'
            manifest['error'] = 'Raw annotation hash changed'
        save()
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()

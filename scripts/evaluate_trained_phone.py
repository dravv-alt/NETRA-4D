"""Evaluate an experimental phone checkpoint on the bounded Video 3 source.

This is training-session recall, not held-out model quality.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from drishti.data.video_policy import bounded_video_source
from drishti.providers.phone_crop import detect_phone_candidates, PersonCropMemory
from audit_cvat_video import digest
from evaluate_cvat_boxes import evaluate, load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=Path('outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--person-proposals', type=Path, help='Detector JSONL with people boxes; annotations must never supply crop locations')
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--roi-hold-seconds',type=float,default=0.,help='Hold detector person ROIs briefly; phone detections are always recomputed')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    video, bound = bounded_video_source('VIDEO_03_mobile_phone_exam_hall_01.mkv',2200)
    model = YOLO(str(args.model))
    if set(model.names.values()) != {'phone'}:
        raise ValueError(f'Expected phone-only candidate, found {model.names}')
    proposals = {}
    if args.person_proposals:
        for row in load_rows(args.person_proposals):
            index = row['frame_index']
            if index in proposals:
                raise ValueError('Duplicate person-proposal frame')
            proposals[index] = row['people']
        if set(proposals) != set(range(bound)):
            raise ValueError('Person proposals must cover the exact bounded video range')
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    crop_memory = PersonCropMemory(args.roi_hold_seconds)
    predictions = []
    started = perf_counter()
    try:
        with (args.output/'detections.jsonl').open('w',encoding='utf-8') as stream:
            for index in range(bound):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'Video ended before bound at {index}')
                people = proposals.get(index,[])
                height,width = frame.shape[:2]
                regions = crop_memory.regions(people,width,height,index/fps,args.crop_size) if proposals else None
                objects = detect_phone_candidates(model,frame,[p['bbox_xyxy_px'] for p in people],device='cuda:0',full_image_size=args.image_size,confidence=.25,crop_size=args.crop_size,use_crops=bool(proposals),regions_override=regions)
                row = {'frame_index':index,'people':[],'objects':objects}
                stream.write(json.dumps(row)+'\n')
                predictions.append(row)
                if (index+1) % 250 == 0:
                    print('evaluated',index+1,'/',bound,flush=True)
    finally:
        capture.release()
    report = evaluate(load_rows(args.labels),predictions,bound)
    report['per_class'].pop('Person')
    report.update({'checkpoint':str(args.model),'checkpoint_sha256':digest(args.model),'labels_sha256':digest(args.labels),'video_sha256':digest(Path(video)),'image_size':args.image_size,'confidence_threshold':.25,'elapsed_sec':perf_counter()-started,'evaluation_scope':'TRAINING_SESSION_RECALL: same physical phone/camera/session used to bootstrap training; no generalization claim','release_approved':False})
    report['inference_mode'] = 'full_frame_plus_detector_person_crops' if proposals else 'full_frame'
    if proposals:
        report['person_proposals_sha256'] = digest(args.person_proposals)
        report['person_proposal_source'] = str(args.person_proposals)
        report['crop_size_minimum'] = args.crop_size
        report['crop_image_size'] = 640
        report['crop_location_policy'] = 'Detector boxes only; ground truth used only for post-inference scoring'
        report['roi_hold_seconds'] = args.roi_hold_seconds
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

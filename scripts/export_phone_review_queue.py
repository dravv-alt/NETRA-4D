"""Export phone proposals on frames without Phone labels as NEEDS_REVIEW, never negatives."""
import argparse
import csv
import json
from pathlib import Path

from evaluate_cvat_boxes import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--labels',type=Path,default=Path('outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl'))
    args = parser.parse_args()
    output = args.run/'phone_proposals_needs_review.jsonl'
    samples = args.run/'phone_review_sample.csv'
    if output.exists() or samples.exists():
        raise FileExistsError('Review export already exists')
    labelled_frames = {r['frame_index'] for r in load_rows(args.labels) if r['class']=='Phone' and not r['outside']}
    rows = []
    for frame in load_rows(args.run/'detections.jsonl'):
        if frame['frame_index'] in labelled_frames:
            continue
        for obj in frame['objects']:
            if obj['object_class'] != 'MOBILE_PHONE':
                continue
            rows.append({'frame_index':frame['frame_index'],'timestamp_sec':frame['timestamp_sec'],'bbox_xyxy_px':obj['bbox_xyxy_px'],'raw_confidence':obj['confidence'],'inference_metadata':obj['metadata'],'review_status':'NEEDS_REVIEW','reason':'Frame has no visible Phone box annotation: UNKNOWN, not a false positive or negative label','evidence_video':str(args.run/'annotated.mp4'),'model_provenance':frame['model_provenance']})
    with output.open('w',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    by_window = {}
    for row in rows:
        key = int(row['timestamp_sec']//3)
        if key not in by_window or row['raw_confidence'] > by_window[key]['raw_confidence']:
            by_window[key] = row
    with samples.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['frame_index','timestamp_sec','raw_confidence','bbox_xyxy_px','reviewer_decision','notes'])
        writer.writeheader()
        for row in sorted(by_window.values(),key=lambda r:r['frame_index']):
            writer.writerow({k:row[k] for k in ('frame_index','timestamp_sec','raw_confidence','bbox_xyxy_px')})
    print(json.dumps({'proposals_needs_review':len(rows),'frames_with_unknown_phone_proposals':len({r['frame_index'] for r in rows}),'representative_review_samples':len(by_window),'negative_examples_created':0}))


if __name__ == '__main__':
    main()

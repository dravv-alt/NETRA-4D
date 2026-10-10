"""Diagnose visible-phone misses using detector-only ROIs; does not alter labels or run settings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from build_video03_human_review import digest,read_rows,write_jsonl,write_json,XML_HASH
from audit_cvat_video import iou
from drishti.providers.phone_crop import PersonCropMemory,detect_phone_candidates


def coverage(box,region):
    intersection=max(0,min(box[2],region[2])-max(box[0],region[0]))*max(0,min(box[3],region[3])-max(box[1],region[1]))
    return intersection/max(1e-9,(box[2]-box[0])*(box[3]-box[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    args=parser.parse_args()
    package=args.package.resolve()
    run=package.parent
    output=package/'phone_miss_probe.jsonl'
    report=package/'PHONE_MISS_DIAGNOSTICS.md'
    if output.exists() or report.exists():
        raise FileExistsError('Phone miss diagnostics already exist')
    failures=[r for r in read_rows(package/'failure_frames.jsonl') if r['class']=='Phone']
    manifest=json.loads((run/'COMPLETE.json').read_text())
    record=manifest['models']['phone_candidate']
    raw_xml=ROOT/'MANUAL_ANNOTATIONS(CVAT)/VIDEO_3_EXAM_HALL_PHONE_USE/annotations.xml'
    before={name:digest(run/name) for name in ('detections.jsonl','phone_proposals_needs_review.jsonl','phone_review_sample.csv')}
    if digest(raw_xml)!=XML_HASH or digest(record['path'])!=record['sha256'] or digest(manifest['source'])!=manifest['source_sha256']:
        raise ValueError('Source/checkpoint hash mismatch')
    frames={r['frame_index']:r for r in read_rows(run/'detections.jsonl')}
    target_frames={r['frame_index'] for r in failures}
    memory=PersonCropMemory(1.)
    regions={}
    for index in range(2200):
        proposals=memory.regions(frames[index]['people'],1280,720,index/25)
        if index in target_frames: regions[index]=proposals
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=YOLO(record['path'])
    cap=cv2.VideoCapture(manifest['source'])
    rows=[]
    try:
        for failure in failures:
            index=failure['frame_index']
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,image=cap.read()
            if not ok: raise RuntimeError(f'Cannot read frame {index}')
            # Lower threshold for diagnostics ONLY. Production conf .25 is untouched.
            predicted=detect_phone_candidates(model,image,[],device=device,confidence=.05,regions_override=regions[index])
            matches=[p for p in predicted if iou(failure['bbox_xyxy_px'],p['bbox_xyxy_px'])>=.5]
            best=max(matches,key=lambda p:p['confidence'],default=None)
            fraction=max((coverage(failure['bbox_xyxy_px'],p['crop_xyxy']) for p in regions[index]),default=0.)
            row={'frame_index':index,'track_id':failure['track_id'],'manual_bbox_xyxy_px':failure['bbox_xyxy_px'],
                'max_detector_crop_coverage_fraction':fraction,'detector_crop_regions':regions[index],
                'best_matching_prediction_at_diagnostic_confidence_0_05':best,
                'raw_candidates':predicted,'model_sha256':record['sha256'],
                'diagnostic_confidence':.05,'production_confidence_unchanged':.25,
                'crops_source':'verified detector observations and one-second ROI memory, never CVAT geometry',
                'status':'REQUIRES_HUMAN_REVIEW','ground_truth_modified':False}
            row['hypothesis']=('CROP_COVERAGE_GAP' if fraction<.95 else
                'SUBTHRESHOLD_MATCH' if best and best['confidence']<.25 else
                'NON_REPRODUCIBLE_RETAINED_MISS' if best else 'NO_MATCH_EVEN_AT_DIAGNOSTIC_0_05')
            rows.append(row)
    finally:
        cap.release()
    summary={'visible_phone_misses_probed':len(rows),
        'hypotheses':dict(Counter(r['hypothesis'] for r in rows)),
        'late_segment_hypotheses':dict(Counter(r['hypothesis'] for r in rows if 1664<=r['frame_index']<=1739)),
        'scope':'diagnostic hypotheses, not confirmed annotation errors or behavioural-event metrics',
        'training_performed':False,'annotation_changes_applied':0}
    write_jsonl(output,rows)
    write_json(package/'phone_miss_probe_summary.json',summary)
    text=['# Visible-phone miss diagnostics','',json.dumps(summary,indent=2),'',
        'The production threshold and inference artifacts are unchanged. The .05 probe is diagnostic only; no proposal was labelled, and no threshold recommendation or new quality claim is made.',
        'ROIs are reconstructed from verified person detections and one-second crop memory. Manual phone boxes are used solely for post-inference coverage/IoU measurements, never to generate crops.',
        'Partial phone visibility at frame/image borders and geometry/occlusion remain human review questions.', '',
        '| Frame | CVAT visibility track | Maximum crop coverage | Best match confidence at IoU >= .5 | Hypothesis | Example |',
        '| --- | --- | ---: | ---: | --- | --- |']
    for row in rows:
        best=row['best_matching_prediction_at_diagnostic_confidence_0_05']
        score=f"{best['confidence']:.4f}" if best else 'none'
        index=row['frame_index']
        text.append(f"| {index} | {row['track_id']} | {row['max_detector_crop_coverage_fraction']:.3f} | {score} | {row['hypothesis']} | [frame {index}](examples/frame_{index:04d}.jpg) |")
    report.write_text('\n'.join(text)+'\n',encoding='utf-8')
    if digest(raw_xml)!=XML_HASH or any(digest(run/name)!=value for name,value in before.items()):
        raise RuntimeError('Source integrity changed')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

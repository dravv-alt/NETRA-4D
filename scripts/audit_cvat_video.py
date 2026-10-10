"""Audit immutable CVAT tracks and export bounded, explicitly annotated evidence.

No extrapolation beyond exported boxes; missing annotations remain unknown.
Detection evaluation reports recall against labelled boxes, never invents negatives.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from drishti.data.video_policy import bounded_video_source


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda x: max(0, x[2] - x[0]) * max(0, x[3] - x[1])
    return overlap / max(1e-9, area(a) + area(b) - overlap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-frames', type=int, default=2200)
    parser.add_argument('--image-stride', type=int, default=25)
    args = parser.parse_args()
    source, args.max_frames = bounded_video_source(args.video, args.max_frames)
    args.video = Path(source)
    if args.max_frames <= 0 or args.image_stride <= 0:
        parser.error('frame count and image stride must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    original_hash = digest(args.annotations)
    root = ET.parse(args.annotations).getroot()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError('Cannot open video')
    width, height = (int(capture.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(args.video)], text=True))
    verified_total = int(probe['streams'][0]['nb_read_frames'])
    schema = {}
    for label in root.findall('./meta/job/labels/label'):
        schema[label.findtext('name')] = {a.findtext('name'): a.findtext('values', '').splitlines() for a in label.findall('./attributes/attribute')}
    rows = defaultdict(list)
    issues = []
    tracks = []
    distributions = defaultdict(Counter)
    keyframes = 0
    outside = 0
    seen_ids = set()
    all_frames = []
    for track in root.findall('track'):
        tid, label = int(track.get('id')), track.get('label')
        if tid in seen_ids:
            issues.append({'type': 'duplicate_track_id', 'track_id': tid})
        seen_ids.add(tid)
        boxes = sorted(track.findall('box'), key=lambda b: int(b.get('frame')))
        tracks.append({'track_id': tid, 'class': label, 'first': int(boxes[0].get('frame')), 'last': int(boxes[-1].get('frame')), 'explicit_boxes': len(boxes)})
        previous = None
        for box in boxes:
            frame = int(box.get('frame'))
            all_frames.append(frame)
            if frame >= args.max_frames:
                continue
            bounds = [float(box.get(k)) for k in ('xtl', 'ytl', 'xbr', 'ybr')]
            attrs = {a.get('name'): a.text for a in box.findall('attribute')}
            is_outside = box.get('outside') == '1'
            row = {'frame_index': frame, 'track_id': tid, 'class': label, 'bbox_xyxy_px': bounds, 'attributes': attrs, 'outside': is_outside, 'occluded': box.get('occluded') == '1', 'keyframe': box.get('keyframe') == '1', 'source': track.get('source'), 'annotation_status': 'HUMAN_REVIEWED_EXPORT'}
            keyframes += row['keyframe']
            outside += is_outside
            if not all(math.isfinite(x) for x in bounds) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                issues.append({'type': 'invalid_box', 'frame': frame, 'track_id': tid})
                row['geometry_valid'] = False
            else:
                row['geometry_valid'] = True
            if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > width or bounds[3] > height:
                issues.append({'type': 'out_of_bounds', 'frame': frame, 'track_id': tid})
                row['geometry_valid'] = False
            for name, values in schema.get(label, {}).items():
                if name not in attrs or attrs[name] not in values:
                    issues.append({'type': 'missing_or_invalid_attribute', 'frame': frame, 'track_id': tid, 'attribute': name, 'value': attrs.get(name)})
            if not is_outside:
                for name, value in attrs.items():
                    distributions[name][value] += 1
            if previous:
                pf, pb, po = previous
                if frame <= pf:
                    issues.append({'type': 'duplicate_frame', 'frame': frame, 'track_id': tid})
                elif frame > pf + 1:
                    issues.append({'type': 'explicit_box_gap', 'start': pf + 1, 'end': frame - 1, 'track_id': tid, 'previous_box_outside': po, 'interpretation': 'expected outside interval' if po else 'review required; no interpolation applied'})
                if not is_outside and not po and frame == pf + 1:
                    distance = math.hypot((bounds[0] + bounds[2] - pb[0] - pb[2]) / 2, (bounds[1] + bounds[3] - pb[1] - pb[3]) / 2)
                    size = max(1, math.hypot(pb[2] - pb[0], pb[3] - pb[1]))
                    ratio = ((bounds[2]-bounds[0]) * (bounds[3]-bounds[1])) / max(1e-9, (pb[2]-pb[0]) * (pb[3]-pb[1]))
                    if distance / size > .5 or ratio > 2 or ratio < .5:
                        issues.append({'type': 'jump_or_size_change_proposal', 'frame': frame, 'track_id': tid})
            previous = (frame, bounds, is_outside)
            rows[frame].append(row)
    duplicate_candidates = []
    for frame, objects in rows.items():
        active = [r for r in objects if not r['outside']]
        for i, a in enumerate(active):
            for b in active[i+1:]:
                if a['class'] == b['class'] and iou(a['bbox_xyxy_px'], b['bbox_xyxy_px']) > .9:
                    duplicate_candidates.append({'frame': frame, 'tracks': [a['track_id'], b['track_id']]})
    report = {'video': str(args.video), 'video_sha256': digest(args.video), 'annotations': str(args.annotations), 'annotation_sha256': original_hash, 'format': 'CVAT XML '+root.findtext('version'), 'frame_numbering': 'zero-based; bound is exclusive', 'total_frames_metadata': total, 'fps': fps, 'resolution': [width,height], 'processed_range': [0,args.max_frames-1], 'requested_trusted_range_inclusive': [0,2200], 'actual_export_range': [min(all_frames),max(all_frames)], 'unknown_tail': [max(all_frames)+1,total-1], 'frames_with_explicit_boxes': len(rows), 'frames_with_visible_boxes': sum(any(not r['outside'] for r in rs) for rs in rows.values()), 'track_counts': dict(Counter(t['class'] for t in tracks)), 'tracks': tracks, 'keyframes': keyframes, 'outside_markers': outside, 'attribute_schema': schema, 'attribute_distributions_visible_boxes': {k:dict(v) for k,v in distributions.items()}, 'issues': issues, 'duplicate_track_proposals': duplicate_candidates, 'identity_switches': 'not established automatically; geometry checks are review proposals', 'negative_annotation_policy': 'No annotation does not mean absent. No track extrapolation.', 'evaluation_scope': 'Single-session diagnostic, not held-out product validation'}
    report['total_frames_verified'] = verified_total
    report['unknown_tail'] = [max(all_frames)+1, verified_total-1]
    report['visible_box_counts'] = dict(Counter(r['class'] for rs in rows.values() for r in rs if not r['outside']))
    with (args.output/'trusted_boxes.jsonl').open('w',encoding='utf-8') as stream:
        for frame in sorted(rows):
            for row in rows[frame]:
                stream.write(json.dumps(row)+'\n')
    writer = cv2.VideoWriter(str(args.output/'trusted_overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    if not writer.isOpened():
        raise RuntimeError('Cannot create overlay')
    images_dir = args.output/'images'
    images_dir.mkdir()
    categories = [{'id':i+1,'name':name} for i,name in enumerate(schema)]
    category_ids = {c['name']:c['id'] for c in categories}
    coco = {'info': {'description':'Explicit CVAT boxes only; missing objects UNKNOWN; attributes and track IDs retained'}, 'images': [], 'annotations': [], 'categories': categories}
    decoded = 0
    try:
        for frame_index in range(args.max_frames):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'Video ended at {frame_index}, before requested bound')
            active = [r for r in rows.get(frame_index,[]) if not r['outside'] and r['geometry_valid']]
            if frame_index % args.image_stride == 0 and active:
                filename = f'frame_{frame_index:06d}.jpg'
                if not cv2.imwrite(str(images_dir/filename),image):
                    raise RuntimeError('Image write failed')
                coco['images'].append({'id':frame_index,'file_name':'images/'+filename,'width':width,'height':height,'frame_index':frame_index,'timestamp_sec':frame_index/fps,'annotation_completeness':'UNKNOWN'})
                for row in active:
                    x,y,x2,y2 = row['bbox_xyxy_px']
                    coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':frame_index,'category_id':category_ids[row['class']],'bbox':[x,y,x2-x,y2-y],'area':(x2-x)*(y2-y),'iscrowd':0,'track_id':row['track_id'],'attributes':row['attributes'],'occluded':row['occluded']})
            for row in active:
                x,y,x2,y2 = map(round,row['bbox_xyxy_px'])
                color = (0,220,255) if row['class']=='Phone' else (0,220,0)
                cv2.rectangle(image,(x,y),(x2,y2),color,2)
                cv2.putText(image,f"{row['class']} #{row['track_id']}",(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            cv2.putText(image,f"Frame {frame_index} | explicit CVAT boxes: {len(active)} | missing labels UNKNOWN",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            writer.write(image)
            decoded += 1
    finally:
        capture.release()
        writer.release()
    report['decoded_frames'] = decoded
    (args.output/'coco.json').write_text(json.dumps(coco,indent=2),encoding='utf-8')
    if digest(args.annotations) != original_hash:
        raise RuntimeError('Raw annotations changed during processing')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('issues','attribute_schema','attribute_distributions_visible_boxes','duplicate_track_proposals')},indent=2))


if __name__ == '__main__':
    main()

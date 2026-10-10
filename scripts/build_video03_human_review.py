"""Immutable-source review package; no automatic labels, annotation edits or training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_cvat_video import iou

XML_HASH = '301620001d90d0f2fb06deb3535e98dc67c6c641f7a10e84b5b01c2db1452382'
STATUSES = ('UNKNOWN', 'CONFIRMED_PHONE', 'NOT_PHONE', 'AMBIGUOUS')


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k)) if isinstance(row.get(k), (dict,list)) else row.get(k) for k in fields})


def enrich_proposals(export, predictions, export_hash):
    """Exact joins only; never guess a missing association or model fingerprint."""
    enriched = []
    used = set()
    for index, row in enumerate(export):
        frame = predictions[row['frame_index']]
        candidates = [(ordinal,obj) for ordinal,obj in enumerate(frame['objects'])
            if obj['object_class'] == 'MOBILE_PHONE'
            and obj['bbox_xyxy_px'] == row['bbox_xyxy_px']
            and obj['confidence'] == row['raw_confidence']
            and obj['metadata'] == row['inference_metadata']
            and (row['frame_index'],ordinal) not in used]
        if len(candidates) != 1:
            raise ValueError(f'Non-unique exact provenance join at export row {index}: {len(candidates)}')
        ordinal, obj = candidates[0]
        used.add((row['frame_index'],ordinal))
        weight = obj['metadata']['model_weight']
        matching_models = [m for m in frame['model_provenance'].values() if m['path'] == weight]
        if len(matching_models) != 1:
            raise ValueError(f'Missing/ambiguous model provenance for {weight}')
        model_hash = matching_models[0]['sha256']
        proposal_id = 'proposal_' + sha256(f'{export_hash}:{index}:{row["frame_index"]}:{ordinal}:{model_hash}'.encode()).hexdigest()[:24]
        enriched.append({**row, 'proposal_id':proposal_id, 'source_export_row':index,
            'source_detection_object_index':ordinal, 'review_status':'UNKNOWN',
            'source_review_status':row['review_status'], 'model_sha256':model_hash,
            'associated_entity_id':obj['associated_entity_id'],
            'associated_seat_id':obj['associated_seat_id'],
            'association_confidence':obj['association_confidence'],
            'association_method':obj['metadata'].get('association_method'),
            'raw_detection':obj})
    return enriched


def spatial_link(a, b, max_gap=10):
    """Conservative apparent-object grouping, not a physical identity classifier."""
    gap = b['frame_index'] - a['frame_index']
    if not 1 <= gap <= max_gap:
        return None
    ba, bb = a['bbox_xyxy_px'], b['bbox_xyxy_px']
    wa,ha = ba[2]-ba[0],ba[3]-ba[1]
    wb,hb = bb[2]-bb[0],bb[3]-bb[1]
    if min(wa,ha,wb,hb) <= 0:
        return None
    ratio = wb*hb/(wa*ha)
    if not .35 <= ratio <= 2.85:
        return None
    distance = math.hypot((ba[0]+ba[2]-bb[0]-bb[2])/2, (ba[1]+ba[3]-bb[1]-bb[3])/2)
    overlap = iou(ba,bb)
    if overlap < .2 and distance > min(80., .6*max(math.hypot(wa,ha),math.hypot(wb,hb))):
        return None
    # Association is soft evidence only: person tracker IDs can change mid-segment.
    penalty = .15 if a.get('associated_entity_id') != b.get('associated_entity_id') else 0.
    return 1-overlap + distance/100 + gap*.01 + penalty


def temporal_clusters(rows, max_gap=10):
    clusters = []
    frames = defaultdict(list)
    for row in rows:
        frames[row['frame_index']].append(row)
    for frame_index in sorted(frames):
        current = sorted(frames[frame_index], key=lambda r:r['proposal_id'])
        edges = []
        for ci, cluster in enumerate(clusters):
            last = cluster[-1]
            if frame_index-last['frame_index'] > max_gap:
                continue
            for ri,row in enumerate(current):
                score = spatial_link(last,row,max_gap)
                if score is not None:
                    edges.append((score,ci,ri))
        assigned_clusters, assigned_rows = set(), set()
        for _,ci,ri in sorted(edges):
            if ci not in assigned_clusters and ri not in assigned_rows:
                clusters[ci].append(current[ri])
                assigned_clusters.add(ci)
                assigned_rows.add(ri)
        clusters.extend([[row] for ri,row in enumerate(current) if ri not in assigned_rows])
    return clusters


def quantiles(values):
    import numpy as np
    return {k:float(v) for k,v in zip(('min','p25','median','p75','max'), np.quantile(values,[0,.25,.5,.75,1]))}


def match_rows(labels, candidates):
    """Same confidence-ordered, one-to-one IoU >= .5 protocol as verified metrics."""
    matched = {}
    used = set()
    for candidate in sorted(candidates, key=lambda p:p.get('detection_confidence',p.get('confidence',0)), reverse=True):
        best = max(((iou(candidate['bbox_xyxy_px'],r['bbox_xyxy_px']),i) for i,r in enumerate(labels) if i not in used),default=(0,-1))
        if best[0] >= .5:
            used.add(best[1])
            matched[best[1]] = (candidate,best[0])
    return matched


def diagnose(labels, predictions, reasoning):
    by_frame = defaultdict(list)
    for row in labels:
        if 0 <= row['frame_index'] < 2200 and not row['outside'] and row['geometry_valid']:
            by_frame[row['frame_index']].append(row)
    failures, matches = [], []
    for frame_index, truth in sorted(by_frame.items()):
        frame = predictions[frame_index]
        for family in ('Person','Phone'):
            subset = [r for r in truth if (r['class'] in ('Student','Invigilator') if family=='Person' else r['class']=='Phone')]
            candidates = frame['people'] if family=='Person' else [o for o in frame['objects'] if o['object_class']=='MOBILE_PHONE']
            assigned = match_rows(subset,candidates)
            for index,row in enumerate(subset):
                best_candidate = max(candidates,key=lambda p:iou(p['bbox_xyxy_px'],row['bbox_xyxy_px']),default=None)
                best_iou = iou(best_candidate['bbox_xyxy_px'],row['bbox_xyxy_px']) if best_candidate else 0.
                diagnostic = {'frame_index':frame_index,'timestamp_sec':frame_index/25,
                    'track_id':row['track_id'],'class':row['class'],'bbox_xyxy_px':row['bbox_xyxy_px'],
                    'attributes':row.get('attributes',{}),'occluded':row.get('occluded'),
                    'best_iou':best_iou,'best_detection':best_candidate,
                    'annotation_change_applied':False}
                if index in assigned:
                    detection,overlap = assigned[index]
                    diagnostic.update(matched_detection=detection,matched_iou=overlap)
                    if family == 'Person':
                        tracker = detection['entity_local_id']
                        diagnostic['candidate_ids'] = [c['candidate_id'] for c in reasoning.get(frame_index,{}).get('candidates',[]) if c['tracker_id']==tracker]
                    matches.append(diagnostic)
                else:
                    diagnostic['diagnostic_category'] = ('ASSIGNMENT_COMPETITION' if best_iou>=.5 else
                        'LOCALIZATION_OR_GEOMETRY_REVIEW' if best_iou>=.1 else 'NO_OVERLAPPING_RETAINED_DETECTION')
                    diagnostic['confirmed_cause'] = None
                    diagnostic['requires_human_review'] = True
                    failures.append(diagnostic)
    return failures,matches


def contiguous_ranges(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for key,group in sorted(grouped.items()):
        segments = []
        for row in sorted(group,key=lambda r:r['frame_index']):
            if not segments or row['frame_index'] > segments[-1][-1]['frame_index']+1:
                segments.append([])
            segments[-1].append(row)
        for segment in segments:
            output.append({**dict(zip(keys,key)), 'start_frame':segment[0]['frame_index'],
                'end_frame':segment[-1]['frame_index'],'frame_count':len(segment),
                'representative_frame':segment[len(segment)//2]['frame_index'],
                'examples':[segment[0]['frame_index'],segment[len(segment)//2]['frame_index'],segment[-1]['frame_index']]})
    return output


def probe_pose(source, frames, model_record, predictions, labels_by_frame):
    """Selected-frame raw predictor diagnostics, never retraining or threshold changes."""
    import cv2
    import torch
    from ultralytics import YOLO
    if digest(model_record['path']) != model_record['sha256']:
        raise ValueError('Pose checkpoint differs from verified run')
    model = YOLO(model_record['path'])
    torch.set_num_threads(4)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cap = cv2.VideoCapture(str(source))
    result = []
    try:
        for frame_index in sorted(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
            ok,frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read diagnostic frame {frame_index}')
            options = {'conf':.25,'iou':.5,'verbose':False,'imgsz':640}
            if device.startswith('cuda'):
                options['device'] = device
            predicted = model.predict(frame,**options)[0]
            boxes = [{'bbox_xyxy_px':box.xyxy[0].tolist(),'confidence':float(box.conf.item())} for box in predicted.boxes]
            for label in labels_by_frame[frame_index]:
                if label['class'] not in ('Student','Invigilator') or label['outside']:
                    continue
                before = max((iou(label['bbox_xyxy_px'],p['bbox_xyxy_px']) for p in boxes),default=0.)
                after = max((iou(label['bbox_xyxy_px'],p['bbox_xyxy_px']) for p in predictions[frame_index]['people']),default=0.)
                result.append({'frame_index':frame_index,'track_id':label['track_id'],
                    'raw_predict_best_iou':before,'retained_track_best_iou':after,
                    'interpretation': 'RAW_BOX_PRESENT_BUT_NOT_RETAINED: tracker filtering/association suspect; no confirmed cause' if before>=.5 and after<.5 else
                    'RAW_PREDICTOR_MISS_OR_LOCALIZATION: inspect occlusion and annotation extent' if before<.5 else 'BOX_MATCH_PRESENT',
                    'raw_boxes':boxes,'diagnostic_only':True,'imgsz':640,'confidence':.25,
                    'model_sha256':model_record['sha256']})
    finally:
        cap.release()
    return result


def build(args):
    run = args.run.resolve()
    output = args.output.resolve()
    xml = ROOT/'MANUAL_ANNOTATIONS(CVAT)/VIDEO_3_EXAM_HALL_PHONE_USE/annotations.xml'
    if digest(xml) != XML_HASH:
        raise ValueError('Raw CVAT hash mismatch; stopping without modifications')
    files = {name:run/name for name in ('phone_proposals_needs_review.jsonl','phone_review_sample.csv','detections.jsonl','unified_reasoning.jsonl','COMPLETE.json','annotated.mp4','metrics.json')}
    files['trusted_boxes.jsonl'] = args.labels.resolve()
    initial_hashes = {name:digest(path) for name,path in files.items()}
    manifest = json.loads(files['COMPLETE.json'].read_text())
    source = Path(manifest['source'])
    if digest(source) != manifest['source_sha256'] or manifest['frames'] != 2200:
        raise ValueError('Canonical source/run mismatch')
    predictions = {r['frame_index']:r for r in read_rows(files['detections.jsonl'])}
    reasoning = {r['frame_index']:r for r in read_rows(files['unified_reasoning.jsonl'])}
    if set(predictions) != set(range(2200)) or set(reasoning) != set(range(2200)):
        raise ValueError('Incomplete frame coverage')
    labels = read_rows(args.labels)
    labelled_phone_frames = {r['frame_index'] for r in labels if r['class']=='Phone' and not r['outside']}
    original = read_rows(files['phone_proposals_needs_review.jsonl'])
    if len(original) != 849 or len({r['frame_index'] for r in original}) != 846:
        raise ValueError('Unknown queue differs from expected verified export')
    if any(r['frame_index'] in labelled_phone_frames for r in original):
        raise ValueError('Unknown export contains visible phone labelled frames')
    proposals = enrich_proposals(original,predictions,initial_hashes['phone_proposals_needs_review.jsonl'])
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite review package: {output}')
    output.mkdir(parents=True)
    segments = []
    for group in temporal_clusters(proposals,args.max_gap):
        sid = 'segment_'+sha256(group[0]['proposal_id'].encode()).hexdigest()[:16]
        confidences = [p['raw_confidence'] for p in group]
        unassociated = sum(p['associated_entity_id'] is None for p in group)
        ttl = sum((p['inference_metadata'].get('crop_proposal_age_sec') or 0)>0 for p in group)
        associations = Counter(str(p['associated_entity_id']) for p in group)
        boundary = any(min(abs(p['frame_index']-f) for f in labelled_phone_frames)<=25 for p in (group[0],group[-1]))
        flags = []
        if len(group)>=25: flags.append('persistent_unknown_object')
        if unassociated: flags.append('unassociated_proposals')
        if len(associations)>1: flags.append('association_changes')
        if ttl: flags.append('retained_crop_proposals')
        if boundary: flags.append('near_manual_phone_visibility_boundary')
        # Review workload/ambiguity heuristic, not a model or risk probability.
        priority = 100*(.5*min(1,len(group)/100)+.2*unassociated/len(group)+.15*ttl/len(group)+.15*int(boundary))
        summary = {'segment_id':sid,'start_frame':group[0]['frame_index'],'end_frame':group[-1]['frame_index'],
            'start_sec':group[0]['timestamp_sec'],'end_sec':group[-1]['timestamp_sec'],
            'proposal_count':len(group),'proposal_ids':[p['proposal_id'] for p in group],
            'review_status':'UNKNOWN','confidence_distribution':quantiles(confidences),
            'associated_entity_distribution':dict(associations),'flags':flags,
            'review_priority':round(priority,3),'priority_semantics':'workload/context heuristic; not probability',
            'physical_identity':'unverified apparent-object segment',
            'representative_proposal_ids':list(dict.fromkeys([group[0]['proposal_id'],group[len(group)//2]['proposal_id'],
                max(group,key=lambda p:p['raw_confidence'])['proposal_id'],group[-1]['proposal_id']]))}
        for p in group:
            p['segment_id'] = sid
        segments.append(summary)
    segments.sort(key=lambda s:(-s['review_priority'],s['start_frame']))
    failures,matches = diagnose(labels,predictions,reasoning)
    ranges = contiguous_ranges(failures,('class','track_id','diagnostic_category'))
    track_diagnostics = []
    for track_id in (0,1,2):
        good = sorted((m for m in matches if m['track_id']==track_id and m['class'] in ('Student','Invigilator')),key=lambda r:r['frame_index'])
        changes = [{'frame_index':b['frame_index'],'previous_frame':a['frame_index'],
            'old_tracker_id':a['matched_detection']['entity_local_id'],'new_tracker_id':b['matched_detection']['entity_local_id']}
            for a,b in zip(good,good[1:]) if a['matched_detection']['entity_local_id'] != b['matched_detection']['entity_local_id']]
        track_diagnostics.append({'track_id':track_id,'matched_boxes':len(good),
            'unique_tracker_ids':sorted({m['matched_detection']['entity_local_id'] for m in good}),
            'tracker_id_changes':changes,'matched_frames_missing_candidate_mapping':[m['frame_index'] for m in good if not m['candidate_ids']],
            'causality_note':'Identity changes do not explain geometric box recall; association quality is a separate unvalidated task'})
    selected_ranges = sorted((r for r in ranges if r['class']!='Phone'),key=lambda r:-r['frame_count'])[:20]
    example_frames = {f for r in selected_ranges for f in r['examples']}
    example_frames.update(r['frame_index'] for r in failures if r['class']=='Phone')
    proposal_map = {p['proposal_id']:p for p in proposals}
    for segment in segments:
        example_frames.update(proposal_map[p]['frame_index'] for p in segment['representative_proposal_ids'])
    labels_by_frame = defaultdict(list)
    for r in labels: labels_by_frame[r['frame_index']].append(r)
    pose_samples = sorted({f for r in selected_ranges for f in r['examples']})
    probes = probe_pose(source,pose_samples,manifest['models']['pose'],predictions,labels_by_frame) if args.probe_pose else []
    write_jsonl(output/'proposals.jsonl',proposals)
    write_jsonl(output/'segments.jsonl',segments)
    write_jsonl(output/'failure_frames.jsonl',failures)
    write_jsonl(output/'pose_probe.jsonl',probes)
    write_json(output/'tracking_diagnostics.json',track_diagnostics)
    write_csv(output/'segments_to_review.csv',segments,('segment_id','review_priority','start_frame','end_frame','proposal_count','flags','review_status'))
    write_csv(output/'failure_ranges.csv',sorted(ranges,key=lambda r:(r['track_id'],r['start_frame'])),('class','track_id','diagnostic_category','start_frame','end_frame','frame_count','representative_frame','examples'))
    write_csv(output/'proposal_decision_template.csv',proposals,('proposal_id','segment_id','frame_index','timestamp_sec','bbox_xyxy_px','raw_confidence','associated_entity_id','association_confidence','model_sha256','review_status'))
    with files['phone_review_sample.csv'].open(encoding='utf-8',newline='') as stream:
        original_samples = list(csv.DictReader(stream))
    mapped_samples = []
    for row in original_samples:
        candidates = [p for p in proposals if p['frame_index']==int(row['frame_index']) and p['bbox_xyxy_px']==json.loads(row['bbox_xyxy_px']) and abs(p['raw_confidence']-float(row['raw_confidence']))<1e-12]
        if len(candidates)!=1: raise ValueError('CSV sample cannot be mapped exactly')
        p = candidates[0]
        mapped_samples.append({**row,'proposal_id':p['proposal_id'],'segment_id':p['segment_id'],'review_status':'UNKNOWN'})
    write_csv(output/'original_samples_mapped.csv',mapped_samples,tuple(mapped_samples[0]))
    import cv2
    cap = cv2.VideoCapture(str(files['annotated.mp4']))
    images = output/'examples'
    images.mkdir()
    try:
        for index in sorted(example_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,image = cap.read()
            if not ok: raise RuntimeError(f'Cannot extract example {index}')
            for r in labels_by_frame[index]:
                if not r['outside']:
                    x1,y1,x2,y2 = map(round,r['bbox_xyxy_px'])
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(image,f"MANUAL {r['class']} track {r['track_id']}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
            cv2.putText(image,f'Frame {index} | GREEN = unchanged manual boxes | other overlays = existing inference',(10,45),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            if not cv2.imwrite(str(images/f'frame_{index:04d}.jpg'),image):
                raise RuntimeError('Cannot save review example')
    finally:
        cap.release()
    if not args.skip_preview:
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(files['annotated.mp4']),'-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'preview.mp4')],check=True)
    bins = {name:sum(lo<=p['raw_confidence']<hi for p in proposals) for name,lo,hi in
        [('[.25,.40)',.25,.4),('[.40,.60)',.4,.6),('[.60,.80)',.6,.8),('[.80,1.00]',.8,1.000001)]}
    summary = {'unknown_proposals':len(proposals),'unknown_frames':len({p['frame_index'] for p in proposals}),
        'temporal_clusters':len(segments),'cluster_gap_frames':args.max_gap,'confidence_distribution':quantiles([p['raw_confidence'] for p in proposals]),
        'confidence_bins':bins,'person_association_distribution':dict(Counter(str(p['associated_entity_id']) for p in proposals)),
        'association_method_distribution':dict(Counter(p['association_method'] for p in proposals)),
        'inference_view_distribution':dict(Counter(p['inference_metadata']['inference_view'] for p in proposals)),
        'failure_counts':dict(Counter(f"{r['class']}_track_{r['track_id']}:{r['diagnostic_category']}" for r in failures)),
        'pose_probe_interpretations':dict(Counter(p['interpretation'] for p in probes)),
        'review_states':{s:len(proposals) if s=='UNKNOWN' else 0 for s in STATUSES},
        'annotation_changes_applied':0,'training_performed':False,'source_hashes':initial_hashes,'raw_cvat_sha256':XML_HASH,
        'examples':sorted(example_frames),'geometry_cause_confirmed':False,
        'scope':'Single-video review diagnostics; no behavioural accuracy or precision/F1/mAP/FP-hour claims'}
    write_json(output/'summary.json',summary)
    data = {'summary':summary,'segments':segments,'proposals':proposals,'failure_ranges':ranges,
        'failure_frames':failures,'tracking_diagnostics':track_diagnostics,'pose_probes':probes}
    write_json(output/'review_data.json',data)
    template = ROOT/'scripts/templates/video03_review.html'
    (output/'index.html').write_text(template.read_text(encoding='utf-8'),encoding='utf-8')
    render_report(output,summary,segments,ranges,failures,track_diagnostics,probes)
    if digest(xml)!=XML_HASH or any(digest(path)!=initial_hashes[name] for name,path in files.items()):
        raise RuntimeError('Source immutability check failed')
    write_json(output/'READY_FOR_HUMAN_REVIEW.json',{'status':'WAITING_FOR_HUMAN_REVIEW','raw_cvat_sha256':XML_HASH,
        'review_data_sha256':digest(output/'review_data.json'),'source_hashes':initial_hashes,
        'all_proposals_initial_status':'UNKNOWN','training_authorized':False})
    print(json.dumps(summary,indent=2))


def render_report(output,summary,segments,ranges,failures,tracking,probes):
    lines = ['# Video 3 human review report','',
        'Status: **WAITING FOR HUMAN REVIEW**. No CVAT edits, generated negative labels, or training.', '',
        f"Unknown proposals: **{summary['unknown_proposals']}** on **{summary['unknown_frames']} frames**; apparent-object temporal clusters: **{len(segments)}**.",
        'All proposal states start UNKNOWN. Manual choices: UNKNOWN, CONFIRMED_PHONE, NOT_PHONE, AMBIGUOUS.', '',
        '## Review workflow','',
        'From the repository root, run:', '',
        f'```powershell\npython scripts/serve_video03_human_review.py --package "{output.relative_to(ROOT).as_posix()}"\n```','',
        'Open http://127.0.0.1:8765. Review segments, scrub the video and inspect every proposal. Enter a reviewer ID; decisions are explicit clicks and append-only. Segment-wide decisions require an explicit acknowledgement that all constituent proposals were reviewed. Mixed segments must be reviewed per proposal.',
        'The 22 old CSV samples are mapped in `original_samples_mapped.csv`; they are a convenience sample, not exhaustive coverage.',
        'No export is authorized for training. The decision ledger does not modify source files or automatically create labels.', '',
        '## Grouping and prioritization','',
        'Adjacent proposals link at gaps <= 10 frames (0.4 seconds), IoU >= .2 or close centre motion, bounded area change, and one proposal per frame per segment. Tracker association is a soft cue, not an identity constraint. No gaps are interpolated; every retained row is an original proposal. Clusters describe apparent objects, not proven physical identity.',
        'Review priority is an explicit workload/context heuristic: proposal coverage, missing associations, retained-crop use, and proximity to manually annotated phone visibility boundaries. It is not a confidence, risk or behavioural probability.', '',
        '## Confidence and association distributions','',
        f"Raw confidence quantiles: `{json.dumps(summary['confidence_distribution'])}`.",
        f"Bins: `{json.dumps(summary['confidence_bins'])}`.",
        f"Person association IDs (detector-local, not CVAT student IDs): `{json.dumps(summary['person_association_distribution'])}`.",
        f"Association methods: `{json.dumps(summary['association_method_distribution'])}`.",
        f"Crop views: `{json.dumps(summary['inference_view_distribution'])}`.", '',
        '## Prioritized phone segments','',
        '| Segment | Frames | Proposals | Priority | Flags |','| --- | --- | ---: | ---: | --- |']
    lines += [f"| {s['segment_id']} | {s['start_frame']}–{s['end_frame']} | {s['proposal_count']} | {s['review_priority']} | {', '.join(s['flags']) or 'isolated proposal'} |" for s in segments]
    lines += ['', 'All 849 frame numbers, boxes, IDs, confidence, person associations, crop provenance and model hashes are retained in `proposals.jsonl`. All ranges and examples are in `segments.jsonl` and `segments_to_review.csv`.', '',
        '## Person/student recall investigation','',
        'Confidence-ordered one-to-one IoU matching reproduces the verified box protocol. Identity IDs do not enter box matching: an association failure cannot by itself explain the 53.99%/85.44% geometric recalls.',
        'NO_OVERLAPPING_RETAINED_DETECTION means no retained box with IoU >= .1; it does not prove a raw detector miss. LOCALIZATION_OR_GEOMETRY_REVIEW means a nearby box exists but IoU < .5; the detector extent, visibility and manual annotation extent must be visually checked. Neither condition authorizes a ground-truth edit.', '',
        f"Failure counts: `{json.dumps(summary['failure_counts'])}`.", '',
        '| Manual person track | Matched boxes | Distinct detector IDs | ID changes across successive matched observations | Matched frames without candidate mapping |',
        '| --- | ---: | ---: | ---: | ---: |']
    lines += [f"| {t['track_id']} | {t['matched_boxes']} | {len(t['unique_tracker_ids'])} | {len(t['tracker_id_changes'])} | {len(t['matched_frames_missing_candidate_mapping'])} |" for t in tracking]
    lines += ['', 'These ID changes include gaps; they are fragmentation/reassociation indicators, not verified identity switches. Invigilator exclusion from candidate reasoning may be intentional. Full frame lists are in `tracking_diagnostics.json`.', '',
        '### Largest person failure ranges','', '| Track | Frames | Count | Diagnostic | Example |','| --- | --- | ---: | --- | --- |']
    top = sorted((r for r in ranges if r['class']!='Phone'),key=lambda r:-r['frame_count'])[:20]
    lines += [f"| {r['track_id']} | {r['start_frame']}–{r['end_frame']} | {r['frame_count']} | {r['diagnostic_category']} | [frame {r['representative_frame']}](examples/frame_{r['representative_frame']:04d}.jpg) |" for r in top]
    occlusion = Counter((r['track_id'],str(r.get('occluded')),str(r['attributes'].get('visibility','missing'))) for r in failures if r['class']!='Phone')
    lines += ['', f'Occlusion metadata on failed person frames: `{dict(occlusion)}`. These are existing annotator attributes, not independently verified occlusion causes.', '',
        '### Raw pose probe','',
        f"Selected representative frames rerun without tracking at the verified predictor settings (640, conf .25, IoU .5): `{json.dumps(summary['pose_probe_interpretations'])}`.",
        'A raw matching box absent from the retained run points to tracker filtering/reassociation as a hypothesis. A raw non-match points to detector localization/visibility, subject to geometry review. The sparse probe is not an exhaustive causal audit; full probes/boxes are preserved in `pose_probe.jsonl`.', '',
        '## Phone recall failure ranges','',
        'CVAT tracks 3 and 4 are visibility segments of the **same physical phone**, per the annotator; they are not two phones.', '',
        '| CVAT segment | Missed frame range | Count | Diagnostic | Example |','| --- | --- | ---: | --- | --- |']
    lines += [f"| {r['track_id']} | {r['start_frame']}–{r['end_frame']} | {r['frame_count']} | {r['diagnostic_category']} | [frame {r['representative_frame']}](examples/frame_{r['representative_frame']:04d}.jpg) |" for r in ranges if r['class']=='Phone']
    lines += ['', 'The 1664–1739 visibility window matches 58/76 frames, contributing 18/27 phone-box misses. Other windows contribute 6 (144–439), 3 (1290–1389), and 0 (1552–1579) misses. Each missed phone frame has an example image; inspect crop coverage and detector geometry before selecting a cause.', '',
        '## Recommendations for manual annotation review (not applied)','',
        '- Confirm object identity on each unknown segment; only an explicit reviewer decision may mark a proposal NOT_PHONE. Absence of a Phone box is never negative evidence.',
        '- A reviewed NOT_PHONE decision applies only to that proposal/object box, never to an entire frame or another object. UNKNOWN and AMBIGUOUS are not training labels.',
        '- If a reviewed object is a phone outside existing visibility windows, propose an additional visibility interval and attributes for annotator approval; keep the immutable XML unchanged.',
        '- For low-IoU person boxes, compare visible-body versus amodal/full-body conventions. Correct geometry only if the annotator confirms an actual error; otherwise fix the model/evaluation contract, not ground truth.',
        '- Inspect the listed person gaps for occlusion, merged boxes, localization and tracker filtering. Do not infer occlusion from a miss alone.',
        '- Review the late-phone examples for visibility, crop coverage, nearby people and annotation alignment. Propose outside/occlusion changes only if individually confirmed.',
        '- Treat frames beyond the last explicit CVAT visibility marker as unknown coverage unless manually certified; the 2,200-frame source is trusted, but annotation absence is not exhaustive negative coverage.', '',
        '## Integrity and limitations','',
        f'Raw XML SHA256: `{XML_HASH}`. Original export/CSV, labels, detections, source and verified video are hash checked and unmodified.',
        'All shown scores are raw detector or diagnostic measures. No behavioural-event accuracy is claimed. Precision, F1, mAP and false positives per hour are intentionally not reported. No training was started. Stop here for human review.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/unified_pipeline/video03_review_verified_20261003')
    parser.add_argument('--labels',type=Path,default=ROOT/'outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/unified_pipeline/video03_review_verified_20261003/human_review_v1')
    parser.add_argument('--max-gap',type=int,default=10)
    parser.add_argument('--probe-pose',action='store_true')
    parser.add_argument('--skip-preview',action='store_true')
    args = parser.parse_args()
    if not 1<=args.max_gap<=25: parser.error('max gap must be between 1 and 25 frames')
    build(args)

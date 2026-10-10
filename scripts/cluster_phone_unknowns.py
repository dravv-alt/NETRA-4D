"""Deterministically cluster UNKNOWN phone proposals; does not create labels."""
from __future__ import annotations
import argparse, json
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_video03_human_review import digest, enrich_proposals, quantiles, read_rows, temporal_clusters

KNOWN_WINDOWS=((144,439),(1290,1389),(1552,1579),(1664,1739))

def near_window(start,end,margin):
    return any(start<=b+margin and end>=a-margin for a,b in KNOWN_WINDOWS)

def clusters(run: Path, max_gap: int, spatial_margin: int):
    # spatial_margin is recorded for forward-compatible configuration; the reused
    # conservative spatial linker owns its explicit pixel/IoU thresholds.
    proposal_file=run/'phone_proposals_needs_review.jsonl'; detection_file=run/'detections.jsonl'; complete=run/'COMPLETE.json'
    if not all(p.exists() for p in (proposal_file,detection_file,complete)): raise FileNotFoundError('Expected frozen-run evidence files are missing')
    manifest=json.loads(complete.read_text(encoding='utf8'))
    if manifest.get('frames')!=2200: raise ValueError('Only the frozen 2,200-frame Video 3 run is supported')
    raw=read_rows(proposal_file)
    if len(raw)!=849: raise ValueError(f'Expected 849 UNKNOWN proposals, found {len(raw)}')
    predictions={r['frame_index']:r for r in read_rows(detection_file)}
    if set(predictions)!=set(range(2200)): raise ValueError('Detection coverage is incomplete')
    proposal_hash=digest(proposal_file); proposals=enrich_proposals(raw,predictions,proposal_hash)
    output=[]
    for group in temporal_clusters(proposals,max_gap=max_gap):
        start,end=group[0]['frame_index'],group[-1]['frame_index']; conf=[p['raw_confidence'] for p in group]
        pid='cluster_'+sha256(group[0]['proposal_id'].encode()).hexdigest()[:16]
        reps=[group[0],group[len(group)//2],max(group,key=lambda p:p['raw_confidence']),group[-1]]
        rep_ids=list(dict.fromkeys(p['proposal_id'] for p in reps))
        assoc=Counter(str(p.get('associated_entity_id')) for p in group)
        close=near_window(start,end,25)
        priority=(4 if close else 0, len(group), max(conf), max(assoc.values()))
        output.append({'cluster_id':pid,'frame_start':start,'frame_end':end,'duration_frames':end-start+1,
          'duration_seconds':(end-start+1)/25,'proposal_count':len(group),'max_confidence':max(conf),
          'mean_confidence':sum(conf)/len(conf),'person_association_distribution':dict(assoc),
          'spatial_summary':{'linker':'build_video03_human_review.spatial_link','max_gap_frames':max_gap,
            'spatial_margin_px_recorded':spatial_margin,'physical_identity':'unverified apparent-object cluster'},
          'representative_frames':[p['frame_index'] for p in reps if p['proposal_id'] in rep_ids],
          'representative_proposal_ids':rep_ids,'proposal_ids':[p['proposal_id'] for p in group],
          'review_status':'UNKNOWN','review_priority_key':list(priority),'priority_reason':'near known phone window' if close else 'longer/high-confidence/association workload ordering',
          'provenance':{'source_run':str(run.resolve()),'source_export_sha256':proposal_hash,
             'detections_sha256':digest(detection_file),'model_sha256s':sorted({p['model_sha256'] for p in group})}})
    return sorted(output,key=lambda x:(-x['review_priority_key'][0],-x['review_priority_key'][1],-x['review_priority_key'][2],x['frame_start'],x['cluster_id']))

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir',type=Path,default=ROOT/'outputs/unified_pipeline/video03_review_verified_20261003'); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-gap-frames',type=int,default=10); p.add_argument('--spatial-margin-px',type=int,default=80)
 a=p.parse_args()
 if a.output.exists(): raise FileExistsError(f'Refusing to overwrite {a.output}')
 if not 1<=a.max_gap_frames<=25 or a.spatial_margin_px<=0: p.error('invalid clustering thresholds')
 rows=clusters(a.run_dir.resolve(),a.max_gap_frames,a.spatial_margin_px)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',encoding='utf8') as f:
  for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')
 print(json.dumps({'status':'COMPLETE','clusters':len(rows),'proposals':sum(x['proposal_count'] for x in rows),'output':str(a.output)},indent=2))
if __name__=='__main__': main()

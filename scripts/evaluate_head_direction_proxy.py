"""Evaluate the release head-direction proxy against compatible CVAT head_dir labels."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_cvat_video import iou,digest
MAP={'LEFT':'left','RIGHT':'right','FORWARD':'front','UP':'up','DOWN':'down'}
CLASSES=tuple(MAP.values())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--detections',type=Path,required=True);p.add_argument('--reasoning',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 truth=defaultdict(list)
 for r in rows(a.labels):
  label=r.get('attributes',{}).get('head_dir')
  if r['class']=='Student' and not r['outside'] and r['geometry_valid'] and label in CLASSES: truth[r['frame_index']].append(r)
 reason={r['frame_index']:{c['tracker_id']:c['head_direction_proxy'] for c in r['candidates']} for r in rows(a.reasoning)}
 pairs=[]; excluded=Counter()
 for pred in rows(a.detections):
  frame=pred['frame_index']; used=set()
  for person in sorted(pred['people'],key=lambda x:x['detection_confidence'],reverse=True):
   opts=[(iou(person['bbox_xyxy_px'],x['bbox_xyxy_px']),i) for i,x in enumerate(truth[frame]) if i not in used]
   score,index=max(opts,default=(0,-1))
   if score<.5:continue
   used.add(index); proxy=reason.get(frame,{}).get(person['entity_local_id'])
   if proxy not in MAP: excluded['unsupported_or_missing_proxy']+=1;continue
   pairs.append((truth[frame][index]['attributes']['head_dir'],MAP[proxy]))
 total=len(pairs); correct=sum(t==q for t,q in pairs); matrix={t:{q:0 for q in CLASSES} for t in CLASSES}
 for t,q in pairs:matrix[t][q]+=1
 per={}
 for c in CLASSES:
  tp=matrix[c][c];fp=sum(matrix[t][c] for t in CLASSES if t!=c);fn=sum(matrix[c][q] for q in CLASSES if q!=c)
  per[c]={'support':sum(matrix[c].values()),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
 report={'status':'COMPLETE','metric':'head-direction proxy conditional on matched student detections; not gaze','iou_threshold':.5,'label_mapping':MAP,'excluded_truth_labels':['back','unknown','missing'],'matched_compatible_samples':total,'accuracy':correct/total if total else None,'macro_precision':sum(x['precision'] for x in per.values() if x['precision'] is not None)/len([x for x in per.values() if x['precision'] is not None]),'macro_recall':sum(x['recall'] for x in per.values() if x['recall'] is not None)/len([x for x in per.values() if x['recall'] is not None]),'macro_f1':sum(x['f1'] for x in per.values() if x['f1'] is not None)/len([x for x in per.values() if x['f1'] is not None]),'per_class':per,'confusion_matrix':matrix,'excluded':dict(excluded),'limitations':['Same-video diagnostic only.','Conditional on IoU-matched person detections; does not measure person-detection recall.','Not an eye-gaze or gaze-target metric.'] ,'labels_sha256':digest(a.labels),'detections_sha256':digest(a.detections),'reasoning_sha256':digest(a.reasoning)}
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

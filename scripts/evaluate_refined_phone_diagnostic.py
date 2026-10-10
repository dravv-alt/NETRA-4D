"""Proposal-derived diagnostic evaluation with reviewed ambiguous proposals ignored."""
from __future__ import annotations
import argparse,json,statistics,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_cvat_video import iou,digest
from serve_video03_human_review import ReviewStore
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def ap(preds,gts,ignore,thr):
 used=set();tp=[];fp=[]
 for p in sorted(preds,key=lambda x:-x['confidence']):
  f=p['frame']; candidates=[(iou(p['bbox'],b),j) for j,b in enumerate(gts[f]) if (f,j) not in used]; score,j=max(candidates,default=(0,-1))
  if score>=thr: used.add((f,j));tp.append(1);fp.append(0)
  elif max((iou(p['bbox'],b) for b in ignore.get(f,[])),default=0)>=thr: continue
  else:tp.append(0);fp.append(1)
 if not gts or not tp:return None
 cumt=[];cumf=[];a=b=0
 for x,y in zip(tp,fp):a+=x;b+=y;cumt.append(a);cumf.append(b)
 rec=[x/sum(map(len,gts.values())) for x in cumt];prec=[x/(x+y) for x,y in zip(cumt,cumf)]
 return sum(max([p for r,p in zip(rec,prec) if r>=k/100] or [0]) for k in range(101))/101
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 gt=defaultdict(list);base=rows(ROOT/'outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl')
 base_keys={(r['frame_index'],tuple(r['bbox_xyxy_px'])) for r in base if r['class']=='Phone' and not r['outside']}
 for r in rows(a.labels):
  if r['class']=='Phone' and not r['outside'] and r['geometry_valid']:gt[r['frame_index']].append(r['bbox_xyxy_px'])
 store=ReviewStore(a.review);state=store.state();data=json.loads((a.review/'review_data.json').read_text());pm={x['proposal_id']:x for x in data['proposals']}; ignore=defaultdict(list);notp=[]
 for pid,status in state['statuses'].items():
  r=pm[pid]
  if status=='AMBIGUOUS':ignore[r['frame_index']].append(r['bbox_xyxy_px'])
  if status=='NOT_PHONE':notp.append(r)
 preds=[{'frame':r['frame_index'],'bbox':o['bbox_xyxy_px'],'confidence':o['confidence'],'person_id':o.get('associated_entity_id')} for r in rows(a.run/'detections.jsonl') for o in r['objects'] if o['object_class']=='MOBILE_PHONE']
 used=set(); errors=[];ious=[];tp=fp=0
 for pr in sorted(preds,key=lambda x:-x['confidence']):
  f=pr['frame']; best,j=max(((iou(pr['bbox'],b),j) for j,b in enumerate(gt[f]) if (f,j) not in used),default=(0,-1));amb=max((iou(pr['bbox'],b) for b in ignore.get(f,[])),default=0)
  if best>=.5: used.add((f,j));tp+=1;ious.append(best);cls='TP';g=gt[f][j]
  elif amb>=.5:cls='AMBIGUOUS';g=None
  else:fp+=1;cls='FP';g=None
  errors.append({'frame':f,'prediction_bbox':pr['bbox'],'confidence':pr['confidence'],'person_id':pr['person_id'],'ground_truth_bbox':g,'IoU':best,'classification':cls,'provenance':'PROPOSAL_DERIVED' if g and (f,tuple(g)) not in base_keys else 'INDEPENDENT_MANUAL'})
 for f,boxes in gt.items():
  for j,b in enumerate(boxes):
   if (f,j) not in used:errors.append({'frame':f,'prediction_bbox':None,'confidence':None,'person_id':None,'ground_truth_bbox':b,'IoU':0,'classification':'FN','provenance':'PROPOSAL_DERIVED' if (f,tuple(b)) not in base_keys else 'INDEPENDENT_MANUAL'})
 fn=sum(1 for e in errors if e['classification']=='FN'); valid=tp+fp; precision=tp/valid if valid else None;recall=tp/(tp+fn) if tp+fn else None;f1=2*precision*recall/(precision+recall) if precision and recall else 0
 # FP episode grouping uses reviewed NOT_PHONE proposal temporal gaps <=10.
 fpframes=sorted({r['frame_index'] for r in notp});episodes=0;last=None
 for f in fpframes:
  if last is None or f>last+10:episodes+=1
  last=f
 aps={str(t):ap(preds,gt,ignore,t) for t in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]}
 report={'status':'COMPLETE','evaluation_type':'VIDEO03_DIAGNOSTIC_PROPOSAL_DERIVED','matching':'confidence-descending one-to-one IoU matching; ambiguous reviewed proposals ignored','tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1,'AP50':aps['0.5'],'mAP50_95':sum(aps.values())/len(aps),'mean_iou':statistics.mean(ious),'median_iou':statistics.median(ious),'fp_reviewed_not_phone':{'detections':len(notp),'frames':len(fpframes),'episodes':episodes,'duration_hours':88/3600,'VIDEO03_DIAGNOSTIC_FP_per_hour':episodes/(88/3600)},'ambiguous':{'proposals':sum(v=='AMBIGUOUS' for v in state['statuses'].values()),'frames':len(ignore),'episodes':'not counted as FP'},'error_table':'phone_diagnostic_error_table.jsonl','limitations':['Proposal-derived confirmed GT; not independent model benchmark.','FP includes all unmatched non-ambiguous predictions; NOT_PHONE counts are separately reported.','No behavioural claims or held-out generalization.']}
 with (a.output/'phone_diagnostic_error_table.jsonl').open('w') as f:
  for e in errors:f.write(json.dumps(e)+'\n')
 (a.output/'phone_diagnostic_evaluation.json').write_text(json.dumps(report,indent=2));(a.output/'phone_diagnostic_evaluation.md').write_text('\n'.join(['# Phone diagnostic evaluation','',f"TP {tp}; FP {fp}; FN {fn}",f"Precision {precision:.4f}; Recall {recall:.4f}; F1 {f1:.4f}",f"AP50 {aps['0.5']:.4f}; mAP50-95 {report['mAP50_95']:.4f}",f"Mean IoU {report['mean_iou']:.4f}; median IoU {report['median_iou']:.4f}",f"Reviewed NOT_PHONE: {len(notp)} detections, {len(fpframes)} frames, {episodes} episodes; VIDEO03 diagnostic FP/hr {report['fp_reviewed_not_phone']['VIDEO03_DIAGNOSTIC_FP_per_hour']:.2f}",'','Proposal-derived diagnostic only; not independent benchmark.'])+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

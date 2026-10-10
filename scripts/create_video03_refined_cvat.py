"""Create a bounded CVAT refinement candidate from append-only human decisions.

The raw XML is never written.  Confirmed review proposals provide geometry; no
interpolation is introduced across unreviewed frames.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from serve_video03_human_review import ReviewStore

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); den=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/den if den else 0.0

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--base-xml',type=Path,default=ROOT/'datasets/video03_0_2199_derivative/annotations_0_2199.xml'); p.add_argument('--review-package',type=Path,default=ROOT/'outputs/unified_pipeline/video03_review_verified_20261003/human_review_v1'); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 if a.output_dir.exists(): raise FileExistsError(f'Refusing to overwrite {a.output_dir}')
 store=ReviewStore(a.review_package); state=store.state(); data=json.loads((a.review_package/'review_data.json').read_text(encoding='utf8')); proposals={x['proposal_id']:x for x in data['proposals']}
 confirmed=[proposals[k] for k,v in state['statuses'].items() if v=='CONFIRMED_PHONE']; ambiguous=[k for k,v in state['statuses'].items() if v=='AMBIGUOUS']; not_phone=[k for k,v in state['statuses'].items() if v=='NOT_PHONE']
 if len(confirmed)+len(ambiguous)+len(not_phone)!=len(proposals): raise RuntimeError('Unreviewed proposals remain')
 tree=ET.parse(a.base_xml); root=tree.getroot(); tracks=root.findall('track'); next_id=max(int(t.get('id')) for t in tracks)+1
 # Preserve every explicitly reviewed geometry. Separate runs prevent CVAT from
 # interpolating through gaps the reviewer did not confirm.
 all_per_frame=defaultdict(list)
 for row in confirmed: all_per_frame[row['frame_index']].append(row)
 merged=[]; confirmed_final=[]; duplicate_frames=[]
 for frame,rows in sorted(all_per_frame.items()):
  rows=sorted(rows,key=lambda x:(-x['raw_confidence'],x['proposal_id']))
  winner=rows[0]; confirmed_final.append(winner); extras=rows[1:]
  if extras and all(iou(winner['bbox_xyxy_px'],x['bbox_xyxy_px'])>=.5 for x in extras): merged.append({'frame_index':frame,'kept_proposal_id':winner['proposal_id'],'merged_duplicate_proposal_ids':[x['proposal_id'] for x in extras]})
  elif extras: confirmed_final.extend(extras); duplicate_frames.append({'frame_index':frame,'proposal_ids':[x['proposal_id'] for x in rows],'status':'REQUIRES_GEOMETRY_MERGE_REVIEW'})
 by_segment=defaultdict(list)
 for row in confirmed_final: by_segment[row['segment_id']].append(row)
 candidate_tracks=[]
 for segment, rows in sorted(by_segment.items()):
  per_frame=defaultdict(list)
  for r in rows: per_frame[r['frame_index']].append(r)
  # A parallel detection on one frame remains separate rather than silently dropped.
  lanes=[]
  for r in sorted(rows,key=lambda x:(x['frame_index'],x['proposal_id'])):
   placed=False
   for lane in lanes:
    if r['frame_index']==lane[-1]['frame_index']+1: lane.append(r); placed=True; break
   if not placed: lanes.append([r])
  for lane in lanes:
   t=ET.SubElement(root,'track',{'id':str(next_id),'label':'Phone','source':'manual'}); next_id+=1
   for r in lane:
    x1,y1,x2,y2=r['bbox_xyxy_px']; ET.SubElement(t,'box',{'frame':str(r['frame_index']),'outside':'0','occluded':'0','keyframe':'1','xtl':str(x1),'ytl':str(y1),'xbr':str(x2),'ybr':str(y2),'z_order':'0'})
   candidate_tracks.append({'track_id':int(t.get('id')),'source_segment_id':segment,'frame_start':lane[0]['frame_index'],'frame_end':lane[-1]['frame_index'],'boxes':len(lane),'proposal_ids':[x['proposal_id'] for x in lane]})
 a.output_dir.mkdir(parents=True); xml=a.output_dir/'video03_cvat_refined_reviewed_boxes.xml'; tree.write(xml,encoding='utf-8',xml_declaration=True)
 report={'status':'CANDIDATE_REFINED_CVAT_READY_FOR_IMPORT','base_xml_sha256':sha(a.base_xml),'refined_xml_sha256':sha(xml),'raw_xml_sha256':'301620001d90d0f2fb06deb3535e98dc67c6c641f7a10e84b5b01c2db1452382','review_data_sha256':state['review_data_sha256'],'reviewed_proposals':len(proposals),'confirmed_phone_proposals':len(confirmed),'confirmed_phone_boxes_added':len(confirmed_final),'same_frame_duplicate_proposals_merged':merged,'not_phone_excluded':len(not_phone),'ambiguous_excluded':len(ambiguous),'new_phone_tracks':len(candidate_tracks),'duplicate_confirmed_frames_requiring_import_review':duplicate_frames,'candidate_tracks':candidate_tracks,'created_utc':datetime.now(timezone.utc).isoformat(),'warning':'Candidate uses human-confirmed proposal geometry. Raw XML unchanged; no metrics calculated.'}
 (a.output_dir/'REFINEMENT_MANIFEST.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps({k:v for k,v in report.items() if k!='candidate_tracks'},indent=2))
if __name__=='__main__':main()

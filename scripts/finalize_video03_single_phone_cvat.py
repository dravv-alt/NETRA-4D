"""Build one bounded Phone track from raw visible boxes and human-confirmed proposals."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from create_video03_refined_cvat import sha
from serve_video03_human_review import ReviewStore

def main():
 p=argparse.ArgumentParser(); p.add_argument('--base-xml',type=Path,default=ROOT/'datasets/video03_0_2199_derivative/annotations_0_2199.xml'); p.add_argument('--review-package',type=Path,default=ROOT/'outputs/unified_pipeline/video03_review_verified_20261003/human_review_v1'); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 if a.output_dir.exists(): raise FileExistsError(a.output_dir)
 tree=ET.parse(a.base_xml); root=tree.getroot(); store=ReviewStore(a.review_package); state=store.state(); data=json.loads((a.review_package/'review_data.json').read_text()); props={x['proposal_id']:x for x in data['proposals']}
 # Existing visible Phone boxes plus human-confirmed reviewed proposal boxes.
 boxes={}
 for track in root.findall('track'):
  if track.get('label')=='Phone':
   for b in track.findall('box'):
    if b.get('outside')!='1': boxes[int(b.get('frame'))]=[float(b.get(k)) for k in ('xtl','ytl','xbr','ybr')]
 confirmed=[props[k] for k,v in state['statuses'].items() if v=='CONFIRMED_PHONE']
 merged=[]
 for frame in sorted({x['frame_index'] for x in confirmed}):
  rows=sorted([x for x in confirmed if x['frame_index']==frame],key=lambda x:(-x['raw_confidence'],x['proposal_id']))
  boxes[frame]=rows[0]['bbox_xyxy_px']
  if len(rows)>1: merged.append({'frame_index':frame,'kept_proposal_id':rows[0]['proposal_id'],'discarded_duplicate_ids':[x['proposal_id'] for x in rows[1:]],'reason':'one physical phone confirmed by reviewer'})
 # Replace only the two legacy visibility tracks with one physical-phone track.
 for t in list(root.findall('track')):
  if t.get('label')=='Phone': root.remove(t)
 t=ET.SubElement(root,'track',{'id':'3','label':'Phone','source':'manual'})
 frames=sorted(boxes)
 for i,frame in enumerate(frames):
  x1,y1,x2,y2=boxes[frame]; ET.SubElement(t,'box',{'frame':str(frame),'outside':'0','occluded':'0','keyframe':'1','xtl':str(x1),'ytl':str(y1),'xbr':str(x2),'ybr':str(y2),'z_order':'0'})
  nxt=frames[i+1] if i+1<len(frames) else 2200
  if nxt>frame+1: ET.SubElement(t,'box',{'frame':str(frame+1),'outside':'1','occluded':'0','keyframe':'1','xtl':str(x1),'ytl':str(y1),'xbr':str(x2),'ybr':str(y2),'z_order':'0'})
 a.output_dir.mkdir(parents=True); out=a.output_dir/'video03_cvat_refined_single_phone.xml'; tree.write(out,encoding='utf-8',xml_declaration=True)
 report={'status':'READY_FOR_REFINED_EVALUATION','raw_xml_sha256':'301620001d90d0f2fb06deb3535e98dc67c6c641f7a10e84b5b01c2db1452382','base_xml_sha256':sha(a.base_xml),'refined_xml_sha256':sha(out),'single_physical_phone':True,'visible_phone_boxes':len(boxes),'human_confirmed_proposals':len(confirmed),'merged_duplicate_proposals':merged,'ambiguous_excluded':sum(v=='AMBIGUOUS' for v in state['statuses'].values()),'not_phone_excluded':sum(v=='NOT_PHONE' for v in state['statuses'].values())}
 (a.output_dir/'REFINEMENT_MANIFEST.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__':main()

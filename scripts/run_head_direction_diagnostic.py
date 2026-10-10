"""Read-only Video 3 head-direction convention and visualization diagnostic."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import cv2

CLASSES=("left","right","front","up","down")
TRANSFORMS={
 "A_baseline": {c:c for c in CLASSES},
 "B_left_right_inversion": {"left":"right","right":"left","front":"front","up":"up","down":"down"},
 "C_up_down_inversion": {"left":"left","right":"right","front":"front","up":"down","down":"up"},
 "D_horizontal_vertical_inversion": {"left":"right","right":"left","front":"front","up":"down","down":"up"},
 "E_yaw_sign_inversion": {"left":"right","right":"left","front":"front","up":"up","down":"down"},
 "F_pitch_sign_inversion": {"left":"left","right":"right","front":"front","up":"down","down":"up"},
 "G_180_yaw_rotation_representable": {"left":"right","right":"left","front":"front","up":"up","down":"down"},
 "H_xy_axis_swap_clockwise": {"left":"up","right":"down","front":"front","up":"right","down":"left"},
}
COLORS={"nose":(0,0,255),"left_eye":(255,0,0),"right_eye":(0,255,255)}
def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def score(rows, mapping):
 m={t:{p:0 for p in CLASSES} for t in CLASSES}
 for r in rows:
  predicted = "front" if r['predicted_direction'] == "FORWARD" else r['predicted_direction'].lower()
  m[r['cvat_head_dir']][mapping[predicted]] += 1
 per={}
 for c in CLASSES:
  tp=m[c][c];fp=sum(m[t][c] for t in CLASSES if t!=c);fn=sum(m[c][p] for p in CLASSES if p!=c)
  per[c]={"support":sum(m[c].values()),"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
 n=len(rows);acc=sum(m[c][c] for c in CLASSES)/n
 return {"accuracy":acc,"macro_precision":sum(x['precision'] or 0 for x in per.values())/5,"macro_recall":sum(x['recall'] or 0 for x in per.values())/5,"macro_f1":sum(x['f1'] or 0 for x in per.values())/5,"per_class":per,"confusion_matrix":m}
def threshold_score(rows,yaw,down,up):
 derived=[]
 for r in rows:
  y,p=r['smoothed_yaw_proxy'],r['pitch_proxy']
  q='right' if y>yaw else 'left' if y<-yaw else 'down' if p>down else 'up' if p<up else 'front'
  derived.append({**r,'predicted_direction':q})
 return score(derived,TRANSFORMS['A_baseline'])
def put(im,text,y,color=(255,255,255)): cv2.putText(im,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
def draw(frame,r):
 h,w=frame.shape[:2]; out=frame.copy(); kp=r['raw_keypoints_normalized']; pts={}
 for n in ("nose","left_eye","right_eye"):
  q=kp.get(n)
  if q: pts[n]=(round(q['x']*w),round(q['y']*h));cv2.circle(out,pts[n],5,COLORS[n],-1)
 if all(x in pts for x in pts):
  mid=((pts['left_eye'][0]+pts['right_eye'][0])//2,(pts['left_eye'][1]+pts['right_eye'][1])//2);cv2.line(out,pts['left_eye'],pts['right_eye'],(255,255,255),2);cv2.arrowedLine(out,mid,pts['nose'],(0,0,255),2,tipLength=.25)
 put(out,f"frame={r['frame']} id={r['person_id']} CVAT={r['cvat_head_dir']} PRED={r['predicted_direction']}",28)
 put(out,f"yaw={r['smoothed_yaw_proxy']:.3f} pitch={r['pitch_proxy']:.3f} face_conf={r['face_keypoint_min_confidence']:.3f}",54)
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--intermediates',type=Path,required=True);a.add_argument('--video',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 if x.output.exists():raise FileExistsError(x.output)
 rows=read(x.intermediates);results={name:{"mapping":mp,**score(rows,mp)} for name,mp in TRANSFORMS.items()}
 thresholds={f"yaw_{y}_down_{d}_up_{u}":threshold_score(rows,y,d,u) for y,d,u in ((.2,.6,-.2),(.4,.85,-.4),(.6,1.1,-.6),(.8,1.3,-.8))}
 x.output.mkdir(parents=True);(x.output/'transformation_metrics.json').write_text(json.dumps({"samples":len(rows),"note":"Transforms are diagnostic only; no calibration or model change.","results":results},indent=2));(x.output/'threshold_sensitivity_video03_development_calibration.json').write_text(json.dumps({"status":"VIDEO03_DEVELOPMENT_CALIBRATION_ONLY","warning":"Computed from diagnostic replay intermediates; not independent evaluation and not suitable for selecting production thresholds.","results":thresholds},indent=2))
 targets=[("front","right"),("front","down"),("down","right"),("left",None),("right",None)];selected=[]
 for truth,pred in targets:
  selected.extend([r for r in rows if r['cvat_head_dir']==truth and (pred is None or r['predicted_direction'].lower()==pred)][:12])
 byframe={r['frame']:[] for r in selected}
 for r in selected:byframe[r['frame']].append(r)
 cap=cv2.VideoCapture(str(x.video));idx=0;written=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  if idx in byframe:
   for j,r in enumerate(byframe[idx]):
    name=f"frame_{idx:04d}_person_{r['person_id']}_{j}.jpg";cv2.imwrite(str(x.output/name),draw(f,r));written.append(name)
  idx+=1
 cap.release()
 (x.output/'representatives.json').write_text(json.dumps({"count":len(written),"files":written,"selection":"first 12 per requested truth/prediction stratum; raw pose geometry only"},indent=2))
 print(json.dumps({"samples":len(rows),"representatives":len(written),"baseline":results['A_baseline']},indent=2))
if __name__=='__main__':main()

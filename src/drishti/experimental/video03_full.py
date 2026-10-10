"""Video-03 frozen-baseline augmentation experiment.

This module deliberately replays release detections rather than silently replacing
the release path.  It produces observable evidence and review candidates only;
it never emits a cheating verdict or a probability of cheating.
"""
from __future__ import annotations

import hashlib, json, platform, subprocess, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
from drishti.engine.adaptive_escalation import (
    AdaptiveSpecialistEscalationController,
    load_specialist_escalation_config,
)

ROOT = Path(__file__).resolve().parents[3]
VIDEO03 = ROOT / "input" / "videos" / "VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4"
DEFAULT_BASELINE = ROOT / "outputs" / "unified_pipeline" / "video03_review_verified_20261003"
EXPECTED_VIDEO_HASH = "7fba8bdda53c2524204822cbf630ff47a69123a2924e378644a471c6b8f9383b"

def _sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def _write(f, row: Dict[str,Any]) -> None: f.write(json.dumps(row, sort_keys=True)+"\n")
def _dist(a,b): return float(np.hypot(a[0]-b[0],a[1]-b[1]))
def _centre(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def _iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); den=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/den if den else 0.0
def _git() -> Optional[str]:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: return None
def _kp(person, name, w,h):
    p=person.get("keypoints_2d",{}).get(name)
    if not p or p.get("confidence",0)<.25: return None
    return (p["x"]*w,p["y"]*h,p["confidence"])
def _head(person,w,h):
    nose=_kp(person,"nose",w,h); ls=_kp(person,"left_shoulder",w,h); rs=_kp(person,"right_shoulder",w,h)
    if not (nose and ls and rs): return "UNKNOWN"
    delta=nose[0]-(ls[0]+rs[0])/2
    return "RIGHT" if delta>8 else "LEFT" if delta<-8 else "FORWARD_OR_UNKNOWN"

def run_video03_full_experimental(video_source: str, output_dir: str, *, baseline_run: str|Path=DEFAULT_BASELINE,
                                  enable_rtmpose=False, enable_hands=False, enable_interaction=False,
                                  enable_fusion=False, enable_3d=False, max_frames=2200, specialist_stride=1,
                                  specialist_policy: str="conditional_escalation",
                                  specialist_policy_config: str|Path=ROOT/"configs"/"specialist_escalation.yaml") -> tuple[str,str]:
    """Run a bounded, non-destructive augmentation experiment over frozen baseline outputs."""
    source=Path(video_source).resolve(); base=Path(baseline_run).resolve(); out=Path(output_dir).resolve()
    if out.exists(): raise FileExistsError(f"Refusing to overwrite experiment output: {out}")
    if _sha(source)!=EXPECTED_VIDEO_HASH: raise ValueError("VIDEO_03 hash mismatch; refusing an unproven source")
    if max_frames <=0 or max_frames>2200: max_frames=2200
    det=base/"detections.jsonl"; reasoning=base/"unified_reasoning.jsonl"; metrics=base/"metrics.json"
    if not all(p.exists() for p in (det,reasoning,metrics)): raise FileNotFoundError("Baseline run is incomplete")
    out.mkdir(parents=True)
    escalation_config=load_specialist_escalation_config(specialist_policy_config, specialist_policy)
    config={"profile":"video03-full-experimental","mode":"frozen_baseline_augmentation_replay","frame_range":[0,max_frames-1],
      "enabled_components":{"rtmpose":enable_rtmpose,"hands_experimental":enable_hands,"interaction_graph":enable_interaction,"evidence_fusion":enable_fusion,"3d_experimental":enable_3d},
      "disabled_components":[x for x,v in {"rtmpose":enable_rtmpose,"hands_experimental":enable_hands,"interaction_graph":enable_interaction,"evidence_fusion":enable_fusion,"3d_experimental":enable_3d}.items() if not v],
      "specialist_stride_frames":max(1, specialist_stride),
      "specialist_policy_name": specialist_policy,
      "specialist_policy": escalation_config.to_dict(),
      "execution_devices":{"frozen_yolo_baseline":"cuda:0 provenance from baseline run","rtmpose":"CUDA requested; provider resolved at runtime","mediapipe_hands":"MediaPipe CPU task runtime"},
      "safety":"Evidence and review candidates only; no cheating classifier or probability; UNKNOWN is never converted to NOT_PHONE."}
    cfg_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    model_paths={"yolo26_pose":ROOT/"models/weights/yolo26n-pose.pt","rtmpose":ROOT/"models/weights/rtmpose_m/rtmpose_m.onnx","mediapipe_hands":ROOT/"models/weights/hand_landmarker.task","multihmr":ROOT/"models/multiHMR/multiHMR_672_S.pt"}
    model_hashes={k:_sha(p) for k,p in model_paths.items() if p.exists()}
    start=time.perf_counter(); cap=cv2.VideoCapture(str(source)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS)
    rtm=None; hands=None
    if enable_rtmpose:
        from drishti.engine.specialists import RTMPoseONNXRunner
        rtm=RTMPoseONNXRunner()
        if not rtm.is_available: rtm=None
    config["execution_devices"]["rtmpose"] = ("cuda" if rtm and "CUDAExecutionProvider" in getattr(rtm.session, "get_providers", lambda:[])() else "cpu")
    if enable_hands:
        from drishti.engine.specialists import MediaPipeHandRunner
        hands=MediaPipeHandRunner()
        if not hands.is_available: hands=None
    reason_by_frame={int(x["frame_index"]):x for x in map(json.loads,reasoning.read_text(encoding="utf8").splitlines())}
    histories=defaultdict(lambda:{"head":None,"head_run":0,"phone_run":0,"neighbour_run":0,"state":"NORMAL_ACTIVITY","last":-2})
    escalation_controllers=defaultdict(lambda: AdaptiveSpecialistEscalationController(escalation_config))
    counts=Counter(); agreement=[]; states=[]
    files={n:(out/n).open("w",encoding="utf8") for n in ["pose_comparison.jsonl","hand_evidence.jsonl","interaction_graph.jsonl","temporal_evidence.jsonl","hypotheses.jsonl","state_transitions.jsonl","evidence_fusion.jsonl","review_queue.jsonl","3d_pose_evidence.jsonl","specialist_escalation.jsonl"]}
    if enable_3d:
      # Checkpoint availability is not a validated camera-to-track adapter.
      for fr in (151, 895, 1464, 1733):
        if fr < max_frames: _write(files["3d_pose_evidence.jsonl"], {"frame_index":fr,"status":"PENDING_ADAPTER_VALIDATION","selection":"predeclared representative case","body_orientation":None,"torso_orientation":None,"head_body_relative_orientation":None,"reconstruction_quality":None,"reason":"no validated Multi-HMR mesh-to-camera-track adapter","provenance":{"multihmr_hash":model_hashes.get("multihmr"),"config_hash":cfg_hash}})
    try:
      with det.open(encoding="utf8") as dfile:
       for line in dfile:
        row=json.loads(line); fr=int(row["frame_index"])
        if fr>=max_frames: break
        ok,frame=cap.read()
        if not ok: raise RuntimeError(f"Cannot decode frame {fr}")
        people=row.get("people",[]); objects=row.get("objects",[])
        candidate_rows={c.get("tracker_id"):c for c in reason_by_frame.get(fr,{}).get("candidates",[])}
        candidates={tid:c.get("candidate_id",f"tracker:{tid}") for tid,c in candidate_rows.items()}
        frame_hands=defaultdict(list); frame_inter=defaultdict(list)
        for p in people:
          tid=p.get("entity_local_id"); cid=candidates.get(tid,f"tracker:{tid}"); box=p["bbox_xyxy_px"]
          ykp=p.get("keypoints_2d",{}); rkp=[]; score=None; dtype="NOT_ESCALATED"; rtm_observed=False
          baseline_uncertainty=float(candidate_rows.get(tid,{}).get("uncalibrated_uncertainty_proxy", 1.0-p.get("detection_confidence",1.0)))
          person_confidence=float(p.get("detection_confidence",p.get("confidence",1.0)))
          quality_signal=max(0.0,min(1.0,max(baseline_uncertainty,1.0-person_confidence)))
          decision=escalation_controllers[tid].step(fr,fr/fps,quality_signal)
          specialist_types=[]
          if decision.invoke_specialist and rtm: specialist_types.append("RTMPOSE")
          if decision.invoke_specialist and hands: specialist_types.append("MEDIAPIPE_HANDS")
          _write(files["specialist_escalation.jsonl"],{"frame_index":fr,"timestamp_sec":fr/fps,"track_id":tid,"candidate_id":cid,"baseline_quality_evidence":{"uncertainty_proxy":baseline_uncertainty,"person_detection_confidence":person_confidence,"quality_signal":quality_signal},"trigger_state":decision.to_dict(),"specialist_invoked":decision.invoke_specialist,"specialist_types":specialist_types,"provenance":{"policy_name":specialist_policy,"policy_config":escalation_config.to_dict(),"config_hash":cfg_hash}})
          if decision.invoke_specialist: counts["specialist_invocations"]+=1
          if rtm and decision.invoke_specialist and fr % max(1,specialist_stride) == 0:
            x1,y1,x2,y2=map(int,box); pad=max(4,int(.08*max(x2-x1,y2-y1))); x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w,x2+pad); y2=min(h,y2+pad)
            try:
              raw=rtm.infer_crop(frame[y1:y2,x1:x2]); rkp=[{"name":q.name,"x":x1+q.x*(x2-x1),"y":y1+q.y*(y2-y1),"confidence":q.confidence} for q in raw]
              pairs=[]
              for q in rkp:
                a=ykp.get(q["name"])
                if a and a.get("confidence",0)>=.25 and q["confidence"]>=.25: pairs.append(_dist((a["x"]*w,a["y"]*h),(q["x"],q["y"]))/max(w,h))
              score=float(np.mean(pairs)) if pairs else None; dtype="AGREEMENT_AVAILABLE" if pairs else "INSUFFICIENT_MUTUAL_KEYPOINTS"; rtm_observed=bool(raw); agreement.extend(pairs); counts["rtmpose_invocations"]+=1
            except Exception as e: dtype="RTMPOSE_RUNTIME_FAILURE"; counts["rtmpose_failures"]+=1
          _write(files["pose_comparison.jsonl"],{"frame_index":fr,"timestamp_sec":fr/fps,"track_id":tid,"candidate_id":cid,"yolo26_keypoints":ykp,"rtmpose_keypoints":rkp,"agreement_score_normalized_distance":score,"disagreement_type":dtype,"escalation":decision.to_dict(),"provenance":{"baseline_model_hash":model_hashes.get("yolo26_pose"),"rtmpose_hash":model_hashes.get("rtmpose"),"config_hash":cfg_hash}})
          for side in ("left","right"):
            wrist=_kp(p,f"{side}_wrist",w,h); rec={"frame_index":fr,"timestamp_sec":fr/fps,"track_id":tid,"candidate_id":cid,"hand_side":side,"associated_object":"UNKNOWN","association_confidence":0.0,"model_provenance":{"mediapipe_hash":model_hashes.get("mediapipe_hands"),"config_hash":cfg_hash}}
            if not hands or not wrist: rec.update({"status":"UNKNOWN","reason":"hand model disabled/unavailable or wrist keypoint unavailable","keypoints":[]})
            elif not decision.invoke_specialist: rec.update({"status":"NOT_ESCALATED","reason":"sustained trigger not active","keypoints":[]})
            elif fr % max(1,specialist_stride) != 0: rec.update({"status":"NOT_SAMPLED","reason":"specialist interval; no observation inferred","keypoints":[]})
            else:
              x,y,_=wrist; r=72; x1,y1,x2,y2=max(0,int(x-r)),max(0,int(y-r)),min(w,int(x+r)),min(h,int(y+r))
              try:
                hs=hands.infer_hand(frame[y1:y2,x1:x2],side)
                landmarks = hs.landmarks.values() if hs is not None else []
                pts=[{"x":x1+a.x*(x2-x1),"y":y1+a.y*(y2-y1),"z":a.z,"confidence":a.confidence} for a in landmarks]
                rec.update({"status":"OBSERVED" if pts else "UNKNOWN","keypoints":pts,"crop_provenance":{"bbox_xyxy_px":[x1,y1,x2,y2],"source":"baseline_yolo26_wrist"},"confidence":hs.confidence})
                counts["hands_invocations"]+=1
                if pts: frame_hands[tid].append((side,pts[0],rec)); counts["hands_observed"]+=1
              except Exception: rec.update({"status":"UNKNOWN","reason":"hand_model_runtime_failure","keypoints":[]}); counts["hand_failures"]+=1
            _write(files["hand_evidence.jsonl"],rec)
        # Relationships are geometry, never a behavioural verdict.
        for p in people:
          tid=p.get("entity_local_id"); cid=candidates.get(tid,f"tracker:{tid}"); pb=p["bbox_xyxy_px"]
          assoc=[o for o in objects if o.get("associated_entity_id")==tid] if enable_interaction else []
          for o in assoc:
            rr={"frame_index":fr,"timestamp_sec":fr/fps,"source":{"track_id":tid,"candidate_id":cid},"relation":"associated_object","target":{"object_class":o.get("object_class"),"bbox_xyxy_px":o.get("bbox_xyxy_px")},"distance_px":_dist(_centre(pb),_centre(o["bbox_xyxy_px"])),"confidence":o.get("association_confidence",0),"temporal_duration_sec":1/fps,"provenance":{"source":"frozen_baseline_association","config_hash":cfg_hash}}
            frame_inter[tid].append(rr); _write(files["interaction_graph.jsonl"],rr); counts["object_relations"]+=1
          for q in (people if enable_interaction else []):
            if q is p: continue
            qid=q.get("entity_local_id"); d=_dist(_centre(pb),_centre(q["bbox_xyxy_px"]))
            if d<350:
              rr={"frame_index":fr,"timestamp_sec":fr/fps,"source":{"track_id":tid,"candidate_id":cid},"relation":"near_neighbour","target":{"track_id":qid},"distance_px":d,"confidence":max(0,1-d/350),"temporal_duration_sec":1/fps,"provenance":{"source":"person_box_geometry","config_hash":cfg_hash}}
              frame_inter[tid].append(rr); _write(files["interaction_graph.jsonl"],rr); counts["neighbour_relations"]+=1
        for p in people:
          tid=p.get("entity_local_id"); cid=candidates.get(tid,f"tracker:{tid}"); hist=histories[tid]; head=_head(p,w,h); rels=frame_inter[tid]
          phone=any(r["target"].get("object_class") in ("PHONE","PHONE_CANDIDATE") for r in rels); neigh=any(r["relation"]=="near_neighbour" for r in rels)
          hist["head_run"]=hist["head_run"]+1 if head==hist["head"] and head!="UNKNOWN" else 1; hist["head"]=head
          hist["phone_run"]=hist["phone_run"]+1 if phone else 0; hist["neighbour_run"]=hist["neighbour_run"]+1 if neigh and head in ("LEFT","RIGHT") else 0
          evidence={"pose_evidence":{"head_direction_proxy":head,"persistence_sec":hist["head_run"]/fps},"hand_evidence":{"observed":bool(frame_hands[tid]),"status":"OBSERVED" if frame_hands[tid] else "UNKNOWN"},"phone_evidence":{"associated_phone":phone,"duration_sec":hist["phone_run"]/fps},"neighbour_evidence":{"near_neighbour":neigh,"attention_proxy_duration_sec":hist["neighbour_run"]/fps},"object_evidence":{"relations":len(rels)},"specialist_evidence":{"escalation":decision.to_dict(),"rtmpose_keypoints_observed":rtm_observed,"hands_observed":bool(frame_hands[tid])},"occlusion":{"status":"UNKNOWN"}}
          _write(files["temporal_evidence.jsonl"],{"frame_index":fr,"timestamp_sec":fr/fps,"track_id":tid,"candidate_id":cid,"evidence":evidence,"provenance":{"config_hash":cfg_hash}})
          # Internal scores are deliberately not probabilities and do not name cheating.
          supports=[]; missing=[]
          if phone: supports.append("baseline-associated phone candidate")
          if frame_hands[tid]: supports.append("hand landmark observation")
          if hist["neighbour_run"]>=50: supports.append("sustained neighbour-attention geometry proxy")
          if not supports: missing.append("corroborating observable evidence")
          hyp="phone_object_interaction" if phone else ("neighbour_attention" if neigh else "normal_environmental_attention")
          review=phone and bool(frame_hands[tid]) and hist["neighbour_run"]>=25
          state="REVIEW_REQUIRED" if review else ("OBJECT_INTERACTION" if phone else ("SUSTAINED_NEIGHBOUR_ATTENTION" if hist["neighbour_run"]>=50 else ("ATTENTION_SHIFT" if head!= "UNKNOWN" else "UNKNOWN")))
          if state!=hist["state"]:
            _write(files["state_transitions.jsonl"],{"frame_index":fr,"timestamp_sec":fr/fps,"track_id":tid,"candidate_id":cid,"from_state":hist["state"],"to_state":state,"observable_conditions":supports or missing,"provenance":{"config_hash":cfg_hash}}); hist["state"]=state; states.append(state)
          hrow={"frame_index":fr,"timestamp_sec":fr/fps,"track_id":tid,"candidate_id":cid,"hypothesis":hyp,"supporting_evidence":supports,"contradictory_evidence":[],"missing_evidence":missing,"internal_uncalibrated_evidence_score":len(supports)/3,"score_semantics":"uncalibrated evidence aggregation; not a probability of cheating","status":"review_required" if review else ("candidate" if supports else "unresolved"),"provenance":{"config_hash":cfg_hash}}
          _write(files["hypotheses.jsonl"],hrow)
          frow={"frame_index":fr,"timestamp_sec":fr/fps,"track_id":tid,"candidate_id":cid,"evidence_vector":evidence,"plausible_hypotheses":[hyp,"insufficient_evidence"],"state":state,"provenance":{"config_hash":cfg_hash}}
          if enable_fusion: _write(files["evidence_fusion.jsonl"],frow)
          if enable_fusion and review: _write(files["review_queue.jsonl"],dict(frow, review_reason="multiple independent observable evidence streams", status="PENDING_HUMAN_REVIEW")); counts["review_required"]+=1
    finally:
      cap.release()
      for f in files.values(): f.close()
    baseline_metrics=json.loads(metrics.read_text(encoding="utf8")); runtime=time.perf_counter()-start
    result={"status":"COMPLETE","experiment_type":"augmentation replay, not a replacement baseline inference run","source_video_sha256":EXPECTED_VIDEO_HASH,"baseline_run":str(base),"baseline_detection_sha256":_sha(det),"configuration":config,"configuration_sha256":cfg_hash,"model_hashes":model_hashes,"git_commit":_git(),"timestamp_utc":datetime.now(timezone.utc).isoformat(),"runtime_sec":runtime,"processed_frames":max_frames,"fps":max_frames/runtime,"experimental_observations":dict(counts),"pose_agreement":{"mean_normalized_distance":float(np.mean(agreement)) if agreement else None,"sampled_keypoint_pairs":len(agreement),"semantics":"inter-model agreement only; no pose ground truth was supplied"},"states_observed":dict(Counter(states)),"baseline_metrics_copied_without_reinterpretation":baseline_metrics.get("per_class"),"metrics_unavailable":{"hand_accuracy":"no hand labels","interaction_accuracy":"no reviewed relationship labels","behavioural_accuracy":"no behavioural temporal labels","precision_f1_map":"negative coverage is not exhaustive"},"limitations":["RTMPose and MediaPipe outputs are experimental and unvalidated on this camera.","No output is a cheating verdict or cheating probability.","3D checkpoint is represented but its camera-to-track adapter is pending validation; no orientation is claimed."]}
    (out/"experiment_manifest.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf8")
    (out/"COMPLETE.json").write_text(json.dumps({"status":"COMPLETE","manifest":"experiment_manifest.json"},indent=2),encoding="utf8")
    return str(out), str(out/"experiment_manifest.json")

"""
Master Multi-Dataset Ensemble Evaluation Runner for NETRA-4D.
Evaluates the 4 specialized YOLO26m models trained today:
  1. yolo26m_cheating.pt (Exam Cheating: students_cheating vs normal)
  2. yolo26m_chit.pt (Chit Paper Detector)
  3. yolo26m_phone.pt (Mobile Phone Detector)
  4. yolo26m_exam_multimodal.pt (Multi-Modal: chit-paper, hand-signs, head_movement, mobilephone)
on the remaining 4 benchmark videos: VIDEO_02, VIDEO_03, VIDEO_04, and VIDEO_05.
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VIDEOS = {
    "VIDEO_02": {
        "path": "outputs/unified_pipeline/VIDEO_02/.staging/04.CCTV Candidate Talking.mkv",
        "title": "Candidate Talking / Collusion",
        "primary_behavior": "TALKING / PEEKING"
    },
    "VIDEO_03": {
        "path": "outputs/unified_pipeline/VIDEO_03/.staging/01.Candidate was found using a mobile phone in the examination hall..mkv",
        "title": "Candidate Mobile Phone Usage (01)",
        "primary_behavior": "MOBILE PHONE"
    },
    "VIDEO_04": {
        "path": "outputs/unified_pipeline/VIDEO_04/.staging/02.Candidate was found using a mobile phone in the examination hall..mkv",
        "title": "Candidate Mobile Phone Usage (02)",
        "primary_behavior": "MOBILE PHONE"
    },
    "VIDEO_05": {
        "path": "outputs/unified_pipeline/VIDEO_05/.staging/03.CCTV Mobile Usage.mkv",
        "title": "CCTV Mobile Usage (03)",
        "primary_behavior": "MOBILE PHONE"
    }
}


def load_ensemble():
    print(f"[NETRA-4D] Loading 4 trained specialist models onto {DEVICE}...")
    pose = YOLO("models/weights/yolo26n-pose.pt").to(DEVICE)
    cheating = YOLO("models/weights/yolo26m_cheating.pt").to(DEVICE)
    chit = YOLO("models/weights/yolo26m_chit.pt").to(DEVICE)
    phone = YOLO("models/weights/yolo26m_phone.pt").to(DEVICE)
    multimodal = YOLO("models/weights/yolo26m_exam_multimodal.pt").to(DEVICE)
    print("  -> All 4 models loaded successfully!")
    return pose, cheating, chit, phone, multimodal


def evaluate_video(vid_id, info, pose_model, cheating_model, chit_model, phone_model, multimodal_model, sample_step=15):
    vid_path = info["path"]
    if not os.path.exists(vid_path):
        print(f"[ERROR] Video path not found: {vid_path}")
        return None

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps

    out_dir = ROOT / "outputs" / "ensemble_eval" / vid_id
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Evaluating {vid_id}: {info['title']}")
    print(f"Specs: {w_frame}x{h_frame} | {total_frames} frames @ {fps:.1f} fps ({duration_s:.1f}s)")
    print(f"Sampling step: every {sample_step} frames (~{sample_step/fps:.2f}s)")
    print(f"=======================================================")

    detections_timeline = []
    peak_events = []
    saved_annotated_count = 0

    frame_idx = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            ts = round(frame_idx / fps, 2)
            frame_events = []

            # 1. Pose: Detect candidates
            pose_res = pose_model.predict(frame, imgsz=1280, conf=0.15, verbose=False)[0]
            cand_boxes = pose_res.boxes.xyxy.cpu().numpy() if pose_res.boxes is not None else np.empty((0, 4))
            cand_kpts = pose_res.keypoints.data.cpu().numpy() if pose_res.keypoints is not None else np.empty((0, 17, 3))

            annotated_frame = frame.copy()

            for c_idx, (b, kp) in enumerate(zip(cand_boxes, cand_kpts)):
                bx1, by1, bx2, by2 = [int(v) for v in b]
                cand_roi = frame[max(0, by1):min(h_frame, by2), max(0, bx1):min(w_frame, bx2)]
                if cand_roi.size == 0:
                    continue

                # A. Cheating model on candidate
                c_res = cheating_model.predict(cand_roi, conf=0.20, verbose=False)[0]
                for cbx, cconf, ccls in zip(c_res.boxes.xyxy.cpu().numpy(), c_res.boxes.conf.cpu().numpy(), c_res.boxes.cls.int().cpu().numpy()):
                    if ccls == 1:  # students_cheating
                        gx1, gy1 = bx1 + int(cbx[0]), by1 + int(cbx[1])
                        gx2, gy2 = bx1 + int(cbx[2]), by1 + int(cbx[3])
                        frame_events.append({
                            "type": "STUDENTS_CHEATING",
                            "model": "yolo26m_cheating",
                            "confidence": float(cconf),
                            "candidate_idx": c_idx,
                            "bbox": [gx1, gy1, gx2, gy2]
                        })

                # B. Multi-modal model on candidate
                m_res = multimodal_model.predict(cand_roi, conf=0.18, verbose=False)[0]
                for mbx, mconf, mcls in zip(m_res.boxes.xyxy.cpu().numpy(), m_res.boxes.conf.cpu().numpy(), m_res.boxes.cls.int().cpu().numpy()):
                    cls_name = multimodal_model.names.get(int(mcls), "")
                    gx1, gy1 = bx1 + int(mbx[0]), by1 + int(mbx[1])
                    gx2, gy2 = bx1 + int(mbx[2]), by1 + int(mbx[3])
                    frame_events.append({
                        "type": cls_name.upper(),
                        "model": "yolo26m_exam_multimodal",
                        "confidence": float(mconf),
                        "candidate_idx": c_idx,
                        "bbox": [gx1, gy1, gx2, gy2]
                    })

                # C. Hand & Desk ROI: Check wrists for phone or chit
                lw, rw = kp[9], kp[10]
                wrists = [pt for pt in (lw, rw) if pt[2] > 0.2]
                if wrists:
                    wx = int(np.mean([pt[0] for pt in wrists]))
                    wy = int(np.mean([pt[1] for pt in wrists]))
                    hx1 = max(0, wx - 65)
                    hy1 = max(0, wy - 65)
                    hx2 = min(w_frame, wx + 65)
                    hy2 = min(h_frame, wy + 65)
                    hand_roi = frame[hy1:hy2, hx1:hx2]

                    if hand_roi.size > 0:
                        # Phone detector
                        p_res = phone_model.predict(hand_roi, conf=0.20, verbose=False)[0]
                        for pbx, pconf, pcls in zip(p_res.boxes.xyxy.cpu().numpy(), p_res.boxes.conf.cpu().numpy(), p_res.boxes.cls.int().cpu().numpy()):
                            if pcls == 1:  # phone
                                gx1, gy1 = hx1 + int(pbx[0]), hy1 + int(pbx[1])
                                gx2, gy2 = hx1 + int(pbx[2]), hy1 + int(pbx[3])
                                frame_events.append({
                                    "type": "MOBILE_PHONE",
                                    "model": "yolo26m_phone",
                                    "confidence": float(pconf),
                                    "candidate_idx": c_idx,
                                    "bbox": [gx1, gy1, gx2, gy2]
                                })

                        # Chit detector
                        ch_res = chit_model.predict(hand_roi, conf=0.20, verbose=False)[0]
                        for chbx, chconf in zip(ch_res.boxes.xyxy.cpu().numpy(), ch_res.boxes.conf.cpu().numpy()):
                            gx1, gy1 = hx1 + int(chbx[0]), hy1 + int(chbx[1])
                            gx2, gy2 = hx1 + int(chbx[2]), hy1 + int(chbx[3])
                            frame_events.append({
                                "type": "PAPER_CHIT",
                                "model": "yolo26m_chit",
                                "confidence": float(chconf),
                                "candidate_idx": c_idx,
                                "bbox": [gx1, gy1, gx2, gy2]
                            })

            # Record timeline entry
            if frame_events:
                detections_timeline.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": ts,
                    "event_count": len(frame_events),
                    "events": frame_events
                })

                # Sort events by confidence and draw top events
                for ev in frame_events:
                    bx1, gy1, bx2, gy2 = ev["bbox"]
                    etype = ev["type"]
                    conf = ev["confidence"]
                    
                    if "PHONE" in etype:
                        color = (0, 0, 255)  # Red
                    elif "CHIT" in etype:
                        color = (0, 128, 255)  # Orange
                    elif "CHEATING" in etype:
                        color = (0, 0, 220)
                    else:
                        color = (0, 255, 255)  # Yellow
                        
                    cv2.rectangle(annotated_frame, (bx1, gy1), (bx2, gy2), color, 2)
                    label = f"{etype} [{int(conf*100)}%]"
                    cv2.putText(annotated_frame, label, (bx1, max(15, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                # Save peak candidate frames
                top_conf = max(e["confidence"] for e in frame_events)
                if top_conf >= 0.35 and saved_annotated_count < 8:
                    out_img_path = out_dir / f"peak_evidence_f{frame_idx:04d}_{ts:.1f}s.jpg"
                    cv2.imwrite(str(out_img_path), annotated_frame)
                    peak_events.append({
                        "frame": frame_idx,
                        "timestamp": ts,
                        "image": str(out_img_path),
                        "top_event": max(frame_events, key=lambda x: x["confidence"])
                    })
                    saved_annotated_count += 1

        frame_idx += 1

    cap.release()
    elapsed = round(time.time() - t0, 1)

    # Summary analysis
    event_types = {}
    for entry in detections_timeline:
        for ev in entry["events"]:
            et = ev["type"]
            event_types[et] = event_types.get(et, 0) + 1

    summary = {
        "asset_id": vid_id,
        "title": info["title"],
        "primary_behavior": info["primary_behavior"],
        "duration_sec": duration_s,
        "total_frames": total_frames,
        "sampled_frames": frame_idx // sample_step,
        "elapsed_time_sec": elapsed,
        "fps_processing": round((frame_idx // sample_step) / elapsed, 1) if elapsed > 0 else 0,
        "total_cheat_detections": sum(event_types.values()),
        "detections_by_type": event_types,
        "peak_evidence_images": [p["image"] for p in peak_events]
    }

    # Save JSON summary
    json_path = out_dir / "ensemble_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": summary,
            "peak_events": peak_events,
            "timeline": detections_timeline
        }, f, indent=2)

    print(f"\n[DONE] Evaluated {vid_id} in {elapsed}s!")
    print(f"  -> Total Cheat Detections: {sum(event_types.values())}")
    print(f"  -> Detections Breakdown: {event_types}")
    print(f"  -> Peak Evidence Images Saved: {len(peak_events)}")
    print(f"  -> JSON: {json_path}")
    return summary


def main():
    pose, cheating, chit, phone, multimodal = load_ensemble()

    master_results = {}
    for vid_id in ["VIDEO_02", "VIDEO_03", "VIDEO_04", "VIDEO_05"]:
        info = VIDEOS[vid_id]
        res = evaluate_video(
            vid_id=vid_id,
            info=info,
            pose_model=pose,
            cheating_model=cheating,
            chit_model=chit,
            phone_model=phone,
            multimodal_model=multimodal,
            sample_step=15
        )
        if res:
            master_results[vid_id] = res

    # Master report
    report_path = ROOT / "outputs" / "ensemble_eval" / "master_ensemble_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(master_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"[ALL COMPLETE] Master evaluation report saved to: {report_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()

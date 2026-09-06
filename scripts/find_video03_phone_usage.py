"""
Investigate exact phone usage in VIDEO_03.
Finds when 'the guy' uses the phone, crops evidence, and logs timeline.
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("c:/Users/bhavv/OneDrive/Desktop/Dravvya(4th Sem)/Hackathons/NETRA-4D")
vid_path = str(ROOT / "outputs/unified_pipeline/VIDEO_03/.staging/01.Candidate was found using a mobile phone in the examination hall..mkv")
out_dir = ROOT / "outputs/investigation_video03"
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(vid_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
duration = total_frames / fps

print(f"VIDEO_03: {total_frames} frames, {fps} fps, {duration:.1f}s")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
phone_model = YOLO(str(ROOT / "models/weights/yolo26m_phone.pt")).to(device)
cheating_model = YOLO(str(ROOT / "models/weights/yolo26m_cheating.pt")).to(device)
multimodal_model = YOLO(str(ROOT / "models/weights/yolo26m_exam_multimodal.pt")).to(device)

evidence_found = []

# Scan every 25 frames (1.0 sec)
for fno in range(0, total_frames, 25):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        break
    sec = fno / fps
    h, w = frame.shape[:2]

    # Guy in the foreground (x > 600, y > 400)
    guy_roi = frame[int(h * 0.45):h, int(w * 0.55):w]

    # Run phone detector on guy ROI
    p_res = phone_model.predict(guy_roi, conf=0.20, verbose=False)[0]
    
    # Run multimodal on guy ROI
    m_res = multimodal_model.predict(guy_roi, conf=0.20, verbose=False)[0]

    detections = []
    if p_res.boxes is not None and len(p_res.boxes) > 0:
        for box, conf, cls in zip(p_res.boxes.xyxy.cpu().numpy(), p_res.boxes.conf.cpu().numpy(), p_res.boxes.cls.int().cpu().numpy()):
            if cls == 1:  # phone
                gx1 = int(w * 0.55) + int(box[0])
                gy1 = int(h * 0.45) + int(box[1])
                gx2 = int(w * 0.55) + int(box[2])
                gy2 = int(h * 0.45) + int(box[3])
                detections.append(("PHONE_SPEC", float(conf), [gx1, gy1, gx2, gy2]))

    if m_res.boxes is not None and len(m_res.boxes) > 0:
        for box, conf, cls in zip(m_res.boxes.xyxy.cpu().numpy(), m_res.boxes.conf.cpu().numpy(), m_res.boxes.cls.int().cpu().numpy()):
            name = multimodal_model.names.get(int(cls), "")
            if "phone" in name.lower():
                gx1 = int(w * 0.55) + int(box[0])
                gy1 = int(h * 0.45) + int(box[1])
                gx2 = int(w * 0.55) + int(box[2])
                gy2 = int(h * 0.45) + int(box[3])
                detections.append(("PHONE_MM", float(conf), [gx1, gy1, gx2, gy2]))

    if detections:
        top_det = max(detections, key=lambda x: x[1])
        evidence_found.append({
            "frame": fno,
            "sec": round(sec, 1),
            "type": top_det[0],
            "conf": round(top_det[1], 3),
            "box": top_det[2]
        })
        annotated = frame.copy()
        cv2.rectangle(annotated, (top_det[2][0], top_det[2][1]), (top_det[2][2], top_det[2][3]), (0, 0, 255), 2)
        cv2.putText(annotated, f"{top_det[0]} [{int(top_det[1]*100)}%]", 
                    (top_det[2][0], max(20, top_det[2][1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"guy_phone_f{fno:04d}_{sec:.1f}s.jpg"), annotated)

cap.release()
print(f"Total detections of guy using phone: {len(evidence_found)}")
for ev in evidence_found:
    print(f"  Frame {ev['frame']} ({ev['sec']}s): {ev['type']} Conf={ev['conf']}")

# Also extract candidate thumbnails every 10 seconds to visually inspect what the guy is doing
cap = cv2.VideoCapture(vid_path)
for sec in range(0, int(duration), 10):
    fno = int(sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    guy_crop = frame[int(h * 0.45):h, int(w * 0.55):w]
    cv2.imwrite(str(out_dir / f"timeline_guy_{sec:03d}s.jpg"), guy_crop)
cap.release()
print("[DONE] Extracted timeline samples.")

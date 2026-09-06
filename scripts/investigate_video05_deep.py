"""
Deep Scan of VIDEO_05 to locate exact moments when candidate at Desk 61
removes the phone from under the desk and holds it in his hands.
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("c:/Users/bhavv/OneDrive/Desktop/Dravvya(4th Sem)/Hackathons/NETRA-4D")
vid_path = str(ROOT / "outputs/unified_pipeline/VIDEO_05/.staging/03.CCTV Mobile Usage.mkv")
out_dir = ROOT / "outputs/investigation_video05"
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(vid_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
duration = total_frames / fps
print(f"VIDEO_05: {total_frames} frames, {fps:.2f} fps, {duration:.1f}s")

# Extract crops of Desk 61 every 5 seconds across the entire 281s
# In VIDEO_05: Desk 61 is on the left (x: 0..450, y: 250..720)
for sec in range(0, int(duration), 5):
    fno = int(sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    # Desk 61 candidate crop
    d61_crop = frame[int(h*0.3):h, 0:int(w*0.4)]
    cv2.imwrite(str(out_dir / f"desk61_{sec:03d}s.jpg"), d61_crop)

cap.release()
print("[DONE] Extracted Desk 61 crops every 5 seconds.")

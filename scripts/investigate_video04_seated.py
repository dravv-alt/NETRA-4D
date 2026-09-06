"""
Investigate seated candidates in VIDEO_04.
Only monitor students when they are seated in front of their desks.
Extract full frames and seated crops.
"""
import os
import cv2
from pathlib import Path

ROOT = Path("c:/Users/bhavv/OneDrive/Desktop/Dravvya(4th Sem)/Hackathons/NETRA-4D")
vid_path = str(ROOT / "outputs/unified_pipeline/VIDEO_04/.staging/02.Candidate was found using a mobile phone in the examination hall..mkv")
out_dir = ROOT / "outputs/investigation_video04"
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(vid_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
duration = total_frames / fps
print(f"VIDEO_04: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

# Extract full frames every 15 seconds across the 212s
for sec in range(0, int(duration), 15):
    fno = int(sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(str(out_dir / f"full_{sec:03d}s.jpg"), frame)

cap.release()
print("[DONE] Extracted full frames for VIDEO_04.")

import cv2
import numpy as np
from ultralytics import YOLO

video_file = 'outputs/unified_pipeline/VIDEO_01/.staging/Seat No. 12 was seen taking a piece of paper from the desk.mkv'
cap = cv2.VideoCapture(video_file)
pose_model = YOLO('models/weights/yolo26n-pose.pt')

SEAT_ANCHORS = {
    'Desk_12': (0.129, 0.417),  # Striped shirt guy (cheating student)
    'Desk_03': (0.532, 0.511),  # Light-blue shirt guy (Desk 3)
    'Desk_10': (0.609, 0.372),  # Guy behind Desk 3
    'Desk_09': (0.402, 0.183),  # Desk 9
    'Desk_01': (0.116, 0.712),  # Front bottom-left
}

frames_to_test = [100, 500, 850, 960, 970, 990, 1050, 1122, 1200, 1300, 1400, 1800]
print("Testing spatial anchor matching across full video...")

for fno in frames_to_test:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        break
    
    pose_res = pose_model.predict(frame, imgsz=1280, conf=0.15, verbose=False)[0]
    boxes = pose_res.boxes.xyxy.cpu().numpy()
    
    matched = {}
    for b in boxes:
        cx = (b[0] + b[2]) / 2.0 / 1280.0
        cy = (b[1] + b[3]) / 2.0 / 720.0
        
        best_seat = None
        min_d = 0.16
        for seat, (scx, scy) in SEAT_ANCHORS.items():
            d = np.hypot(cx - scx, cy - scy)
            if d < min_d:
                min_d = d
                best_seat = seat
                
        if best_seat and best_seat not in matched:
            matched[best_seat] = (min_d, [int(v) for v in b])
            
    d12_status = "MATCHED" if "Desk_12" in matched else "MISSED"
    d3_status = "MATCHED" if "Desk_03" in matched else "MISSED"
    d12_d = matched["Desk_12"][0] if "Desk_12" in matched else 0.0
    d3_d = matched["Desk_03"][0] if "Desk_03" in matched else 0.0
    print(f"Frame {fno:04d}: Desk_12={d12_status} (dist={d12_d:.3f}) | Desk_03={d3_status} (dist={d3_d:.3f})")

cap.release()

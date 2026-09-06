import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from drishti.spatial.cctv_calibrator import CCTVReverseCalibrator

def main():
    video_path = ROOT / "outputs" / "unified_pipeline" / "VIDEO_02" / ".staging" / "04.CCTV Candidate Talking.mkv"
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame!")
        return
    calibrator = CCTVReverseCalibrator()
    res = calibrator.calibrate_from_frame(frame, camera_id="CAM-02")
    print("Dynamic CCTV Calibration Result:")
    print(f"  Anchors detected: {res['detected_anchor_count']}")
    print(f"  Terminals generated: {res['detected_terminal_count']}")
    for t in res['terminals']:
        sp = t['spatialPosition']
        print(f"    Seat {t['id']}: candidate={t['candidateId']} ({sp['x']}%, {sp['y']}%) facing {sp['facing']}")

if __name__ == '__main__':
    main()

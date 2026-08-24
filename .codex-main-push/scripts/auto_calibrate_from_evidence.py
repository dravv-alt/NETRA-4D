"""Create a labelled AUTOMATIC CBT map from existing real detection JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.cbt.scene import AutomaticWorkstationCalibrator
from drishti.schemas import ObjectObservation, PersonObservation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path, help="Completed VIDEO_XX evidence directory")
    parser.add_argument("--camera-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-sec", type=float, default=30.0)
    parser.add_argument("--minimum-observations", type=int, default=12)
    args = parser.parse_args()
    video_dir = args.video_dir.resolve()
    video_id = video_dir.name
    detections = video_dir / "detections.jsonl"
    summary_path = video_dir / "summary.json"
    if not detections.is_file() or not summary_path.is_file():
        raise SystemExit(f"Missing detections.jsonl or summary.json in {video_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video = summary.get("video", summary)
    width, height = int(video["width"]), int(video["height"])
    calibrator = AutomaticWorkstationCalibrator(
        args.camera_id or video_id, video_id, width, height,
        warmup_sec=0.0, minimum_observations=args.minimum_observations,
    )
    for line in detections.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        timestamp = float(payload.get("timestamp_sec", 0.0))
        if timestamp > args.maximum_sec:
            break
        if not payload.get("decode_ok", True):
            continue
        calibrator.observe(
            timestamp,
            [PersonObservation.model_validate(item) for item in payload.get("people", [])],
            [ObjectObservation.model_validate(item) for item in payload.get("objects", [])],
        )
    scene = calibrator.finalize()
    if scene is None:
        raise SystemExit(
            f"Automatic calibration found insufficient stable person/monitor pairs in {video_id}; "
            "the runtime will safely use baseline reasoning unless a human map is supplied."
        )
    output = (ROOT / (args.output or Path(f"drishti4d/configs/workstations/{video_id}.yaml"))).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(scene.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "calibration_status": scene.calibration_status,
        "workstations": len(scene.workstations), "people_clusters": len(calibrator.people),
        "monitor_clusters": len(calibrator.monitors),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

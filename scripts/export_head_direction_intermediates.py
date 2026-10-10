"""Replay and export the exact head-direction proxy inputs for a completed run.

This is an audit tool, not a model-training or metric-changing tool.  It
replays the same YOLO26Adapter + BiomechanicalFeatureExtractor calculation
against stored detections and compares the reconstructed direction with the
direction recorded by the run before exporting CVAT-compatible samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_cvat_video import iou, digest
from src.drishti.engine.features import BiomechanicalFeatureExtractor
from src.drishti.engine.observation import YOLO26Adapter


LABEL_MAP = {"LEFT": "left", "RIGHT": "right", "FORWARD": "front", "UP": "up", "DOWN": "down"}
COCO_NAMES = YOLO26Adapter.COCO_KEYPOINT_NAMES


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keypoints_as_array(person: dict, width: int, height: int) -> np.ndarray:
    """Restore the pixel-space 17x3 array consumed by YOLO26Adapter."""
    result = []
    for name in COCO_NAMES:
        point = person.get("keypoints_2d", {}).get(name)
        if point is None:
            result.append([0.0, 0.0, 0.0])
        else:
            result.append([point["x"] * width, point["y"] * height, point["confidence"]])
    return np.asarray(result, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--reasoning", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    truth = defaultdict(list)
    for row in read_rows(args.labels):
        label = row.get("attributes", {}).get("head_dir")
        if row.get("class") == "Student" and not row.get("outside") and row.get("geometry_valid") and label in LABEL_MAP.values():
            truth[row["frame_index"]].append(row)

    reasoning = {row["frame_index"]: row for row in read_rows(args.reasoning)}
    extractor = BiomechanicalFeatureExtractor(smoothing_tau=0.12)
    exported, direction_mismatches, missing_proxy = [], 0, 0

    # Replay all stored candidates in recorded candidate order.  Candidate IDs,
    # timestamps, normalized keypoints, and smoothing parameters match the run.
    for detection_frame in read_rows(args.detections):
        frame = detection_frame["frame_index"]
        recorded = reasoning.get(frame, {}).get("candidates", [])
        people_by_track = {person["entity_local_id"]: person for person in detection_frame.get("people", [])}
        replay = {}
        for candidate in recorded:
            person = people_by_track.get(candidate["tracker_id"])
            if person is None:
                continue
            obs = YOLO26Adapter.adapt(
                frame_idx=frame,
                timestamp_sec=detection_frame["timestamp_sec"],
                frame_shape=(args.height, args.width),
                tracked_box=person["bbox_xyxy_px"],
                track_id=person["entity_local_id"],
                keypoints_raw=keypoints_as_array(person, args.width, args.height),
                det_confidence=person["detection_confidence"],
                detected_objects=[],
            )
            replay[candidate["tracker_id"]] = (candidate, person, extractor.extract(candidate["candidate_id"], obs))

        used = set()
        for person in sorted(detection_frame.get("people", []), key=lambda item: item["detection_confidence"], reverse=True):
            options = [(iou(person["bbox_xyxy_px"], row["bbox_xyxy_px"]), index)
                       for index, row in enumerate(truth[frame]) if index not in used]
            score, index = max(options, default=(0.0, -1))
            if score < 0.5:
                continue
            used.add(index)
            item = replay.get(person["entity_local_id"])
            if item is None:
                missing_proxy += 1
                continue
            candidate, raw_person, features = item
            recorded_direction = candidate.get("head_direction_proxy")
            if recorded_direction not in LABEL_MAP:
                missing_proxy += 1
                continue
            reconstructed_direction = features.head.direction.value
            if reconstructed_direction != recorded_direction:
                direction_mismatches += 1
            kp = raw_person["keypoints_2d"]
            nose, left_eye, right_eye = (kp.get(name) for name in ("nose", "left_eye", "right_eye"))
            # The vector is in normalized image coordinates and is intentionally
            # pre-EMA, exposing the actual geometry behind raw_yaw/raw_pitch.
            eye_mid = [(left_eye["x"] + right_eye["x"]) / 2, (left_eye["y"] + right_eye["y"]) / 2]
            vector = [nose["x"] - eye_mid[0], nose["y"] - eye_mid[1]]
            eye_span_x = abs(right_eye["x"] - left_eye["x"])
            raw_yaw = vector[0] / (eye_span_x / 2) if eye_span_x >= 1e-4 else None
            raw_pitch = vector[1] / eye_span_x if eye_span_x >= 1e-4 else None
            exported.append({
                "frame": frame,
                "timestamp_sec": detection_frame["timestamp_sec"],
                "person_id": person["entity_local_id"],
                "candidate_id": candidate["candidate_id"],
                "cvat_head_dir": truth[frame][index]["attributes"]["head_dir"],
                "person_iou_with_cvat": score,
                "raw_keypoints_normalized": {name: kp.get(name) for name in COCO_NAMES},
                "derived_head_vector_normalized": {"nose_minus_eye_mid": vector, "eye_mid": eye_mid, "eye_span_x": eye_span_x},
                "raw_yaw_proxy": raw_yaw,
                "smoothed_yaw_proxy": features.head.yaw_proxy,
                "pitch_proxy": features.head.pitch_proxy,
                "predicted_direction": recorded_direction,
                "reconstructed_direction": reconstructed_direction,
                "replay_matches_recorded_direction": reconstructed_direction == recorded_direction,
                "face_keypoint_min_confidence": features.head.confidence,
                "head_proxy_valid": features.head.is_valid,
            })

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "head_direction_intermediates.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in exported), encoding="utf-8")
    manifest = {
        "status": "COMPLETE",
        "samples_exported": len(exported),
        "replay_direction_mismatches": direction_mismatches,
        "missing_or_unsupported_proxy": missing_proxy,
        "labels_sha256": digest(args.labels),
        "detections_sha256": digest(args.detections),
        "reasoning_sha256": digest(args.reasoning),
        "implementation": {
            "keypoints": ["nose", "left_eye", "right_eye"],
            "raw_yaw": "(nose_x - mean(left_eye_x,right_eye_x)) / (abs(right_eye_x-left_eye_x)/2)",
            "pitch": "(nose_y - mean(left_eye_y,right_eye_y)) / abs(right_eye_x-left_eye_x)",
            "smoothing": "Keypoint EMA tau=0.12 sec; yaw EMA tau=0.15 sec",
            "thresholds": {"RIGHT": "> 0.40", "LEFT": "< -0.40", "DOWN": "> 0.85 if yaw is in front band", "UP": "< -0.40 if yaw is in front band", "FORWARD": "otherwise"},
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

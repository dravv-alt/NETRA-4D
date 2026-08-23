"""Run real pose and object models on the five user-supplied Drive videos.

No detections, boxes, labels, events, benchmark scores, or hardware metrics are
fabricated. Drive videos are staged one at a time outside the repository and
deleted immediately after processing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset
from drishti.providers.object_provider import ObjectDisambiguationProvider
from drishti.providers.yolo26_provider import COCO_KEYPOINT_NAMES, YOLO26PoseProvider


SKELETON = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str, require_cuda: bool) -> str:
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA was required but is unavailable. The run was stopped; it was not silently "
            "executed on CPU or replaced with simulated detections."
        )
    if requested == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        raise RuntimeError(f"Requested {requested}, but torch.cuda.is_available() is False.")
    return requested


def sample_indices(frame_count: int, max_frames: int) -> List[int]:
    if frame_count <= 0:
        return list(range(max_frames))
    count = min(max_frames, frame_count)
    if count == 1:
        return [frame_count // 2]
    return sorted({int(value) for value in np.linspace(0, frame_count - 1, count)})


def draw_verified_overlay(frame, people, objects):
    output = frame.copy()
    height, width = output.shape[:2]
    for person in people:
        x1, y1, x2, y2 = (int(value) for value in person.bbox_xyxy_px)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            output,
            f"person {person.detection_confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
        points = person.keypoints_2d or {}
        for left_name, right_name in SKELETON:
            left, right = points.get(left_name), points.get(right_name)
            if left and right and left.confidence >= 0.25 and right.confidence >= 0.25:
                cv2.line(
                    output,
                    (int(left.x * width), int(left.y * height)),
                    (int(right.x * width), int(right.y * height)),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        for point in points.values():
            if point.confidence >= 0.25:
                cv2.circle(output, (int(point.x * width), int(point.y * height)), 3, (0, 140, 255), -1)

    for detected in objects:
        x1, y1, x2, y2 = (int(value) for value in detected.bbox_xyxy_px)
        raw_class = detected.metadata.get("raw_model_class", detected.object_class.value)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(
            output,
            f"{raw_class} {detected.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 120, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def model_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model weight not found: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def run(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device, args.require_cuda)
    pose_path = (ROOT / args.pose_model).resolve()
    object_path = (ROOT / args.object_model).resolve()
    manifest_path = (ROOT / args.manifest).resolve()
    output_root = (ROOT / args.output_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    preview_dir = run_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=False)

    pose = YOLO26PoseProvider(model_name=str(pose_path), device=device, conf_thresh=args.confidence)
    objects = ObjectDisambiguationProvider(model_name=str(object_path), device=device, conf_thresh=args.confidence)
    pose.start()
    objects.start()

    selected = set(args.asset or [])
    assets = [asset for asset in load_drive_manifest(manifest_path) if not selected or asset.asset_id in selected]
    if selected.difference({asset.asset_id for asset in assets}):
        raise ValueError(f"Unknown asset IDs: {sorted(selected.difference({a.asset_id for a in assets}))}")

    run_record: Dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_kind": "verified_real_model_inference",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "claims": {
            "pose_boxes_and_keypoints": "model output",
            "object_boxes_and_classes": "model output",
            "malpractice_classification": "not evaluated",
            "ground_truth_metrics": "not available",
            "smplx_3d": "not executed; no configured live Multi-HMR image model",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "requested_device": args.device,
            "actual_device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": {
            "pose": model_record(pose_path),
            "object": model_record(object_path),
        },
        "dataset_manifest": str(manifest_path.relative_to(ROOT)),
        "assets": [],
    }

    try:
        for asset in assets:
            with materialize_drive_asset(asset) as (video_path, source_provenance):
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"OpenCV could not decode {asset.asset_id}: {video_path}")
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                asset_result: Dict[str, Any] = {
                    "source": source_provenance,
                    "video": {
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "frame_count": frame_count,
                        "duration_sec": frame_count / fps if fps > 0 else None,
                    },
                    "sampled_frames": [],
                }

                for frame_index in sample_indices(frame_count, args.max_frames):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Failed to decode {asset.asset_id} frame {frame_index}")
                    timestamp_sec = frame_index / fps if fps > 0 else 0.0

                    started = perf_counter()
                    people, _ = pose.process_frame(
                        frame_index, timestamp_sec, frame, camera_id=asset.asset_id, video_id=asset.filename
                    )
                    _, detected_objects = objects.process_frame(
                        frame_index,
                        timestamp_sec,
                        frame,
                        camera_id=asset.asset_id,
                        video_id=asset.filename,
                        person_obs_list=people,
                    )
                    latency_ms = (perf_counter() - started) * 1000.0

                    preview_name = f"{asset.asset_id}_frame_{frame_index:06d}.jpg"
                    cv2.imwrite(
                        str(preview_dir / preview_name),
                        draw_verified_overlay(frame, people, detected_objects),
                    )
                    asset_result["sampled_frames"].append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": timestamp_sec,
                            "combined_inference_ms": latency_ms,
                            "people": [item.model_dump(mode="json") for item in people],
                            "objects": [item.model_dump(mode="json") for item in detected_objects],
                            "preview": f"previews/{preview_name}",
                        }
                    )
                capture.release()
                run_record["assets"].append(asset_result)
    finally:
        pose.close()
        objects.close()

    run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_record["summary"] = {
        "videos_processed": len(run_record["assets"]),
        "frames_processed": sum(len(item["sampled_frames"]) for item in run_record["assets"]),
        "person_detections": sum(
            len(frame["people"])
            for item in run_record["assets"]
            for frame in item["sampled_frames"]
        ),
        "object_detections": sum(
            len(frame["objects"])
            for item in run_record["assets"]
            for frame in item["sampled_frames"]
        ),
    }
    result_path = run_dir / "run.json"
    result_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(result_path), "summary": run_record["summary"], "device": device}, indent=2))
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="drishti4d/configs/datasets.json")
    parser.add_argument("--pose-model", default="models/weights/yolo26n-pose.pt")
    parser.add_argument("--object-model", default="models/weights/yolo26n.pt")
    parser.add_argument("--output-dir", default="outputs/verified")
    parser.add_argument("--asset", action="append", help="Process only this asset ID; repeatable")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    run(parsed)

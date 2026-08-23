"""Add temporal baseline review overlays to an already verified NETRA-4D run.

This is a compute-efficient evidence replay: it consumes the original run's
per-frame model JSON and privacy-safe annotated MP4.  It does not rerun YOLO,
invent detections, or read unblurred source frames.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drishti.schemas import ObjectObservation, PersonObservation
from drishti.temporal import BaselineTemporalEngine, draw_temporal_review_events
from scripts.run_verified_inference import file_sha256


def json_line(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


def replay_video(source_dir: Path, destination_dir: Path, maximum_frames: int = 0) -> dict:
    complete = source_dir / "COMPLETE.json"
    if not complete.is_file():
        raise RuntimeError(f"Source is not complete: {source_dir}")
    video_id = source_dir.name
    source_video = source_dir / f"{video_id}_annotated.mp4"
    detections_path = source_dir / "detections.jsonl"
    if not source_video.is_file() or not detections_path.is_file():
        raise FileNotFoundError(f"Missing source video or detections in {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=False)
    incomplete = destination_dir / "INCOMPLETE.txt"
    incomplete.write_text("Evidence replay is incomplete.\n", encoding="utf-8")
    output_video = destination_dir / f"{video_id}_annotated.mp4"
    events_path = destination_dir / "review_events.jsonl"
    event_frames_dir = destination_dir / "review_event_frames"

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frames = min(source_frames, maximum_frames) if maximum_frames else source_frames
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create {output_video}")

    engine = BaselineTemporalEngine()
    emitted = Counter()
    frame_count = 0
    try:
        with (
            detections_path.open("r", encoding="utf-8") as detection_file,
            events_path.open("w", encoding="utf-8") as event_file,
        ):
            while frame_count < target_frames:
                ok, frame = capture.read()
                evidence_line = detection_file.readline()
                if not ok or frame is None or not evidence_line:
                    raise RuntimeError(f"Video/evidence ended at frame {frame_count} in {video_id}")
                evidence = json.loads(evidence_line)
                if int(evidence["frame_index"]) != frame_count:
                    raise RuntimeError(f"Evidence frame mismatch in {video_id}: {evidence['frame_index']} != {frame_count}")
                people = [PersonObservation.model_validate(item) for item in evidence.get("people", [])]
                objects = [ObjectObservation.model_validate(item) for item in evidence.get("objects", [])]
                privacy = [
                    SimpleNamespace(**item)
                    for item in evidence.get("privacy", {}).get("annotations", [])
                ]
                events = engine.update(float(evidence["timestamp_sec"]), people, objects, privacy)
                annotated = draw_temporal_review_events(frame, people, events)
                new_events = [event for event in events if event.is_new]
                for event_index, event in enumerate(new_events):
                    event_file.write(json_line({"frame_index": frame_count, **event.to_dict()}))
                    emitted[event.event_type] += 1
                    event_frames_dir.mkdir(exist_ok=True)
                    event_name = f"frame_{frame_count:06d}_{event_index:02d}_{event.event_type.lower()}.jpg"
                    if not cv2.imwrite(str(event_frames_dir / event_name), annotated):
                        raise RuntimeError(f"Cannot write {event_name}")
                writer.write(annotated)
                frame_count += 1
    finally:
        writer.release()
        capture.release()

    summary = {
        "video_id": video_id,
        "processing": "temporal evidence replay; YOLO was not rerun",
        "source_video": str(source_video),
        "source_video_sha256": file_sha256(source_video),
        "source_detections": str(detections_path),
        "source_detections_sha256": file_sha256(detections_path),
        "frames": frame_count,
        "source_frames": source_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "event_counts": dict(sorted(emitted.items())),
        "output_video": output_video.name,
        "output_video_bytes": output_video.stat().st_size,
        "output_video_sha256": file_sha256(output_video),
    }
    (destination_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    incomplete.unlink()
    (destination_dir / "COMPLETE.json").write_text(
        json.dumps({"complete": True, "verified_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_event_runs"))
    parser.add_argument("--run-id", default="baseline_events_v1")
    parser.add_argument("--asset", action="append")
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    if not (source_run / "run.json").is_file():
        raise SystemExit(f"Missing source run.json: {source_run}")
    destination_run = (ROOT / args.output_dir / args.run_id).resolve()
    destination_run.mkdir(parents=True, exist_ok=False)
    selected = set(args.asset or [])
    source_videos = sorted(path for path in source_run.glob("VIDEO_*" ) if path.is_dir())
    if selected:
        source_videos = [path for path in source_videos if path.name in selected]
    if not source_videos:
        raise SystemExit("No source VIDEO_XX directories selected")

    summaries = []
    for source_dir in source_videos:
        print(f"Replaying {source_dir.name}", flush=True)
        summaries.append(replay_video(
            source_dir, destination_run / source_dir.name, args.max_frames_per_video
        ))
    record = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "run_kind": "temporal_baseline_evidence_replay",
        "source_run": str(source_run),
        "models_rerun": False,
        "fixed_seat_boxes_used": False,
        "privacy": "source annotated video was already privacy masked",
        "assets": summaries,
        "summary": {
            "videos": len(summaries),
            "frames": sum(item["frames"] for item in summaries),
            "events": sum(sum(item["event_counts"].values()) for item in summaries),
        },
    }
    (destination_run / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

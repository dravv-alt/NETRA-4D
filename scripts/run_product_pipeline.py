"""The only supported NETRA batch/live inference entry point.

It runs approved registry models, produces operator-review video and raw JSONL,
persists human-review events, and writes a release manifest.  It intentionally
does not output cheating verdicts or synthetic performance metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from drishti.event_repository import EventRepository
from drishti.data.video_policy import bounded_video_source
from drishti.events import EvidenceKind, EventLifecycle, EventRecord
from drishti.privacy import draw_post_blur_candidate_annotations
from drishti.render.operator_overlay import draw_operator_overlay
from drishti.release import CanonicalInferenceService, ModelRegistry
from drishti.temporal import draw_temporal_review_events
from scripts.run_verified_inference import draw_verified_overlay


def resolve_source(value: str):
    return int(value) if value.isdigit() else value


def resolve_device(requested: str, require_cuda: bool) -> str:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if (require_cuda or requested.startswith("cuda")) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable; no CPU fallback was used.")
    return requested


def publish_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def event_record(event, *, run_id: str, camera_id: str, video_id: str, provenance: dict) -> EventRecord:
    stable = uuid5(NAMESPACE_URL, f"netra4d/{run_id}/{event.event_type}/{event.candidate_ids}/{event.started_at_sec:.3f}")
    kind = EvidenceKind(event.evidence_kind)
    return EventRecord(
        event_id=str(stable), run_id=run_id, camera_id=camera_id, video_id=video_id,
        event_type=event.event_type, label=event.label, lifecycle=EventLifecycle.CLOSED,
        evidence_kind=kind, evidence_score=event.evidence_score, candidate_ids=list(event.candidate_ids),
        started_at_sec=event.started_at_sec, ended_at_sec=event.timestamp_sec,
        duration_sec=event.duration_sec, source_signals=list(event.source_signals),
        model_provenance=provenance,
        reasoning_provenance={"engine": "BaselineTemporalEngine", "output_semantics": "review evidence, not cheating probability"},
        review_priority=event.evidence_score, uncertainty=max(0.0, 1.0 - event.evidence_score), details=event.details,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local video path, webcam index, or RTSP URL")
    parser.add_argument("--camera-id", default="CAM01")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--output-dir", default="outputs/product_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--database", default="outputs/netra4d_events.sqlite")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--phone-review-confidence", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means read until source ends or interrupted")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Use 'none' only for explicitly frame-local output")
    parser.add_argument("--live-publish-dir", default=None, help="Optional directory for atomically published latest.jpg/latest.json")
    parser.add_argument("--monitoring-control-file", default=None, help="When absent, capture continues but inference is paused")
    parser.add_argument('--phone-role', default=None, help='Optional phone-only registry role using full-frame plus detector person crops')
    parser.add_argument('--allow-experimental-models', action='store_true', help='Explicitly enable an unapproved phone role for an experimental evidence run')
    parser.add_argument('--cpu-threads', type=int, default=4, help='Torch CPU threads for preprocessing and model operations')
    parser.add_argument('--phone-roi-hold-seconds',type=float,default=1.,help='Detector person crop retention, seconds; never retains phone detections')
    parser.add_argument('--enable-preprocessing', action='store_true', help='Apply configured adaptive enhancement to inference only; evidence rendering remains raw')
    parser.add_argument('--preprocessing-config', default=None, help='Optional adaptive-preprocessing YAML configuration')
    parser.add_argument('--overlay-style', choices=('operator', 'debug'), default='operator', help='Operator is compact review evidence; debug draws all raw model primitives.')
    parser.add_argument('--privacy-mask', action='store_true', help='Apply face masking to the rendered output. Off by default for local operator review.')
    args = parser.parse_args()
    args.source, args.max_frames = bounded_video_source(args.source, args.max_frames, args.video_id)
    if args.cpu_threads <= 0:
        raise ValueError('cpu threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    if not 0 <= args.confidence <= 1 or not 0 <= args.phone_review_confidence <= 1:
        raise ValueError("confidence values must be within [0, 1]")
    device = resolve_device(args.device, args.require_cuda)
    registry = ModelRegistry(ROOT / args.registry, ROOT)
    if args.phone_role:
        registry.get(args.phone_role, require_approved=not args.allow_experimental_models)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("product_%Y%m%dT%H%M%SZ")
    run_dir = (ROOT / args.output_dir / run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    video_id = args.video_id or Path(str(args.source)).stem or run_id
    capture = cv2.VideoCapture(resolve_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Source did not expose a valid frame geometry")
    writer = cv2.VideoWriter(str(run_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Unable to create annotated.mp4")
    service = CanonicalInferenceService(registry, device=device, tracker=None if args.tracker == "none" else args.tracker, confidence=args.confidence, phone_review_confidence=args.phone_review_confidence,phone_role=args.phone_role,allow_experimental_models=args.allow_experimental_models,phone_roi_hold_seconds=args.phone_roi_hold_seconds,enable_preprocessing=args.enable_preprocessing,preprocessing_config=args.preprocessing_config,enable_privacy_mask=args.privacy_mask)
    repository = EventRepository((ROOT / args.database).resolve())
    counters = {"frames": 0, "people": 0, "objects": 0, "events": 0, "new_events": 0}
    publish_dir = (ROOT / args.live_publish_dir).resolve() if args.live_publish_dir else None
    control_file = (ROOT / args.monitoring_control_file).resolve() if args.monitoring_control_file else None
    if publish_dir:
        publish_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    complete = False
    try:
        service.start()
        with (run_dir / "detections.jsonl").open("w", encoding="utf-8") as detections, (run_dir / "review_events.jsonl").open("w", encoding="utf-8") as events:
            while not args.max_frames or counters["frames"] < args.max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    complete = True
                    break
                index = counters["frames"]
                if control_file is not None and not control_file.is_file():
                    counters["frames"] += 1
                    continue
                frame_started = perf_counter()
                result = service.process(index, index / fps, frame, camera_id=args.camera_id, video_id=video_id)
                if args.overlay_style == 'debug':
                    rendered = draw_verified_overlay(result.privacy_frame, result.people, result.objects)
                    rendered = draw_post_blur_candidate_annotations(rendered, result.people, result.privacy_annotations)
                    rendered = draw_temporal_review_events(rendered, result.people, result.review_events)
                else:
                    rendered = draw_operator_overlay(result.privacy_frame, result.people, result.objects, result.review_events)
                writer.write(rendered)
                detections.write(json.dumps({"frame_index": index, "timestamp_sec": index / fps, "people": [row.model_dump(mode="json") for row in result.people], "objects": [row.model_dump(mode="json") for row in result.objects], "model_provenance": result.model_provenance, "quality_report": result.quality_report}, separators=(",", ":")) + "\n")
                for event in result.review_events:
                    record = event_record(event, run_id=run_id, camera_id=args.camera_id, video_id=video_id, provenance=result.model_provenance)
                    repository.upsert(record)
                    events.write(record.model_dump_json() + "\n")
                    counters["events"] += 1
                    counters["new_events"] += int(event.is_new)
                counters["people"] += len(result.people)
                counters["objects"] += len(result.objects)
                counters["frames"] += 1
                if publish_dir:
                    encoded_ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not encoded_ok:
                        raise RuntimeError("Unable to encode live annotated frame")
                    publish_atomic(publish_dir / "latest.jpg", encoded.tobytes())
                    publish_atomic(publish_dir / "latest.json", json.dumps({
                        "run_id": run_id, "frame_index": index, "timestamp_sec": index / fps,
                        "latency_ms": (perf_counter() - frame_started) * 1000.0,
                        "effective_fps": (counters["frames"] + 1) / max(perf_counter() - started, 0.001),
                        "people": [item.model_dump(mode="json") for item in result.people],
                        "objects": [item.model_dump(mode="json") for item in result.objects],
                        "active_review_events": [item.to_dict() for item in result.review_events],
                        "temporal_evidence": result.temporal_evidence,
                        "model_provenance": result.model_provenance,
                    }).encode("utf-8"))
    finally:
        capture.release(); writer.release(); service.close()
    completed = complete or args.max_frames > 0
    manifest = {"schema_version": "1.0", "run_id": run_id, "status": "COMPLETE" if completed else "INTERRUPTED", "bounded_run": bool(args.max_frames), "source": str(args.source), "camera_id": args.camera_id, "video_id": video_id, "fps": fps, "width": width, "height": height, "device": device, "elapsed_sec": perf_counter() - started, "counters": counters, "models": service.provenance(), "experimental_models_enabled":any(not m['release_approved'] for m in service.provenance().values()),"phone_inference_mode":"full_frame_plus_detector_person_crops" if args.phone_role else "baseline_coco","phone_confidence":args.confidence,"cpu_threads":args.cpu_threads, "claim": "review evidence only; no cheating verdict or quality metric is implied"}
    manifest['phone_roi_hold_seconds'] = args.phone_roi_hold_seconds if args.phone_role else 0.
    manifest['phone_retention_policy'] = 'Only detector crop proposals retained; each phone box requires fresh inference'
    manifest['preprocessing'] = service.preprocessing_summary()
    manifest['rendering'] = {'overlay_style': args.overlay_style, 'privacy_mask_enabled': args.privacy_mask}
    (run_dir / ("COMPLETE.json" if manifest["status"] == "COMPLETE" else "INCOMPLETE.json")).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

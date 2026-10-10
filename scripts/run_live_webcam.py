"""Bounded, privacy-safe live NETRA-4D webcam/RTSP inference runner."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import sys
import threading
from time import monotonic, perf_counter, sleep
from uuid import uuid4

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drishti.event_repository import EventRepository
from drishti.cbt import AutomaticCBTBehaviouralEngine, CBTBehaviouralReasoner, draw_cbt_context, load_scene_map
from drishti.events import EvidenceKind, EventLifecycle, EventRecord, SpecialistStatus
from drishti.privacy import apply_privacy_mask_and_collect_annotations, draw_post_blur_candidate_annotations
from drishti.preprocessing import PreprocessingPipeline
from drishti.providers.object_provider import ObjectDisambiguationProvider
from drishti.providers.yolo26_provider import YOLO26PoseProvider
from drishti.temporal import BaselineTemporalEngine, draw_temporal_review_events
from scripts.run_verified_inference import draw_verified_overlay, model_record, resolve_device


@dataclass
class FramePacket:
    index: int
    captured_at: float
    frame: np.ndarray


class BoundedCapture:
    def __init__(self, source, maximum_queue: int = 2, pace_file: bool = False):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open live source {source!r}")
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=maximum_queue)
        self.stopped = threading.Event()
        self.dropped = 0
        self.pace_file = pace_file
        self.frame_interval = 1.0 / (float(self.capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0)
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()

    def _read(self):
        index = 0
        while not self.stopped.is_set():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.stopped.set()
                break
            packet = FramePacket(index, monotonic(), frame)
            if self.pace_file:
                # Uploaded videos are an evidence source: back-pressure the decoder so
                # every source frame is processed, even when inference is slower than FPS.
                while not self.stopped.is_set():
                    try:
                        self.queue.put(packet, timeout=0.25)
                        break
                    except queue.Full:
                        continue
            else:
                # A camera/RTSP feed must stay live, so discard stale buffered frames.
                try:
                    self.queue.put_nowait(packet)
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
                    self.queue.put_nowait(packet)
            index += 1

    def get(self, timeout: float = 1.0) -> FramePacket | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        self.capture.release()


class BrowserFrameCapture:
    """Consume the newest browser-published JPEG without allowing latency buildup."""

    def __init__(self, directory: Path, width: int, height: int, fps: float, maximum_queue: int = 2):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps or 5.0
        self.frame_interval = 1.0 / self.fps
        self.capture = self
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=maximum_queue)
        self.stopped = threading.Event()
        self.dropped = 0
        self.thread = threading.Thread(target=self._read, daemon=True)

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if property_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def start(self):
        self.thread.start()

    def _read(self):
        latest = self.directory / "latest.jpg"
        stop = self.directory / "STOP"
        last_mtime = -1
        index = 0
        while not self.stopped.is_set() and not stop.exists():
            try:
                mtime = latest.stat().st_mtime_ns
            except OSError:
                sleep(0.02)
                continue
            if mtime == last_mtime:
                sleep(0.01)
                continue
            frame = cv2.imread(str(latest))
            if frame is None:
                sleep(0.01)
                continue
            last_mtime = mtime
            packet = FramePacket(index, monotonic(), frame)
            try:
                self.queue.put_nowait(packet)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
                self.queue.put_nowait(packet)
            index += 1
        self.stopped.set()

    def get_frame(self, timeout: float = 1.0) -> FramePacket | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)


def publish_atomic(path: Path, payload: bytes) -> bool:
    """Publish live state without crashing when a Windows reader briefly locks the target."""
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    for _attempt in range(8):
        try:
            temporary.replace(path)
            return True
        except OSError:
            sleep(0.01)
    temporary.unlink(missing_ok=True)
    return False


def write_clip(
    path: Path, frames: list[np.ndarray], fps: float, ended_at_sec: float,
    record: EventRecord, repository: EventRepository,
) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create event clip {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    record.clip_path = str(path.relative_to(ROOT))
    record.lifecycle = EventLifecycle.CLOSED
    record.ended_at_sec = max(record.started_at_sec, ended_at_sec)
    record.duration_sec = record.ended_at_sec - record.started_at_sec
    if record.started_at_utc is not None:
        record.ended_at_utc = record.started_at_utc + timedelta(seconds=record.duration_sec)
    repository.upsert(record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Webcam index, RTSP URL, or video path")
    parser.add_argument("--browser-frame-dir", type=Path, help="Directory receiving atomic latest.jpg browser frames")
    parser.add_argument("--browser-width", type=int, default=640)
    parser.add_argument("--browser-height", type=int, default=480)
    parser.add_argument("--browser-fps", type=float, default=5.0)
    parser.add_argument("--camera-id", default="LIVE_CAM_01")
    parser.add_argument("--pose-model", default="models/weights/yolo26n-pose.pt")
    parser.add_argument("--object-model", default="models/weights/yolo26n.pt")
    parser.add_argument("--chit-model")
    parser.add_argument("--workstation-map", type=Path, help="Optional human-reviewed CBT scene YAML override")
    parser.add_argument("--require-workstation-map", action="store_true")
    parser.add_argument("--workstation-mode", choices=("auto", "human", "off"), default="auto")
    parser.add_argument("--auto-calibration-warmup", type=float, default=8.0)
    parser.add_argument("--phone-proposal-confidence", type=float, default=0.10)
    parser.add_argument("--phone-event-confidence", type=float, default=0.20)
    parser.add_argument("--tiled-small-object-inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tile-every", type=int, default=4)
    parser.add_argument("--hand-low-duration", type=float, default=1.25)
    parser.add_argument("--interaction-duration", type=float, default=0.75)
    parser.add_argument("--screen-duration", type=float, default=0.80)
    parser.add_argument("--interaction-max-distance", type=float, default=0.42)
    parser.add_argument("--screen-max-distance", type=float, default=0.55)
    parser.add_argument("--gaze-min-confidence", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--maximum-frames", type=int, default=0)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--record-stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-roll-sec", type=float, default=2.0)
    parser.add_argument("--post-roll-sec", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/live_runs"))
    parser.add_argument("--database", type=Path, default=Path("outputs/netra4d_events.sqlite"))
    parser.add_argument("--preprocessing-config", default=None, help="Path to preprocessing YAML config")
    parser.add_argument("--live-publish-dir", type=Path, help="Atomically publish latest.jpg and latest.json for the API")
    parser.add_argument("--monitoring-control-file", type=Path, help="When absent, keep the live feed running but skip inference")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = resolve_device(args.device, args.require_cuda)
    scene_path = (ROOT / args.workstation_map).resolve() if args.workstation_map else None
    if (args.require_workstation_map or args.workstation_mode == "human") and (scene_path is None or not scene_path.is_file()):
        raise FileNotFoundError("Human/required workstation mode needs an existing --workstation-map YAML")
    if scene_path and not scene_path.is_file():
        raise FileNotFoundError(f"CBT workstation map does not exist: {scene_path}")
    scene = load_scene_map(scene_path) if scene_path and args.workstation_mode != "off" else None
    if scene and scene.camera_id != args.camera_id:
        raise ValueError(f"Workstation map camera_id {scene.camera_id!r} does not match {args.camera_id!r}")
    run_started_utc = datetime.now(timezone.utc)
    run_id = run_started_utc.strftime("live_%Y%m%dT%H%M%SZ")
    run_dir = (ROOT / args.output_dir / run_id).resolve()
    keyframe_dir, clip_dir = run_dir / "keyframes", run_dir / "clips"
    keyframe_dir.mkdir(parents=True)
    clip_dir.mkdir()
    repository = EventRepository((ROOT / args.database).resolve())
    pose_path, object_path = (ROOT / args.pose_model).resolve(), (ROOT / args.object_model).resolve()
    pose = YOLO26PoseProvider(str(pose_path), tracker="bytetrack.yaml", device=device)
    objects = ObjectDisambiguationProvider(
        model_name=str(object_path), device=device, enable_candidate_crops=True,
        phone_proposal_confidence=args.phone_proposal_confidence, crop_every=2,
        enable_tiled_inference=args.tiled_small_object_inference, tile_every=args.tile_every,
        chit_model_name=str((ROOT / args.chit_model).resolve()) if args.chit_model else None,
    )
    pose.start()
    objects.start()
    preprocessing = PreprocessingPipeline(config=args.preprocessing_config)
    provenance = {
        "pose": model_record(pose_path),
        "object": model_record(object_path),
        "chit": model_record((ROOT / args.chit_model).resolve()) if args.chit_model else None,
        "device": device,
    }
    manual_reasoner = CBTBehaviouralReasoner(scene) if scene else None
    baseline_engine = BaselineTemporalEngine(
            phone_confidence=args.phone_event_confidence,
            hand_low_duration_sec=args.hand_low_duration,
            interaction_duration_sec=args.interaction_duration,
            screen_duration_sec=args.screen_duration,
            interaction_max_distance=args.interaction_max_distance,
            screen_max_distance=args.screen_max_distance,
            gaze_min_confidence=args.gaze_min_confidence,
    )
    source_path = Path(str(source)) if isinstance(source, str) and not str(source).lower().startswith(("rtsp://", "rtmp://")) else None
    lossless_file_source = bool(source_path and source_path.is_file())
    browser_frame_dir = (ROOT / args.browser_frame_dir).resolve() if args.browser_frame_dir else None
    capture = (
        BrowserFrameCapture(browser_frame_dir, args.browser_width, args.browser_height, args.browser_fps)
        if browser_frame_dir else BoundedCapture(source, maximum_queue=2, pace_file=lossless_file_source)
    )
    if scene:
        scene.validate_frame_geometry(
            int(capture.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    capture_width = int(capture.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(capture.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if manual_reasoner:
        temporal = manual_reasoner
    elif args.workstation_mode == "auto" and capture_width > 0 and capture_height > 0:
        temporal = AutomaticCBTBehaviouralEngine(
            args.camera_id, run_id, capture_width, capture_height,
            calibration_warmup_sec=args.auto_calibration_warmup,
            baseline_engine=baseline_engine,
        )
    else:
        temporal = baseline_engine
    fps = float(capture.capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    pre_roll = deque(maxlen=max(1, int(fps * args.pre_roll_sec)))
    active: dict[str, dict] = {}
    latencies, processed = [], 0
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="event-clip")
    futures = []
    stream_path = run_dir / "live_annotated.mp4"
    publish_dir = (ROOT / args.live_publish_dir).resolve() if args.live_publish_dir else None
    monitoring_control_file = (ROOT / args.monitoring_control_file).resolve() if args.monitoring_control_file else None
    if publish_dir:
        publish_dir.mkdir(parents=True, exist_ok=True)
    stream_writer = None
    last_timestamp = 0.0
    started = monotonic()
    capture.start()
    try:
        while not capture.stopped.is_set() or not capture.queue.empty():
            packet = capture.get_frame() if isinstance(capture, BrowserFrameCapture) else capture.get()
            if packet is None:
                continue
            if monitoring_control_file and not monitoring_control_file.is_file():
                # Keep capture current for a live source, but do not compute or persist paused frames.
                continue
            frame_started = perf_counter()
            # Preserve exact media timestamps for uploaded files. Live sources use wall time.
            timestamp = packet.index / fps if lossless_file_source else monotonic() - started
            last_timestamp = timestamp
            enhanced_frame, q_report = preprocessing.process(packet.frame, packet.index)
            people, _ = pose.process_frame(packet.index, timestamp, enhanced_frame, args.camera_id, run_id)
            _, detected = objects.process_frame(packet.index, timestamp, enhanced_frame, args.camera_id, run_id, people)
            masked, privacy = apply_privacy_mask_and_collect_annotations(packet.frame, people)
            events = temporal.update(timestamp, people, detected, privacy)
            annotated = draw_verified_overlay(masked, people, detected)
            annotated = draw_post_blur_candidate_annotations(annotated, people, privacy)
            active_cbt = (
                temporal.active_reasoner
                if isinstance(temporal, AutomaticCBTBehaviouralEngine)
                else temporal if isinstance(temporal, CBTBehaviouralReasoner) else None
            )
            if active_cbt:
                annotated = draw_cbt_context(annotated, active_cbt, timestamp_sec=timestamp)
            annotated = draw_temporal_review_events(annotated, people, events)
            if events and preprocessing.config.save_flagged_frames:
                event_tags = "_".join(str(getattr(e, "event_type", "EVENT")) for e in events[:2])
                clean_tag = "".join(c if c.isalnum() or c == "_" else "_" for c in event_tags)[:40]
                preprocessing.save_flagged_event_frame(
                    orig_bgr=packet.frame,
                    enh_bgr=enhanced_frame,
                    frame_idx=packet.index,
                    reason=f"LIVE_{clean_tag}",
                    metadata={"timestamp": timestamp, "event_count": len(events), "quality_score": q_report.overall_score}
                )
            if args.record_stream and stream_writer is None:
                height, width = annotated.shape[:2]
                try:
                    stream_writer = cv2.VideoWriter(
                        str(stream_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    if not stream_writer.isOpened():
                        stream_writer = None
                except Exception:
                    stream_writer = None
            if stream_writer is not None:
                stream_writer.write(annotated)
            pre_roll.append(annotated.copy())

            new_event_ids = []
            for event in (item for item in events if item.is_new):
                event_id = str(uuid4())
                keyframe = keyframe_dir / f"{event_id}.jpg"
                cv2.imwrite(str(keyframe), annotated)
                record = EventRecord(
                    event_id=event_id, run_id=run_id, camera_id=args.camera_id, video_id=run_id,
                    event_type=event.event_type, label=event.label, lifecycle=EventLifecycle.OPEN,
                    evidence_kind=EvidenceKind(event.evidence_kind), evidence_score=event.evidence_score,
                    candidate_ids=list(event.candidate_ids), started_at_sec=event.started_at_sec,
                    workstation_ids=[value for value in (
                        event.details.get("source_workstation"), event.details.get("target_workstation")
                    ) if value],
                    started_at_utc=run_started_utc + timedelta(seconds=event.started_at_sec),
                    duration_sec=event.duration_sec, source_signals=list(event.source_signals),
                    keyframe_path=str(keyframe.relative_to(ROOT)),
                    model_provenance=provenance,
                    reasoning_provenance={
                        "engine": type(temporal).__name__,
                        "workstation_map": str(scene_path.relative_to(ROOT)) if scene_path else None,
                    },
                    review_priority=event.details.get("review_priority", event.evidence_score),
                    uncertainty=event.details.get("uncertainty"),
                    specialist_status=SpecialistStatus.REQUESTED if event.details.get("specialist_request") else SpecialistStatus.NOT_REQUIRED,
                    specialist_request=event.details.get("specialist_request"),
                    details=event.details,
                )
                repository.upsert(record)
                new_event_ids.append(event_id)
                active[event_id] = {
                    "record": record,
                    "frames": list(pre_roll),
                    "close_at": timestamp + args.post_roll_sec,
                    "skip_current": True,
                }
            for event_id, state in list(active.items()):
                if state.pop("skip_current", False):
                    pass
                else:
                    state["frames"].append(annotated.copy())
                if timestamp >= state["close_at"]:
                    clip = clip_dir / f"{event_id}.mp4"
                    futures.append(executor.submit(
                        write_clip, clip, state["frames"], fps, timestamp, state["record"], repository
                    ))
                    del active[event_id]

            latency_ms = (perf_counter() - frame_started) * 1000
            latencies.append(latency_ms)
            processed += 1
            if publish_dir:
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise RuntimeError("Unable to encode live annotated frame")
                publish_atomic(publish_dir / "latest.jpg", encoded.tobytes())
                ok_clean, encoded_clean = cv2.imencode(".jpg", masked, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok_clean:
                    publish_atomic(publish_dir / "latest_clean.jpg", encoded_clean.tobytes())
                snapshot = temporal.frame_snapshot() if hasattr(temporal, "frame_snapshot") else None
                telemetry = {
                    "run_id": run_id, "source": str(args.source), "frame_index": packet.index,
                    "timestamp_sec": timestamp, "latency_ms": latency_ms,
                    "effective_fps": processed / max(timestamp, 0.001), "dropped_frames": capture.dropped,
                    "input_policy": "lossless_backpressure" if lossless_file_source else "live_drop_oldest",
                    "people": [item.model_dump(mode="json") for item in people],
                    "objects": [item.model_dump(mode="json") for item in detected],
                    "active_review_events": [item.to_dict() for item in events],
                    "new_event_ids": new_event_ids, "cbt_reasoning": snapshot,
                }
                publish_atomic(publish_dir / "latest.json", json.dumps(telemetry).encode("utf-8"))
            if args.display:
                cv2.imshow("NETRA-4D LIVE", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.maximum_frames and processed >= args.maximum_frames:
                break
    finally:
        capture.close()
        pose.close()
        objects.close()
        if stream_writer is not None:
            stream_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        for event_id, state in active.items():
            clip = clip_dir / f"{event_id}.mp4"
            futures.append(executor.submit(
                write_clip, clip, state["frames"], fps, last_timestamp, state["record"], repository
            ))
        executor.shutdown(wait=True)
        for future in futures:
            future.result()

    elapsed = monotonic() - started
    metrics = {
        "run_id": run_id, "source": str(args.source), "device": device,
        "processed_frames": processed, "dropped_frames": capture.dropped,
        "elapsed_sec": elapsed, "effective_fps": processed / elapsed if elapsed else 0,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else 0,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0,
        "queue_capacity": 2,
        "stale_frame_policy": "lossless_backpressure" if lossless_file_source else "drop_oldest",
        "preprocessing_stats": preprocessing.get_stats_summary(),
        "annotated_stream_path": str(stream_path.relative_to(ROOT)) if stream_path.is_file() else None,
        "reasoning_engine": type(temporal).__name__,
        "workstation_map": str(scene_path.relative_to(ROOT)) if scene_path else None,
    }
    automatic_scene = temporal.scene if isinstance(temporal, AutomaticCBTBehaviouralEngine) else None
    if automatic_scene:
        automatic_map = run_dir / "automatic_workstation_map.yaml"
        automatic_map.write_text(yaml.safe_dump(automatic_scene.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
        metrics["workstation_map"] = str(automatic_map.relative_to(ROOT))
        metrics["automatic_calibration_status"] = temporal.status
    active_graph_reasoner = (
        temporal if isinstance(temporal, CBTBehaviouralReasoner)
        else temporal if isinstance(temporal, AutomaticCBTBehaviouralEngine) else None
    )
    if active_graph_reasoner:
        graph_path = run_dir / "cbt_graph.json"
        graph_path.write_text(json.dumps(active_graph_reasoner.graph_snapshot(), indent=2), encoding="utf-8")
        metrics["cbt_graph_path"] = str(graph_path.relative_to(ROOT))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

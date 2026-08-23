"""Render every frame of Drive-hosted videos with genuine model detections.

Each video receives its own output folder. Annotated video frames contain real
YOLO pose/person and object outputs plus explicitly labelled temporal review
proxies derived from those observations. Still images are written only when
the corresponding evidence gate is satisfied. A review proxy is never proof
of behaviour or malpractice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset
from drishti.cbt import AutomaticCBTBehaviouralEngine, CBTBehaviouralReasoner, draw_cbt_context, load_scene_map
from drishti.providers.object_provider import ObjectDisambiguationProvider
from drishti.providers.yolo26_provider import YOLO26PoseProvider
from drishti.privacy import (
    apply_privacy_mask_and_collect_annotations,
    draw_post_blur_candidate_annotations,
)
from drishti.temporal import BaselineTemporalEngine, draw_temporal_review_events
from scripts.run_verified_inference import (
    draw_verified_overlay,
    file_sha256,
    model_record,
    resolve_device,
)


def high_confidence_phone_detections(objects: list[Any], threshold: float) -> list[Any]:
    return [
        item
        for item in objects
        if item.metadata.get("raw_model_class") == "cell phone"
        and item.confidence >= threshold
    ]


def json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


def undecodable_frame(width: int, height: int, frame_index: int) -> Any:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame, "SOURCE FRAME UNDECODABLE - NO MODEL INFERENCE",
        (30, max(50, height // 2)), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"frame {frame_index}", (30, max(85, height // 2 + 38)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def render(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device, args.require_cuda)
    manifest_path = (ROOT / args.manifest).resolve()
    pose_path = (ROOT / args.pose_model).resolve()
    object_path = (ROOT / args.object_model).resolve()
    output_root = (ROOT / args.output_dir).resolve()
    staging_root = output_root / ".staging"
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id

    assets = load_drive_manifest(manifest_path)
    selected = set(args.asset or [])
    known = {asset.asset_id for asset in assets}
    if selected - known:
        raise ValueError(f"Unknown asset IDs: {sorted(selected - known)}")
    assets = [asset for asset in assets if not selected or asset.asset_id in selected]
    scene_configs = {}
    for asset in assets:
        scene_path = (ROOT / args.workstation_map_dir / f"{asset.asset_id}.yaml").resolve()
        if (args.require_workstation_map or args.workstation_mode == "human") and not scene_path.is_file():
            raise FileNotFoundError(f"Required calibrated CBT workstation map is missing: {scene_path}")
        scene = load_scene_map(scene_path) if scene_path.is_file() and args.workstation_mode != "off" else None
        if scene and scene.video_id and scene.video_id != asset.asset_id:
            raise ValueError(f"Workstation map video_id {scene.video_id!r} does not match {asset.asset_id!r}")
        scene_configs[asset.asset_id] = (scene_path, scene)

    run_dir.mkdir(parents=True, exist_ok=False)

    tracker = None if args.tracker == "none" else args.tracker
    pose = YOLO26PoseProvider(
        str(pose_path), conf_thresh=args.confidence, device=device, tracker=tracker
    )
    object_model = ObjectDisambiguationProvider(
        model_name=str(object_path), conf_thresh=args.confidence, device=device,
        enable_candidate_crops=args.small_object_crops,
        phone_proposal_confidence=args.phone_proposal_confidence,
        crop_every=args.small_object_crop_every,
        enable_tiled_inference=args.tiled_small_object_inference,
        tile_every=args.tile_every,
        chit_model_name=str((ROOT / args.chit_model).resolve()) if args.chit_model else None,
        chit_confidence=args.chit_confidence,
    )
    pose.start()
    object_model.start()

    run_record: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_kind": "full_video_real_model_inference",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination_layout": "one_folder_per_source_video",
        "phone_still_rule": {
            "raw_model_class": "cell phone",
            "minimum_confidence": args.phone_confidence,
            "meaning": "high-confidence phone detection; not a phone-usage classification",
        },
        "high_confidence_event_still_rule": {
            "minimum_evidence_confidence": args.high_confidence_event_threshold,
            "meaning": "sustained model/geometry evidence for human review; not cheating probability",
        },
        "privacy_pipeline": {
            "order": ["pose_keypoints", "gaze_proxy", "face_or_head_blur", "post_blur_annotation"],
            "face_region": "pose facial keypoints; person-head fallback when facial keypoints are weak",
            "gaze_label": "2-D head-direction proxy from eye/nose pose geometry, not eye tracking",
            "candidate_id_source": "ByteTrack" if tracker else "frame-local index",
        },
        "baseline_temporal_events": {
            "enabled": not args.disable_baseline_events,
            "execution": "CPU-only geometry and temporal persistence after real YOLO/ByteTrack observations",
            "phone_review_minimum_confidence": args.baseline_phone_confidence,
            "thresholds": {
                "hand_low_duration_sec": args.hand_low_duration,
                "interaction_duration_sec": args.interaction_duration,
                "screen_duration_sec": args.screen_duration,
                "interaction_max_distance": args.interaction_max_distance,
                "screen_max_distance": args.screen_max_distance,
                "gaze_min_confidence": args.gaze_min_confidence,
            },
            "labels": {
                "PHONE_USE_REVIEW": "model cell-phone evidence plus candidate/wrist association",
                "LOW_HAND_POSSIBLE_CHIT_REVIEW": "behavioral proxy; no chit object claim",
                "CANDIDATE_INTERACTION_REVIEW": "pairwise mutual head-direction/proximity proxy",
                "OTHER_SCREEN_LOOK_REVIEW": "laptop plus head-direction geometry proxy; not iris gaze",
            },
                "chair_overlay": "assigned workstations only for the first configured seconds",
        },
        "cbt_peak_reasoner": {
            "default_activation": "automatic model-assisted warmup from real person and monitor observations",
            "human_map_is_optional_override": True,
            "workstation_map_directory": str((ROOT / args.workstation_map_dir).resolve()),
            "require_workstation_map": args.require_workstation_map,
            "mode": args.workstation_mode,
            "fallback": "automatic model-assisted warmup, then baseline if stable workstations cannot be recovered",
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
            "chit": model_record((ROOT / args.chit_model).resolve()) if args.chit_model else None,
        },
        "assets": [],
    }

    try:
        for asset in assets:
            scene_path, scene = scene_configs[asset.asset_id]
            manual_reasoner = CBTBehaviouralReasoner(scene) if scene else None
            video_dir = run_dir / asset.asset_id
            video_dir.mkdir()
            incomplete = video_dir / "INCOMPLETE.txt"
            incomplete.write_text("This folder is not upload-ready until COMPLETE.json exists.\n", encoding="utf-8")
            detections_path = video_dir / "detections.jsonl"
            events_path = video_dir / "review_events.jsonl"
            output_video = video_dir / f"{asset.asset_id}_annotated.mp4"
            phone_dir = video_dir / "phone_detection_frames"
            event_frame_dir = video_dir / "review_event_frames"
            started_video = perf_counter()
            counters = {
                "frames": 0, "decoded_frames": 0, "decode_failures": 0,
                "people": 0, "objects": 0, "phone_stills": 0, "privacy_masks": 0,
                "gaze_proxy_known": 0,
                "review_events": 0, "review_event_frames": 0,
                "high_confidence_event_frames": 0,
            }

            with materialize_drive_asset(asset, temp_root=staging_root) as (source_path, provenance):
                capture = cv2.VideoCapture(str(source_path))
                if not capture.isOpened():
                    raise RuntimeError(f"Unable to decode {asset.asset_id}: {source_path}")
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                render_limit = (
                    min(expected_frames, args.max_frames_per_video)
                    if expected_frames and args.max_frames_per_video
                    else (args.max_frames_per_video or expected_frames)
                )
                if fps <= 0 or width <= 0 or height <= 0:
                    raise RuntimeError(f"Invalid video metadata for {asset.asset_id}")
                if scene:
                    scene.validate_frame_geometry(width, height)
                writer = cv2.VideoWriter(
                    str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    capture.release()
                    raise RuntimeError(f"OpenCV could not create {output_video}")
                baseline_engine = None if args.disable_baseline_events else BaselineTemporalEngine(
                    phone_confidence=args.baseline_phone_confidence,
                    hand_low_duration_sec=args.hand_low_duration,
                    interaction_duration_sec=args.interaction_duration,
                    screen_duration_sec=args.screen_duration,
                    interaction_max_distance=args.interaction_max_distance,
                    screen_max_distance=args.screen_max_distance,
                    gaze_min_confidence=args.gaze_min_confidence,
                )
                if manual_reasoner:
                    reasoning_engine = manual_reasoner
                elif args.workstation_mode == "auto" and baseline_engine:
                    reasoning_engine = AutomaticCBTBehaviouralEngine(
                        asset.asset_id, asset.asset_id, width, height,
                        calibration_warmup_sec=args.auto_calibration_warmup,
                        baseline_engine=baseline_engine,
                    )
                else:
                    reasoning_engine = baseline_engine

                try:
                    with (
                        detections_path.open("w", encoding="utf-8") as detection_file,
                        events_path.open("w", encoding="utf-8") as event_file,
                    ):
                        frame_index = 0
                        seek_before_read = False
                        high_confidence_captured: set[tuple[str, tuple[int, ...]]] = set()
                        while frame_index < render_limit if render_limit else True:
                            if seek_before_read:
                                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                            ok, frame = capture.read()
                            if not ok:
                                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                                ok, frame = capture.read()
                            if not ok or frame is None:
                                diagnostic = undecodable_frame(width, height, frame_index)
                                writer.write(diagnostic)
                                detection_file.write(json_line({
                                    "frame_index": frame_index,
                                    "timestamp_sec": frame_index / fps,
                                    "decode_ok": False,
                                    "inference_executed": False,
                                    "people": [], "objects": [],
                                    "high_confidence_phone_still": None,
                                }))
                                counters["frames"] += 1
                                counters["decode_failures"] += 1
                                frame_index += 1
                                seek_before_read = True
                                continue
                            seek_before_read = False
                            timestamp_sec = frame_index / fps
                            inference_started = perf_counter()
                            people, _ = pose.process_frame(
                                frame_index, timestamp_sec, frame,
                                camera_id=asset.asset_id, video_id=asset.filename,
                            )
                            _, objects = object_model.process_frame(
                                frame_index, timestamp_sec, frame,
                                camera_id=asset.asset_id, video_id=asset.filename,
                                person_obs_list=people,
                            )
                            privacy_masked, privacy_annotations = apply_privacy_mask_and_collect_annotations(
                                frame, people, min_keypoint_confidence=args.face_keypoint_confidence
                            )
                            review_events = reasoning_engine.update(
                                timestamp_sec, people, objects, privacy_annotations
                            ) if reasoning_engine else []
                            annotated = draw_verified_overlay(privacy_masked, people, objects)
                            annotated = draw_post_blur_candidate_annotations(
                                annotated, people, privacy_annotations
                            )
                            active_cbt = (
                                reasoning_engine.active_reasoner
                                if isinstance(reasoning_engine, AutomaticCBTBehaviouralEngine)
                                else reasoning_engine if isinstance(reasoning_engine, CBTBehaviouralReasoner) else None
                            )
                            if active_cbt:
                                annotated = draw_cbt_context(
                                    annotated, active_cbt,
                                    timestamp_sec=timestamp_sec,
                                    chair_display_seconds=args.chair_display_seconds,
                                )
                            annotated = draw_temporal_review_events(annotated, people, review_events)
                            events_to_log = []
                            for event in review_events:
                                event_key = (event.event_type, event.candidate_ids)
                                crossed_high_confidence = (
                                    event.evidence_score >= args.high_confidence_event_threshold
                                    and event_key not in high_confidence_captured
                                )
                                if event.is_new or crossed_high_confidence:
                                    events_to_log.append((event, crossed_high_confidence))
                            for event_index, (event, crossed_high_confidence) in enumerate(events_to_log):
                                saved_event_still = None
                                if crossed_high_confidence:
                                    event_frame_dir.mkdir(exist_ok=True)
                                    event_name = (
                                        f"frame_{frame_index:06d}_{event_index:02d}_"
                                        f"t_{timestamp_sec:.3f}_conf_{event.evidence_score:.3f}_"
                                        f"{event.event_type.lower()}.jpg"
                                    )
                                    saved_event_still = f"review_event_frames/{event_name}"
                                    if not cv2.imwrite(str(video_dir / saved_event_still), annotated):
                                        raise RuntimeError(f"Failed to write {event_name}")
                                    high_confidence_captured.add((event.event_type, event.candidate_ids))
                                    counters["review_event_frames"] += 1
                                    counters["high_confidence_event_frames"] += 1
                                event_payload = {
                                    "frame_index": frame_index,
                                    "log_reason": "high_confidence_crossing" if crossed_high_confidence else "temporal_activation",
                                    "high_confidence_snapshot": saved_event_still,
                                    **event.to_dict(),
                                }
                                event_file.write(json_line(event_payload))
                                counters["review_events"] += 1
                            writer.write(annotated)
                            phones = high_confidence_phone_detections(objects, args.phone_confidence)
                            saved_still = None
                            if phones:
                                phone_dir.mkdir(exist_ok=True)
                                maximum = max(item.confidence for item in phones)
                                saved_still = (
                                    f"phone_detection_frames/frame_{frame_index:06d}_"
                                    f"t_{timestamp_sec:.3f}_conf_{maximum:.3f}.jpg"
                                )
                                if not cv2.imwrite(str(video_dir / saved_still), annotated):
                                    raise RuntimeError(f"Failed to write {saved_still}")
                                counters["phone_stills"] += 1
                            detection_file.write(json_line({
                                "frame_index": frame_index,
                                "timestamp_sec": timestamp_sec,
                                "decode_ok": True,
                                "inference_executed": True,
                                "inference_ms": (perf_counter() - inference_started) * 1000.0,
                                "people": [item.model_dump(mode="json") for item in people],
                                "objects": [item.model_dump(mode="json") for item in objects],
                                "privacy": {
                                    "calculation_order": ["gaze_proxy", "face_or_head_blur", "post_blur_annotation"],
                                    "annotations": [annotation.__dict__ for annotation in privacy_annotations],
                                },
                                "active_review_events": [event.to_dict() for event in review_events],
                                "cbt_reasoning": reasoning_engine.frame_snapshot() if isinstance(reasoning_engine, (AutomaticCBTBehaviouralEngine, CBTBehaviouralReasoner)) else None,
                                "high_confidence_phone_still": saved_still,
                            }))
                            counters["frames"] += 1
                            counters["decoded_frames"] += 1
                            counters["people"] += len(people)
                            counters["objects"] += len(objects)
                            counters["privacy_masks"] += len(privacy_annotations)
                            counters["gaze_proxy_known"] += sum(
                                item.gaze_proxy != "UNKNOWN" for item in privacy_annotations
                            )
                            frame_index += 1
                            if frame_index % args.progress_every == 0:
                                print(
                                    f"{asset.asset_id}: {frame_index}/{expected_frames or '?'} "
                                    f"frames, phone stills={counters['phone_stills']}", flush=True
                                )
                finally:
                    capture.release()
                    writer.release()

            automatic_scene = (
                reasoning_engine.scene
                if isinstance(reasoning_engine, AutomaticCBTBehaviouralEngine) else None
            )
            automatic_map_path = None
            if automatic_scene:
                automatic_map_path = video_dir / "automatic_workstation_map.yaml"
                automatic_map_path.write_text(
                    yaml.safe_dump(automatic_scene.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
                )
            summary = {
                "asset_id": asset.asset_id,
                "source": provenance,
                "video": {
                    "width": width, "height": height, "fps": fps,
                    "source_frame_count": expected_frames,
                    "requested_render_frame_count": render_limit,
                    "rendered_frame_count": counters["frames"],
                },
                "counts": counters,
                "reasoning_engine": type(reasoning_engine).__name__ if reasoning_engine else None,
                "workstation_map": str(scene_path.relative_to(ROOT)) if manual_reasoner else (
                    automatic_map_path.name if automatic_map_path else None
                ),
                "automatic_calibration_status": reasoning_engine.status if isinstance(reasoning_engine, AutomaticCBTBehaviouralEngine) else None,
                "cbt_graph": reasoning_engine.graph_snapshot() if isinstance(reasoning_engine, (AutomaticCBTBehaviouralEngine, CBTBehaviouralReasoner)) else None,
                "elapsed_sec": perf_counter() - started_video,
                "outputs": {
                    "annotated_video": output_video.name,
                    "detections": detections_path.name,
                    "review_events": events_path.name,
                    "review_event_frame_directory": event_frame_dir.name if event_frame_dir.exists() else None,
                    "phone_still_directory": phone_dir.name if phone_dir.exists() else None,
                },
                "decode_policy": (
                    "Undecodable source positions are red diagnostic frames with no "
                    "detections or inference; scene pixels are never fabricated or duplicated."
                ),
                "integrity": {
                    "annotated_video_bytes": output_video.stat().st_size,
                    "annotated_video_sha256": file_sha256(output_video),
                    "detections_sha256": file_sha256(detections_path),
                },
            }
            if counters["frames"] == 0:
                raise RuntimeError(f"No frames rendered for {asset.asset_id}")
            if render_limit and counters["frames"] != render_limit:
                raise RuntimeError(
                    f"Frame count mismatch for {asset.asset_id}: "
                    f"expected {render_limit}, rendered {counters['frames']}"
                )
            (video_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            incomplete.unlink()
            (video_dir / "COMPLETE.json").write_text(
                json.dumps({"complete": True, "verified_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
                encoding="utf-8",
            )
            run_record["assets"].append(summary)
            (run_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
            print(f"Completed {asset.asset_id}: {output_video}", flush=True)
    finally:
        pose.close()
        object_model.close()

    run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_record["summary"] = {
        "videos": len(run_record["assets"]),
        "frames": sum(item["counts"]["frames"] for item in run_record["assets"]),
        "phone_stills": sum(item["counts"]["phone_stills"] for item in run_record["assets"]),
        "review_events": sum(item["counts"]["review_events"] for item in run_record["assets"]),
    }
    result = run_dir / "run.json"
    result.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(result), "summary": run_record["summary"]}, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="drishti4d/configs/datasets.json")
    parser.add_argument("--pose-model", default="models/weights/yolo26n-pose.pt")
    parser.add_argument("--object-model", default="models/weights/yolo26n.pt")
    parser.add_argument("--output-dir", default="outputs/full_video_runs")
    parser.add_argument("--run-id")
    parser.add_argument("--asset", action="append", help="Process only this asset; repeatable")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--phone-confidence", type=float, default=0.70)
    parser.add_argument(
        "--tracker", default="bytetrack.yaml",
        help="Ultralytics tracker for stable candidate IDs; pass 'none' for frame-local IDs",
    )
    parser.add_argument("--face-keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--disable-baseline-events", action="store_true")
    parser.add_argument("--workstation-map-dir", type=Path, default=Path("drishti4d/configs/workstations"))
    parser.add_argument("--require-workstation-map", action="store_true")
    parser.add_argument("--workstation-mode", choices=("auto", "human", "off"), default="auto")
    parser.add_argument("--auto-calibration-warmup", type=float, default=8.0)
    parser.add_argument("--baseline-phone-confidence", type=float, default=0.20)
    parser.add_argument("--hand-low-duration", type=float, default=1.25)
    parser.add_argument("--interaction-duration", type=float, default=0.75)
    parser.add_argument("--screen-duration", type=float, default=0.80)
    parser.add_argument("--interaction-max-distance", type=float, default=0.42)
    parser.add_argument("--screen-max-distance", type=float, default=0.55)
    parser.add_argument("--gaze-min-confidence", type=float, default=0.35)
    parser.add_argument("--small-object-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phone-proposal-confidence", type=float, default=0.10)
    parser.add_argument("--small-object-crop-every", type=int, default=2)
    parser.add_argument("--tiled-small-object-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-every", type=int, default=4)
    parser.add_argument("--chit-model", help="Real custom YOLO checkpoint with an explicit chit/paper class")
    parser.add_argument("--chit-confidence", type=float, default=0.25)
    parser.add_argument("--high-confidence-event-threshold", type=float, default=0.70)
    parser.add_argument("--chair-display-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-frames-per-video", type=int, default=0,
        help="Bounded verification only; 0 renders the complete source video",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if not 0.0 <= parsed.phone_confidence <= 1.0:
        raise SystemExit("--phone-confidence must be between 0 and 1")
    if not 0.0 <= parsed.face_keypoint_confidence <= 1.0:
        raise SystemExit("--face-keypoint-confidence must be between 0 and 1")
    if parsed.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    if not 0.0 <= parsed.baseline_phone_confidence <= 1.0:
        raise SystemExit("--baseline-phone-confidence must be between 0 and 1")
    if any(value < 0 for value in (parsed.hand_low_duration, parsed.interaction_duration, parsed.screen_duration)):
        raise SystemExit("Temporal durations cannot be negative")
    if any(not 0.0 <= value <= 1.0 for value in (parsed.interaction_max_distance, parsed.screen_max_distance, parsed.gaze_min_confidence)):
        raise SystemExit("Temporal distance/confidence thresholds must be between 0 and 1")
    if parsed.max_frames_per_video < 0:
        raise SystemExit("--max-frames-per-video cannot be negative")
    if not 0.0 <= parsed.high_confidence_event_threshold <= 1.0:
        raise SystemExit("--high-confidence-event-threshold must be between 0 and 1")
    if parsed.chair_display_seconds < 0:
        raise SystemExit("--chair-display-seconds cannot be negative")
    if parsed.small_object_crop_every < 1:
        raise SystemExit("--small-object-crop-every must be positive")
    if parsed.tile_every < 1:
        raise SystemExit("--tile-every must be positive")
    render(parsed)

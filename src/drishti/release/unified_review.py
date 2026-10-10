"""Video 3 unified review: shared inference plus NETRA's own reasoning components.

Annotations are never read by this pipeline. Behavioural scores are uncalibrated
review proxies, and are not exposed as probabilities or automatic verdicts.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from .models import ModelRegistry, file_sha256
from .pipeline import CanonicalInferenceService
from ..data.video_policy import bounded_video_source
from ..engine.observation import DetectedObjectObservation, YOLO26Adapter
from ..engine.candidate_state import CandidateStateManager
from ..engine.temporal_buffer import TemporalBufferManager
from ..engine.features import BiomechanicalFeatureExtractor
from ..engine.primitive_events import PrimitiveEventDetector
from ..engine.temporal_patterns import PersistenceDetector, RepetitionDetector
from ..engine.hypotheses import CompetingHypothesisEngine
from ..engine.state_machine import CandidateStateMachine
from ..engine.zones import SemanticZoneEngine
from ..engine.contextual_physics import ContextualPhysicsEngine
from ..pipeline.environment import EnvironmentProfiler
from ..pipeline.role_registry import ImmutableRoleRegistry, RoleEnum

ROOT = Path(__file__).resolve().parents[3]
PHONE_ROLE = "phone_cvat_video03_100epochs"


class UnifiedReviewEngine:
    """Adapter from shared tracked observations into the existing NETRA engines."""

    def __init__(self):
        self.environment = EnvironmentProfiler(warmup_frames=30)
        self.roles = ImmutableRoleRegistry(warmup_frames=45)
        self.candidates = CandidateStateManager(warmup_frames=45)
        self.zones = SemanticZoneEngine()
        self.contextual_physics = ContextualPhysicsEngine()
        self.buffer = TemporalBufferManager(max_duration_sec=5.)
        self._head_diagnostics = []
        self._processing_frame_index = None
        self.features = BiomechanicalFeatureExtractor(smoothing_tau=.12, head_diagnostic_sink=self._record_head_diagnostic)
        self.primitives = PrimitiveEventDetector()
        self.persistence = PersistenceDetector()
        self.repetition = RepetitionDetector(window_duration_sec=90., repetition_threshold=3)
        self.hypotheses = CompetingHypothesisEngine()
        self.machines = {}

    def _record_head_diagnostic(self, row):
        row["processing_frame_index"] = self._processing_frame_index
        self._head_diagnostics.append(row)

    def drain_head_diagnostics(self):
        rows, self._head_diagnostics = self._head_diagnostics, []
        return rows

    def drain_contextual_physics(self):
        return self.contextual_physics.drain_graph_snapshot()

    def process(self, index, timestamp, frame, result):
        self._processing_frame_index = index
        height, width = frame.shape[:2]
        objects = [DetectedObjectObservation(
            class_name="cell phone" if o.object_class.value == "MOBILE_PHONE" else str(o.metadata.get("raw_model_class", o.object_class.value)),
            confidence=o.confidence, bbox_xyxy_norm=o.bbox_xyxy_norm,
            associated_track_id=o.associated_entity_id, metadata=o.metadata,
        ) for o in result.objects]
        raw = []
        for p in result.people:
            keypoints = np.array([[p.keypoints_2d[name].x * width, p.keypoints_2d[name].y * height,
                                   p.keypoints_2d[name].confidence] if name in p.keypoints_2d else [0., 0., 0.]
                                  for name in YOLO26Adapter.COCO_KEYPOINT_NAMES])
            raw.append({"track_id": p.entity_local_id, "bbox": p.bbox_xyxy_px,
                        "keypoints": keypoints, "confidence": p.detection_confidence})
        self.environment.observe(raw, objects)
        self.roles.observe_frame(index, raw, frame_bgr=frame)
        observations, proctor = [], None
        for p in raw:
            if self.roles.get_role_for_track(p["track_id"]) == RoleEnum.INVIGILATOR:
                b = p["bbox"]
                proctor = ((b[0]+b[2])/(2*width), (b[1]+b[3])/(2*height))
            else:
                observations.append(YOLO26Adapter.adapt(index, timestamp, (height,width),
                    p["bbox"], p["track_id"], p["keypoints"], p["confidence"],
                    [o for o in objects if o.associated_track_id == p["track_id"]]))
        self.zones.update_invigilator_position(proctor)
        candidates = self.candidates.update_frame_observations(index, timestamp, observations)
        self.contextual_physics.update(candidates, freeze=self.candidates.is_finalized)
        rows = []
        for candidate_id, state in candidates.items():
            if not state.is_visible or not state.current_observation or state.identity.is_invigilator:
                continue
            cx, cy = state.identity.anchor_center_norm
            bw, bh = state.identity.anchor_box_size_norm
            if candidate_id not in self.zones.candidate_desk_zones:
                self.zones.register_candidate_desk(candidate_id, (max(0,cx-bw/2), max(0,cy-bh/4),
                    min(1,cx+bw/2), min(1,cy+bh*.75)))
            obs = state.current_observation
            self.buffer.push(candidate_id, obs)
            features = self.features.extract(candidate_id, obs)
            neighbour, neighbour_id = self.zones.is_gaze_aligned_with_neighbour(candidate_id, features.head.direction.value)
            near = self.zones.is_invigilator_near_candidate(candidate_id)
            completed = self.primitives.process(features, board_aligned=False,
                invigilator_nearby=near, neighbour_seat_aligned=neighbour)
            patterns = []
            for event in list(self.primitives.active_events.values()):
                if event.candidate_id == candidate_id:
                    pattern = self.persistence.check_active_event(event, timestamp)
                    if pattern:
                        patterns.append(pattern)
            for event in completed:
                pattern = self.repetition.register_completed_event(event)
                if pattern:
                    patterns.append(pattern)
            phone = any(o.class_name == "cell phone" for o in obs.nearby_objects)
            # No synthetic HEAD_TURN gets registered when no observed pattern exists.
            evaluation = self.hypotheses.evaluate(candidate_id, patterns[-1] if patterns else None,
                is_invigilator_nearby=near, is_phone_detected=phone,
                sensor_quality_ratio=obs.tracking_confidence)
            machine = self.machines.setdefault(candidate_id, CandidateStateMachine(candidate_id))
            stage = machine.update(timestamp, evaluation).value
            rows.append({"candidate_id": candidate_id, "tracker_id": obs.track_id,
                "review_state": stage, "phone_candidate_nearby": phone,
                "head_direction_proxy": features.head.direction.value,
                "seat_context": self.contextual_physics.context_for(candidate_id),
                "neighbour_orientation_proxy": neighbour, "neighbour_candidate": neighbour_id,
                "pattern_types": [p.pattern_type.value for p in patterns],
                "uncalibrated_review_priority": evaluation.suspicion_score,
                "uncalibrated_uncertainty_proxy": evaluation.uncertainty_score,
                "source_signals": {"phone": "fresh associated object detection" if phone else None,
                    "behaviour": "pose-derived temporal proxies"},
                "counter_evidence": evaluation.counter_evidence,
                "output_semantics": "experimental behavioural review proxies, not cheating probabilities"})
        if index >= 45:
            self.zones.auto_infer_neighbours()
        return rows


def run_video03_review(
    source, output_dir, *, max_frames=0, allow_experimental_models=False,
    enable_preprocessing=False, preprocessing_config=None,
):
    from scripts.run_verified_inference import draw_verified_overlay
    from ..privacy import draw_post_blur_candidate_annotations
    from ..temporal import draw_temporal_review_events
    from scripts.run_product_pipeline import event_record
    from ..event_repository import EventRepository

    source, limit = bounded_video_source(source, max_frames, "VIDEO_03")
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    registry = ModelRegistry(ROOT / "configs/model_registry.yaml", ROOT)
    # Fail approval/file checks before creating an output directory.
    registry.get(PHONE_ROLE, require_approved=not allow_experimental_models)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    output.mkdir(parents=True)
    manifest = {"profile": "video03-review", "schema_version": "1.0", "source": str(source),
        "source_sha256": file_sha256(Path(source)), "frame_limit": limit, "frames": 0,
        "device": device, "phone_confidence": .25, "phone_roi_hold_seconds": 1.,
        "experimental_models_enabled": True, "labels_used_for_inference": False,
        "disabled_components": {"Multi-HMR/SMPL-X": "not evaluated for this task; no 3D inference",
            "legacy cheating/chit/multimodal checkpoints": "unvalidated and not selected",
            "RTMPose/MediaPipe/hand contour specialists": "not evaluated; cannot corroborate phone detections",
            "enhancement": (
                "enabled for this experimental inference comparison"
                if enable_preprocessing else "raw input preserves evaluated detector distribution"
            )},
        "preprocessing": {"enabled": bool(enable_preprocessing), "config_path": preprocessing_config},
        "claim": "Human review evidence only; no automated verdict or calibrated behavioural probability"}
    service = CanonicalInferenceService(registry, device=device, confidence=.25,
        phone_role=PHONE_ROLE, allow_experimental_models=allow_experimental_models,
        phone_roi_hold_seconds=1., enable_preprocessing=enable_preprocessing,
        preprocessing_config=preprocessing_config)
    engine = UnifiedReviewEngine()
    capture = cv2.VideoCapture(str(source))
    writer = None
    started = perf_counter()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected = limit
        manifest.update(fps=fps, expected_frames=expected, models=service.provenance())
        writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
        if not writer.isOpened():
            raise RuntimeError("Unable to create evidence video")
        repository = EventRepository(output / "events.sqlite")
        service.start()
        with (output/"detections.jsonl").open("w",encoding="utf-8") as detections, (output/"unified_reasoning.jsonl").open("w",encoding="utf-8") as reasoning, (output/"review_events.jsonl").open("w",encoding="utf-8") as events, (output/"head_direction_diagnostics.jsonl").open("w",encoding="utf-8") as head_diagnostics, (output/"frame_quality.jsonl").open("w",encoding="utf-8") as quality, (output/"contextual_physics.jsonl").open("w",encoding="utf-8") as contextual_physics:
            for index in range(limit):
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = index/fps
                result = service.process(index,timestamp,frame,camera_id="VIDEO03_CAM",video_id="VIDEO_03")
                rows = engine.process(index,timestamp,frame,result)
                for diagnostic in engine.drain_head_diagnostics():
                    head_diagnostics.write(json.dumps(diagnostic)+"\n")
                for context in engine.drain_contextual_physics():
                    contextual_physics.write(json.dumps({"frame_index": index, "timestamp_sec": timestamp, **context}) + "\n")
                rendered = draw_verified_overlay(result.privacy_frame,result.people,result.objects)
                rendered = draw_post_blur_candidate_annotations(rendered,result.people,result.privacy_annotations)
                rendered = draw_temporal_review_events(rendered,result.people,result.review_events)
                cv2.putText(rendered,"UNIFIED REVIEW | experimental proxies, not verdicts",(12,height-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                writer.write(rendered)
                detections.write(json.dumps({"frame_index":index,"timestamp_sec":timestamp,
                    "people":[p.model_dump(mode="json") for p in result.people],
                    "objects":[o.model_dump(mode="json") for o in result.objects],
                    "model_provenance":result.model_provenance,
                    "quality_report":result.quality_report})+"\n")
                if result.quality_report is not None:
                    quality.write(json.dumps({"frame_index": index, "timestamp_sec": timestamp,
                        "quality_report": result.quality_report}) + "\n")
                reasoning.write(json.dumps({"frame_index":index,"timestamp_sec":timestamp,
                    "environment_proxy":engine.environment.environment_type.value,"candidates":rows})+"\n")
                for event in result.review_events:
                    record = event_record(event,run_id=output.name,camera_id="VIDEO03_CAM",video_id="VIDEO_03",provenance=result.model_provenance)
                    repository.upsert(record)
                    events.write(record.model_dump_json()+"\n")
                manifest["frames"] += 1
                if (index+1)%100 == 0 or index==0:
                    print(f"[Unified review] {index+1}/{expected} frames",flush=True)
        if manifest["frames"] != expected:
            raise RuntimeError(f"Premature video end: {manifest['frames']}/{expected}")
        manifest.update(status="COMPLETE", environment_profile=engine.environment.get_summary(),
            candidate_identities=len(engine.candidates.candidates),
            preprocessing=service.preprocessing_summary())
    except BaseException as error:
        manifest.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        service.close()
        manifest["elapsed_sec"] = perf_counter()-started
        manifest["processed_fps"] = manifest["frames"]/max(manifest["elapsed_sec"],1e-9)
        if "preprocessing" not in manifest or "stats" not in manifest["preprocessing"]:
            manifest["preprocessing"] = service.preprocessing_summary()
        (output/("COMPLETE.json" if manifest.get("status")=="COMPLETE" else "INCOMPLETE.json")).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)
    return str(output/"annotated.mp4"),str(output/"unified_reasoning.jsonl")

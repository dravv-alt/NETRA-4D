"""One supported frame-inference path for batch files and live sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..privacy import apply_privacy_mask_and_collect_annotations, collect_candidate_annotations
from ..providers.object_provider import ObjectDisambiguationProvider
from ..providers.yolo26_provider import YOLO26PoseProvider
from ..providers.phone_review_provider import PhoneReviewProvider
from ..temporal import BaselineTemporalEngine, ReviewEvent
from ..preprocessing import PreprocessingPipeline
from .models import ModelRegistry


@dataclass
class FrameInference:
    people: list[Any]
    objects: list[Any]
    privacy_frame: np.ndarray
    privacy_annotations: list[Any]
    review_events: list[ReviewEvent]
    model_provenance: dict[str, Any]
    quality_report: dict[str, Any] | None = None
    temporal_evidence: dict[str, Any] | None = None


class CanonicalInferenceService:
    """Owns providers and temporal state; it never emits a cheating verdict."""

    def __init__(
        self, registry: ModelRegistry, *, device: str, tracker: str | None = "bytetrack.yaml",
        confidence: float = 0.25, phone_review_confidence: float = 0.20,
        chit_role: str | None = None,
        phone_role: str | None = None, allow_experimental_models: bool = False,
        phone_roi_hold_seconds: float = 1.,
        enable_preprocessing: bool = False, preprocessing_config: str | None = None,
        enable_privacy_mask: bool = False,
    ):
        self.registry = registry
        self.pose_record = registry.get("pose")
        self.object_record = registry.get("objects")
        self.chit_record = registry.get(chit_role) if chit_role else None
        self.phone_record = registry.get(phone_role, require_approved=not allow_experimental_models) if phone_role else None
        self.phone = PhoneReviewProvider(self.phone_record.path,device=device,confidence=confidence,roi_hold_seconds=phone_roi_hold_seconds) if self.phone_record else None
        self.pose = YOLO26PoseProvider(str(self.pose_record.path), conf_thresh=confidence, device=device, tracker=tracker)
        self.objects = ObjectDisambiguationProvider(
            model_name=str(self.object_record.path), conf_thresh=confidence, device=device,
            chit_model_name=str(self.chit_record.path) if self.chit_record else None,
        )
        self.temporal = BaselineTemporalEngine(phone_confidence=phone_review_confidence)
        self.preprocessing = (
            PreprocessingPipeline(config=preprocessing_config)
            if enable_preprocessing else None
        )
        self.enable_privacy_mask = enable_privacy_mask
        self._provenance = registry.provenance(("pose", "objects"))
        if self.chit_record:
            self._provenance["chit"] = self.chit_record.provenance()
        if self.phone_record:
            self._provenance['phone_candidate'] = self.phone_record.provenance()

    def provenance(self):
        return dict(self._provenance)

    def preprocessing_summary(self) -> dict[str, Any]:
        """Return explicit run provenance; raw input remains the safe default."""
        if self.preprocessing is None:
            return {"enabled": False, "mode": "raw_input"}
        return {
            "enabled": True,
            "config": self.preprocessing.config.to_dict(),
            "stats": self.preprocessing.get_stats_summary(),
        }

    def start(self) -> None:
        self.pose.start()
        self.objects.start()
        if self.phone:
            self.phone.start()

    def close(self) -> None:
        self.objects.close()
        self.pose.close()
        if self.phone:
            self.phone.close()

    def process(self, frame_index: int, timestamp_sec: float, frame_bgr: np.ndarray, *, camera_id: str, video_id: str) -> FrameInference:
        # Enhancement affects inference only. Privacy masking and review imagery
        # intentionally use the unmodified decoded frame.
        inference_frame = frame_bgr
        quality_report = None
        if self.preprocessing is not None:
            inference_frame, report = self.preprocessing.process(frame_bgr, frame_idx=frame_index)
            quality_report = report.to_dict()
        people, _ = self.pose.process_frame(frame_index, timestamp_sec, inference_frame, camera_id, video_id)
        _, objects = self.objects.process_frame(frame_index, timestamp_sec, inference_frame, camera_id, video_id, people)
        if self.phone:
            objects.extend(self.phone.process_frame(frame_index,timestamp_sec,inference_frame,people,camera_id=camera_id,video_id=video_id))
            retained = []
            for obj in sorted(objects,key=lambda p:p.confidence,reverse=True):
                if any(other.object_class == obj.object_class and ObjectDisambiguationProvider._iou(obj.bbox_xyxy_px,other.bbox_xyxy_px) >= .5 for other in retained):
                    continue
                retained.append(obj)
            objects = retained
        if self.enable_privacy_mask:
            privacy_frame, annotations = apply_privacy_mask_and_collect_annotations(frame_bgr, people)
        else:
            # Local review mode: retain a readable frame while preserving the
            # same geometric inputs used by temporal evidence processing.
            privacy_frame = frame_bgr.copy()
            annotations = collect_candidate_annotations(frame_bgr, people)
        events = self.temporal.update(timestamp_sec, people, objects, annotations)
        return FrameInference(
            people, objects, privacy_frame, annotations, events, self._provenance,
            quality_report, self.temporal.temporal_evidence(timestamp_sec),
        )

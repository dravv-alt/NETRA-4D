"""Workstation-relative CBT behavioural evidence graph.

This module reasons over calibrated workstation semantics and real perception
observations. Scores are review-priority evidence, never cheating probability.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from drishti.temporal import BaselineTemporalEngine, ReviewEvent
from .scene import (
    AutomaticWorkstationCalibrator, CBTSceneMap, CandidateBinding,
    CandidateWorkstationBinder, Polygon, Workstation,
)


class AtomicEventType(str, Enum):
    OWN_SCREEN_ATTENTION = "OWN_SCREEN_ATTENTION"
    OWN_SCREEN_DISENGAGEMENT = "OWN_SCREEN_DISENGAGEMENT"
    RETURN_TO_OWN_SCREEN = "RETURN_TO_OWN_SCREEN"
    NEIGHBOUR_SCREEN_ATTENTION = "NEIGHBOUR_SCREEN_ATTENTION"
    HAND_BELOW_DESK = "HAND_BELOW_DESK"
    PHONE_VISIBLE = "PHONE_VISIBLE"
    PHONE_ASSOCIATED_WITH_HAND = "PHONE_ASSOCIATED_WITH_HAND"
    CHIT_VISIBLE = "CHIT_VISIBLE"
    HAND_CROSSES_SEAT_BOUNDARY = "HAND_CROSSES_SEAT_BOUNDARY"
    MUTUAL_ATTENTION = "MUTUAL_ATTENTION"
    NORMAL_INPUT_ACTIVITY = "NORMAL_INPUT_ACTIVITY"
    STATIC_WORKSTATION_OBJECT = "STATIC_WORKSTATION_OBJECT"
    TRACK_OCCLUSION = "TRACK_OCCLUSION"


class PeakLifecycle(str, Enum):
    NORMAL = "NORMAL"
    OBSERVATION = "OBSERVATION"
    CANDIDATE = "CANDIDATE"
    WATCH = "WATCH"
    SPECIALIST_REVIEW = "SPECIALIST_REVIEW"
    REVIEW = "REVIEW"
    COOLDOWN = "COOLDOWN"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class AtomicEvidence:
    event_type: AtomicEventType
    candidate_ids: tuple[int, ...]
    timestamp_sec: float
    confidence: float
    workstation_id: str | None
    target_workstation_id: str | None = None
    uncertainty: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateDeviceBinding:
    """One-to-one candidate to assigned workstation screen mapping."""

    candidate_id: int
    device_id: str
    workstation_id: str
    confidence: float
    method: str
    detected_class: str | None = None
    device_bbox_xyxy_norm: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "device_id": self.device_id,
            "workstation_id": self.workstation_id,
            "confidence": self.confidence,
            "method": self.method,
            "detected_class": self.detected_class,
            "device_bbox_xyxy_norm": list(self.device_bbox_xyxy_norm) if self.device_bbox_xyxy_norm else None,
        }


@dataclass
class CandidateBaseline:
    started_at_sec: float
    head_samples: list[float] = field(default_factory=list)
    torso_samples: list[float] = field(default_factory=list)
    ready: bool = False

    def update(self, head: float | None, torso: float | None, timestamp_sec: float, learning_sec: float) -> None:
        if self.ready:
            return
        if head is not None:
            self.head_samples.append(head)
        if torso is not None:
            self.torso_samples.append(torso)
        self.ready = timestamp_sec - self.started_at_sec >= learning_sec and len(self.head_samples) >= 8

    @property
    def head_mean(self) -> float:
        return float(np.median(self.head_samples)) if self.head_samples else 0.0

    @property
    def head_scale(self) -> float:
        if len(self.head_samples) < 3:
            return 0.25
        median = self.head_mean
        mad = float(np.median(np.abs(np.asarray(self.head_samples) - median)))
        return max(0.18, 1.4826 * mad)


@dataclass
class InteractionEdge:
    source_candidate_id: int
    target_candidate_id: int
    target_workstation_id: str
    attention_count: int = 0
    cumulative_attention_sec: float = 0.0
    mutual_attention_count: int = 0
    hand_boundary_count: int = 0
    temporal_overlap_sec: float = 0.0
    uncertainty_sum: float = 0.0
    last_attention_sec: float = -1e9

    @property
    def mean_uncertainty(self) -> float:
        return self.uncertainty_sum / self.attention_count if self.attention_count else 1.0


@dataclass
class BehaviourHypothesisState:
    candidate_id: int
    phone_use_evidence: float = 0.0
    chit_transfer_evidence: float = 0.0
    neighbour_screen_evidence: float = 0.0
    candidate_interaction_evidence: float = 0.0
    normal_activity_evidence: float = 0.0
    uncertainty: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "phone_use_evidence": self.phone_use_evidence,
            "chit_transfer_evidence": self.chit_transfer_evidence,
            "neighbour_screen_evidence": self.neighbour_screen_evidence,
            "candidate_interaction_evidence": self.candidate_interaction_evidence,
            "normal_activity_evidence": self.normal_activity_evidence,
            "uncertainty": self.uncertainty,
        }


@dataclass
class _Excursion:
    target_workstation_id: str
    started_at_sec: float
    last_seen_sec: float
    support_sum: float = 0.0
    observations: int = 0
    uncertainty_sum: float = 0.0
    sustained_emitted: bool = False


@dataclass
class _PhoneSequence:
    phase: str = "NORMAL"
    started_at_sec: float = 0.0
    phone_first_seen_sec: float | None = None
    phone_last_seen_sec: float | None = None
    use_emitted: bool = False


@dataclass
class _ObjectZoneHistory:
    first_seen_sec: float
    last_seen_sec: float
    observations: int = 1


@dataclass(frozen=True)
class SpecialistRequest:
    question: str
    candidate_ids: tuple[int, ...]
    pre_roll_sec: float
    post_roll_sec: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_role": "geometry_uncertainty_resolver",
            "question": self.question,
            "candidate_ids": list(self.candidate_ids),
            "pre_roll_sec": self.pre_roll_sec,
            "post_roll_sec": self.post_roll_sec,
            "reason": self.reason,
        }


def _point(person: Any, name: str, minimum: float = 0.30) -> tuple[float, float, float] | None:
    point = (person.keypoints_2d or {}).get(name)
    if point is None or point.confidence < minimum:
        return None
    return float(point.x), float(point.y), float(point.confidence)


def _head_signature(person: Any) -> tuple[float | None, float]:
    left, right, nose = (_point(person, name) for name in ("left_eye", "right_eye", "nose"))
    if not left or not right or not nose:
        return None, 0.0
    span = abs(right[0] - left[0])
    if span < 0.003:
        return None, 0.0
    return (nose[0] - (left[0] + right[0]) / 2.0) / (span / 2.0), min(left[2], right[2], nose[2])


def _torso_signature(person: Any) -> tuple[float | None, float]:
    left_shoulder, right_shoulder = _point(person, "left_shoulder"), _point(person, "right_shoulder")
    left_hip, right_hip = _point(person, "left_hip"), _point(person, "right_hip")
    if not all((left_shoulder, right_shoulder, left_hip, right_hip)):
        return None, 0.0
    shoulder_mid = (left_shoulder[0] + right_shoulder[0]) / 2.0
    hip_mid = (left_hip[0] + right_hip[0]) / 2.0
    width = max(abs(right_shoulder[0] - left_shoulder[0]), 0.01)
    return (shoulder_mid - hip_mid) / width, min(
        left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]
    )


def _candidate_head_origin(person: Any) -> tuple[float, float]:
    nose = _point(person, "nose", 0.15)
    if nose:
        return nose[0], nose[1]
    x1, y1, x2, y2 = person.bbox_xyxy_norm
    return (x1 + x2) / 2.0, y1 + 0.18 * (y2 - y1)


def _direction(source_x: float, target_x: float, tolerance: float = 0.025) -> str:
    if target_x < source_x - tolerance:
        return "LEFT"
    if target_x > source_x + tolerance:
        return "RIGHT"
    return "FORWARD"


class CBTBehaviouralReasoner:
    """Peak-stage, workstation-relative evidence reasoner for four CBT sequences."""

    def __init__(
        self,
        scene: CBTSceneMap,
        *,
        baseline_learning_sec: float = 20.0,
        neighbour_persistence_sec: float = 2.0,
        repeated_excursions: int = 3,
        repetition_window_sec: float = 90.0,
        phone_use_persistence_sec: float = 0.50,
        uncertainty_route_threshold: float = 0.45,
    ):
        self.scene = scene
        self.binder = CandidateWorkstationBinder(scene)
        self.baseline_learning_sec = baseline_learning_sec
        self.neighbour_persistence_sec = neighbour_persistence_sec
        self.repeated_excursions = repeated_excursions
        self.repetition_window_sec = repetition_window_sec
        self.phone_use_persistence_sec = phone_use_persistence_sec
        self.uncertainty_route_threshold = uncertainty_route_threshold
        self.baselines: dict[int, CandidateBaseline] = {}
        self.hypotheses: dict[int, BehaviourHypothesisState] = {}
        self.edges: dict[tuple[int, int], InteractionEdge] = {}
        self._excursions: dict[int, _Excursion] = {}
        self._excursion_history: dict[tuple[int, str], deque[tuple[float, float]]] = defaultdict(deque)
        self._phone: dict[int, _PhoneSequence] = defaultdict(_PhoneSequence)
        self._chit_cross_started: dict[tuple[int, str], float] = {}
        self._object_zone_history: dict[tuple[int, str, str, int, int], _ObjectZoneHistory] = {}
        self._last_attention: dict[tuple[int, str], float] = {}
        self._emission_cooldown: dict[tuple[str, tuple[int, ...]], float] = {}
        self.last_atomic_evidence: list[AtomicEvidence] = []
        self.last_bindings: dict[int, CandidateBinding] = {}
        self.last_device_bindings: dict[int, CandidateDeviceBinding] = {}
        self.lifecycle: dict[int, PeakLifecycle] = {}
        self.lifecycle_history: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def _workstation_objects(self, objects: list[Any], station: Workstation) -> list[Any]:
        regions = [station.seat, station.desk, station.own_monitor]
        return [item for item in objects if any(region and region.contains_box_center(item.bbox_xyxy_norm) for region in regions)]

    def _bind_candidate_devices(self, objects: list[Any]) -> dict[int, CandidateDeviceBinding]:
        """Assign each workstation screen to at most one currently bound candidate."""

        winner_by_station: dict[str, CandidateBinding] = {}
        for binding in self.last_bindings.values():
            if not binding.workstation_id:
                continue
            current = winner_by_station.get(binding.workstation_id)
            if current is None or (binding.confidence, -binding.candidate_id) > (current.confidence, -current.candidate_id):
                winner_by_station[binding.workstation_id] = binding

        device_classes = {"laptop", "monitor", "tv"}
        result: dict[int, CandidateDeviceBinding] = {}
        for workstation_id, binding in winner_by_station.items():
            station = self.scene.by_id()[workstation_id]
            detected = [
                item for item in objects
                if str(item.metadata.get("raw_model_class", "")).lower() in device_classes
                and station.own_monitor.contains_box_center(item.bbox_xyxy_norm)
            ]
            best = max(detected, key=lambda item: float(item.confidence), default=None)
            if best is not None:
                confidence = min(binding.confidence, float(best.confidence))
                method = "candidate_workstation_plus_current_device_detection"
                detected_class = str(best.metadata.get("raw_model_class", "")).lower()
                device_box = tuple(float(value) for value in best.bbox_xyxy_norm)
            else:
                confidence = binding.confidence * (0.85 if self.scene.calibration_status == "VALIDATED" else 0.65)
                method = "candidate_workstation_plus_scene_monitor_region"
                detected_class = None
                xs = [point[0] for point in station.own_monitor.points]
                ys = [point[1] for point in station.own_monitor.points]
                device_box = (min(xs), min(ys), max(xs), max(ys))
            candidate_id = binding.candidate_id
            result[candidate_id] = CandidateDeviceBinding(
                candidate_id=candidate_id,
                device_id=f"PC-{candidate_id}",
                workstation_id=workstation_id,
                confidence=confidence,
                method=method,
                detected_class=detected_class,
                device_bbox_xyxy_norm=device_box,
            )
        return result

    @staticmethod
    def _inside(region: Polygon | None, point: tuple[float, float] | None) -> bool:
        return bool(region and point and region.contains(point))

    def _attention(
        self, person: Any, annotation: Any, station: Workstation, baseline: CandidateBaseline,
        head_value: float | None, head_confidence: float,
    ) -> tuple[str, str | None, float, float, dict[str, Any]]:
        gaze = getattr(annotation, "gaze_proxy", "UNKNOWN")
        gaze_conf = float(getattr(annotation, "gaze_proxy_confidence", 0.0))
        origin = _candidate_head_origin(person)
        deviation = abs(head_value - baseline.head_mean) / baseline.head_scale if head_value is not None and baseline.ready else 0.0
        neighbour_candidates = []
        stations = self.scene.by_id()
        for neighbour_id in station.neighbour_ids:
            neighbour = stations[neighbour_id]
            target_direction = _direction(origin[0], neighbour.own_monitor.center[0])
            if gaze == target_direction and gaze in {"LEFT", "RIGHT"}:
                alignment = 1.0 / (1.0 + abs(neighbour.own_monitor.center[0] - origin[0]))
                neighbour_candidates.append((alignment, neighbour_id))
        uncertainty = 1.0 - min(head_confidence, gaze_conf)
        if neighbour_candidates and (not baseline.ready or deviation >= 1.0):
            support, target = max(neighbour_candidates)
            confidence = min(head_confidence, gaze_conf, 0.55 + 0.45 * support)
            return "NEIGHBOUR_SCREEN", target, confidence, uncertainty, {"head_baseline_deviation": deviation}
        if gaze == "FORWARD" or (baseline.ready and deviation < 1.0):
            return "OWN_SCREEN", station.workstation_id, max(head_confidence, gaze_conf) * 0.85, uncertainty, {"head_baseline_deviation": deviation}
        for aisle_index, aisle in enumerate(self.scene.aisle_regions):
            if gaze == _direction(origin[0], aisle.center[0]) and gaze != "UNKNOWN":
                return "AISLE", f"aisle:{aisle_index}", gaze_conf * 0.7, uncertainty, {"head_baseline_deviation": deviation}
        return "UNKNOWN", None, max(head_confidence, gaze_conf) * 0.4, max(uncertainty, 0.55), {"head_baseline_deviation": deviation}

    def _atomic_for_candidate(
        self, timestamp: float, person: Any, annotation: Any, binding: CandidateBinding,
        objects: list[Any], people_by_station: dict[str, int],
    ) -> list[AtomicEvidence]:
        candidate_id = int(person.entity_local_id)
        if not binding.workstation_id:
            return [AtomicEvidence(
                AtomicEventType.TRACK_OCCLUSION, (candidate_id,), timestamp,
                float(person.detection_confidence), None, uncertainty=1.0,
                details={"reason": "candidate_not_bound_to_calibrated_workstation"},
            )]
        station = self.scene.by_id()[binding.workstation_id]
        head_value, head_conf = _head_signature(person)
        torso_value, torso_conf = _torso_signature(person)
        baseline = self.baselines.setdefault(candidate_id, CandidateBaseline(timestamp))
        baseline.update(head_value, torso_value, timestamp, self.baseline_learning_sec)
        attention, target, support, uncertainty, attention_details = self._attention(
            person, annotation, station, baseline, head_value, head_conf
        )
        evidence = []
        if attention == "OWN_SCREEN":
            evidence.append(AtomicEvidence(
                AtomicEventType.OWN_SCREEN_ATTENTION, (candidate_id,), timestamp, support,
                station.workstation_id, station.workstation_id, uncertainty,
                {**attention_details, "baseline_ready": baseline.ready},
            ))
        else:
            evidence.append(AtomicEvidence(
                AtomicEventType.OWN_SCREEN_DISENGAGEMENT, (candidate_id,), timestamp, support,
                station.workstation_id, target, uncertainty,
                {**attention_details, "attention_target": attention, "baseline_ready": baseline.ready},
            ))
        if attention == "NEIGHBOUR_SCREEN" and target:
            target_candidate = people_by_station.get(target)
            ids = (candidate_id, target_candidate) if target_candidate is not None else (candidate_id,)
            evidence.append(AtomicEvidence(
                AtomicEventType.NEIGHBOUR_SCREEN_ATTENTION, ids, timestamp, support,
                station.workstation_id, target, uncertainty,
                {**attention_details, "head_support": head_conf, "torso_support": torso_conf},
            ))

        wrist_points = [point for name in ("left_wrist", "right_wrist") if (point := _point(person, name))]
        wrists = [(point[0], point[1]) for point in wrist_points]
        below = any(self._inside(station.under_desk, point) for point in wrists)
        if not below:
            hips = [point for name in ("left_hip", "right_hip") if (point := _point(person, name))]
            hip_y = float(np.mean([point[1] for point in hips])) if hips else None
            below = hip_y is not None and any(point[1] >= hip_y for point in wrists)
        if below:
            evidence.append(AtomicEvidence(
                AtomicEventType.HAND_BELOW_DESK, (candidate_id,), timestamp,
                min((point[2] for point in wrist_points), default=0.0), station.workstation_id,
                uncertainty=1.0 - min((point[2] for point in wrist_points), default=0.0),
                details={"region_source": "calibrated_under_desk" if station.under_desk else "hip_relative_fallback"},
            ))
        normal_input = any(
            self._inside(station.keyboard, point) or self._inside(station.mouse, point)
            for point in wrists
        )
        if normal_input:
            evidence.append(AtomicEvidence(
                AtomicEventType.NORMAL_INPUT_ACTIVITY, (candidate_id,), timestamp, 0.85,
                station.workstation_id, uncertainty=0.15,
            ))
        for point in wrists:
            if not station.seat.contains(point):
                target_station = next(
                    (candidate for candidate in self.scene.workstations if candidate.workstation_id != station.workstation_id and candidate.seat.contains(point)),
                    None,
                )
                if target_station:
                    evidence.append(AtomicEvidence(
                        AtomicEventType.HAND_CROSSES_SEAT_BOUNDARY, (candidate_id,), timestamp, 0.70,
                        station.workstation_id, target_station.workstation_id, uncertainty=0.30,
                    ))

        for detected in self._workstation_objects(objects, station):
            raw_class = detected.metadata.get("raw_model_class")
            center_regions = {
                "monitor": station.own_monitor.contains_box_center(detected.bbox_xyxy_norm),
                "keyboard": bool(station.keyboard and station.keyboard.contains_box_center(detected.bbox_xyxy_norm)),
                "mouse": bool(station.mouse and station.mouse.contains_box_center(detected.bbox_xyxy_norm)),
            }
            counter_evidence = any(center_regions.values()) and raw_class == "cell phone"
            x1, y1, x2, y2 = detected.bbox_xyxy_norm
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            zone_name = next((name for name, active in center_regions.items() if active), "workstation")
            history_key = (candidate_id, str(raw_class), zone_name, round(center_x * 40), round(center_y * 40))
            history = self._object_zone_history.get(history_key)
            if history is None:
                history = _ObjectZoneHistory(timestamp, timestamp)
                self._object_zone_history[history_key] = history
            else:
                history.last_seen_sec = timestamp
                history.observations += 1
            stationary = history.observations >= 5 and history.last_seen_sec - history.first_seen_sec >= 2.0
            if detected.is_static_equipment or counter_evidence:
                evidence.append(AtomicEvidence(
                    AtomicEventType.STATIC_WORKSTATION_OBJECT, (candidate_id,), timestamp,
                    float(detected.confidence) * (1.0 if stationary or detected.is_static_equipment else 0.65),
                    station.workstation_id, uncertainty=0.10 if stationary else 0.35,
                    details={"raw_class": raw_class, "calibrated_regions": center_regions, "stationary_confirmed": stationary, "observations": history.observations},
                ))
            if raw_class == "cell phone" and not stationary:
                zone_prior = 0.85 if center_regions["monitor"] else 0.65 if center_regions["mouse"] or center_regions["keyboard"] else 0.0
                adjusted_confidence = float(detected.confidence) * (1.0 - zone_prior)
                ownership = (float(detected.association_confidence) if detected.associated_entity_id == candidate_id else 0.0) * (1.0 - zone_prior)
                evidence.append(AtomicEvidence(
                    AtomicEventType.PHONE_VISIBLE, (candidate_id,), timestamp,
                    adjusted_confidence, station.workstation_id,
                    uncertainty=max(zone_prior, 1.0 - max(adjusted_confidence, ownership)),
                    details={"ownership_confidence": ownership, "association_method": detected.metadata.get("association_method"), "normal_device_zone_prior": zone_prior},
                ))
                if ownership >= 0.25:
                    evidence.append(AtomicEvidence(
                        AtomicEventType.PHONE_ASSOCIATED_WITH_HAND, (candidate_id,), timestamp,
                        min(float(detected.confidence), max(ownership, 0.25)), station.workstation_id,
                        uncertainty=1.0 - ownership,
                        details={"normal_device_zone_prior": zone_prior, "ownership_confidence": ownership},
                    ))
            object_class = getattr(detected.object_class, "value", detected.object_class)
            if object_class == "PAPER_CHIT":
                evidence.append(AtomicEvidence(
                    AtomicEventType.CHIT_VISIBLE, (candidate_id,), timestamp,
                    float(detected.confidence), station.workstation_id,
                    uncertainty=1.0 - float(detected.confidence),
                    details={"checkpoint": detected.metadata.get("custom_chit_checkpoint")},
                ))
        return evidence

    def _emit(
        self, timestamp: float, event_type: str, label: str, candidate_ids: tuple[int, ...],
        score: float, started: float, uncertainty: float, signals: tuple[str, ...], details: dict[str, Any],
    ) -> ReviewEvent | None:
        key = (event_type, candidate_ids)
        if timestamp - self._emission_cooldown.get(key, -1e9) < 3.0:
            return None
        self._emission_cooldown[key] = timestamp
        lifecycle = PeakLifecycle.SPECIALIST_REVIEW if uncertainty >= self.uncertainty_route_threshold else PeakLifecycle.REVIEW
        if lifecycle == PeakLifecycle.SPECIALIST_REVIEW:
            question = (
                "Resolve head/torso direction relative to the target workstation"
                if "SCREEN" in event_type or "INTERACTION" in event_type
                else "Resolve wrist, pelvis, and workstation-boundary geometry"
            )
            details = {**details, "specialist_request": SpecialistRequest(
                question, candidate_ids, 2.0, 3.0, "behaviourally relevant evidence has weak 2D geometry",
            ).as_dict()}
        details = {
            **details,
            "peak_lifecycle": lifecycle.value,
            "review_priority": float(np.clip(score * (1.0 - 0.35 * uncertainty), 0.0, 1.0)),
            "uncertainty": uncertainty,
            "important": "Review priority is not cheating probability.",
        }
        return ReviewEvent(
            event_type=event_type,
            label=label,
            candidate_ids=candidate_ids,
            evidence_kind="MODEL_PLUS_GEOMETRY_PROXY",
            evidence_score=float(np.clip(score, 0.0, 1.0)),
            started_at_sec=started,
            timestamp_sec=timestamp,
            duration_sec=max(0.0, timestamp - started),
            source_signals=signals,
            is_new=True,
            details=details,
        )

    def _sequence_events(self, timestamp: float, atomic: list[AtomicEvidence]) -> list[ReviewEvent]:
        output = []
        by_candidate: dict[int, list[AtomicEvidence]] = defaultdict(list)
        for item in atomic:
            by_candidate[item.candidate_ids[0]].append(item)
        for candidate_id, items in by_candidate.items():
            types = {item.event_type for item in items}
            neighbour = next((item for item in items if item.event_type == AtomicEventType.NEIGHBOUR_SCREEN_ATTENTION), None)
            own = AtomicEventType.OWN_SCREEN_ATTENTION in types
            if neighbour and neighbour.target_workstation_id:
                excursion = self._excursions.get(candidate_id)
                if excursion is None or excursion.target_workstation_id != neighbour.target_workstation_id:
                    excursion = _Excursion(neighbour.target_workstation_id, timestamp, timestamp)
                    self._excursions[candidate_id] = excursion
                excursion.last_seen_sec = timestamp
                excursion.support_sum += neighbour.confidence
                excursion.uncertainty_sum += neighbour.uncertainty
                excursion.observations += 1
                duration = timestamp - excursion.started_at_sec
                if duration >= 7.0 and not excursion.sustained_emitted:
                    excursion.sustained_emitted = True
                    event = self._emit(
                        timestamp, "NEIGHBOUR_SCREEN_OVERWATCH_REVIEW", "NEIGHBOUR SCREEN OVERWATCH REVIEW [PROXY]",
                        neighbour.candidate_ids, excursion.support_sum / excursion.observations,
                        excursion.started_at_sec, excursion.uncertainty_sum / excursion.observations,
                        ("workstation binding", "own-screen disengagement", "directed neighbour-monitor attention", "sustained temporal excursion"),
                        {"source_workstation": neighbour.workstation_id, "target_workstation": neighbour.target_workstation_id, "sequence": ["OWN_SCREEN_DISENGAGEMENT", "NEIGHBOUR_SCREEN_ATTENTION"]},
                    )
                    if event:
                        output.append(event)
            elif own and candidate_id in self._excursions:
                excursion = self._excursions.pop(candidate_id)
                duration = excursion.last_seen_sec - excursion.started_at_sec
                history = self._excursion_history[(candidate_id, excursion.target_workstation_id)]
                history.append((timestamp, duration))
                while history and timestamp - history[0][0] > self.repetition_window_sec:
                    history.popleft()
                if duration >= self.neighbour_persistence_sec or len(history) >= self.repeated_excursions:
                    target_candidate = next(
                        (binding.candidate_id for binding in self.last_bindings.values() if binding.workstation_id == excursion.target_workstation_id),
                        None,
                    )
                    ids = (candidate_id, target_candidate) if target_candidate is not None else (candidate_id,)
                    mean_support = excursion.support_sum / max(excursion.observations, 1)
                    uncertainty = excursion.uncertainty_sum / max(excursion.observations, 1)
                    event = self._emit(
                        timestamp, "NEIGHBOUR_SCREEN_OVERWATCH_REVIEW", "NEIGHBOUR SCREEN OVERWATCH REVIEW [PROXY]",
                        ids, min(1.0, mean_support + 0.08 * max(0, len(history) - 1)),
                        excursion.started_at_sec, uncertainty,
                        ("workstation binding", "own-screen baseline", "directed neighbour-monitor excursion", "return to own screen", "repetition history"),
                        {"source_workstation": self.last_bindings[candidate_id].workstation_id, "target_workstation": excursion.target_workstation_id, "excursion_duration_sec": duration, "excursions_in_90_sec": len(history), "sequence": ["OWN_SCREEN", "DISENGAGEMENT", "NEIGHBOUR_SCREEN", "RETURN_TO_OWN_SCREEN"]},
                    )
                    if event:
                        output.append(event)
                    if target_candidate is not None:
                        edge = self.edges.setdefault((candidate_id, target_candidate), InteractionEdge(candidate_id, target_candidate, excursion.target_workstation_id))
                        edge.attention_count += 1
                        edge.cumulative_attention_sec += duration
                        edge.uncertainty_sum += uncertainty
                        edge.last_attention_sec = timestamp

            phone_state = self._phone[candidate_id]
            hand_below = AtomicEventType.HAND_BELOW_DESK in types
            phone = next((item for item in items if item.event_type == AtomicEventType.PHONE_ASSOCIATED_WITH_HAND), None)
            if phone_state.phase == "NORMAL" and hand_below:
                phone_state.phase, phone_state.started_at_sec = "RETRIEVING", timestamp
            phone_context_allowed = bool(phone and (
                hand_below or phone.details.get("normal_device_zone_prior", 0.0) < 0.30
            ))
            if phone_context_allowed and phone_state.phase in {"NORMAL", "RETRIEVING", "CONCEALING"}:
                phone_state.phase = "USING"
                phone_state.started_at_sec = min(phone_state.started_at_sec or timestamp, timestamp)
                phone_state.phone_first_seen_sec = phone_state.phone_first_seen_sec or timestamp
            if phone:
                phone_state.phone_last_seen_sec = timestamp
            if phone_state.phase == "USING" and phone_state.phone_first_seen_sec is not None:
                duration = timestamp - phone_state.phone_first_seen_sec
                if duration >= self.phone_use_persistence_sec and not phone_state.use_emitted:
                    phone_state.use_emitted = True
                    event = self._emit(
                        timestamp, "PHONE_RETRIEVE_USE_CONCEAL_REVIEW", "PHONE RETRIEVE / USE REVIEW",
                        (candidate_id,), phone.confidence if phone else 0.0, phone_state.started_at_sec,
                        phone.uncertainty if phone else 0.5,
                        ("calibrated workstation", "hand below desk/retrieval context", "YOLO phone", "wrist ownership", "temporal persistence"),
                        {"sequence_phase": "PHONE_USE", "phone_visible_duration_sec": duration},
                    )
                    if event:
                        output.append(event)
            if phone_state.phase == "USING" and not phone and hand_below:
                phone_state.phase = "CONCEALING"
            if phone_state.phase == "CONCEALING" and not hand_below:
                phone_state.phase = "NORMAL"
                phone_state.phone_first_seen_sec = None
                phone_state.phone_last_seen_sec = None
                phone_state.use_emitted = False

            chit = next((item for item in items if item.event_type == AtomicEventType.CHIT_VISIBLE), None)
            boundary = next((item for item in items if item.event_type == AtomicEventType.HAND_CROSSES_SEAT_BOUNDARY), None)
            if chit and boundary:
                target_candidate = next(
                    (binding.candidate_id for binding in self.last_bindings.values() if binding.workstation_id == boundary.target_workstation_id),
                    None,
                )
                ids = (candidate_id, target_candidate) if target_candidate is not None else (candidate_id,)
                cross_key = (candidate_id, boundary.target_workstation_id or "UNKNOWN")
                started = self._chit_cross_started.setdefault(cross_key, timestamp)
                if timestamp - started >= 0.30:
                    event = self._emit(
                        timestamp, "CHIT_TRANSFER_REVIEW", "CHIT / PAPER TRANSFER REVIEW",
                        ids, min(chit.confidence, boundary.confidence), started,
                        max(chit.uncertainty, boundary.uncertainty),
                        ("custom chit checkpoint", "candidate ownership", "hand crosses calibrated workstation boundary"),
                        {"source_workstation": boundary.workstation_id, "target_workstation": boundary.target_workstation_id},
                    )
                    if event:
                        output.append(event)
                if target_candidate is not None:
                    edge = self.edges.setdefault((candidate_id, target_candidate), InteractionEdge(candidate_id, target_candidate, boundary.target_workstation_id or "UNKNOWN"))
                    edge.hand_boundary_count += 1

        for (source, target), edge in list(self.edges.items()):
            reverse = self.edges.get((target, source))
            if reverse and abs(edge.last_attention_sec - reverse.last_attention_sec) <= 10.0:
                edge.mutual_attention_count = min(edge.attention_count, reverse.attention_count)
                if edge.mutual_attention_count >= 2 and source < target:
                    uncertainty = max(edge.mean_uncertainty, reverse.mean_uncertainty)
                    event = self._emit(
                        timestamp, "COORDINATED_CANDIDATE_INTERACTION_REVIEW", "COORDINATED CANDIDATE INTERACTION REVIEW [PROXY]",
                        (source, target), min(1.0, 0.45 + 0.12 * edge.mutual_attention_count),
                        min(edge.last_attention_sec, reverse.last_attention_sec), uncertainty,
                        ("bidirectional workstation-directed attention", "repeated temporal excursions", "cross-candidate graph"),
                        {"forward_attention_count": edge.attention_count, "reverse_attention_count": reverse.attention_count, "mutual_attention_count": edge.mutual_attention_count},
                    )
                    if event:
                        output.append(event)
        return output

    def _update_hypotheses(self, atomic: list[AtomicEvidence]) -> None:
        grouped: dict[int, list[AtomicEvidence]] = defaultdict(list)
        for item in atomic:
            grouped[item.candidate_ids[0]].append(item)
        for candidate_id, items in grouped.items():
            types = {item.event_type for item in items}
            phone = max((item.confidence for item in items if item.event_type == AtomicEventType.PHONE_ASSOCIATED_WITH_HAND), default=0.0)
            chit = max((item.confidence for item in items if item.event_type == AtomicEventType.CHIT_VISIBLE), default=0.0)
            neighbour = max((item.confidence for item in items if item.event_type == AtomicEventType.NEIGHBOUR_SCREEN_ATTENTION), default=0.0)
            counter = max((item.confidence for item in items if item.event_type in {AtomicEventType.NORMAL_INPUT_ACTIVITY, AtomicEventType.STATIC_WORKSTATION_OBJECT}), default=0.0)
            uncertainty = float(np.mean([item.uncertainty for item in items])) if items else 1.0
            self.hypotheses[candidate_id] = BehaviourHypothesisState(
                candidate_id=candidate_id,
                phone_use_evidence=max(0.0, phone - 0.65 * counter),
                chit_transfer_evidence=chit,
                neighbour_screen_evidence=max(0.0, neighbour - 0.25 * counter),
                candidate_interaction_evidence=max((edge.attention_count / 5.0 for edge in self.edges.values() if edge.source_candidate_id == candidate_id), default=0.0),
                normal_activity_evidence=max(counter, 0.8 if AtomicEventType.OWN_SCREEN_ATTENTION in types else 0.0),
                uncertainty=uncertainty,
            )
            state = self.hypotheses[candidate_id]
            dominant = max(
                state.phone_use_evidence, state.chit_transfer_evidence,
                state.neighbour_screen_evidence, state.candidate_interaction_evidence,
            )
            next_lifecycle = (
                PeakLifecycle.NORMAL if dominant < 0.20 else
                PeakLifecycle.OBSERVATION if dominant < 0.35 else
                PeakLifecycle.CANDIDATE if dominant < 0.50 else
                PeakLifecycle.WATCH if dominant < 0.65 else
                PeakLifecycle.SPECIALIST_REVIEW if state.uncertainty >= self.uncertainty_route_threshold else PeakLifecycle.REVIEW
            )
            previous = self.lifecycle.get(candidate_id)
            if previous != next_lifecycle:
                self.lifecycle[candidate_id] = next_lifecycle
                self.lifecycle_history[candidate_id].append({
                    "from": previous.value if previous else None,
                    "to": next_lifecycle.value,
                    "timestamp_sec": max((item.timestamp_sec for item in items), default=0.0),
                })

    def update(
        self, timestamp_sec: float, people: list[Any], objects: list[Any], privacy_annotations: list[Any]
    ) -> list[ReviewEvent]:
        self.last_bindings = self.binder.bind(people)
        self.last_device_bindings = self._bind_candidate_devices(objects)
        people_by_station = {
            binding.workstation_id: candidate_id
            for candidate_id, binding in self.last_bindings.items() if binding.workstation_id
        }
        annotations = {
            int(person.entity_local_id): annotation
            for person, annotation in zip(people, privacy_annotations)
        }
        atomic = []
        for person in people:
            candidate_id = int(person.entity_local_id)
            annotation = annotations.get(candidate_id)
            if annotation is None:
                annotation = type("UnknownAttention", (), {"gaze_proxy": "UNKNOWN", "gaze_proxy_confidence": 0.0})()
            atomic.extend(self._atomic_for_candidate(
                timestamp_sec, person, annotation, self.last_bindings[candidate_id], objects, people_by_station
            ))
        for item in list(atomic):
            if item.event_type == AtomicEventType.OWN_SCREEN_ATTENTION and item.candidate_ids[0] in self._excursions:
                atomic.append(AtomicEvidence(
                    AtomicEventType.RETURN_TO_OWN_SCREEN, item.candidate_ids, timestamp_sec,
                    item.confidence, item.workstation_id, item.workstation_id,
                    item.uncertainty, {"closed_target_workstation": self._excursions[item.candidate_ids[0]].target_workstation_id},
                ))
        self.last_atomic_evidence = atomic
        self._update_hypotheses(atomic)
        events = self._sequence_events(timestamp_sec, atomic)
        for event in events:
            event.details["hypotheses"] = {
                str(candidate_id): self.hypotheses[candidate_id].as_dict()
                for candidate_id in event.candidate_ids if candidate_id in self.hypotheses
            }
            event.details["workstation_map"] = {
                "camera_id": self.scene.camera_id,
                "video_id": self.scene.video_id,
                "schema_version": self.scene.schema_version,
                "calibration_source_sha256": self.scene.provenance.source_frame_sha256,
            }
            event.details["candidate_lifecycle_history"] = {
                str(candidate_id): self.lifecycle_history.get(candidate_id, [])[-8:]
                for candidate_id in event.candidate_ids
            }
            event.details["candidate_device_bindings"] = {
                str(candidate_id): self.last_device_bindings[candidate_id].as_dict()
                for candidate_id in event.candidate_ids if candidate_id in self.last_device_bindings
            }
        return events

    def graph_snapshot(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"candidate_id": candidate_id, "workstation_id": self.last_bindings.get(candidate_id).workstation_id if candidate_id in self.last_bindings else None, "device_id": self.last_device_bindings.get(candidate_id).device_id if candidate_id in self.last_device_bindings else None, "lifecycle": self.lifecycle.get(candidate_id, PeakLifecycle.NORMAL).value, "hypotheses": state.as_dict()}
                for candidate_id, state in sorted(self.hypotheses.items())
            ],
            "edges": [
                {
                    "source_candidate_id": edge.source_candidate_id,
                    "target_candidate_id": edge.target_candidate_id,
                    "target_workstation_id": edge.target_workstation_id,
                    "attention_count": edge.attention_count,
                    "cumulative_attention_sec": edge.cumulative_attention_sec,
                    "mutual_attention_count": edge.mutual_attention_count,
                    "hand_boundary_count": edge.hand_boundary_count,
                    "uncertainty": edge.mean_uncertainty,
                }
                for edge in self.edges.values()
            ],
        }

    def frame_snapshot(self) -> dict[str, Any]:
        return {
            "engine": "CBTBehaviouralReasoner",
            "bindings": {
                str(candidate_id): {
                    "workstation_id": binding.workstation_id,
                    "confidence": binding.confidence,
                    "method": binding.method,
                }
                for candidate_id, binding in self.last_bindings.items()
            },
            "candidate_device_bindings": {
                str(candidate_id): binding.as_dict()
                for candidate_id, binding in self.last_device_bindings.items()
            },
            "atomic_evidence": [
                {
                    "event_type": item.event_type.value,
                    "candidate_ids": list(item.candidate_ids),
                    "timestamp_sec": item.timestamp_sec,
                    "confidence": item.confidence,
                    "workstation_id": item.workstation_id,
                    "target_workstation_id": item.target_workstation_id,
                    "uncertainty": item.uncertainty,
                    "details": item.details,
                }
                for item in self.last_atomic_evidence
            ],
            "hypotheses": {
                str(candidate_id): state.as_dict() for candidate_id, state in self.hypotheses.items()
            },
        }


def draw_cbt_context(
    frame: np.ndarray,
    reasoner: CBTBehaviouralReasoner,
    *,
    timestamp_sec: float = 0.0,
    chair_display_seconds: float = 5.0,
) -> np.ndarray:
    """Draw assigned chairs briefly, then retain candidate/device context only."""
    output = frame.copy()
    height, width = output.shape[:2]

    def points(region: Polygon) -> np.ndarray:
        return np.asarray([(round(x * width), round(y * height)) for x, y in region.points], dtype=np.int32)

    assigned_station_ids = {
        binding.workstation_id
        for binding in reasoner.last_bindings.values()
        if binding.workstation_id
    }
    for station_id in sorted(assigned_station_ids):
        station = reasoner.scene.by_id()[station_id]
        if timestamp_sec < chair_display_seconds:
            cv2.polylines(output, [points(station.seat)], True, (0, 190, 80), 1, cv2.LINE_AA)
        # The assigned screen remains visible after chair geometry is removed.
        cv2.polylines(output, [points(station.own_monitor)], True, (255, 200, 0), 2, cv2.LINE_AA)
    for candidate_id, binding in reasoner.last_bindings.items():
        if not binding.workstation_id:
            continue
        station = reasoner.scene.by_id()[binding.workstation_id]
        x, y = station.seat.center
        cv2.putText(output, f"C{candidate_id}->{binding.workstation_id} {binding.confidence:.2f}",
                    (round(x * width), round(y * height)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"C{candidate_id}->{binding.workstation_id} {binding.confidence:.2f}",
                    (round(x * width), round(y * height)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (20, 80, 20), 1, cv2.LINE_AA)
        device = reasoner.last_device_bindings.get(candidate_id)
        if device:
            monitor_x, monitor_y = station.own_monitor.center
            label = f"{device.device_id} <-> C{candidate_id} {device.confidence:.2f}"
            cv2.putText(output, label, (round(monitor_x * width), round(monitor_y * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output, label, (round(monitor_x * width), round(monitor_y * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 90, 0), 1, cv2.LINE_AA)
    return output


class AutomaticCBTBehaviouralEngine:
    """Default engine: auto-calibrate, then promote to workstation reasoning."""

    def __init__(
        self, camera_id: str, video_id: str, reference_width: int, reference_height: int,
        *, calibration_warmup_sec: float = 8.0, baseline_engine: BaselineTemporalEngine | None = None,
    ):
        self.calibrator = AutomaticWorkstationCalibrator(
            camera_id, video_id, reference_width, reference_height,
            warmup_sec=calibration_warmup_sec,
        )
        self.baseline_engine = baseline_engine or BaselineTemporalEngine()
        self.reasoner: CBTBehaviouralReasoner | None = None
        self.last_attempt_sec = -1e9
        self.status = "WARMING_UP"

    def update(self, timestamp_sec, people, objects, privacy_annotations) -> list[ReviewEvent]:
        if self.reasoner is not None:
            return self.reasoner.update(timestamp_sec, people, objects, privacy_annotations)
        self.calibrator.observe(timestamp_sec, people, objects)
        if self.calibrator.ready(timestamp_sec) and timestamp_sec - self.last_attempt_sec >= 5.0:
            self.last_attempt_sec = timestamp_sec
            scene = self.calibrator.finalize()
            if scene is not None:
                self.reasoner = CBTBehaviouralReasoner(scene)
                self.status = "AUTOMATIC_MAP_ACTIVE"
                return self.reasoner.update(timestamp_sec, people, objects, privacy_annotations)
            self.status = "INSUFFICIENT_STABLE_WORKSTATIONS_USING_BASELINE"
        return self.baseline_engine.update(timestamp_sec, people, objects, privacy_annotations)

    @property
    def active_reasoner(self) -> CBTBehaviouralReasoner | None:
        return self.reasoner

    @property
    def scene(self) -> CBTSceneMap | None:
        return self.reasoner.scene if self.reasoner else None

    def frame_snapshot(self) -> dict[str, Any]:
        if self.reasoner:
            return {**self.reasoner.frame_snapshot(), "automatic_calibration_status": self.status}
        return {
            "engine": "AutomaticCBTBehaviouralEngine",
            "automatic_calibration_status": self.status,
            "frames_observed": self.calibrator.frames_observed,
            "person_clusters": len(self.calibrator.people),
            "monitor_clusters": len(self.calibrator.monitors),
        }

    def graph_snapshot(self) -> dict[str, Any]:
        if self.reasoner:
            return self.reasoner.graph_snapshot()
        return {"nodes": [], "edges": [], "status": self.status}

"""Lightweight temporal review events built only from real model observations.

The engine deliberately calls geometric behavior inferences *proxies*.  It
does not convert a low hand into a detected chit or head direction into proven
screen copying.  The implementation is CPU-only and suitable for the live
fast path after YOLO pose/object inference and ByteTrack association.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class ReviewEvent:
    event_type: str
    label: str
    candidate_ids: tuple[int, ...]
    evidence_kind: str
    evidence_score: float
    started_at_sec: float
    timestamp_sec: float
    duration_sec: float
    source_signals: tuple[str, ...]
    is_new: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_ids"] = list(self.candidate_ids)
        payload["source_signals"] = list(self.source_signals)
        return payload


@dataclass
class _Proposal:
    event_type: str
    label: str
    candidate_ids: tuple[int, ...]
    evidence_kind: str
    score: float
    minimum_duration_sec: float
    source_signals: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _EventState:
    proposal: _Proposal
    first_seen_sec: float
    last_seen_sec: float
    active: bool = False
    emitted: bool = False


def _point(person: Any, name: str, minimum_confidence: float = 0.30) -> tuple[float, float, float] | None:
    value = (person.keypoints_2d or {}).get(name)
    if value is None or value.confidence < minimum_confidence:
        return None
    return float(value.x), float(value.y), float(value.confidence)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _gaze_toward(
    gaze: tuple[str, float], source_x: float, target_x: float, minimum_confidence: float = 0.35
) -> bool:
    direction, confidence = gaze
    if confidence < minimum_confidence or direction not in {"LEFT", "RIGHT"}:
        return False
    return (target_x < source_x and direction == "LEFT") or (target_x > source_x and direction == "RIGHT")


class BaselineTemporalEngine:
    """Stateful, low-cost review-event engine for the primary renderer."""

    def __init__(
        self,
        *,
        phone_confidence: float = 0.20,
        maximum_gap_sec: float = 0.35,
        hand_low_duration_sec: float = 1.25,
        interaction_duration_sec: float = 0.75,
        screen_duration_sec: float = 0.80,
        interaction_max_distance: float = 0.42,
        screen_max_distance: float = 0.55,
        gaze_min_confidence: float = 0.35,
    ) -> None:
        self.phone_confidence = phone_confidence
        self.maximum_gap_sec = maximum_gap_sec
        self.hand_low_duration_sec = hand_low_duration_sec
        self.interaction_duration_sec = interaction_duration_sec
        self.screen_duration_sec = screen_duration_sec
        self.interaction_max_distance = interaction_max_distance
        self.screen_max_distance = screen_max_distance
        self.gaze_min_confidence = gaze_min_confidence
        self._states: dict[tuple[str, tuple[int, ...], str], _EventState] = {}

    @staticmethod
    def _gaze_by_candidate(people: list[Any], privacy_annotations: list[Any]) -> dict[int, tuple[str, float]]:
        return {
            int(person.entity_local_id): (annotation.gaze_proxy, float(annotation.gaze_proxy_confidence))
            for person, annotation in zip(people, privacy_annotations)
        }

    def _phone_proposals(self, objects: list[Any]) -> list[_Proposal]:
        proposals = []
        for detected in objects:
            if detected.metadata.get("raw_model_class") != "cell phone":
                continue
            if detected.confidence < self.phone_confidence or detected.associated_entity_id is None:
                continue
            candidate_id = int(detected.associated_entity_id)
            proposals.append(_Proposal(
                event_type="PHONE_USE_REVIEW",
                label="PHONE USE REVIEW",
                candidate_ids=(candidate_id,),
                evidence_kind="MODEL_PLUS_GEOMETRY",
                score=float(min(detected.confidence, max(detected.association_confidence, 0.25))),
                minimum_duration_sec=0.30,
                source_signals=("YOLO cell phone", "phone-person/wrist association", "temporal persistence"),
                details={
                    "object_confidence": float(detected.confidence),
                    "association_confidence": float(detected.association_confidence),
                    "association_method": detected.metadata.get("association_method"),
                },
            ))
        return proposals

    @staticmethod
    def _chit_proposals(objects: list[Any]) -> list[_Proposal]:
        proposals = []
        for detected in objects:
            object_class = getattr(detected, "object_class", None)
            if getattr(object_class, "value", object_class) != "PAPER_CHIT":
                continue
            if detected.associated_entity_id is None:
                continue
            proposals.append(_Proposal(
                event_type="CHIT_USE_REVIEW",
                label="CHIT / PAPER USE REVIEW",
                candidate_ids=(int(detected.associated_entity_id),),
                evidence_kind="MODEL_PLUS_GEOMETRY",
                score=float(min(detected.confidence, max(detected.association_confidence, 0.25))),
                minimum_duration_sec=0.30,
                source_signals=("custom chit checkpoint", "paper-person/wrist association", "temporal persistence"),
                details={
                    "object_confidence": float(detected.confidence),
                    "checkpoint": detected.metadata.get("custom_chit_checkpoint"),
                    "inference_view": detected.metadata.get("inference_view"),
                },
            ))
        return proposals

    def _low_hand_proposals(self, people: list[Any]) -> list[_Proposal]:
        proposals = []
        for person in people:
            hips = [point for name in ("left_hip", "right_hip") if (point := _point(person, name))]
            wrists = [point for name in ("left_wrist", "right_wrist") if (point := _point(person, name))]
            if not wrists:
                continue
            x1, y1, x2, y2 = person.bbox_xyxy_norm
            reference_y = float(np.mean([point[1] for point in hips])) if hips else y1 + 0.62 * (y2 - y1)
            low = [point for point in wrists if point[1] >= reference_y]
            if not low:
                continue
            proposals.append(_Proposal(
                event_type="LOW_HAND_POSSIBLE_CHIT_REVIEW",
                label="LOW HAND / POSSIBLE CHIT REVIEW [PROXY]",
                candidate_ids=(int(person.entity_local_id),),
                evidence_kind="BEHAVIORAL_PROXY",
                score=float(min(point[2] for point in low)),
                minimum_duration_sec=self.hand_low_duration_sec,
                source_signals=("YOLO pose wrist", "hip/person-relative low-hand geometry", "temporal persistence"),
                details={"important": "No chit or paper object was detected; this is a concealment proxy."},
            ))
        return proposals

    def _interaction_proposals(self, people: list[Any], gaze: dict[int, tuple[str, float]]) -> list[_Proposal]:
        proposals = []
        ordered = sorted(people, key=lambda person: person.entity_local_id)
        for index, first in enumerate(ordered):
            first_id = int(first.entity_local_id)
            first_center = _center(first.bbox_xyxy_norm)
            for second in ordered[index + 1:]:
                second_id = int(second.entity_local_id)
                second_center = _center(second.bbox_xyxy_norm)
                distance = float(np.linalg.norm(np.asarray(first_center) - np.asarray(second_center)))
                if distance > self.interaction_max_distance:
                    continue
                mutual = (
                    _gaze_toward(gaze.get(first_id, ("UNKNOWN", 0.0)), first_center[0], second_center[0], self.gaze_min_confidence)
                    and _gaze_toward(gaze.get(second_id, ("UNKNOWN", 0.0)), second_center[0], first_center[0], self.gaze_min_confidence)
                )
                if not mutual:
                    continue
                score = min(gaze[first_id][1], gaze[second_id][1], max(0.0, 1.0 - distance / self.interaction_max_distance))
                proposals.append(_Proposal(
                    event_type="CANDIDATE_INTERACTION_REVIEW",
                    label="CANDIDATE INTERACTION REVIEW [PROXY]",
                    candidate_ids=(first_id, second_id),
                    evidence_kind="BEHAVIORAL_PROXY",
                    score=float(score),
                    minimum_duration_sec=self.interaction_duration_sec,
                    source_signals=("two ByteTrack candidates", "mutual head-direction proxy", "candidate proximity", "temporal persistence"),
                    details={
                        "normalized_center_distance": distance,
                        "gaze_proxy_confidence": min(gaze[first_id][1], gaze[second_id][1]),
                        "configured_max_distance": self.interaction_max_distance,
                        "configured_gaze_min_confidence": self.gaze_min_confidence,
                    },
                ))
        return proposals

    def _screen_proposals(
        self, people: list[Any], objects: list[Any], gaze: dict[int, tuple[str, float]]
    ) -> list[_Proposal]:
        proposals = []
        laptops = [item for item in objects if item.metadata.get("raw_model_class") == "laptop"]
        for laptop_index, laptop in enumerate(laptops):
            target_center = _center(laptop.bbox_xyxy_norm)
            owner = laptop.associated_entity_id
            for person in people:
                candidate_id = int(person.entity_local_id)
                if owner is not None and candidate_id == int(owner):
                    continue
                person_center = _center(person.bbox_xyxy_norm)
                distance = float(np.linalg.norm(np.asarray(person_center) - np.asarray(target_center)))
                if distance > self.screen_max_distance or not _gaze_toward(
                    gaze.get(candidate_id, ("UNKNOWN", 0.0)), person_center[0], target_center[0], self.gaze_min_confidence
                ):
                    continue
                proposals.append(_Proposal(
                    event_type="OTHER_SCREEN_LOOK_REVIEW",
                    label="OTHER SCREEN LOOK REVIEW [PROXY]",
                    candidate_ids=(candidate_id,),
                    evidence_kind="MODEL_PLUS_GEOMETRY_PROXY",
                    score=float(min(laptop.confidence, gaze[candidate_id][1], max(0.0, 1.0 - distance / self.screen_max_distance))),
                    minimum_duration_sec=self.screen_duration_sec,
                    source_signals=("YOLO laptop", "candidate head-direction proxy", "screen-candidate geometry", "temporal persistence"),
                    details={
                        "laptop_index": laptop_index,
                        "possible_owner_candidate_id": int(owner) if owner is not None else None,
                        "target_bbox_xyxy_norm": list(laptop.bbox_xyxy_norm),
                        "normalized_center_distance": distance,
                        "gaze_proxy_confidence": gaze[candidate_id][1],
                        "configured_max_distance": self.screen_max_distance,
                        "configured_gaze_min_confidence": self.gaze_min_confidence,
                        "important": "Head direction is not iris gaze; event requires human review.",
                    },
                ))
        return proposals

    def update(
        self,
        timestamp_sec: float,
        people: list[Any],
        objects: list[Any],
        privacy_annotations: list[Any],
    ) -> list[ReviewEvent]:
        gaze = self._gaze_by_candidate(people, privacy_annotations)
        proposals = (
            self._phone_proposals(objects)
            + self._chit_proposals(objects)
            + self._low_hand_proposals(people)
            + self._interaction_proposals(people, gaze)
            + self._screen_proposals(people, objects, gaze)
        )
        seen: set[tuple[str, tuple[int, ...], str]] = set()
        for proposal in proposals:
            detail_key = str(proposal.details.get("laptop_index", ""))
            key = (proposal.event_type, proposal.candidate_ids, detail_key)
            seen.add(key)
            state = self._states.get(key)
            if state is None or timestamp_sec - state.last_seen_sec > self.maximum_gap_sec:
                state = _EventState(proposal, timestamp_sec, timestamp_sec)
                self._states[key] = state
            else:
                state.proposal = proposal
                state.last_seen_sec = timestamp_sec
            if timestamp_sec - state.first_seen_sec >= proposal.minimum_duration_sec:
                state.active = True

        active_events: list[ReviewEvent] = []
        expired = []
        for key, state in self._states.items():
            if key not in seen and timestamp_sec - state.last_seen_sec > self.maximum_gap_sec:
                expired.append(key)
                continue
            if not state.active:
                continue
            is_new = not state.emitted
            state.emitted = True
            proposal = state.proposal
            duration_sec = max(0.0, timestamp_sec - state.first_seen_sec)
            # Confidence in an observed pattern may strengthen when the same
            # model/geometry evidence persists.  This is deliberately bounded
            # and remains an evidence score, never a cheating probability.
            persistence_window = max(proposal.minimum_duration_sec * 4.0, 1.0)
            persistence_boost = 0.25 * min(1.0, duration_sec / persistence_window)
            sustained_score = min(1.0, proposal.score + persistence_boost)
            active_events.append(ReviewEvent(
                event_type=proposal.event_type,
                label=proposal.label,
                candidate_ids=proposal.candidate_ids,
                evidence_kind=proposal.evidence_kind,
                evidence_score=sustained_score,
                started_at_sec=state.first_seen_sec,
                timestamp_sec=timestamp_sec,
                duration_sec=duration_sec,
                source_signals=proposal.source_signals,
                is_new=is_new,
                details=proposal.details,
            ))
        for key in expired:
            del self._states[key]
        return sorted(active_events, key=lambda event: (event.candidate_ids, event.event_type))


def draw_temporal_review_events(frame: np.ndarray, people: list[Any], events: list[ReviewEvent]) -> np.ndarray:
    """Draw review labels after privacy masking and candidate annotations."""

    output = frame.copy()
    people_by_id = {int(person.entity_local_id): person for person in people}
    height, width = output.shape[:2]
    for row, event in enumerate(events[:8]):
        color = (0, 0, 255) if "MODEL" in event.evidence_kind else (0, 165, 255)
        banner = (
            f"{event.label} | Candidate {','.join(map(str, event.candidate_ids))} | "
            f"evidence {event.evidence_score:.2f} | sustained {event.duration_sec:.1f}s"
        )
        y = min(height - 8, 24 + row * 24)
        cv2.rectangle(output, (4, y - 18), (min(width - 4, 12 + len(banner) * 9), y + 4), (15, 15, 15), -1)
        cv2.putText(output, banner, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        boxes = []
        for candidate_id in event.candidate_ids:
            person = people_by_id.get(candidate_id)
            if person is None:
                continue
            box = tuple(int(value) for value in person.bbox_xyxy_px)
            boxes.append(box)
            cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), color, 3)
        if len(boxes) == 2:
            first_center = tuple(int(value) for value in _center(boxes[0]))
            second_center = tuple(int(value) for value in _center(boxes[1]))
            cv2.line(output, first_center, second_center, color, 2, cv2.LINE_AA)
    return output

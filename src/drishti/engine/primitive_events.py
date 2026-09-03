"""Primitive Event Extraction and Temporal Spanning Engine.

Conforms to docs/ARCHITECTURE.md Layer 3: Temporal Behaviour Engine.
Detects atomic biomechanical events with explicit start, end, and duration spans:
- HEAD_TURN, TORSO_ROTATION, HAND_MOVEMENT, HAND_BELOW_DESK
- DUAL_HAND_CONVERGENCE, HAND_TO_OBJECT, NEIGHBOUR_ORIENTATION
- BOARD_ORIENTATION, INVIGILATOR_INTERACTION

Strict Invariant: Every PrimitiveEvent is an actual temporal span with
(start_time_sec, end_time_sec, duration_sec), not an instantaneous point observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from .features import CandidateBiomechanicalFeatures, GazeDirection


class PrimitiveEventType(str, Enum):
    """Catalog of atomic physical movements."""
    HEAD_TURN = "HEAD_TURN"
    TORSO_ROTATION = "TORSO_ROTATION"
    HAND_MOVEMENT = "HAND_MOVEMENT"
    HAND_BELOW_DESK = "HAND_BELOW_DESK"
    DUAL_HAND_CONVERGENCE = "DUAL_HAND_CONVERGENCE"
    HAND_TO_OBJECT = "HAND_TO_OBJECT"
    NEIGHBOUR_ORIENTATION = "NEIGHBOUR_ORIENTATION"
    BOARD_ORIENTATION = "BOARD_ORIENTATION"
    INVIGILATOR_INTERACTION = "INVIGILATOR_INTERACTION"


@dataclass
class PrimitiveEvent:
    """A continuous temporal interval during which a specific biomechanical state held."""
    event_id: str
    candidate_id: str
    event_type: PrimitiveEventType
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    confidence: float
    peak_magnitude: float
    details: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def update(self, current_time: float, magnitude: float, conf: float) -> None:
        self.end_time_sec = current_time
        self.duration_sec = max(0.0, current_time - self.start_time_sec)
        self.peak_magnitude = max(self.peak_magnitude, magnitude)
        self.confidence = (self.confidence + conf) / 2.0

    def close(self, end_time: float) -> None:
        self.end_time_sec = end_time
        self.duration_sec = max(0.0, end_time - self.start_time_sec)
        self.is_active = False


class PrimitiveEventDetector:
    """Detects onset, duration, and termination of primitive events for candidates."""

    def __init__(
        self,
        head_turn_yaw_threshold: float = 0.38,
        torso_rotation_threshold: float = 0.32,
        hand_convergence_threshold: float = 0.35,
        object_interaction_dist_threshold: float = 0.35,
        min_activation_conf: float = 0.40,
    ):
        self.head_turn_yaw_threshold = head_turn_yaw_threshold
        self.torso_rotation_threshold = torso_rotation_threshold
        self.hand_convergence_threshold = hand_convergence_threshold
        self.object_interaction_dist_threshold = object_interaction_dist_threshold
        self.min_activation_conf = min_activation_conf

        # Active events indexed by (candidate_id, PrimitiveEventType)
        self.active_events: Dict[tuple[str, PrimitiveEventType], PrimitiveEvent] = {}

    def process(
        self,
        features: CandidateBiomechanicalFeatures,
        board_aligned: bool = False,
        invigilator_nearby: bool = False,
        neighbour_seat_aligned: bool = False,
    ) -> List[PrimitiveEvent]:
        """Evaluate current frame features and return newly completed or active events."""
        t = features.timestamp_sec
        cand_id = features.candidate_id
        completed_events: List[PrimitiveEvent] = []

        # Current active criteria flags
        is_head_turned = (
            features.head.is_valid
            and abs(features.head.yaw_proxy) >= self.head_turn_yaw_threshold
            and features.head.confidence >= self.min_activation_conf
        )
        is_torso_turned = (
            features.torso.is_valid
            and abs(features.torso.orientation_ratio) >= self.torso_rotation_threshold
            and features.torso.confidence >= self.min_activation_conf
        )
        is_hand_below = (
            features.hands.left_hand_below_lap or features.hands.right_hand_below_lap
        )
        is_hands_converged = (
            features.hands.is_converged
            and features.hands.dual_hand_convergence_ratio <= self.hand_convergence_threshold
        )
        is_near_object = (
            features.hands.min_object_distance_ratio <= self.object_interaction_dist_threshold
        )
        is_neighbour_look = is_head_turned and neighbour_seat_aligned and not board_aligned
        is_board_look = (is_head_turned or features.head.direction == GazeDirection.UP) and board_aligned
        is_invig_look = is_head_turned and invigilator_nearby

        criteria = [
            (PrimitiveEventType.HEAD_TURN, is_head_turned, abs(features.head.yaw_proxy), features.head.confidence, {"direction": features.head.direction.value}),
            (PrimitiveEventType.TORSO_ROTATION, is_torso_turned, abs(features.torso.orientation_ratio), features.torso.confidence, {}),
            (PrimitiveEventType.HAND_BELOW_DESK, is_hand_below, 1.0, features.hands.confidence, {}),
            (PrimitiveEventType.DUAL_HAND_CONVERGENCE, is_hands_converged, 1.0 - features.hands.dual_hand_convergence_ratio, features.hands.confidence, {"ratio": features.hands.dual_hand_convergence_ratio}),
            (PrimitiveEventType.HAND_TO_OBJECT, is_near_object, 1.0 - features.hands.min_object_distance_ratio, features.hands.confidence, {"object": features.hands.nearest_object_class}),
            (PrimitiveEventType.NEIGHBOUR_ORIENTATION, is_neighbour_look, abs(features.head.yaw_proxy), features.head.confidence, {}),
            (PrimitiveEventType.BOARD_ORIENTATION, is_board_look, abs(features.head.yaw_proxy), features.head.confidence, {}),
            (PrimitiveEventType.INVIGILATOR_INTERACTION, is_invig_look, abs(features.head.yaw_proxy), features.head.confidence, {}),
        ]

        for ev_type, condition, magnitude, conf, details in criteria:
            key = (cand_id, ev_type)
            active = self.active_events.get(key)

            if condition:
                if active is None:
                    # New event started
                    event_id = f"{cand_id}_{ev_type.value}_{uuid.uuid4().hex[:8]}"
                    new_ev = PrimitiveEvent(
                        event_id=event_id,
                        candidate_id=cand_id,
                        event_type=ev_type,
                        start_time_sec=t,
                        end_time_sec=t,
                        duration_sec=0.0,
                        confidence=conf,
                        peak_magnitude=magnitude,
                        details=details,
                        is_active=True
                    )
                    self.active_events[key] = new_ev
                else:
                    # Continue existing event
                    active.update(t, magnitude, conf)
            else:
                if active is not None:
                    # Event concluded
                    active.close(t)
                    completed_events.append(active)
                    del self.active_events[key]

        return completed_events

    def close_all(self, current_time: float) -> List[PrimitiveEvent]:
        """Close any lingering active events at stream termination."""
        completed = []
        for ev in self.active_events.values():
            ev.close(current_time)
            completed.append(ev)
        self.active_events.clear()
        return completed

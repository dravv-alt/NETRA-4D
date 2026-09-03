"""Temporal Pattern Aggregation: Duration, Persistence, and Repetition.

Conforms to docs/ARCHITECTURE.md Layer 3: Temporal Behaviour Engine.
Enforces the Foundational Axiom: 'Behaviour != Single Frame'.
Implements:
1. Brief filter (fleeting movements < brief_threshold are filtered out as routine physiology).
2. PersistenceDetector (unbroken continuous hold exceeding duration threshold).
3. RepetitionDetector (sliding temporal window frequency counter).

Strict Invariant: REPETITION != PERSISTENCE.
- Repetition measures recurrence frequency over an extended window (e.g. 90s).
- Persistence measures continuous unbroken duration of an ongoing state.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from .primitive_events import PrimitiveEvent, PrimitiveEventType


class TemporalPatternType(str, Enum):
    BRIEF_ROUTINE = "BRIEF_ROUTINE"           # Fleeting, physiological (< brief_threshold)
    SUSTAINED_HOLD = "SUSTAINED_HOLD"         # Continuous unbroken hold (>= persistence_threshold)
    REPEATED_EPISODE = "REPEATED_EPISODE"     # Recurrent episodes (>= count in sliding window)
    REPEATED_AND_SUSTAINED = "REPEATED_AND_SUSTAINED"  # High recurrence + long duration


@dataclass
class AggregatedTemporalPattern:
    """High-level temporal pattern extracted from primitive events."""
    pattern_type: TemporalPatternType
    candidate_id: str
    event_type: PrimitiveEventType
    timestamp_sec: float
    duration_sec: float
    recurrence_count_in_window: int
    mean_confidence: float
    peak_magnitude: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


class PersistenceDetector:
    """Monitors active events to detect unbroken, sustained holds."""

    def __init__(self, persistence_thresholds: Optional[Dict[PrimitiveEventType, float]] = None):
        # Default persistence duration thresholds (seconds)
        self.thresholds: Dict[PrimitiveEventType, float] = persistence_thresholds or {
            PrimitiveEventType.HEAD_TURN: 1.8,
            PrimitiveEventType.TORSO_ROTATION: 2.2,
            PrimitiveEventType.HAND_BELOW_DESK: 1.5,
            PrimitiveEventType.DUAL_HAND_CONVERGENCE: 1.8,
            PrimitiveEventType.HAND_TO_OBJECT: 1.2,
            PrimitiveEventType.NEIGHBOUR_ORIENTATION: 2.0,
            PrimitiveEventType.BOARD_ORIENTATION: 3.0,
            PrimitiveEventType.INVIGILATOR_INTERACTION: 1.5,
        }
        self.emitted_events: set[str] = set()

    def check_active_event(
        self,
        event: PrimitiveEvent,
        current_time: float
    ) -> Optional[AggregatedTemporalPattern]:
        """Check if an active event has reached the sustained persistence threshold."""
        if not event.is_active or event.event_id in self.emitted_events:
            return None

        threshold = self.thresholds.get(event.event_type, 2.0)
        current_duration = current_time - event.start_time_sec

        if current_duration >= threshold:
            self.emitted_events.add(event.event_id)
            return AggregatedTemporalPattern(
                pattern_type=TemporalPatternType.SUSTAINED_HOLD,
                candidate_id=event.candidate_id,
                event_type=event.event_type,
                timestamp_sec=current_time,
                duration_sec=current_duration,
                recurrence_count_in_window=1,
                mean_confidence=event.confidence,
                peak_magnitude=event.peak_magnitude,
                description=(
                    f"Sustained unbroken {event.event_type.value} hold "
                    f"({current_duration:.2f}s >= threshold {threshold:.2f}s)"
                ),
                details={"event_id": event.event_id, "threshold_sec": threshold}
            )

        return None


class RepetitionDetector:
    """Tracks recurrence frequency of completed events within a sliding window (e.g. 90s)."""

    def __init__(
        self,
        window_duration_sec: float = 90.0,
        repetition_threshold: int = 3,
        brief_suppress_threshold_sec: float = 0.60
    ):
        self.window_duration_sec = window_duration_sec
        self.repetition_threshold = repetition_threshold
        self.brief_suppress_threshold_sec = brief_suppress_threshold_sec

        # History: (candidate_id, PrimitiveEventType) -> deque of (timestamp_sec, duration_sec, magnitude)
        self.history: Dict[Tuple[str, PrimitiveEventType], Deque[Tuple[float, float, float]]] = defaultdict(deque)

    def register_completed_event(
        self,
        event: PrimitiveEvent
    ) -> Optional[AggregatedTemporalPattern]:
        """Record completed event and evaluate recurrence frequency across sliding window."""
        t = event.end_time_sec
        key = (event.candidate_id, event.event_type)
        dq = self.history[key]

        # Purge entries older than the sliding window
        while dq and (t - dq[0][0]) > self.window_duration_sec:
            dq.popleft()

        # Brief pulse filter: Don't register micro-physiological twitches as recurring patterns
        if event.duration_sec < self.brief_suppress_threshold_sec:
            return AggregatedTemporalPattern(
                pattern_type=TemporalPatternType.BRIEF_ROUTINE,
                candidate_id=event.candidate_id,
                event_type=event.event_type,
                timestamp_sec=t,
                duration_sec=event.duration_sec,
                recurrence_count_in_window=len(dq) + 1,
                mean_confidence=event.confidence,
                peak_magnitude=event.peak_magnitude,
                description=f"Brief routine movement ({event.duration_sec:.2f}s < {self.brief_suppress_threshold_sec:.2f}s)",
                details={"brief": True}
            )

        # Add valid event to temporal history
        dq.append((t, event.duration_sec, event.peak_magnitude))
        count = len(dq)

        if count >= self.repetition_threshold:
            durations = [item[1] for item in dq]
            mean_dur = float(np.mean(durations))
            max_mag = float(np.max([item[2] for item in dq]))

            pattern_type = (
                TemporalPatternType.REPEATED_AND_SUSTAINED
                if mean_dur >= 1.5 else TemporalPatternType.REPEATED_EPISODE
            )

            return AggregatedTemporalPattern(
                pattern_type=pattern_type,
                candidate_id=event.candidate_id,
                event_type=event.event_type,
                timestamp_sec=t,
                duration_sec=event.duration_sec,
                recurrence_count_in_window=count,
                mean_confidence=event.confidence,
                peak_magnitude=max_mag,
                description=(
                    f"Recurrent {event.event_type.value} ({count} episodes in "
                    f"{self.window_duration_sec:.0f}s window, mean duration: {mean_dur:.2f}s)"
                ),
                details={
                    "window_duration_sec": self.window_duration_sec,
                    "count": count,
                    "mean_duration_sec": mean_dur
                }
            )

        return None

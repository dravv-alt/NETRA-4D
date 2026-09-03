"""Unit tests for Phase C: Primitive Events, Duration, Repetition & Persistence.

Verifies:
1. PrimitiveEvent temporal span creation (start, end, duration).
2. The 'Behaviour != Single Frame' axiom (brief movement filtered as routine).
3. PersistenceDetector triggering on continuous unbroken holds.
4. RepetitionDetector tracking frequency across sliding 90s windows.
5. Strict Invariant: REPETITION != PERSISTENCE (distinct independent mechanisms).
"""

import pytest

from drishti.engine.features import (
    CandidateBiomechanicalFeatures,
    GazeDirection,
    HandDynamicsFeatures,
    HeadFeatures,
    TorsoFeatures,
)
from drishti.engine.primitive_events import (
    PrimitiveEvent,
    PrimitiveEventDetector,
    PrimitiveEventType,
)
from drishti.engine.temporal_patterns import (
    AggregatedTemporalPattern,
    PersistenceDetector,
    RepetitionDetector,
    TemporalPatternType,
)


def mock_features(
    timestamp: float,
    cand_id: str = "Cand_01",
    head_yaw: float = 0.0,
    hands_converged: bool = False,
    hand_below_lap: bool = False,
) -> CandidateBiomechanicalFeatures:
    head = HeadFeatures(
        yaw_proxy=head_yaw,
        pitch_proxy=0.0,
        direction=GazeDirection.RIGHT if head_yaw > 0.35 else GazeDirection.FORWARD,
        angular_velocity=0.0,
        confidence=0.90,
        is_valid=True
    )
    torso = TorsoFeatures(
        orientation_ratio=0.0,
        tilt_angle_deg=0.0,
        is_turned=False,
        confidence=0.90,
        is_valid=True
    )
    hands = HandDynamicsFeatures(
        dual_hand_convergence_ratio=0.15 if hands_converged else 0.80,
        is_converged=hands_converged,
        max_hand_velocity=0.0,
        left_hand_below_lap=hand_below_lap,
        right_hand_below_lap=False,
        min_object_distance_ratio=1.0,
        nearest_object_class=None,
        confidence=0.90
    )
    return CandidateBiomechanicalFeatures(
        timestamp_sec=timestamp,
        candidate_id=cand_id,
        head=head,
        torso=torso,
        hands=hands
    )


def test_primitive_event_lifecycle_and_duration():
    detector = PrimitiveEventDetector(head_turn_yaw_threshold=0.35)

    # Frame 1: Normal posture at t=1.0s (no event)
    f1 = mock_features(1.0, head_yaw=0.0)
    completed = detector.process(f1)
    assert len(completed) == 0
    assert len(detector.active_events) == 0

    # Frame 2: Turns head at t=1.1s (event starts)
    f2 = mock_features(1.1, head_yaw=0.55)
    detector.process(f2)
    key = ("Cand_01", PrimitiveEventType.HEAD_TURN)
    assert key in detector.active_events
    active_ev = detector.active_events[key]
    assert active_ev.start_time_sec == 1.1
    assert active_ev.is_active is True

    # Frame 3: Holds turn at t=2.0s
    f3 = mock_features(2.0, head_yaw=0.60)
    detector.process(f3)
    assert active_ev.duration_sec == pytest.approx(0.90)

    # Frame 4: Returns to center at t=2.5s (event concludes!)
    f4 = mock_features(2.5, head_yaw=0.0)
    completed = detector.process(f4)
    assert len(completed) == 1
    ev = completed[0]
    assert ev.event_type == PrimitiveEventType.HEAD_TURN
    assert ev.start_time_sec == 1.1
    assert ev.end_time_sec == 2.5
    assert ev.duration_sec == pytest.approx(1.40)
    assert ev.is_active is False
    assert len(detector.active_events) == 0


def test_persistence_detector_triggers_only_on_unbroken_hold():
    persistence = PersistenceDetector()

    # Create an event that starts at t=0.0
    ev = PrimitiveEvent(
        event_id="ev_001",
        candidate_id="Cand_01",
        event_type=PrimitiveEventType.HEAD_TURN,
        start_time_sec=0.0,
        end_time_sec=1.0,
        duration_sec=1.0,
        confidence=0.9,
        peak_magnitude=0.6,
        is_active=True
    )

    # At t=1.0 (threshold is 1.8s) -> Not sustained yet
    p1 = persistence.check_active_event(ev, current_time=1.0)
    assert p1 is None

    # At t=2.0 (duration = 2.0s >= 1.8s threshold) -> Triggers SUSTAINED_HOLD!
    ev.update(2.0, magnitude=0.65, conf=0.9)
    p2 = persistence.check_active_event(ev, current_time=2.0)
    assert p2 is not None
    assert p2.pattern_type == TemporalPatternType.SUSTAINED_HOLD
    assert p2.duration_sec >= 1.8

    # Subsequent check on same active event should not duplicate alert
    p3 = persistence.check_active_event(ev, current_time=2.5)
    assert p3 is None


def test_repetition_detector_counts_episodes_in_window():
    repetition = RepetitionDetector(window_duration_sec=60.0, repetition_threshold=3, brief_suppress_threshold_sec=0.5)

    def make_event(start, end):
        return PrimitiveEvent(
            event_id=f"ev_{start}",
            candidate_id="Cand_01",
            event_type=PrimitiveEventType.HEAD_TURN,
            start_time_sec=start,
            end_time_sec=end,
            duration_sec=end - start,
            confidence=0.9,
            peak_magnitude=0.5,
            is_active=False
        )

    # Episode 1: duration 0.8s
    p1 = repetition.register_completed_event(make_event(0.0, 0.8))
    assert p1 is None  # count = 1 < 3

    # Episode 2: duration 0.7s at t=10.0s
    p2 = repetition.register_completed_event(make_event(10.0, 10.7))
    assert p2 is None  # count = 2 < 3

    # Episode 3: duration 0.9s at t=25.0s -> Reaches threshold 3!
    p3 = repetition.register_completed_event(make_event(25.0, 25.9))
    assert p3 is not None
    assert p3.pattern_type == TemporalPatternType.REPEATED_EPISODE
    assert p3.recurrence_count_in_window == 3


def test_invariant_repetition_distinct_from_persistence():
    """A continuous long hold must NOT be treated as multiple repetitions, and
    multiple brief glances must NOT be treated as a sustained hold."""
    persistence = PersistenceDetector()
    repetition = RepetitionDetector(window_duration_sec=60.0, repetition_threshold=3, brief_suppress_threshold_sec=0.5)

    # Case A: One continuous 4.0 second hold
    single_long_event = PrimitiveEvent(
        event_id="long_ev",
        candidate_id="Cand_01",
        event_type=PrimitiveEventType.HEAD_TURN,
        start_time_sec=0.0,
        end_time_sec=4.0,
        duration_sec=4.0,
        confidence=0.9,
        peak_magnitude=0.7,
        is_active=True
    )
    # Persistence triggers
    assert persistence.check_active_event(single_long_event, 4.0) is not None
    # Repetition does NOT trigger because recurrence count is only 1
    single_long_event.is_active = False
    rep_result = repetition.register_completed_event(single_long_event)
    assert rep_result is None

    # Case B: Multiple separate brief movements (< 0.5s each)
    # They should be flagged as BRIEF_ROUTINE and never trigger persistence
    brief1 = PrimitiveEvent("b1", "Cand_02", PrimitiveEventType.HEAD_TURN, 0.0, 0.3, 0.3, 0.9, 0.5, is_active=False)
    brief2 = PrimitiveEvent("b2", "Cand_02", PrimitiveEventType.HEAD_TURN, 2.0, 2.4, 0.4, 0.9, 0.5, is_active=False)
    brief3 = PrimitiveEvent("b3", "Cand_02", PrimitiveEventType.HEAD_TURN, 4.0, 4.3, 0.3, 0.9, 0.5, is_active=False)

    assert persistence.check_active_event(brief1, 0.3) is None
    res1 = repetition.register_completed_event(brief1)
    assert res1.pattern_type == TemporalPatternType.BRIEF_ROUTINE

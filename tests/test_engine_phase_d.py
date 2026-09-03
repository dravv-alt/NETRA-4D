"""Unit tests for Phase D: Zones, Hypotheses, State Machine & Escalation Gate.

Verifies:
1. SemanticZoneEngine polygon containment, neighbour alignment, and boundary crossing.
2. CompetingHypothesisEngine probability distribution, counter-evidence, and dual metrics.
3. CandidateStateMachine auditable lifecycle transitions and cooldown guards.
4. EscalationGate Stage 1 -> Stage 2 -> Stage 3 progressive precision hierarchy.
"""

import pytest

from drishti.engine.primitive_events import PrimitiveEventType
from drishti.engine.temporal_patterns import AggregatedTemporalPattern, TemporalPatternType
from drishti.engine.zones import (
    NormalizedPolygonZone,
    SemanticZoneEngine,
    SemanticZoneType,
    box_to_polygon,
)
from drishti.engine.hypotheses import (
    CompetingHypothesisEngine,
    HypothesisKey,
)
from drishti.engine.state_machine import (
    CandidateBehaviouralState,
    CandidateStateMachine,
)
from drishti.engine.escalation import (
    EscalationGate,
    EscalationStage,
)


def test_semantic_zone_engine_containment_and_neighbour_inference():
    engine = SemanticZoneEngine()

    # Register Candidate 1 desk at [0.10, 0.30] to [0.30, 0.60]
    engine.register_candidate_desk("Cand_01", (0.10, 0.30, 0.30, 0.60))
    # Register Candidate 2 desk laterally adjacent at [0.35, 0.30] to [0.55, 0.60]
    engine.register_candidate_desk("Cand_02", (0.35, 0.30, 0.55, 0.60))

    # Test point containment
    assert engine.identify_point_zone((0.20, 0.45)) == SemanticZoneType.DESK
    assert engine.identify_point_zone((0.90, 0.90)) == SemanticZoneType.UNKNOWN

    # Test neighbour inference
    engine.auto_infer_neighbours(max_lateral_dist=0.30)
    assert "Cand_02" in engine.neighbour_mappings["Cand_01"]
    assert "Cand_01" in engine.neighbour_mappings["Cand_02"]

    # Test gaze alignment
    aligned_right, target = engine.is_gaze_aligned_with_neighbour("Cand_01", "RIGHT")
    assert aligned_right is True
    assert target == "Cand_02"

    aligned_left, _ = engine.is_gaze_aligned_with_neighbour("Cand_01", "LEFT")
    assert aligned_left is False

    # Test hand crossing boundary into neighbour desk
    crossed, crossed_target = engine.is_hand_crossing_boundary("Cand_01", (0.40, 0.45))
    assert crossed is True
    assert crossed_target == "Cand_02"


def test_competing_hypotheses_and_counter_evidence():
    engine = CompetingHypothesisEngine()

    pattern = AggregatedTemporalPattern(
        pattern_type=TemporalPatternType.SUSTAINED_HOLD,
        candidate_id="Cand_01",
        event_type=PrimitiveEventType.HEAD_TURN,
        timestamp_sec=5.0,
        duration_sec=2.5,
        recurrence_count_in_window=1,
        mean_confidence=0.9,
        peak_magnitude=0.6,
        description="Sustained head turn"
    )

    # Case A: Head turn without counter-evidence -> Suspicion elevated
    res_suspicious = engine.evaluate(
        candidate_id="Cand_01",
        pattern=pattern,
        is_board_aligned=False,
        is_invigilator_nearby=False
    )
    assert res_suspicious.suspicion_score > 0.30
    assert len(res_suspicious.supporting_evidence) > 0

    # Case B: Identical head turn, but candidate is looking at the instruction BOARD
    # Counter-evidence MUST shift weight to H2 and neutralize suspicion!
    res_board = engine.evaluate(
        candidate_id="Cand_01",
        pattern=pattern,
        is_board_aligned=True,
        is_invigilator_nearby=False
    )
    assert res_board.dominant_hypothesis == HypothesisKey.H2_BOARD_CONTEXT
    assert res_board.suspicion_score < res_suspicious.suspicion_score
    assert any("board" in item.lower() for item in res_board.counter_evidence)


def test_candidate_state_machine_transitions():
    sm = CandidateStateMachine(candidate_id="Cand_01")
    engine = CompetingHypothesisEngine()

    assert sm.current_state == CandidateBehaviouralState.NORMAL

    pattern_routine = AggregatedTemporalPattern(
        pattern_type=TemporalPatternType.BRIEF_ROUTINE,
        candidate_id="Cand_01",
        event_type=PrimitiveEventType.HEAD_TURN,
        timestamp_sec=1.0,
        duration_sec=0.3,
        recurrence_count_in_window=1,
        mean_confidence=0.9,
        peak_magnitude=0.4,
        description="Brief look"
    )
    eval_routine = engine.evaluate("Cand_01", pattern_routine)
    state = sm.update(1.0, eval_routine)
    assert state == CandidateBehaviouralState.NORMAL

    # Now simulate suspicious sustained hold with cell phone detected
    pattern_cheat = AggregatedTemporalPattern(
        pattern_type=TemporalPatternType.SUSTAINED_HOLD,
        candidate_id="Cand_01",
        event_type=PrimitiveEventType.HAND_TO_OBJECT,
        timestamp_sec=10.0,
        duration_sec=3.0,
        recurrence_count_in_window=1,
        mean_confidence=0.95,
        peak_magnitude=0.8,
        description="Holding phone"
    )
    eval_cheat = engine.evaluate(
        "Cand_01", pattern_cheat, is_phone_detected=True, sensor_quality_ratio=1.0
    )
    state = sm.update(10.0, eval_cheat)
    # High confidence violation transitions directly to HIGH_PRIORITY_REVIEW
    assert state == CandidateBehaviouralState.HIGH_PRIORITY_REVIEW


def test_escalation_gate_routing():
    gate = EscalationGate(
        suspicion_escalation_threshold=0.40,
        uncertainty_escalation_threshold=0.25
    )

    # 1. Clear benign event: low suspicion -> No escalation
    from drishti.engine.hypotheses import EvidenceEvaluationResult, HypothesisDistribution
    benign_eval = EvidenceEvaluationResult(
        candidate_id="Cand_01",
        suspicion_score=0.10,
        uncertainty_score=0.10,
        hypotheses=HypothesisDistribution(),
        supporting_evidence=[],
        counter_evidence=["Normal device use"],
        dominant_hypothesis=HypothesisKey.H4_ERGONOMIC_BASELINE,
        requires_specialist=False
    )
    assert gate.should_escalate_to_stage2(benign_eval) is False

    # 2. Ambiguous event: Notable suspicion (0.55) AND High uncertainty (0.45) -> Escalate to Stage 2
    ambiguous_eval = EvidenceEvaluationResult(
        candidate_id="Cand_02",
        suspicion_score=0.55,
        uncertainty_score=0.45,
        hypotheses=HypothesisDistribution(),
        supporting_evidence=["Hands obscured"],
        counter_evidence=[],
        dominant_hypothesis=HypothesisKey.H1_PRIMARY_SUSPICION,
        requires_specialist=True
    )
    assert gate.should_escalate_to_stage2(ambiguous_eval) is True

    # Mock Stage 2 specialist resolving uncertainty
    class MockCropRes:
        is_suspicious = True
        confidence = 0.85

    gate.stage2_crop_analyzer = lambda crops: MockCropRes()
    stage2_res = gate.evaluate_stage2_specialists(
        "Cand_02", ambiguous_eval, hand_crops_bgr=[None]
    )
    assert stage2_res.stage == EscalationStage.STAGE_2_SPECIALIST
    assert stage2_res.is_resolved is True
    assert stage2_res.resolved_uncertainty < 0.25
    assert stage2_res.resolved_suspicion >= 0.85

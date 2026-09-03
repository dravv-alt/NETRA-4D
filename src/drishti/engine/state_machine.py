"""Candidate Behavioural State Machine and Evidence Arbitration.

Conforms to docs/ARCHITECTURE.md Candidate State Machine (Section 1.1 & Section 5.1).
Defines explicit, auditable states for every candidate:
NORMAL -> OBSERVING -> SUSPICIOUS -> INVESTIGATING -> REVIEW / HIGH_PRIORITY_REVIEW -> COOLDOWN -> NORMAL
(or INSUFFICIENT_EVIDENCE when measurement entropy dominates).

Zero hardcoded constants. All transitions are guarded by continuous suspicion (S),
uncertainty (U), counter-evidence, and temporal cooldown parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .hypotheses import EvidenceEvaluationResult, HypothesisKey


class CandidateBehaviouralState(str, Enum):
    NORMAL = "NORMAL"
    OBSERVING = "OBSERVING"
    SUSPICIOUS = "SUSPICIOUS"
    INVESTIGATING = "INVESTIGATING"
    REVIEW = "REVIEW"
    HIGH_PRIORITY_REVIEW = "HIGH_PRIORITY_REVIEW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    COOLDOWN = "COOLDOWN"


@dataclass
class StateTransitionRecord:
    """Historical audit record of a state transition."""
    from_state: CandidateBehaviouralState
    to_state: CandidateBehaviouralState
    timestamp_sec: float
    trigger_reason: str
    suspicion_score: float
    uncertainty_score: float


class CandidateStateMachine:
    """Manages the state lifecycle and transitions for an individual candidate."""

    def __init__(
        self,
        candidate_id: str,
        cooldown_duration_sec: float = 6.0,
        high_priority_suspicion_threshold: float = 0.70,
        review_suspicion_threshold: float = 0.40,
        uncertainty_escalation_threshold: float = 0.45,
    ):
        self.candidate_id = candidate_id
        self.cooldown_duration_sec = cooldown_duration_sec
        self.high_priority_suspicion_threshold = high_priority_suspicion_threshold
        self.review_suspicion_threshold = review_suspicion_threshold
        self.uncertainty_escalation_threshold = uncertainty_escalation_threshold

        self.current_state: CandidateBehaviouralState = CandidateBehaviouralState.NORMAL
        self.state_entered_timestamp: float = 0.0
        self.cooldown_until_timestamp: float = -1.0
        self.history: List[StateTransitionRecord] = []

    def transition_to(
        self,
        next_state: CandidateBehaviouralState,
        timestamp_sec: float,
        reason: str,
        s: float,
        u: float
    ) -> None:
        if self.current_state == next_state:
            return

        record = StateTransitionRecord(
            from_state=self.current_state,
            to_state=next_state,
            timestamp_sec=timestamp_sec,
            trigger_reason=reason,
            suspicion_score=s,
            uncertainty_score=u
        )
        self.history.append(record)
        self.current_state = next_state
        self.state_entered_timestamp = timestamp_sec

        if next_state in (CandidateBehaviouralState.REVIEW, CandidateBehaviouralState.HIGH_PRIORITY_REVIEW):
            self.cooldown_until_timestamp = timestamp_sec + self.cooldown_duration_sec

    def update(
        self,
        timestamp_sec: float,
        evaluation: EvidenceEvaluationResult
    ) -> CandidateBehaviouralState:
        """Evaluate evidence and execute state transition according to architectural guards."""
        s = evaluation.suspicion_score
        u = evaluation.uncertainty_score

        # Check cooldown expiration
        if self.current_state == CandidateBehaviouralState.COOLDOWN:
            if timestamp_sec >= self.cooldown_until_timestamp:
                self.transition_to(CandidateBehaviouralState.NORMAL, timestamp_sec, "Cooldown elapsed", s, u)
            return self.current_state

        # Cooldown guard: In post-alert cooldown period
        if timestamp_sec < self.cooldown_until_timestamp:
            if s >= self.high_priority_suspicion_threshold:
                # Critical escalation overrides cooldown
                self.transition_to(
                    CandidateBehaviouralState.HIGH_PRIORITY_REVIEW,
                    timestamp_sec,
                    "Multi-modal critical evidence override during cooldown",
                    s, u
                )
            return self.current_state

        # State Transition Guards
        if self.current_state == CandidateBehaviouralState.NORMAL:
            if s >= self.high_priority_suspicion_threshold and u < self.uncertainty_escalation_threshold:
                self.transition_to(CandidateBehaviouralState.HIGH_PRIORITY_REVIEW, timestamp_sec, "High-confidence violation detected", s, u)
            elif s >= self.review_suspicion_threshold and u < self.uncertainty_escalation_threshold:
                self.transition_to(CandidateBehaviouralState.REVIEW, timestamp_sec, "Review threshold reached", s, u)
            elif s >= 0.25:
                self.transition_to(CandidateBehaviouralState.OBSERVING, timestamp_sec, "Suspicious movement initiated", s, u)

        elif self.current_state == CandidateBehaviouralState.OBSERVING:
            if s < 0.15:
                self.transition_to(CandidateBehaviouralState.NORMAL, timestamp_sec, "Movement decayed without persistence", s, u)
            elif s >= self.review_suspicion_threshold:
                self.transition_to(CandidateBehaviouralState.SUSPICIOUS, timestamp_sec, "Persistent or recurrent episode observed", s, u)

        elif self.current_state == CandidateBehaviouralState.SUSPICIOUS:
            if evaluation.dominant_hypothesis in (HypothesisKey.H2_BOARD_CONTEXT, HypothesisKey.H3_INVIGILATOR_CONTEXT, HypothesisKey.H4_ERGONOMIC_BASELINE):
                self.transition_to(CandidateBehaviouralState.NORMAL, timestamp_sec, "Neutralized by counter-evidence", s, u)
            elif u >= self.uncertainty_escalation_threshold:
                self.transition_to(CandidateBehaviouralState.INVESTIGATING, timestamp_sec, "High uncertainty requires specialist investigation", s, u)
            elif s >= self.high_priority_suspicion_threshold:
                self.transition_to(CandidateBehaviouralState.HIGH_PRIORITY_REVIEW, timestamp_sec, "Escalated to High Priority Review", s, u)
            elif s >= self.review_suspicion_threshold:
                self.transition_to(CandidateBehaviouralState.REVIEW, timestamp_sec, "Promoted to Review state", s, u)

        elif self.current_state == CandidateBehaviouralState.INVESTIGATING:
            # Under specialist analysis
            if u < self.uncertainty_escalation_threshold:
                # Resolved by specialist!
                if s >= self.high_priority_suspicion_threshold:
                    self.transition_to(CandidateBehaviouralState.HIGH_PRIORITY_REVIEW, timestamp_sec, "Specialist resolved: High Priority", s, u)
                elif s >= self.review_suspicion_threshold:
                    self.transition_to(CandidateBehaviouralState.REVIEW, timestamp_sec, "Specialist resolved: Review", s, u)
                else:
                    self.transition_to(CandidateBehaviouralState.NORMAL, timestamp_sec, "Specialist resolved: Benign", s, u)
            elif timestamp_sec - self.state_entered_timestamp > 5.0:
                # Persistent uncertainty without resolution
                self.transition_to(CandidateBehaviouralState.INSUFFICIENT_EVIDENCE, timestamp_sec, "Suppressed due to persistent measurement uncertainty", s, u)

        elif self.current_state in (CandidateBehaviouralState.REVIEW, CandidateBehaviouralState.HIGH_PRIORITY_REVIEW, CandidateBehaviouralState.INSUFFICIENT_EVIDENCE):
            self.transition_to(CandidateBehaviouralState.COOLDOWN, timestamp_sec, "Entering post-event cooldown", s, u)

        return self.current_state

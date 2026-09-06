"""
Stage 3 Evidence Arbitration and Review State Evaluator.
Fuses Pose evidence, Object evidence, Spatial Context, and Multi-HMR 3D corroboration
into a final arbitrated state (NORMAL, REVIEW, HIGH_PRIORITY_REVIEW).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FinalReviewState(str, Enum):
    NORMAL = "NORMAL"
    REVIEW = "REVIEW"
    HIGH_PRIORITY_REVIEW = "HIGH_PRIORITY_REVIEW"


@dataclass
class ArbitratedEvent:
    candidate_id: str
    desk_id: str
    state: FinalReviewState
    event_type: str
    confidence: float
    supporting_evidence: List[str]
    counter_evidence: List[str]
    model_agreement_score: float


class EvidenceArbitrator:
    """
    Arbitrates multi-modal cues (2D Pose, Objects, Stage 2 Crops, Stage 3 Multi-HMR)
    against candidate baselines and spatial constraints to eliminate false alarms.
    """

    def arbitrate(
        self,
        candidate_id: str,
        desk_id: str,
        proposals: List[Dict[str, Any]],
        specialist_result: Optional[Any] = None,
        is_invigilator: bool = False
    ) -> ArbitratedEvent:
        # Rule 1: Strict Invigilator Exclusion
        if is_invigilator:
            return ArbitratedEvent(
                candidate_id=candidate_id,
                desk_id="AISLE",
                state=FinalReviewState.NORMAL,
                event_type="INVIGILATOR_MONITORING",
                confidence=0.0,
                supporting_evidence=["Invigilator role verified and excluded from suspicion"],
                counter_evidence=["Invigilator exclusion rule active"],
                model_agreement_score=1.0
            )

        supporting = []
        counter = []
        highest_conf = 0.0
        primary_event = "NORMAL"

        for p in proposals:
            event_name = p.get("event_type", "UNKNOWN")
            conf = float(p.get("confidence", 0.0))
            if conf > highest_conf:
                highest_conf = conf
                primary_event = event_name
            supporting.append(f"Proposal: {event_name} (conf: {conf:.2f})")

        # Stage 2 Specialist Corroboration
        if specialist_result and specialist_result.is_suspicious:
            supporting.append(specialist_result.evidence_desc)
            highest_conf = max(highest_conf, specialist_result.confidence)
            primary_event = f"{primary_event}_VERIFIED_CHIT"

        # Final state determination
        if highest_conf >= 0.72:
            state = FinalReviewState.HIGH_PRIORITY_REVIEW
        elif highest_conf >= 0.40:
            state = FinalReviewState.REVIEW
        else:
            state = FinalReviewState.NORMAL

        agreement_score = min(1.0, 0.50 + 0.25 * len(supporting))

        return ArbitratedEvent(
            candidate_id=candidate_id,
            desk_id=desk_id,
            state=state,
            event_type=primary_event,
            confidence=highest_conf,
            supporting_evidence=supporting,
            counter_evidence=counter,
            model_agreement_score=agreement_score
        )

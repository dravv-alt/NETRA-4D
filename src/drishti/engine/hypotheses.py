"""Competing Hypotheses and Counter-Evidence Framework.

Conforms to docs/ARCHITECTURE.md Layer 3 (Section 4.3) & Layer 4 (Section 5.2).
Models 5 competing hypotheses for every observed behavioural pattern:
- H1: Primary Suspicion (Neighbour screen look, unauthorized notes/chit, illicit device)
- H2: Environmental Context (Looking at instruction board, wall clock, or teacher)
- H3: Invigilator Context (Reacting to approaching proctor, asking exam clarification)
- H4: Ergonomic Baseline (Natural posture reset, neck stretch, physiological movement)
- H5: Sensor / Tracker Uncertainty (Keypoint occlusion, tracker noise, jitter)

Calculates dual independent metrics:
- Suspiciousness (S in [0.0, 1.0]): 'Does this pattern matter?'
- Uncertainty (U in [0.0, 1.0]): 'Do we clearly understand what happened?'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from .primitive_events import PrimitiveEvent, PrimitiveEventType
from .temporal_patterns import AggregatedTemporalPattern, TemporalPatternType


class HypothesisKey(str, Enum):
    H1_PRIMARY_SUSPICION = "H1_PRIMARY_SUSPICION"
    H2_BOARD_CONTEXT = "H2_BOARD_CONTEXT"
    H3_INVIGILATOR_CONTEXT = "H3_INVIGILATOR_CONTEXT"
    H4_ERGONOMIC_BASELINE = "H4_ERGONOMIC_BASELINE"
    H5_SENSOR_UNCERTAINTY = "H5_SENSOR_UNCERTAINTY"


@dataclass
class HypothesisDistribution:
    """Normalized probability distribution across all 5 competing explanations."""
    p_h1: float = 0.05  # Malpractice / illicit action
    p_h2: float = 0.10  # Board / clock look
    p_h3: float = 0.05  # Proctor interaction
    p_h4: float = 0.70  # Normal ergonomics (neutral prior)
    p_h5: float = 0.10  # Sensor uncertainty

    def normalize(self) -> None:
        total = self.p_h1 + self.p_h2 + self.p_h3 + self.p_h4 + self.p_h5
        if total > 0:
            self.p_h1 /= total
            self.p_h2 /= total
            self.p_h3 /= total
            self.p_h4 /= total
            self.p_h5 /= total
        else:
            self.p_h4 = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            HypothesisKey.H1_PRIMARY_SUSPICION.value: round(self.p_h1, 4),
            HypothesisKey.H2_BOARD_CONTEXT.value: round(self.p_h2, 4),
            HypothesisKey.H3_INVIGILATOR_CONTEXT.value: round(self.p_h3, 4),
            HypothesisKey.H4_ERGONOMIC_BASELINE.value: round(self.p_h4, 4),
            HypothesisKey.H5_SENSOR_UNCERTAINTY.value: round(self.p_h5, 4),
        }


@dataclass
class EvidenceEvaluationResult:
    """Outcome of multi-modal evidence arbitration with counter-evidence."""
    candidate_id: str
    suspicion_score: float                  # S in [0.0, 1.0]
    uncertainty_score: float                # U in [0.0, 1.0]
    hypotheses: HypothesisDistribution
    supporting_evidence: List[str]
    counter_evidence: List[str]
    dominant_hypothesis: HypothesisKey
    requires_specialist: bool = False


class CompetingHypothesisEngine:
    """Evaluates behavioural evidence, weights competing hypotheses, and isolates counter-evidence."""

    def evaluate(
        self,
        candidate_id: str,
        pattern: Optional[AggregatedTemporalPattern],
        is_board_aligned: bool = False,
        is_invigilator_nearby: bool = False,
        is_normal_device_interaction: bool = False,
        is_boundary_crossed: bool = False,
        is_phone_detected: bool = False,
        is_chit_detected: bool = False,
        sensor_quality_ratio: float = 1.0,
    ) -> EvidenceEvaluationResult:
        dist = HypothesisDistribution()
        supporting: List[str] = []
        counter: List[str] = []

        # 1. Base weights from temporal pattern
        if pattern is not None and pattern.pattern_type == TemporalPatternType.BRIEF_ROUTINE:
            dist.p_h4 += 1.5  # Brief movements overwhelmingly favor natural ergonomics
            counter.append(f"Brief movement ({pattern.duration_sec:.2f}s) indicates routine physiological adjustment")
        elif pattern is not None and pattern.pattern_type == TemporalPatternType.SUSTAINED_HOLD:
            dist.p_h1 += 1.8
            dist.p_h4 -= 0.3
            supporting.append(f"Sustained unbroken hold ({pattern.duration_sec:.2f}s) in {pattern.event_type.value}")
        elif pattern is not None and pattern.pattern_type in (TemporalPatternType.REPEATED_EPISODE, TemporalPatternType.REPEATED_AND_SUSTAINED):
            dist.p_h1 += 2.2
            dist.p_h4 -= 0.5
            supporting.append(f"Recurrent episodes ({pattern.recurrence_count_in_window}x in sliding window)")

        # 2. Environmental Context: Board Look
        if is_board_aligned:
            dist.p_h2 += 2.5
            dist.p_h1 = max(0.01, dist.p_h1 * 0.20)
            counter.append("Candidate orientation directly aligns with instructor board / front clock")

        # 3. Proctor Context: Invigilator in proximity
        if is_invigilator_nearby:
            dist.p_h3 += 2.0
            dist.p_h1 = max(0.01, dist.p_h1 * 0.30)
            counter.append("Proctor / Invigilator present within immediate interaction radius")

        # 4. Normal Device Usage (Keyboard/Mouse zone)
        if is_normal_device_interaction:
            dist.p_h4 += 1.5
            dist.p_h1 = max(0.01, dist.p_h1 * 0.40)
            counter.append("Hands resting naturally within calibrated keyboard/mouse device area")

        # 5. Unauthorized Physical Indicators
        if is_boundary_crossed:
            dist.p_h1 += 2.8
            dist.p_h4 = max(0.01, dist.p_h4 * 0.20)
            supporting.append("Candidate hand crossed workstation boundary into neighbour zone")

        if is_phone_detected:
            dist.p_h1 += 3.5
            supporting.append("Electronic cell phone detected in candidate desk area")

        if is_chit_detected:
            dist.p_h1 += 3.8
            supporting.append("High-contrast folded paper chit detected during hand interaction")

        # 6. Sensor Quality & Measurement Entropy
        if sensor_quality_ratio < 0.60:
            dist.p_h5 += 1.5 * (1.0 - sensor_quality_ratio)
            counter.append(f"Reduced keypoint coverage / tracking ambiguity ({sensor_quality_ratio:.2f})")

        # Normalize probability mass across H1..H5
        dist.normalize()

        # Compute Dual Independent Scores
        # S = P(H1)
        suspicion = float(np.clip(dist.p_h1, 0.0, 1.0))
        # U = P(H5) + entropy proxy (when competition between H1 and H2/H3/H4 is balanced)
        entropy_proxy = float(4.0 * dist.p_h1 * (dist.p_h2 + dist.p_h3 + dist.p_h4))
        uncertainty = float(np.clip(dist.p_h5 + 0.30 * entropy_proxy, 0.0, 1.0))

        # Dominant explanation
        scores = {
            HypothesisKey.H1_PRIMARY_SUSPICION: dist.p_h1,
            HypothesisKey.H2_BOARD_CONTEXT: dist.p_h2,
            HypothesisKey.H3_INVIGILATOR_CONTEXT: dist.p_h3,
            HypothesisKey.H4_ERGONOMIC_BASELINE: dist.p_h4,
            HypothesisKey.H5_SENSOR_UNCERTAINTY: dist.p_h5,
        }
        dominant = max(scores, key=scores.get)

        # Ambiguity triggers specialist escalation: S >= 0.40 and U >= 0.25
        requires_specialist = (suspicion >= 0.40 and uncertainty >= 0.25)

        return EvidenceEvaluationResult(
            candidate_id=candidate_id,
            suspicion_score=suspicion,
            uncertainty_score=uncertainty,
            hypotheses=dist,
            supporting_evidence=supporting,
            counter_evidence=counter,
            dominant_hypothesis=dominant,
            requires_specialist=requires_specialist
        )

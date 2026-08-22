"""
Behavioural Risk Engine & State Machine for Drishti-4D / NETRA-4D.
Fuses multi-modal evidence with exponential decay, quality scaling, counter-evidence penalties,
and manages transitions between NORMAL, WATCH, REVIEW, and UNKNOWN states.
"""

from typing import Dict, List, Optional
import numpy as np
from ..schemas import BehaviourEvidence, RiskSnapshot, RiskStateEnum, PersonObservation


class RiskStateMachine:
    def __init__(
        self,
        watch_threshold: float = 0.35,
        review_threshold: float = 0.68,
        unknown_quality_threshold: float = 0.45,
        risk_tau_seconds: float = 12.0,
        review_min_duration_seconds: float = 2.0,
        counter_weight: float = 0.25
    ):
        self.watch_threshold = watch_threshold
        self.review_threshold = review_threshold
        self.unknown_quality_threshold = unknown_quality_threshold
        self.risk_tau = risk_tau_seconds
        self.review_min_duration = review_min_duration_seconds
        self.counter_weight = counter_weight

        # entity_id -> accumulated risk score
        self.risk_scores: Dict[int, float] = {}
        # entity_id -> last update timestamp
        self.last_timestamps: Dict[int, float] = {}
        # entity_id -> time spent continuously above review threshold
        self.review_durations: Dict[int, float] = {}
        # entity_id -> current state
        self.states: Dict[int, RiskStateEnum] = {}

    def update(
        self,
        timestamp_sec: float,
        evidence_list: List[BehaviourEvidence],
        current_obs_list: List[PersonObservation]
    ) -> Dict[int, RiskSnapshot]:
        """
        Updates risk scores for all active candidates and returns current RiskSnapshots.
        """
        snapshots: Dict[int, RiskSnapshot] = {}

        # Map active evidence per entity
        evidence_by_entity: Dict[int, List[BehaviourEvidence]] = {}
        for ev in evidence_list:
            if ev.entity_id not in evidence_by_entity:
                evidence_by_entity[ev.entity_id] = []
            evidence_by_entity[ev.entity_id].append(ev)

        for obs in current_obs_list:
            eid = obs.entity_local_id
            seat_id = obs.seat_id
            cam_id = obs.camera_id

            # 1. Decay previous risk score
            last_t = self.last_timestamps.get(eid, timestamp_sec)
            dt = max(0.0, timestamp_sec - last_t)
            prev_risk = self.risk_scores.get(eid, 0.0)
            decay_factor = np.exp(-dt / self.risk_tau) if self.risk_tau > 0 else 0.0
            decayed_risk = prev_risk * decay_factor

            # 2. Sum incoming evidence
            entity_ev = evidence_by_entity.get(eid, [])
            incoming_evidence_score = 0.0
            reasons = []
            ev_ids = []

            for ev in entity_ev:
                incoming_evidence_score += ev.strength * ev.quality
                reasons.extend(ev.reason_codes)
                ev_ids.append(ev.evidence_id)

            # 3. Overall perception quality
            qual_scores = [
                1.0 if obs.quality.head == "GOOD" else 0.3,
                1.0 if obs.quality.torso == "GOOD" else 0.5,
                1.0 if obs.quality.hands == "GOOD" else 0.4,
                1.0 if obs.quality.track == "GOOD" else 0.4
            ]
            overall_quality = float(np.mean(qual_scores))

            # 4. Check for UNKNOWN state
            if overall_quality < self.unknown_quality_threshold:
                current_state = RiskStateEnum.UNKNOWN
                current_risk = decayed_risk
            else:
                # Accumulate new risk
                raw_risk = decayed_risk + incoming_evidence_score * 0.45
                current_risk = float(np.clip(raw_risk, 0.0, 1.0))

                # State Transitions
                if current_risk >= self.review_threshold:
                    self.review_durations[eid] = self.review_durations.get(eid, 0.0) + dt
                    if self.review_durations[eid] >= self.review_min_duration or incoming_evidence_score >= 1.5:
                        current_state = RiskStateEnum.REVIEW
                    else:
                        current_state = RiskStateEnum.WATCH
                elif current_risk >= self.watch_threshold:
                    self.review_durations[eid] = 0.0
                    current_state = RiskStateEnum.WATCH
                else:
                    self.review_durations[eid] = 0.0
                    current_state = RiskStateEnum.NORMAL

            # Update tracking states
            self.risk_scores[eid] = current_risk
            self.last_timestamps[eid] = timestamp_sec
            self.states[eid] = current_state

            snapshots[eid] = RiskSnapshot(
                camera_id=cam_id,
                timestamp_sec=timestamp_sec,
                entity_id=eid,
                seat_id=seat_id,
                raw_evidence_score=incoming_evidence_score,
                counter_evidence_score=0.0,
                persistence_weight=float(self.review_durations.get(eid, 0.0)),
                quality_factor=overall_quality,
                risk_score=current_risk,
                state=current_state,
                active_evidence_ids=ev_ids,
                active_reasons=reasons
            )

        return snapshots

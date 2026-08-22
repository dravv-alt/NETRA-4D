"""
Forensic Review Event Builder and Summary Generator for Drishti-4D / NETRA-4D.
Merges consecutive risk intervals, compiles forensic reason codes,
and builds standardized ReviewEvents for human invigilator audit.
"""

from typing import Dict, List, Optional, Any
import time
from pathlib import Path
import json

from ..schemas import ReviewEvent, RiskSnapshot, BehaviourEvidence, RiskStateEnum, BehaviourEnum


class EventBuilder:
    def __init__(self, merge_gap_seconds: float = 2.0, pre_roll_seconds: float = 5.0, post_roll_seconds: float = 5.0):
        self.merge_gap_seconds = merge_gap_seconds
        self.pre_roll_seconds = pre_roll_seconds
        self.post_roll_seconds = post_roll_seconds

        # Active open candidate events: entity_id -> partial ReviewEvent
        self.open_events: Dict[int, Dict[str, Any]] = {}
        # Completed closed events
        self.completed_events: List[ReviewEvent] = []

    def update(
        self,
        timestamp_sec: float,
        snapshots: Dict[int, RiskSnapshot],
        evidence_list: List[BehaviourEvidence],
        video_id: str = "VIDEO_01"
    ) -> List[ReviewEvent]:
        """
        Updates ongoing event windows and closes events that have concluded.
        Returns newly completed ReviewEvents.
        """
        newly_closed: List[ReviewEvent] = []

        for eid, snap in snapshots.items():
            is_active_risk = snap.state in [RiskStateEnum.REVIEW, RiskStateEnum.WATCH]

            if is_active_risk:
                if eid not in self.open_events:
                    # Open new event window
                    self.open_events[eid] = {
                        "event_id": f"EVT_{snap.camera_id}_{eid}_{int(timestamp_sec * 10)}",
                        "camera_id": snap.camera_id,
                        "video_id": video_id,
                        "entity_id": eid,
                        "seat_id": snap.seat_id,
                        "start_sec": max(0.0, timestamp_sec - self.pre_roll_seconds),
                        "end_sec": timestamp_sec,
                        "peak_risk": snap.risk_score,
                        "qualities": [snap.quality_factor],
                        "evidence_signals": list(snap.active_reasons),
                        "counter_evidence_signals": [],
                        "state": snap.state,
                        "last_active_sec": timestamp_sec
                    }
                else:
                    # Update open event
                    ev = self.open_events[eid]
                    ev["end_sec"] = timestamp_sec
                    ev["peak_risk"] = max(ev["peak_risk"], snap.risk_score)
                    ev["qualities"].append(snap.quality_factor)
                    ev["state"] = snap.state if snap.state == RiskStateEnum.REVIEW else ev["state"]
                    ev["last_active_sec"] = timestamp_sec
                    for r in snap.active_reasons:
                        if r not in ev["evidence_signals"]:
                            ev["evidence_signals"].append(r)
            else:
                # Candidate is normal now; check if we should close an open event
                if eid in self.open_events:
                    ev = self.open_events[eid]
                    gap = timestamp_sec - ev["last_active_sec"]
                    if gap >= self.merge_gap_seconds:
                        # Close and finalize the event
                        closed_event = self._finalize_event(ev, timestamp_sec)
                        newly_closed.append(closed_event)
                        self.completed_events.append(closed_event)
                        del self.open_events[eid]

        return newly_closed

    def _finalize_event(self, raw_ev: Dict[str, Any], current_timestamp: float) -> ReviewEvent:
        """Constructs a typed ReviewEvent with auto-generated summary."""
        signals = raw_ev["evidence_signals"]
        seat = raw_ev["seat_id"] or f"Track #{raw_ev['entity_id']}"
        start_t = raw_ev["start_sec"]
        end_t = raw_ev["end_sec"] + self.post_roll_seconds
        dur = end_t - start_t

        # Classify primary behaviour
        primary_behaviour = BehaviourEnum.NORMAL
        if any("phone" in s.lower() for s in signals):
            primary_behaviour = BehaviourEnum.POSSIBLE_OBJECT_INTERACTION
        elif any("reciprocal" in s.lower() or "talking" in s.lower() for s in signals):
            primary_behaviour = BehaviourEnum.POSSIBLE_NONVERBAL_INTERACTION
        elif any("paper" in s.lower() or "chit" in s.lower() for s in signals):
            primary_behaviour = BehaviourEnum.POSSIBLE_OBJECT_INTERACTION
        elif any("torso" in s.lower() for s in signals):
            primary_behaviour = BehaviourEnum.TORSO_ROTATION_TOWARD_NEIGHBOUR
        elif any("sustained" in s.lower() for s in signals):
            primary_behaviour = BehaviourEnum.SUSTAINED_EXCESSIVE_HEAD_TURN
        elif any("glance" in s.lower() for s in signals):
            primary_behaviour = BehaviourEnum.REPEATED_SIDEWARD_GLANCE

        summary_text = (
            f"Candidate at {seat} flagged for review over a {dur:.1f}s window ({start_t:.1f}s - {end_t:.1f}s). "
            f"Key observations: {'; '.join(signals[:3]) or 'Sustained behavioural anomaly'}."
        )

        return ReviewEvent(
            event_id=raw_ev["event_id"],
            camera_id=raw_ev["camera_id"],
            video_id=raw_ev["video_id"],
            entity_id=raw_ev["entity_id"],
            seat_id=raw_ev["seat_id"],
            start_sec=round(start_t, 2),
            end_sec=round(end_t, 2),
            behaviour=primary_behaviour,
            state=raw_ev["state"],
            peak_risk=round(raw_ev["peak_risk"], 3),
            mean_quality=round(float(sum(raw_ev["qualities"]) / max(1, len(raw_ev["qualities"]))), 3),
            evidence_signals=signals,
            counter_evidence_signals=raw_ev["counter_evidence_signals"],
            summary=summary_text,
            human_status="UNREVIEWED"
        )

    def flush(self, final_timestamp: float) -> List[ReviewEvent]:
        """Flushes and finalizes all remaining open events at video end."""
        remaining: List[ReviewEvent] = []
        for eid, ev in list(self.open_events.items()):
            closed = self._finalize_event(ev, final_timestamp)
            remaining.append(closed)
            self.completed_events.append(closed)
        self.open_events.clear()
        return remaining

    def save_events_json(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [ev.model_dump() for ev in self.completed_events]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

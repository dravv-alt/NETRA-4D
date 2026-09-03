"""Canonical Candidate State and Immutable Identity Manager.

Conforms to docs/ARCHITECTURE.md Layer 1 (Identity Layer) & Layer 2 (Candidate State).
Enforces:
1. The Immutable Identity Rule (Candidate ID, Desk, Device lock permanently).
2. The Recovery Invariant (Track drop -> IDENTITY_UNCERTAIN, never swapped).
3. Invigilator Exclusion (Tracked for context, never scored for suspicion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .observation import VisibilityState, VisionObservation


class IdentityLockStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"    # During warmup/calibration
    LOCKED = "LOCKED"              # Immutable after warmup
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"  # Track lost & re-association ambiguous


@dataclass
class CandidateIdentity:
    """Immutable identity tuple for an examined student or invigilator."""
    candidate_id: str                      # e.g. "Candidate_01" or "INVIGILATOR"
    seat_id: str                           # e.g. "Seat_01"
    workstation_id: str                    # e.g. "Desk_01"
    device_id: Optional[str] = None        # e.g. "PC_01"
    is_invigilator: bool = False
    lock_status: IdentityLockStatus = IdentityLockStatus.PROVISIONAL

    # Spatial anchor (normalized coordinate of seat/desk centroid)
    anchor_center_norm: Tuple[float, float] = (0.5, 0.5)
    anchor_box_size_norm: Tuple[float, float] = (0.2, 0.4)

    # Initial association tracker ID
    primary_track_id: int = -1


@dataclass
class CandidateState:
    """Full candidate state representation aggregated across frames."""
    identity: CandidateIdentity
    current_observation: Optional[VisionObservation] = None
    last_seen_timestamp_sec: float = 0.0
    last_seen_frame_idx: int = 0
    consecutive_missing_frames: int = 0

    # Behavioural tracking state
    behavioural_state: str = "NORMAL"
    suspicion_score: float = 0.0
    uncertainty_score: float = 1.0

    # Running history statistics
    total_observations_count: int = 0
    active_tracker_id: Optional[int] = None

    @property
    def is_visible(self) -> bool:
        return (
            self.current_observation is not None
            and self.current_observation.visibility == VisibilityState.VISIBLE
            and self.consecutive_missing_frames == 0
        )

    @property
    def is_occluded(self) -> bool:
        return (
            self.current_observation is not None
            and self.current_observation.visibility == VisibilityState.OCCLUDED
        ) or (1 <= self.consecutive_missing_frames <= 15)

    @property
    def is_lost(self) -> bool:
        return self.consecutive_missing_frames > 15


class CandidateStateManager:
    """Manages all candidate and invigilator states across time."""

    def __init__(
        self,
        warmup_frames: int = 45,
        max_reid_distance_ratio: float = 1.25,
    ):
        self.warmup_frames = warmup_frames
        self.max_reid_distance_ratio = max_reid_distance_ratio
        self.current_frame = 0
        self.is_finalized = False

        self.candidates: Dict[str, CandidateState] = {}
        self.track_to_candidate_id: Dict[int, str] = {}
        self.invigilator_track_ids: set[int] = set()

        self._candidate_sequence_counter = 0

    def register_candidate_anchor(
        self,
        track_id: int,
        box_norm: Tuple[float, float, float, float],
        desk_id: Optional[str] = None,
        device_id: Optional[str] = None,
        is_invigilator: bool = False
    ) -> CandidateState:
        """Register a new physical seat anchor during discovery or arrival."""
        cx = (box_norm[0] + box_norm[2]) / 2.0
        cy = (box_norm[1] + box_norm[3]) / 2.0
        bw = max(0.01, box_norm[2] - box_norm[0])
        bh = max(0.01, box_norm[3] - box_norm[1])

        if is_invigilator:
            cand_id = "INVIGILATOR"
            seat_id = "AISLE"
            wstation_id = "ROAMING"
            dev_id = None
            self.invigilator_track_ids.add(track_id)
        else:
            self._candidate_sequence_counter += 1
            cand_id = f"Candidate_{self._candidate_sequence_counter:02d}"
            seat_id = f"Seat_{self._candidate_sequence_counter:02d}"
            wstation_id = desk_id or f"Desk_{self._candidate_sequence_counter:02d}"
            dev_id = device_id or f"PC_{self._candidate_sequence_counter:02d}"

        identity = CandidateIdentity(
            candidate_id=cand_id,
            seat_id=seat_id,
            workstation_id=wstation_id,
            device_id=dev_id,
            is_invigilator=is_invigilator,
            lock_status=IdentityLockStatus.PROVISIONAL,
            anchor_center_norm=(cx, cy),
            anchor_box_size_norm=(bw, bh),
            primary_track_id=track_id
        )

        state = CandidateState(
            identity=identity,
            active_tracker_id=track_id
        )
        self.candidates[cand_id] = state
        self.track_to_candidate_id[track_id] = cand_id
        return state

    def update_frame_observations(
        self,
        frame_idx: int,
        timestamp_sec: float,
        observations: List[VisionObservation]
    ) -> Dict[str, CandidateState]:
        """Update all candidate states with new observations for the current frame."""
        self.current_frame = frame_idx
        observed_track_ids = {obs.track_id: obs for obs in observations}
        matched_candidate_ids: set[str] = set()

        # 1. Match known tracker IDs
        for track_id, obs in observed_track_ids.items():
            cand_id = self.track_to_candidate_id.get(track_id)
            if cand_id and cand_id in self.candidates:
                state = self.candidates[cand_id]
                state.current_observation = obs
                state.last_seen_timestamp_sec = timestamp_sec
                state.last_seen_frame_idx = frame_idx
                state.consecutive_missing_frames = 0
                state.total_observations_count += 1
                state.active_tracker_id = track_id
                matched_candidate_ids.add(cand_id)

        # 2. Re-associate dropped tracks using immutable seat anchors
        unmatched_obs = [obs for tid, obs in observed_track_ids.items() if tid not in self.track_to_candidate_id]
        
        for obs in unmatched_obs:
            best_candidate_id = None
            min_normalized_dist = float("inf")

            for cand_id, state in self.candidates.items():
                if cand_id in matched_candidate_ids or state.identity.is_invigilator:
                    continue

                # Measure Mahalanobis/scale-normalized distance to immutable seat anchor
                anchor_cx, anchor_cy = state.identity.anchor_center_norm
                anchor_bw, anchor_bh = state.identity.anchor_box_size_norm
                obs_cx, obs_cy = obs.center_norm

                dx = (obs_cx - anchor_cx) / anchor_bw
                dy = (obs_cy - anchor_cy) / anchor_bh
                dist = np.hypot(dx, dy)

                if dist < min_normalized_dist and dist <= self.max_reid_distance_ratio:
                    min_normalized_dist = dist
                    best_candidate_id = cand_id

            if best_candidate_id:
                # Re-associated successfully to the locked seat anchor
                state = self.candidates[best_candidate_id]
                state.current_observation = obs
                state.last_seen_timestamp_sec = timestamp_sec
                state.last_seen_frame_idx = frame_idx
                state.consecutive_missing_frames = 0
                state.total_observations_count += 1
                state.active_tracker_id = obs.track_id
                self.track_to_candidate_id[obs.track_id] = best_candidate_id
                matched_candidate_ids.add(best_candidate_id)
            else:
                # Dynamically register candidate anchor so no person is ever ignored
                new_state = self.register_candidate_anchor(obs.track_id, obs.bbox_xyxy_norm)
                new_state.current_observation = obs
                new_state.last_seen_timestamp_sec = timestamp_sec
                new_state.last_seen_frame_idx = frame_idx
                new_state.total_observations_count = 1
                if self.is_finalized:
                    new_state.identity.lock_status = IdentityLockStatus.IDENTITY_UNCERTAIN
                    obs.metadata["identity_status"] = IdentityLockStatus.IDENTITY_UNCERTAIN.value
                matched_candidate_ids.add(new_state.identity.candidate_id)

        # 3. Handle missed tracks
        for cand_id, state in self.candidates.items():
            if cand_id not in matched_candidate_ids:
                state.consecutive_missing_frames += 1
                if state.consecutive_missing_frames > 25 and state.identity.lock_status == IdentityLockStatus.LOCKED:
                    state.identity.lock_status = IdentityLockStatus.IDENTITY_UNCERTAIN

        # 4. Finalize locks after warmup
        if not self.is_finalized and self.current_frame >= self.warmup_frames:
            self.finalize_immutable_registry()

        return self.candidates

    def finalize_immutable_registry(self) -> None:
        """Freeze candidate identities and seat anchors permanently."""
        for state in self.candidates.values():
            if state.identity.lock_status != IdentityLockStatus.IDENTITY_UNCERTAIN:
                state.identity.lock_status = IdentityLockStatus.LOCKED
        self.is_finalized = True

    def get_candidate(self, candidate_id: str) -> Optional[CandidateState]:
        return self.candidates.get(candidate_id)

    def get_candidate_for_track(self, track_id: int) -> Optional[CandidateState]:
        cand_id = self.track_to_candidate_id.get(track_id)
        return self.candidates.get(cand_id) if cand_id else None

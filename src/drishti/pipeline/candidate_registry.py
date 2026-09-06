"""
Immutable Candidate ID and 1-to-1 Desk Binding.
Discovers stationary seated candidates, assigns permanent Candidate_XX identifiers,
and establishes an immutable one-to-one lock with calibrated desks and devices.
Once finalized, Candidate ID and Desk assignment CANNOT change or swap.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CandidateLock:
    candidate_id: str          # e.g. "Candidate_01"
    desk_id: str               # e.g. "Desk_07"
    device_id: str             # e.g. "PC_07" / "Laptop_07"
    seat_center_norm: Tuple[float, float]
    initial_track_id: int
    is_locked: bool = True


class ImmutableCandidateRegistry:
    """
    Ensures every seated candidate receives an immutable identifier permanently bound
    1:1 to their physical desk and monitor. Prevents ID switching and swap errors.
    """

    def __init__(self, seat_lock_radius: float = 0.14):
        self.seat_lock_radius = seat_lock_radius
        self.candidate_locks: Dict[str, CandidateLock] = {}
        self.desk_to_candidate: Dict[str, str] = {}
        self.track_to_candidate: Dict[int, str] = {}
        self.candidate_counter = 0
        self.finalized = False

    def register_seated_candidate(
        self,
        track_id: int,
        seat_center_norm: Tuple[float, float],
        desk_id: Optional[str] = None,
        device_id: Optional[str] = None
    ) -> str:
        # Check if already locked to an existing candidate seat position
        for cand_id, lock in self.candidate_locks.items():
            dist = float(np.linalg.norm(np.array(seat_center_norm) - np.array(lock.seat_center_norm)))
            if dist <= self.seat_lock_radius:
                self.track_to_candidate[track_id] = cand_id
                return cand_id

        # New candidate seated
        self.candidate_counter += 1
        cand_id = f"Candidate_{self.candidate_counter:02d}"
        assigned_desk = desk_id or f"Desk_{self.candidate_counter:02d}"
        assigned_dev = device_id or f"PC_{self.candidate_counter:02d}"

        # Enforce strict 1-to-1 Desk Binding Rule
        if assigned_desk in self.desk_to_candidate:
            # Already taken, append suffix
            assigned_desk = f"{assigned_desk}_SeatB"
            assigned_dev = f"{assigned_dev}_B"

        lock = CandidateLock(
            candidate_id=cand_id,
            desk_id=assigned_desk,
            device_id=assigned_dev,
            seat_center_norm=seat_center_norm,
            initial_track_id=track_id,
            is_locked=True
        )

        self.candidate_locks[cand_id] = lock
        self.desk_to_candidate[assigned_desk] = cand_id
        self.track_to_candidate[track_id] = cand_id
        return cand_id

    def get_candidate_for_track(self, track_id: int, center_norm: Optional[Tuple[float, float]] = None) -> str:
        if track_id in self.track_to_candidate:
            return self.track_to_candidate[track_id]

        # If tracker dropped and person re-appears at locked seat center
        if center_norm is not None:
            for cand_id, lock in self.candidate_locks.items():
                dist = float(np.linalg.norm(np.array(center_norm) - np.array(lock.seat_center_norm)))
                if dist <= self.seat_lock_radius:
                    self.track_to_candidate[track_id] = cand_id
                    return cand_id

        # Create new seated candidate if not found
        center = center_norm if center_norm is not None else (0.5, 0.5)
        return self.register_seated_candidate(track_id, center)

    def get_desk_binding(self, candidate_id: str) -> Optional[CandidateLock]:
        return self.candidate_locks.get(candidate_id)

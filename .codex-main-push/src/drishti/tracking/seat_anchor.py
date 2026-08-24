"""
Seat Anchoring & Anonymous Seat Assignment Engine for Drishti-4D.
Maintains temporal hysteresis so track jitter doesn't cause spurious seat reassignments.
Detects seat exchange anomalies (person != seat).
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from ..schemas import TrackState, PersonObservation, CameraProfile
from ..context.camera_profile import point_in_polygon


class SeatAnchorManager:
    def __init__(
        self,
        camera_profile: CameraProfile,
        lock_seconds: float = 2.0,
        release_seconds: float = 3.0,
        min_consecutive_hits: int = 4
    ):
        self.camera_profile = camera_profile
        self.lock_seconds = lock_seconds
        self.release_seconds = release_seconds
        self.min_consecutive_hits = min_consecutive_hits

        # track_id -> TrackState
        self.tracks: Dict[int, TrackState] = {}
        # seat_id -> current assigned track_id
        self.seat_occupancy: Dict[str, Optional[int]] = {s_id: None for s_id in camera_profile.seats.keys()}
        # track_id -> candidate seat tracking: (candidate_seat_id, first_seen_time, consecutive_hits)
        self.candidate_locks: Dict[int, Tuple[str, float, int]] = {}

    def get_anchor_point(self, obs: PersonObservation) -> Tuple[float, float]:
        """
        Derive normalized [x, y] anchor point:
        Uses pelvis / hip midpoint if available, otherwise bottom-center of bbox.
        """
        if obs.keypoints_2d:
            l_hip = obs.keypoints_2d.get("left_hip")
            r_hip = obs.keypoints_2d.get("right_hip")
            if l_hip and r_hip and l_hip.confidence > 0.25 and r_hip.confidence > 0.25:
                return (l_hip.x + r_hip.x) / 2.0, (l_hip.y + r_hip.y) / 2.0

        # Fallback to bottom-center of bbox
        x1, y1, x2, y2 = obs.bbox_xyxy_norm
        return (x1 + x2) / 2.0, y2 * 0.90 + y1 * 0.10

    def find_matching_seat(self, point: Tuple[float, float]) -> Optional[str]:
        """Finds seat polygon containing the anchor point."""
        for seat_id, profile in self.camera_profile.seats.items():
            if point_in_polygon(point, profile.polygon):
                return seat_id
        return None

    def update(self, obs: PersonObservation) -> Optional[str]:
        """
        Updates seat assignment for the given observation.
        Returns the assigned seat_id or None.
        """
        track_id = obs.entity_local_id
        ts = obs.timestamp_sec
        anchor_pt = self.get_anchor_point(obs)
        instant_seat = self.find_matching_seat(anchor_pt)

        # Initialize track if new
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                seat_id=None,
                first_seen_sec=ts,
                last_seen_sec=ts,
                track_quality=obs.detection_confidence,
                history_size=1,
                status="ACTIVE",
                consecutive_seat_hits=0
            )

        state = self.tracks[track_id]
        state.last_seen_sec = ts
        state.history_size += 1

        # Hysteresis Lock Logic
        if instant_seat:
            if state.seat_id == instant_seat:
                # Already locked to this seat
                state.consecutive_seat_hits += 1
                self.candidate_locks.pop(track_id, None)
            else:
                # Candidate seat check
                if track_id in self.candidate_locks:
                    cand_seat, cand_start, hits = self.candidate_locks[track_id]
                    if cand_seat == instant_seat:
                        hits += 1
                        duration = ts - cand_start
                        if duration >= self.lock_seconds and hits >= self.min_consecutive_hits:
                            # Release old seat
                            if state.seat_id and self.seat_occupancy.get(state.seat_id) == track_id:
                                self.seat_occupancy[state.seat_id] = None

                            # Lock new seat
                            state.seat_id = instant_seat
                            state.consecutive_seat_hits = hits
                            self.seat_occupancy[instant_seat] = track_id
                            self.candidate_locks.pop(track_id, None)
                        else:
                            self.candidate_locks[track_id] = (cand_seat, cand_start, hits)
                    else:
                        self.candidate_locks[track_id] = (instant_seat, ts, 1)
                else:
                    self.candidate_locks[track_id] = (instant_seat, ts, 1)
        else:
            # Not in any seat polygon
            self.candidate_locks.pop(track_id, None)

        obs.seat_id = state.seat_id
        return state.seat_id

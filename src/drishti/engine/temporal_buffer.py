"""FPS-Independent, Rolling Temporal Buffer per Candidate.

Conforms to docs/ARCHITECTURE.md Layer 3: Temporal Behavioural Engine.
Maintains timestamp-indexed rolling observation histories (default 5.0s window).
Computes kinematics (velocities, displacements, trajectories) purely as a function
of continuous real-time timestamps (seconds), never relying on discrete frame counts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import numpy as np

from .observation import VisionObservation


@dataclass
class KeypointTrajectoryPoint:
    timestamp_sec: float
    x_norm: float
    y_norm: float
    confidence: float


class CandidateTemporalBuffer:
    """A sliding temporal window of observations for a single candidate."""

    def __init__(self, max_duration_sec: float = 5.0):
        self.max_duration_sec = max_duration_sec
        self.buffer: Deque[VisionObservation] = deque()

    def push(self, observation: VisionObservation) -> None:
        """Add a new observation and purge older observations beyond max_duration_sec."""
        self.buffer.append(observation)
        current_time = observation.timestamp_sec

        # Purge past window bound
        while self.buffer and (current_time - self.buffer[0].timestamp_sec) > self.max_duration_sec:
            self.buffer.popleft()

    def get_window(self, duration_sec: Optional[float] = None) -> List[VisionObservation]:
        """Retrieve observations within the requested recent duration (in seconds)."""
        if not self.buffer:
            return []
        
        target_duration = duration_sec if duration_sec is not None else self.max_duration_sec
        latest_time = self.buffer[-1].timestamp_sec
        min_allowed_time = latest_time - target_duration

        return [obs for obs in self.buffer if obs.timestamp_sec >= min_allowed_time]

    @property
    def latest(self) -> Optional[VisionObservation]:
        return self.buffer[-1] if self.buffer else None

    @property
    def duration_sec(self) -> float:
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1].timestamp_sec - self.buffer[0].timestamp_sec

    @property
    def observation_count(self) -> int:
        return len(self.buffer)

    def extract_joint_trajectory(
        self,
        joint_name: str,
        duration_sec: Optional[float] = None
    ) -> List[KeypointTrajectoryPoint]:
        """Extract continuous trajectory for a specific anatomical keypoint."""
        window = self.get_window(duration_sec)
        trajectory: List[KeypointTrajectoryPoint] = []

        for obs in window:
            kp = obs.get_keypoint(joint_name)
            if kp and kp.is_valid:
                trajectory.append(KeypointTrajectoryPoint(
                    timestamp_sec=obs.timestamp_sec,
                    x_norm=kp.x,
                    y_norm=kp.y,
                    confidence=kp.confidence
                ))

        return trajectory

    def compute_joint_velocity(
        self,
        joint_name: str,
        window_sec: float = 0.5
    ) -> float:
        """Compute the instantaneous velocity (normalized Euclidean distance / sec).
        
        Calculated across the specified temporal window dt using least-squares or endpoints.
        """
        trajectory = self.extract_joint_trajectory(joint_name, duration_sec=window_sec)
        if len(trajectory) < 2:
            return 0.0

        p_first = trajectory[0]
        p_last = trajectory[-1]
        dt = p_last.timestamp_sec - p_first.timestamp_sec

        if dt <= 1e-4:
            return 0.0

        dist = float(np.hypot(p_last.x_norm - p_first.x_norm, p_last.y_norm - p_first.y_norm))
        return dist / dt

    def compute_bounding_box_displacement(
        self,
        duration_sec: Optional[float] = None
    ) -> float:
        """Measure total trajectory movement of candidate center over duration."""
        window = self.get_window(duration_sec)
        if len(window) < 2:
            return 0.0

        centers = np.array([obs.center_norm for obs in window])
        ptp_x = float(np.ptp(centers[:, 0]))
        ptp_y = float(np.ptp(centers[:, 1]))
        return float(np.hypot(ptp_x, ptp_y))

    def clear(self) -> None:
        self.buffer.clear()


class TemporalBufferManager:
    """Maintains individual CandidateTemporalBuffers for all candidates in the exam room."""

    def __init__(self, max_duration_sec: float = 5.0):
        self.max_duration_sec = max_duration_sec
        self.buffers: dict[str, CandidateTemporalBuffer] = {}

    def get_or_create(self, candidate_id: str) -> CandidateTemporalBuffer:
        if candidate_id not in self.buffers:
            self.buffers[candidate_id] = CandidateTemporalBuffer(max_duration_sec=self.max_duration_sec)
        return self.buffers[candidate_id]

    def push(self, candidate_id: str, observation: VisionObservation) -> None:
        buf = self.get_or_create(candidate_id)
        buf.push(observation)

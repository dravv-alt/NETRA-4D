"""
Base Perception Provider Interface for Drishti-4D.
Every perception branch (YOLO26, RTMPose, Multi-HMR, ObjectDetector, Replay)
implements this contract to emit standardized PersonObservation and ObjectObservation records.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from ..schemas import PersonObservation, ObjectObservation


class PerceptionProvider(ABC):
    @abstractmethod
    def start(self):
        """Initialize models, weights, and GPU context."""
        pass

    @abstractmethod
    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray,
        camera_id: str = "CAM01",
        video_id: str = "VIDEO_01"
    ) -> Tuple[List[PersonObservation], List[ObjectObservation]]:
        """Processes a single BGR video frame and emits normalized observations."""
        pass

    @abstractmethod
    def close(self):
        """Release GPU memory and workers."""
        pass

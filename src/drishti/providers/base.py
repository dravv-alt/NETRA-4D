"""Common contract for model-backed perception providers."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from ..schemas import ObjectObservation, PersonObservation


class PerceptionProvider(ABC):
    @abstractmethod
    def start(self) -> None:
        """Load the real model and selected compute device."""

    @abstractmethod
    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray,
        camera_id: str = "CAM01",
        video_id: str = "VIDEO_01",
    ) -> Tuple[List[PersonObservation], List[ObjectObservation]]:
        """Execute a model on one BGR frame and return model-derived records."""

    @abstractmethod
    def close(self) -> None:
        """Release model references and accelerator caches."""

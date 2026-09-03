"""
Environment Profiling and Examination Type Classification.
Detects whether an examination setup is a CBT Computer Lab or a Large Written Classroom
based on the density of active monitors/laptops and seated candidates in initial opening frames.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List
import numpy as np


class EnvironmentType(str, Enum):
    CBT_LAB = "CBT_COMPUTER_LAB"
    LARGE_CLASSROOM = "LARGE_WRITTEN_CLASSROOM"


class EnvironmentProfiler:
    """
    Analyzes opening frames (e.g. first 30-60 frames) to profile room setup,
    monitor counts, and candidate density to dispatch the proper examination pipeline.
    """

    def __init__(self, warmup_frames: int = 30, min_monitors_for_cbt: int = 3):
        self.warmup_frames = warmup_frames
        self.min_monitors_for_cbt = min_monitors_for_cbt
        self.frame_count = 0
        self.monitor_detections_history: List[int] = []
        self.person_detections_history: List[int] = []
        self.finalized = False
        self.environment_type = EnvironmentType.CBT_LAB  # Safe default

    def observe(self, people: List[Any], objects: List[Any]) -> None:
        if self.finalized:
            return

        def _get_cname(o: Any) -> str:
            if isinstance(o, dict):
                return str(o.get("class_name", ""))
            return str(getattr(o, "class_name", ""))

        num_monitors = sum(
            1 for obj in objects
            if _get_cname(obj).lower() in ["laptop", "tv", "monitor", "screen"]
        )
        self.monitor_detections_history.append(num_monitors)
        self.person_detections_history.append(len(people))

        if self.frame_count >= self.warmup_frames:
            self.finalize()

    def finalize(self) -> EnvironmentType:
        median_monitors = float(np.median(self.monitor_detections_history)) if self.monitor_detections_history else 0.0
        median_people = float(np.median(self.person_detections_history)) if self.person_detections_history else 0.0

        if median_monitors >= self.min_monitors_for_cbt:
            self.environment_type = EnvironmentType.CBT_LAB
        else:
            self.environment_type = EnvironmentType.LARGE_CLASSROOM

        self.finalized = True
        return self.environment_type

    def get_summary(self) -> Dict[str, Any]:
        return {
            "environment_type": self.environment_type.value,
            "finalized": self.finalized,
            "warmup_frames_observed": self.frame_count,
            "median_monitors_detected": float(np.median(self.monitor_detections_history)) if self.monitor_detections_history else 0.0,
            "median_candidates_detected": float(np.median(self.person_detections_history)) if self.person_detections_history else 0.0,
        }

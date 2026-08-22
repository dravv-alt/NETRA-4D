"""
Zero-Overhead Observation Stream Replayer for Drishti-4D.
Enables instant replay of persisted JSONL observations without re-running heavy GPU models.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import numpy as np
from .base import PerceptionProvider
from ..schemas import PersonObservation, ObjectObservation


class ReplayProvider(PerceptionProvider):
    def __init__(self, observations_jsonl_path: str):
        self.path = Path(observations_jsonl_path)
        self.observations_by_frame: Dict[int, List[PersonObservation]] = {}
        self.observations_by_time: List[PersonObservation] = []

    def start(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Observation JSONL file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                obs = PersonObservation(**data)
                f_idx = obs.frame_index
                if f_idx not in self.observations_by_frame:
                    self.observations_by_frame[f_idx] = []
                self.observations_by_frame[f_idx].append(obs)
                self.observations_by_time.append(obs)

        print(f"[✔] ReplayProvider loaded {len(self.observations_by_time)} observations across {len(self.observations_by_frame)} frames.")

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray,
        camera_id: str = "CAM01",
        video_id: str = "VIDEO_01"
    ) -> Tuple[List[PersonObservation], List[ObjectObservation]]:
        return self.observations_by_frame.get(frame_idx, []), []

    def close(self):
        self.observations_by_frame.clear()
        self.observations_by_time.clear()

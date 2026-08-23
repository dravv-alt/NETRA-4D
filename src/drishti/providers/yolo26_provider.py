"""Ultralytics YOLO pose provider with no simulated fallback path."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import PerceptionProvider
from ..schemas import ComponentQuality, Keypoint, ObjectObservation, PersonObservation, QualityEnum


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class YOLO26PoseProvider(PerceptionProvider):
    """Run genuine Ultralytics pose inference and ByteTrack association."""

    def __init__(
        self,
        model_name: str = "yolo26n-pose.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50,
        tracker: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.tracker = tracker
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None

    def start(self) -> None:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device!r} was requested, but this PyTorch build/runtime "
                "does not expose CUDA. No CPU or simulated fallback was used."
            )
        from ultralytics import YOLO

        self.model = YOLO(self.model_name)

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray,
        camera_id: str = "CAM01",
        video_id: str = "VIDEO_01",
    ) -> Tuple[List[PersonObservation], List[ObjectObservation]]:
        if self.model is None:
            self.start()

        height, width = frame_bgr.shape[:2]
        if self.tracker:
            results = self.model.track(
                frame_bgr,
                persist=True,
                tracker=self.tracker,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame_bgr,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )

        observations: List[PersonObservation] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            tracked = result.boxes.id is not None
            track_ids = (
                result.boxes.id.int().detach().cpu().numpy()
                if tracked
                else np.arange(1, len(boxes) + 1)
            )
            keypoints = (
                result.keypoints.data.detach().cpu().numpy()
                if result.keypoints is not None
                else None
            )

            for index, (box, confidence, track_id) in enumerate(zip(boxes, confidences, track_ids)):
                x1, y1, x2, y2 = (float(value) for value in box)
                keypoint_map: Dict[str, Keypoint] = {}
                if keypoints is not None and index < len(keypoints):
                    for keypoint_index, values in enumerate(keypoints[index]):
                        if keypoint_index >= len(COCO_KEYPOINT_NAMES):
                            break
                        keypoint_map[COCO_KEYPOINT_NAMES[keypoint_index]] = Keypoint(
                            name=COCO_KEYPOINT_NAMES[keypoint_index],
                            x=float(values[0]) / width,
                            y=float(values[1]) / height,
                            confidence=float(values[2]),
                        )

                def mean_confidence(names: List[str]) -> float:
                    values = [keypoint_map[name].confidence for name in names if name in keypoint_map]
                    return float(np.mean(values)) if values else 0.0

                quality = ComponentQuality(
                    head=QualityEnum.GOOD if mean_confidence(COCO_KEYPOINT_NAMES[:5]) >= 0.40 else QualityEnum.POOR,
                    torso=QualityEnum.GOOD if mean_confidence(["left_shoulder", "right_shoulder", "left_hip", "right_hip"]) >= 0.50 else QualityEnum.POOR,
                    hands=QualityEnum.GOOD if mean_confidence(["left_wrist", "right_wrist"]) >= 0.35 else QualityEnum.POOR,
                    track=QualityEnum.GOOD if tracked else QualityEnum.UNKNOWN,
                )
                observations.append(
                    PersonObservation(
                        provider=Path(self.model_name).stem,
                        camera_id=camera_id,
                        video_id=video_id,
                        frame_index=frame_idx,
                        timestamp_sec=timestamp_sec,
                        entity_local_id=int(track_id),
                        bbox_xyxy_px=(x1, y1, x2, y2),
                        bbox_xyxy_norm=(x1 / width, y1 / height, x2 / width, y2 / height),
                        detection_confidence=float(confidence),
                        keypoints_2d=keypoint_map,
                        quality=quality,
                        metadata={
                            "model_weight": str(self.model_name),
                            "device": self.device,
                            "tracker": self.tracker,
                            "entity_id_source": "tracker" if tracked else "frame_local_index",
                        },
                    )
                )
        return observations, []

    def close(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

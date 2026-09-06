"""Canonical, model-agnostic vision observation contracts and adapters.

Conforms to docs/ARCHITECTURE.md Layer 2: Adaptive Vision Layer.
Normalizes raw outputs from YOLO26-Pose, RTM-Pose, and MediaPipe Hands
into a unified VisionObservation representation. No raw vendor-specific
structures leak into downstream behavioral reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class VisibilityState(str, Enum):
    """Visibility of candidate or limb."""
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"


class ObservationQuality(str, Enum):
    """Quality rating derived dynamically from keypoint visibility & tracking."""
    EXCELLENT = "EXCELLENT"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"


@dataclass(frozen=True)
class NormalizedKeypoint:
    """A single body or hand landmark normalized to [0.0, 1.0] frame dimensions."""
    name: str
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float = 0.0  # Relative depth if estimated, else 0.0
    visibility: float = 1.0
    confidence: float = 1.0

    @property
    def is_valid(self) -> bool:
        return 0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0 and self.confidence > 0.0


# Standard 21 MediaPipe hand landmark names in anatomical sequence
HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]


@dataclass
class HandLandmarksObservation:
    """Detailed 21-landmark articulated hand observation (MediaPipe Hands)."""
    side: str  # "LEFT" or "RIGHT"
    landmarks: Dict[str, NormalizedKeypoint] = field(default_factory=dict)
    palm_center_norm: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    is_detected: bool = False

    def get(self, name: str) -> Optional[NormalizedKeypoint]:
        return self.landmarks.get(name)


@dataclass
class DetectedObjectObservation:
    """Normalized object detected in scene (phone, book, screen, chit, etc.)."""
    class_name: str
    confidence: float
    bbox_xyxy_norm: Tuple[float, float, float, float]
    associated_track_id: Optional[int] = None
    association_confidence: float = 0.0
    is_static: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center_norm(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area_norm(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class VisionObservation:
    """Model-agnostic canonical observation of one candidate in one video frame."""
    timestamp_sec: float
    frame_idx: int
    track_id: int
    bbox_xyxy_norm: Tuple[float, float, float, float]
    detection_confidence: float

    # Keypoints (COCO-17 or WholeBody body joints)
    keypoints: Dict[str, NormalizedKeypoint] = field(default_factory=dict)

    # Detailed Articulated Hands (MediaPipe 21-landmarks)
    left_hand: Optional[HandLandmarksObservation] = None
    right_hand: Optional[HandLandmarksObservation] = None

    # Surrounding or associated objects
    nearby_objects: List[DetectedObjectObservation] = field(default_factory=list)

    # Tracking & State Quality
    visibility: VisibilityState = VisibilityState.VISIBLE
    quality: ObservationQuality = ObservationQuality.ACCEPTABLE
    tracking_confidence: float = 1.0
    provider_name: str = "GENERIC"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center_norm(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width_norm(self) -> float:
        return max(0.0, self.bbox_xyxy_norm[2] - self.bbox_xyxy_norm[0])

    @property
    def height_norm(self) -> float:
        return max(0.0, self.bbox_xyxy_norm[3] - self.bbox_xyxy_norm[1])

    def get_keypoint(self, name: str) -> Optional[NormalizedKeypoint]:
        kp = self.keypoints.get(name)
        return kp if kp and kp.is_valid else None


# =========================================================================
# Adapters to convert upstream provider outputs to canonical VisionObservation
# =========================================================================

class YOLO26Adapter:
    """Converts raw YOLO26 pose + object detection outputs into VisionObservations."""

    COCO_KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    @classmethod
    def adapt(
        cls,
        frame_idx: int,
        timestamp_sec: float,
        frame_shape: Tuple[int, int],  # (height, width)
        tracked_box: Any,             # [x1, y1, x2, y2]
        track_id: int,
        keypoints_raw: Optional[np.ndarray],  # shape (17, 3) or (17, 2)
        det_confidence: float,
        detected_objects: Optional[List[DetectedObjectObservation]] = None,
        provider_name: str = "YOLO26-Pose",
    ) -> VisionObservation:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in tracked_box]
        bbox_norm = (
            np.clip(x1 / w, 0.0, 1.0),
            np.clip(y1 / h, 0.0, 1.0),
            np.clip(x2 / w, 0.0, 1.0),
            np.clip(y2 / h, 0.0, 1.0),
        )

        keypoints_dict: Dict[str, NormalizedKeypoint] = {}
        valid_kps = 0

        if keypoints_raw is not None:
            num_pts = min(len(cls.COCO_KEYPOINT_NAMES), len(keypoints_raw))
            for i in range(num_pts):
                pt = keypoints_raw[i]
                px, py = float(pt[0]), float(pt[1])
                conf = float(pt[2]) if len(pt) >= 3 else 1.0
                kname = cls.COCO_KEYPOINT_NAMES[i]

                norm_kp = NormalizedKeypoint(
                    name=kname,
                    x=float(np.clip(px / w, 0.0, 1.0)),
                    y=float(np.clip(py / h, 0.0, 1.0)),
                    confidence=conf,
                    visibility=1.0 if conf > 0.25 else 0.0
                )
                keypoints_dict[kname] = norm_kp
                if norm_kp.is_valid and conf > 0.3:
                    valid_kps += 1

        # Dynamic quality assessment without hardcoded thresholds
        # In CBT exam hall, candidates are seated behind desks (legs/lower-body occluded)
        total_kps = len(cls.COCO_KEYPOINT_NAMES)
        coverage_ratio = valid_kps / float(total_kps) if total_kps > 0 else 0.0

        if coverage_ratio >= 0.50:
            quality = ObservationQuality.EXCELLENT
            visibility = VisibilityState.VISIBLE
        elif coverage_ratio >= 0.15 or (det_confidence >= 0.25 and valid_kps >= 2):
            quality = ObservationQuality.ACCEPTABLE
            visibility = VisibilityState.VISIBLE
        elif coverage_ratio >= 0.05 or (det_confidence >= 0.12 and valid_kps >= 1):
            quality = ObservationQuality.DEGRADED
            visibility = VisibilityState.VISIBLE
        elif valid_kps > 0 or det_confidence >= 0.10:
            quality = ObservationQuality.DEGRADED
            visibility = VisibilityState.OCCLUDED
        else:
            quality = ObservationQuality.UNUSABLE
            visibility = VisibilityState.LOST

        return VisionObservation(
            timestamp_sec=timestamp_sec,
            frame_idx=frame_idx,
            track_id=track_id,
            bbox_xyxy_norm=bbox_norm,
            detection_confidence=det_confidence,
            keypoints=keypoints_dict,
            nearby_objects=detected_objects or [],
            visibility=visibility,
            quality=quality,
            tracking_confidence=min(1.0, det_confidence * (0.5 + 0.5 * coverage_ratio)),
            provider_name=provider_name
        )


class RTMPoseAdapter:
    """Converts RTM-Pose coordinate classification (SimCC) outputs into VisionObservations."""

    @classmethod
    def decode_simcc(
        cls,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        crop_shape: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
    ) -> List[NormalizedKeypoint]:
        """Decodes SimCC 1D probability representations into normalized keypoints."""
        ch, cw = crop_shape[:2]
        # simcc_x: (1, 17, W_simcc), simcc_y: (1, 17, H_simcc)
        sx = simcc_x[0] if simcc_x.ndim == 3 else simcc_x
        sy = simcc_y[0] if simcc_y.ndim == 3 else simcc_y
        num_kps = sx.shape[0]

        keypoints = []
        for i in range(num_kps):
            kname = YOLO26Adapter.COCO_KEYPOINT_NAMES[i] if i < len(YOLO26Adapter.COCO_KEYPOINT_NAMES) else f"kp_{i}"
            loc_x = float(np.argmax(sx[i])) / simcc_split_ratio
            loc_y = float(np.argmax(sy[i])) / simcc_split_ratio

            # Softmax peak confidence proxy
            max_sx = float(np.max(sx[i]))
            max_sy = float(np.max(sy[i]))
            conf = float(np.clip((max_sx + max_sy) / 2.0, 0.0, 1.0))

            # Normalize to [0.0, 1.0] relative to original crop
            norm_x = float(np.clip(loc_x / max(1.0, float(cw)), 0.0, 1.0))
            norm_y = float(np.clip(loc_y / max(1.0, float(ch)), 0.0, 1.0))

            keypoints.append(NormalizedKeypoint(
                name=kname,
                x=norm_x,
                y=norm_y,
                confidence=conf,
                visibility=float(conf > 0.20)
            ))
        return keypoints

    @classmethod
    def adapt(
        cls,
        frame_idx: int,
        timestamp_sec: float,
        frame_shape: Tuple[int, int],
        tracked_box: Any,
        track_id: int,
        simcc_keypoints: Dict[str, Tuple[float, float, float]],  # name -> (norm_x, norm_y, conf)
        det_confidence: float,
        provider_name: str = "RTM-Pose-SimCC",
    ) -> VisionObservation:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in tracked_box]
        bbox_norm = (
            float(np.clip(x1 / w, 0.0, 1.0)),
            float(np.clip(y1 / h, 0.0, 1.0)),
            float(np.clip(x2 / w, 0.0, 1.0)),
            float(np.clip(y2 / h, 0.0, 1.0)),
        )

        keypoints_dict: Dict[str, NormalizedKeypoint] = {}
        for name, (nx, ny, conf) in simcc_keypoints.items():
            keypoints_dict[name] = NormalizedKeypoint(
                name=name,
                x=float(np.clip(nx, 0.0, 1.0)),
                y=float(np.clip(ny, 0.0, 1.0)),
                confidence=float(conf),
                visibility=1.0 if conf > 0.25 else 0.0
            )

        return VisionObservation(
            timestamp_sec=timestamp_sec,
            frame_idx=frame_idx,
            track_id=track_id,
            bbox_xyxy_norm=bbox_norm,
            detection_confidence=det_confidence,
            keypoints=keypoints_dict,
            visibility=VisibilityState.VISIBLE,
            quality=ObservationQuality.EXCELLENT,
            tracking_confidence=det_confidence,
            provider_name=provider_name
        )

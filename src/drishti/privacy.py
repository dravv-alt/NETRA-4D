"""Privacy masking and post-mask head-direction annotations.

The available pose model provides COCO facial keypoints, not iris landmarks.
Consequently this module reports a *gaze proxy* (2-D head direction inferred
from nose/eye geometry), never eye-tracking or a true gaze estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PrivacyAnnotation:
    candidate_id: str
    face_bbox_xyxy_px: tuple[int, int, int, int]
    face_region_source: str
    gaze_proxy: str
    gaze_proxy_confidence: float


def _point(person: Any, name: str, width: int, height: int, min_confidence: float) -> tuple[float, float, float] | None:
    kpts = getattr(person, "keypoints_2d", None) or getattr(person, "keypoints", None)
    if isinstance(kpts, dict):
        point = kpts.get(name)
        if point is not None and getattr(point, "confidence", 0.0) >= min_confidence:
            return point.x * width, point.y * height, point.confidence
    elif isinstance(person, dict) and person.get("keypoints") is not None:
        kp_arr = person["keypoints"]
        COCO_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        if name in COCO_NAMES:
            idx = COCO_NAMES.index(name)
            if idx < len(kp_arr) and kp_arr[idx][2] >= min_confidence:
                return float(kp_arr[idx][0]), float(kp_arr[idx][1]), float(kp_arr[idx][2])
    return None


def _clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def face_region_from_pose(person: Any, width: int, height: int, *, min_confidence: float = 0.25) -> tuple[tuple[int, int, int, int], str]:
    """Return a conservative face/head mask based only on pose or person box."""

    facial = [
        point for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        if (point := _point(person, name, width, height, min_confidence)) is not None
    ]
    if len(facial) >= 2:
        xs, ys = [item[0] for item in facial], [item[1] for item in facial]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 12.0)
        padding_x, padding_y = span * 0.75, span * 1.05
        return _clamp_box(
            (min(xs) - padding_x, min(ys) - padding_y, max(xs) + padding_x, max(ys) + padding_y),
            width, height,
        ), "pose_facial_keypoints"

    if hasattr(person, "bbox_xyxy_px"):
        x1, y1, x2, y2 = person.bbox_xyxy_px
    elif hasattr(person, "bbox_xyxy_norm"):
        b = person.bbox_xyxy_norm
        x1, y1, x2, y2 = b[0] * width, b[1] * height, b[2] * width, b[3] * height
    elif isinstance(person, dict) and "bbox" in person:
        x1, y1, x2, y2 = person["bbox"]
    else:
        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

    person_height = max(float(y2) - float(y1), 1.0)
    # Privacy fallback covers the likely head region; it does not assert a face was detected.
    return _clamp_box((x1, y1, x2, y1 + person_height * 0.35), width, height), "person_head_region_fallback"


def gaze_proxy_from_pose(person: Any, width: int, height: int, *, min_confidence: float = 0.25) -> tuple[str, float]:
    """Estimate 2-D head direction from real eye/nose pose keypoints.

    This is deliberately conservative: without both eyes and a nose, the
    result is UNKNOWN. It must not be interpreted as iris/eye gaze tracking.
    """

    left = _point(person, "left_eye", width, height, min_confidence)
    right = _point(person, "right_eye", width, height, min_confidence)
    nose = _point(person, "nose", width, height, min_confidence)
    if not left or not right or not nose:
        return "UNKNOWN", 0.0
    eye_span = abs(right[0] - left[0])
    if eye_span < 4.0:
        return "UNKNOWN", 0.0
    eye_center_x = (left[0] + right[0]) / 2.0
    normalized_offset = (nose[0] - eye_center_x) / (eye_span / 2.0)
    confidence = min(left[2], right[2], nose[2])
    if normalized_offset >= 0.28:
        return "RIGHT", confidence
    if normalized_offset <= -0.28:
        return "LEFT", confidence
    return "FORWARD", confidence


def _blur_region(image: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return
    width, height = x2 - x1, y2 - y1
    if width >= 4 and height >= 4:
        # Dynamic dual-stage: pixelate to destroy facial biometric features, then Gaussian blur to smooth
        sw = max(2, width // 14)
        sh = max(2, height // 14)
        small = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        kernel_x = max(25, (width // 3) | 1)
        kernel_y = max(25, (height // 3) | 1)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(pixelated, (kernel_x, kernel_y), 0)
    else:
        kernel_x = max(3, (width // 3) | 1)
        kernel_y = max(3, (height // 3) | 1)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel_x, kernel_y), 0)


def apply_privacy_mask_and_collect_annotations(frame: np.ndarray, people: list[Any], *, min_keypoint_confidence: float = 0.25) -> tuple[np.ndarray, list[PrivacyAnnotation]]:
    """Compute gaze first, blur face/head second, return records for final overlay."""

    height, width = frame.shape[:2]
    masked = frame.copy()
    annotations: list[PrivacyAnnotation] = []
    for person in people:
        gaze_proxy, gaze_confidence = gaze_proxy_from_pose(
            person, width, height, min_confidence=min_keypoint_confidence
        )
        face_box, source = face_region_from_pose(
            person, width, height, min_confidence=min_keypoint_confidence
        )
        _blur_region(masked, face_box)
        meta = getattr(person, "metadata", None) or (person.get("metadata", {}) if isinstance(person, dict) else {})
        entity_source = meta.get("entity_id_source", "tracker" if isinstance(person, dict) and "track_id" in person else "frame_local_index")
        ent_id = getattr(person, "entity_local_id", None)
        if ent_id is None and isinstance(person, dict):
            ent_id = person.get("track_id", person.get("entity_local_id", "0"))
        candidate_id = f"Candidate {ent_id}"
        if entity_source != "tracker":
            candidate_id += " (frame-local)"
        annotations.append(PrivacyAnnotation(
            candidate_id=candidate_id,
            face_bbox_xyxy_px=face_box,
            face_region_source=source,
            gaze_proxy=gaze_proxy,
            gaze_proxy_confidence=gaze_confidence,
        ))
    return masked, annotations


def draw_post_blur_candidate_annotations(frame: np.ndarray, people: list[Any], annotations: list[PrivacyAnnotation]) -> np.ndarray:
    """Draw candidate and gaze-proxy labels only after the face/head is blurred."""

    output = frame.copy()
    for person, annotation in zip(people, annotations):
        bx = getattr(person, "bbox_xyxy_px", None) or (person.get("bbox", (0, 0, 0, 0)) if isinstance(person, dict) else (0, 0, 0, 0))
        x1, y1 = int(bx[0]), int(bx[1])
        fx1, fy1, fx2, fy2 = annotation.face_bbox_xyxy_px
        label = f"{annotation.candidate_id} | GAZE PROXY: {annotation.gaze_proxy}"
        cv2.putText(output, label, (x1, max(18, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, label, (x1, max(18, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (65, 35, 15), 1, cv2.LINE_AA)
        center_x, center_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        if annotation.gaze_proxy in {"LEFT", "RIGHT"}:
            delta = -max(18, (fx2 - fx1) // 2) if annotation.gaze_proxy == "LEFT" else max(18, (fx2 - fx1) // 2)
            cv2.arrowedLine(output, (center_x, center_y), (center_x + delta, center_y), (65, 35, 15), 2, cv2.LINE_AA, tipLength=0.25)
    return output

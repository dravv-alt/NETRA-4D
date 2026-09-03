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


def build_adaptive_face_mask(height: int, width: int, people: list[Any], min_confidence: float = 0.15) -> np.ndarray:
    """
    Constructs an angle-adaptive face-contour mask that covers the face boundary
    for privacy protection while preserving the eyes (for gaze tracking) and neck.
    Returns a single-channel uint8 mask (0 = sharp/preserve, 255 = blur).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for person in people:
        person_mask = np.zeros((height, width), dtype=np.uint8)

        # Extract bounding box
        bx = getattr(person, "bbox_xyxy_px", None)
        if bx is None and hasattr(person, "bbox_xyxy_norm"):
            b = person.bbox_xyxy_norm
            bx = (b[0] * width, b[1] * height, b[2] * width, b[3] * height)
        elif bx is None and isinstance(person, dict) and "bbox" in person:
            bx = person["bbox"]
        elif bx is None:
            continue

        bw = max(10.0, float(bx[2] - bx[0]))
        bh = max(10.0, float(bx[3] - bx[1]))

        # Keypoints
        nose = _point(person, "nose", width, height, min_confidence)
        leye = _point(person, "left_eye", width, height, min_confidence)
        reye = _point(person, "right_eye", width, height, min_confidence)
        lear = _point(person, "left_ear", width, height, min_confidence)
        rear = _point(person, "right_ear", width, height, min_confidence)
        lsh = _point(person, "left_shoulder", width, height, min_confidence)
        rsh = _point(person, "right_shoulder", width, height, min_confidence)

        has_nose = nose is not None
        has_leye = leye is not None
        has_reye = reye is not None
        has_eyes = has_leye or has_reye
        has_shoulders = lsh is not None and rsh is not None

        if has_eyes or has_nose:
            cranial_pts = [p[:2] for p in [nose, leye, reye, lear, rear] if p is not None]
            fc_x = float(np.mean([p[0] for p in cranial_pts]))
            fc_y = float(np.mean([p[1] for p in cranial_pts]))

            if has_leye and has_reye:
                eye_dist = float(np.hypot(reye[0] - leye[0], reye[1] - leye[1]))
                eye_dist = max(eye_dist, bw * 0.16)
                angle_deg = float(np.degrees(np.arctan2(reye[1] - leye[1], reye[0] - leye[0])))
            else:
                eye_dist = bw * 0.20
                angle_deg = 0.0

            # Proportional face ellipse: tightly contours cheeks, mouth, jaw
            axis_x = int(max(6.0, eye_dist * 1.25))
            axis_y = int(max(8.0, eye_dist * 1.50))

            if has_nose and has_eyes:
                mid_eye_y = (leye[1] + reye[1]) / 2.0 if (has_leye and has_reye) else (leye[1] if has_leye else reye[1])
                fc_y += (nose[1] - mid_eye_y) * 0.35

            cv2.ellipse(person_mask, (int(fc_x), int(fc_y)), (axis_x, axis_y), angle_deg, 0, 360, 255, -1)

            # Preserve eyes for gaze tracking
            eye_radius = int(max(3.0, eye_dist * 0.32))
            if has_leye:
                cv2.circle(person_mask, (int(leye[0]), int(leye[1])), eye_radius, 0, -1)
            if has_reye:
                cv2.circle(person_mask, (int(reye[0]), int(reye[1])), eye_radius, 0, -1)
            if has_leye and has_reye:
                mid_eye = (int((leye[0] + reye[0]) / 2.0), int((leye[1] + reye[1]) / 2.0))
                cv2.ellipse(person_mask, mid_eye, (int(eye_dist * 0.55), int(eye_radius * 0.75)), angle_deg, 0, 360, 0, -1)

            # Preserve neck
            if has_shoulders:
                sh_mid_y = (lsh[1] + rsh[1]) / 2.0
                chin_y = fc_y + axis_y * 0.75
                if sh_mid_y > chin_y:
                    neck_poly = np.array([
                        [int(fc_x - axis_x * 1.5), int(chin_y)],
                        [int(fc_x + axis_x * 1.5), int(chin_y)],
                        [int(rsh[0]), int(rsh[1])],
                        [int(lsh[0]), int(lsh[1])]
                    ], dtype=np.int32)
                    cv2.fillPoly(person_mask, [neck_poly], 0)
        elif has_shoulders:
            # Back-facing: cranial cap centered above shoulders
            sh_mid_x = (lsh[0] + rsh[0]) / 2.0
            sh_mid_y = (lsh[1] + rsh[1]) / 2.0
            sh_span = float(np.hypot(rsh[0] - lsh[0], rsh[1] - lsh[1]))
            head_x = int(sh_mid_x)
            head_y = int(sh_mid_y - sh_span * 0.48)
            cw = int(sh_span * 0.26)
            ch = int(sh_span * 0.30)
            cv2.ellipse(person_mask, (head_x, head_y), (cw, ch), 0, 0, 360, 255, -1)
        else:
            # Fallback when keypoints are missing (top 28% of box)
            cw = int(bw * 0.35)
            ch = int(bh * 0.20)
            cx = int((bx[0] + bx[2]) / 2.0)
            cy = int(bx[1] + ch * 0.9)
            cv2.ellipse(person_mask, (cx, cy), (cw, ch), 0, 0, 360, 255, -1)

        mask = cv2.bitwise_or(mask, person_mask)
    return mask


def apply_privacy_mask_and_collect_annotations(frame: np.ndarray, people: list[Any], *, min_keypoint_confidence: float = 0.25) -> tuple[np.ndarray, list[PrivacyAnnotation]]:
    """Compute gaze first, blur face boundary second (preserving eyes and neck), return records for final overlay."""

    height, width = frame.shape[:2]
    annotations: list[PrivacyAnnotation] = []
    for person in people:
        gaze_proxy, gaze_confidence = gaze_proxy_from_pose(
            person, width, height, min_confidence=min_keypoint_confidence
        )
        face_box, source = face_region_from_pose(
            person, width, height, min_confidence=min_keypoint_confidence
        )
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

    # Construct angle-adaptive face-contour mask (preserves eyes and neck)
    mask = build_adaptive_face_mask(height, width, people, min_confidence=min_keypoint_confidence)

    if np.any(mask > 0):
        # Create full-frame heavy dual-stage blur (pixelated + Gaussian)
        sw, sh = max(4, width // 25), max(4, height // 25)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        blurred_frame = cv2.GaussianBlur(pixelated, (31, 31), 0)

        # Smooth feathered alpha blend
        mask_feathered = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha = (mask_feathered.astype(np.float32) / 255.0)[:, :, np.newaxis]
        masked = (alpha * blurred_frame.astype(np.float32) + (1.0 - alpha) * frame.astype(np.float32)).astype(np.uint8)
    else:
        masked = frame.copy()

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

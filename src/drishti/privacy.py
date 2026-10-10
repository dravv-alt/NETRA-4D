"""Privacy masking and post-mask head-direction annotations.

The available pose model provides COCO facial keypoints, not iris landmarks.
Consequently this module reports a *gaze proxy* (2-D head direction inferred
from nose/eye geometry), never eye-tracking or a true gaze estimate.

Mask geometry is angle-adaptive: it degrades gracefully through tiers as
facial keypoints drop out (frontal -> three-quarter/profile -> hard profile
-> back-facing -> unknown), so a profile-view head gets a rotated ellipse
fit to its actual roll instead of a generic axis-aligned blob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

_COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

Point = tuple[float, float, float]  # (x, y, confidence)


@dataclass(frozen=True)
class PrivacyAnnotation:
    candidate_id: str
    face_bbox_xyxy_px: tuple[int, int, int, int]
    face_region_source: str
    gaze_proxy: str
    gaze_proxy_confidence: float


# ---------------------------------------------------------------------------
# Keypoint access (unchanged public behavior; supports both attribute-style
# and dict-style person records)
# ---------------------------------------------------------------------------

def _point(person: Any, name: str, width: int, height: int, min_confidence: float) -> Point | None:
    kpts = getattr(person, "keypoints_2d", None) or getattr(person, "keypoints", None)
    if isinstance(kpts, dict):
        pt = kpts.get(name)
        if pt is not None and getattr(pt, "confidence", 0.0) >= min_confidence:
            return pt.x * width, pt.y * height, pt.confidence
    elif isinstance(person, dict) and person.get("keypoints") is not None:
        kp_arr = person["keypoints"]
        if name in _COCO_NAMES:
            idx = _COCO_NAMES.index(name)
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


def _bbox_of(person: Any, width: int, height: int) -> tuple[float, float, float, float]:
    if hasattr(person, "bbox_xyxy_px"):
        return person.bbox_xyxy_px
    if hasattr(person, "bbox_xyxy_norm"):
        b = person.bbox_xyxy_norm
        return b[0] * width, b[1] * height, b[2] * width, b[3] * height
    if isinstance(person, dict) and "bbox" in person:
        return tuple(person["bbox"])
    return 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Public: conservative face region for annotation placement (unchanged)
# ---------------------------------------------------------------------------

def face_region_from_pose(person: Any, width: int, height: int, *, min_confidence: float = 0.25) -> tuple[tuple[int, int, int, int], str]:
    """Return a conservative face/head mask based only on pose or person box."""
    facial = [
        pt for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        if (pt := _point(person, name, width, height, min_confidence)) is not None
    ]
    if len(facial) >= 2:
        xs, ys = [p[0] for p in facial], [p[1] for p in facial]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 12.0)
        padding_x, padding_y = span * 0.75, span * 1.05
        return _clamp_box(
            (min(xs) - padding_x, min(ys) - padding_y, max(xs) + padding_x, max(ys) + padding_y),
            width, height,
        ), "pose_facial_keypoints"

    x1, y1, x2, y2 = _bbox_of(person, width, height)
    person_height = max(float(y2) - float(y1), 1.0)
    # Privacy fallback covers the likely head region; it does not assert a face was detected.
    return _clamp_box((x1, y1, x2, y1 + person_height * 0.35), width, height), "person_head_region_fallback"


# ---------------------------------------------------------------------------
# Public: gaze proxy (pure keypoint geometry, unchanged)
# ---------------------------------------------------------------------------

def gaze_proxy_from_pose(person: Any, width: int, height: int, *, min_confidence: float = 0.25) -> tuple[str, float]:
    """Estimate 2-D head direction from real eye/nose pose keypoints.

    Deliberately conservative: without both eyes and a nose, the result is
    UNKNOWN. This is a head-direction proxy, not iris/eye gaze tracking.
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


# ---------------------------------------------------------------------------
# Tiered, angle-adaptive face geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FaceGeometry:
    center: tuple[float, float]
    axis_x: int
    axis_y: int
    angle_deg: float
    tier: str


def _roll_from_vector(dx: float, dy: float) -> float:
    """Map any directed line to a roll angle in (-90, 90] degrees.

    Using atan2 directly would flip sign depending on which side of the face
    supplied the line (left eye->ear vs right eye->ear point in opposite
    x-directions). Folding into a half-turn range makes the ellipse rotation
    consistent regardless of which side's keypoints happened to be visible.
    """
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def _face_geometry(
    nose: Point | None,
    leye: Point | None,
    reye: Point | None,
    lear: Point | None,
    rear: Point | None,
    bw: float,
) -> _FaceGeometry | None:
    """Tiered angle-adaptive face ellipse geometry.

    Tier 1 (frontal): both eyes visible -> eye line gives roll directly.
    Tier 2 (three-quarter/profile): one eye + its same-side ear -> eye-ear
        line approximates roll (eye and ear sit on a near-horizontal line
        across yaw, unlike the eye-eye line which disappears in profile).
    Tier 3 (hard profile / eye occluded): nose + an ear -> nose-ear line as
        a last-resort roll estimate.
    Tier 4 (one eye + nose, ear missing/occluded): nose-eye line as a
        weaker roll estimate.
    Tier 5 (nose only, or eyes/ears with no nose): no orientation cue ->
        axis-aligned ellipse.
    """
    cranial = [p for p in (nose, leye, reye, lear, rear) if p is not None]
    if not cranial:
        return None

    fc_x = float(np.mean([p[0] for p in cranial]))
    fc_y = float(np.mean([p[1] for p in cranial]))

    if leye is not None and reye is not None:
        eye_dist = max(float(np.hypot(reye[0] - leye[0], reye[1] - leye[1])), bw * 0.16)
        angle_deg = _roll_from_vector(reye[0] - leye[0], reye[1] - leye[1])
        tier = "frontal"
    elif (leye is not None and lear is not None) or (reye is not None and rear is not None):
        eye, ear = (leye, lear) if (leye is not None and lear is not None) else (reye, rear)
        angle_deg = _roll_from_vector(ear[0] - eye[0], ear[1] - eye[1])
        eye_dist = bw * 0.20
        tier = "profile_eye_ear"
    elif nose is not None and (lear is not None or rear is not None):
        ear = lear if lear is not None else rear
        angle_deg = _roll_from_vector(ear[0] - nose[0], ear[1] - nose[1])
        eye_dist = bw * 0.20
        tier = "nose_ear"
    elif nose is not None and (leye is not None or reye is not None):
        # One eye + nose, no matching ear (e.g. ear occluded by hair) --
        # weaker than the eye-ear line but still better than defaulting to 0.
        eye = leye if leye is not None else reye
        angle_deg = _roll_from_vector(nose[0] - eye[0], nose[1] - eye[1])
        eye_dist = bw * 0.20
        tier = "nose_eye"
    else:
        angle_deg = 0.0
        eye_dist = bw * 0.20
        tier = "sparse"

    axis_x = int(max(6.0, eye_dist * 1.25))
    axis_y = int(max(8.0, eye_dist * 1.50))

    if nose is not None and (leye is not None or reye is not None):
        mid_eye_y = (
            (leye[1] + reye[1]) / 2.0 if (leye is not None and reye is not None)
            else (leye[1] if leye is not None else reye[1])
        )
        fc_y += (nose[1] - mid_eye_y) * 0.35

    return _FaceGeometry((fc_x, fc_y), axis_x, axis_y, angle_deg, tier)


def _draw_face_mask(person_mask: np.ndarray, geom: _FaceGeometry, leye: Point | None, reye: Point | None,
                     lsh: Point | None, rsh: Point | None, eye_dist_hint: float) -> None:
    fc_x, fc_y = geom.center
    cv2.ellipse(person_mask, (int(fc_x), int(fc_y)), (geom.axis_x, geom.axis_y), geom.angle_deg, 0, 360, 255, -1)

    # The rendered privacy boundary covers the complete face, including the
    # eyes.  Geometric gaze-proxy features are collected before this function
    # is called; exposing the eyes in the published frame is neither necessary
    # nor acceptable for privacy.

    # --- Preserve neck: strictly exclude neck/collar region ---
    if lsh is not None and rsh is not None:
        sh_mid_y = (lsh[1] + rsh[1]) / 2.0
        chin_y = fc_y + geom.axis_y * 0.75
        if sh_mid_y > chin_y:
            neck_poly = np.array([
                [int(fc_x - geom.axis_x * 1.5), int(chin_y)],
                [int(fc_x + geom.axis_x * 1.5), int(chin_y)],
                [int(rsh[0]), int(rsh[1])],
                [int(lsh[0]), int(lsh[1])],
            ], dtype=np.int32)
            cv2.fillPoly(person_mask, [neck_poly], 0)


def build_adaptive_face_mask(height: int, width: int, people: list[Any], min_confidence: float = 0.15) -> np.ndarray:
    """
    Constructs an angle-adaptive face-contour mask that hugs the face boundary
    for privacy protection while preserving the eyes (for gaze tracking) and
    neck, across frontal, three-quarter, profile, and back-facing views.
    Returns a single-channel uint8 mask (0 = sharp/preserve, 255 = blur).
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for person in people:
        person_mask = np.zeros((height, width), dtype=np.uint8)

        has_bbox_source = (
            hasattr(person, "bbox_xyxy_px")
            or hasattr(person, "bbox_xyxy_norm")
            or (isinstance(person, dict) and "bbox" in person)
        )
        if not has_bbox_source:
            # No bbox at all -- matches the original behavior of skipping
            # this person rather than guessing a degenerate zero-sized box.
            continue
        bx = _bbox_of(person, width, height)
        bw = max(10.0, float(bx[2] - bx[0]))
        bh = max(10.0, float(bx[3] - bx[1]))

        nose = _point(person, "nose", width, height, min_confidence)
        leye = _point(person, "left_eye", width, height, min_confidence)
        reye = _point(person, "right_eye", width, height, min_confidence)
        lear = _point(person, "left_ear", width, height, min_confidence)
        rear = _point(person, "right_ear", width, height, min_confidence)
        lsh = _point(person, "left_shoulder", width, height, min_confidence)
        rsh = _point(person, "right_shoulder", width, height, min_confidence)

        has_facial_cue = nose is not None or leye is not None or reye is not None
        has_shoulders = lsh is not None and rsh is not None

        if has_facial_cue:
            geom = _face_geometry(nose, leye, reye, lear, rear, bw)
            eye_dist_hint = (
                float(np.hypot(reye[0] - leye[0], reye[1] - leye[1])) if (leye is not None and reye is not None)
                else bw * 0.20
            )
            _draw_face_mask(person_mask, geom, leye, reye, lsh, rsh, eye_dist_hint)
        elif has_shoulders:
            # Back-facing: cranial cap centered above shoulders, rotated to
            # match shoulder tilt instead of always sitting axis-aligned.
            sh_mid_x = (lsh[0] + rsh[0]) / 2.0
            sh_mid_y = (lsh[1] + rsh[1]) / 2.0
            sh_span = float(np.hypot(rsh[0] - lsh[0], rsh[1] - lsh[1]))
            shoulder_angle = _roll_from_vector(rsh[0] - lsh[0], rsh[1] - lsh[1])
            head_x = int(sh_mid_x)
            head_y = int(sh_mid_y - sh_span * 0.48)
            cw = int(max(6.0, sh_span * 0.26))
            ch = int(max(8.0, sh_span * 0.30))
            cv2.ellipse(person_mask, (head_x, head_y), (cw, ch), shoulder_angle, 0, 360, 255, -1)
        else:
            # Zero-keypoint fallback: top slice of the person bbox.
            cw = int(bw * 0.35)
            ch = int(bh * 0.20)
            cx = int((bx[0] + bx[2]) / 2.0)
            cy = int(bx[1] + ch * 0.9)
            cv2.ellipse(person_mask, (cx, cy), (cw, ch), 0, 0, 360, 255, -1)

        mask = cv2.bitwise_or(mask, person_mask)

    return mask


# ---------------------------------------------------------------------------
# Public: full pipeline
# ---------------------------------------------------------------------------

def collect_candidate_annotations(frame: np.ndarray, people: list[Any], *, min_keypoint_confidence: float = 0.25) -> list[PrivacyAnnotation]:
    """Collect internal geometric annotations without changing the image.

    These records are still required by the temporal engine even when a local
    operator deliberately chooses an unmasked review feed.
    """

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

    return annotations


def apply_privacy_mask_and_collect_annotations(frame: np.ndarray, people: list[Any], *, min_keypoint_confidence: float = 0.25) -> tuple[np.ndarray, list[PrivacyAnnotation]]:
    """Compute annotations, then apply the opt-in privacy mask to a copy."""

    height, width = frame.shape[:2]
    annotations = collect_candidate_annotations(frame, people, min_keypoint_confidence=min_keypoint_confidence)
    mask = build_adaptive_face_mask(height, width, people, min_confidence=min_keypoint_confidence)

    if np.any(mask > 0):
        sw, sh = max(4, width // 25), max(4, height // 25)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        blurred_frame = cv2.GaussianBlur(pixelated, (31, 31), 0)

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

"""Temporal and Biomechanical Feature Extraction Engine.

Conforms to docs/ARCHITECTURE.md Layer 2 & Layer 3.
Extracts normalized, dimensionless biomechanical features from VisionObservations:
- Head orientation & yaw/pitch dynamics (normalized by inter-pupillary span)
- Torso bi-planar orientation (normalized by shoulder span)
- Hand dynamics, dual-hand convergence, and lap/under-desk excursions
- MediaPipe 21-landmark articulated hand features (pinch, finger curl)
- Cross-candidate mutual orientation & interpersonal distance

Strict Invariant: All features are dimensionless ratios or normalized angles.
Zero hardcoded pixel measurements or arbitrary camera-dependent constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .observation import NormalizedKeypoint, VisionObservation, HandLandmarksObservation
from .smoothing import KeypointSmoother, SmoothedSignal, TimeWeightedEMA


class GazeDirection(str, Enum):
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    UP = "UP"
    UNKNOWN = "UNKNOWN"


@dataclass
class HeadFeatures:
    """Head orientation and kinematic rotation features."""
    yaw_proxy: float            # Normalized horizontal angle: [-1.0 (left), 1.0 (right)]
    pitch_proxy: float          # Normalized vertical angle: [-1.0 (up), 1.0 (down)]
    direction: GazeDirection    # Discrete quantized gaze proxy
    angular_velocity: float     # Rate of yaw change: d(yaw)/dt (rad/sec proxy)
    confidence: float           # Minimum confidence of facial keypoints used
    is_valid: bool              # True if eye span and nose were reliably detected


@dataclass
class TorsoFeatures:
    """Torso alignment and rotational posture."""
    orientation_ratio: float    # Horizontal shoulder-to-hip displacement / shoulder width
    tilt_angle_deg: float       # Lateral shoulder tilt in degrees
    is_turned: bool             # True if torso is significantly rotated away from neutral
    confidence: float
    is_valid: bool


@dataclass
class DetailedHandFeatures:
    """Articulated finger and palm features derived from MediaPipe 21 landmarks."""
    side: str                   # "LEFT" or "RIGHT"
    pinch_distance_ratio: float # Normalized thumb-to-index distance / palm size
    is_pinching: bool           # True if fingertips are grasping/pinching
    mean_finger_curl: float     # [0.0 (extended) to 1.0 (fully curled/fist)]
    confidence: float
    is_detected: bool


@dataclass
class HandDynamicsFeatures:
    """Dual-hand position, velocities, desk convergence, and object proximity."""
    dual_hand_convergence_ratio: float  # Wrist-to-wrist distance normalized by shoulder width
    is_converged: bool                  # True if hands are touching/cupped together
    max_hand_velocity: float            # Highest instantaneous speed among visible wrists
    left_hand_below_lap: bool           # True if left wrist is positioned below hip line
    right_hand_below_lap: bool          # True if right wrist is positioned below hip line
    min_object_distance_ratio: float    # Distance to nearest detected object / torso length
    nearest_object_class: Optional[str] # Class of closest object
    detailed_left: Optional[DetailedHandFeatures] = None
    detailed_right: Optional[DetailedHandFeatures] = None
    confidence: float = 1.0


@dataclass
class CandidateBiomechanicalFeatures:
    """Composite feature bundle extracted from one candidate at timestamp t."""
    timestamp_sec: float
    candidate_id: str
    head: HeadFeatures
    torso: TorsoFeatures
    hands: HandDynamicsFeatures


class BiomechanicalFeatureExtractor:
    """Extracts scale-invariant biomechanical features for candidates."""

    def __init__(self, smoothing_tau: float = 0.12):
        self.smoother = KeypointSmoother(time_constant_sec=smoothing_tau)
        self.head_yaw_ema: dict[str, TimeWeightedEMA] = {}

    def extract(
        self,
        candidate_id: str,
        observation: VisionObservation
    ) -> CandidateBiomechanicalFeatures:
        t = observation.timestamp_sec

        # 1. Extract Head Features
        head = self._extract_head_features(candidate_id, observation, t)

        # 2. Extract Torso Features
        torso = self._extract_torso_features(observation)

        # 3. Extract Hand Dynamics Features
        hands = self._extract_hand_features(observation, torso, t)

        return CandidateBiomechanicalFeatures(
            timestamp_sec=t,
            candidate_id=candidate_id,
            head=head,
            torso=torso,
            hands=hands
        )

    def _extract_head_features(
        self,
        cand_id: str,
        obs: VisionObservation,
        t: float
    ) -> HeadFeatures:
        nose = obs.get_keypoint("nose")
        left_eye = obs.get_keypoint("left_eye")
        right_eye = obs.get_keypoint("right_eye")

        if not nose or not left_eye or not right_eye:
            return HeadFeatures(
                yaw_proxy=0.0,
                pitch_proxy=0.0,
                direction=GazeDirection.UNKNOWN,
                angular_velocity=0.0,
                confidence=0.0,
                is_valid=False
            )

        # Smooth facial keypoints
        snx, sny, _, _ = self.smoother.smooth_point(f"{cand_id}_nose", nose.x, nose.y, t)
        slx, sly, _, _ = self.smoother.smooth_point(f"{cand_id}_leye", left_eye.x, left_eye.y, t)
        srx, sry, _, _ = self.smoother.smooth_point(f"{cand_id}_reye", right_eye.x, right_eye.y, t)

        # Anatomical eye span
        eye_span_x = abs(srx - slx)
        if eye_span_x < 1e-4:
            return HeadFeatures(0.0, 0.0, GazeDirection.UNKNOWN, 0.0, 0.0, False)

        eye_mid_x = (slx + srx) / 2.0
        eye_mid_y = (sly + sry) / 2.0
        half_span = eye_span_x / 2.0

        # Dimensionless horizontal yaw proxy
        raw_yaw = (snx - eye_mid_x) / half_span
        # Dimensionless vertical pitch proxy
        raw_pitch = (sny - eye_mid_y) / eye_span_x

        # Temporal EMA on head yaw to compute angular velocity
        if cand_id not in self.head_yaw_ema:
            self.head_yaw_ema[cand_id] = TimeWeightedEMA(time_constant_sec=0.15)
        yaw_sig = self.head_yaw_ema[cand_id].update(raw_yaw, t)

        # Direction classification based on continuous dimensionless ratios
        direction = GazeDirection.FORWARD
        if yaw_sig.smoothed_value > 0.40:
            direction = GazeDirection.RIGHT
        elif yaw_sig.smoothed_value < -0.40:
            direction = GazeDirection.LEFT
        elif raw_pitch > 0.85:
            direction = GazeDirection.DOWN
        elif raw_pitch < -0.40:
            direction = GazeDirection.UP

        conf = min(nose.confidence, left_eye.confidence, right_eye.confidence)

        return HeadFeatures(
            yaw_proxy=yaw_sig.smoothed_value,
            pitch_proxy=raw_pitch,
            direction=direction,
            angular_velocity=abs(yaw_sig.velocity),
            confidence=conf,
            is_valid=True
        )

    def _extract_torso_features(self, obs: VisionObservation) -> TorsoFeatures:
        ls = obs.get_keypoint("left_shoulder")
        rs = obs.get_keypoint("right_shoulder")
        lh = obs.get_keypoint("left_hip")
        rh = obs.get_keypoint("right_hip")

        if not ls or not rs:
            return TorsoFeatures(0.0, 0.0, False, 0.0, False)

        shoulder_span = float(np.hypot(rs.x - ls.x, rs.y - ls.y))
        if shoulder_span < 1e-4:
            return TorsoFeatures(0.0, 0.0, False, 0.0, False)

        # Shoulder line lateral tilt angle (degrees)
        tilt_deg = float(np.degrees(np.arctan2(rs.y - ls.y, rs.x - ls.x)))

        # Torso bi-planar rotation if hips are visible
        orientation_ratio = 0.0
        is_turned = False
        conf = min(ls.confidence, rs.confidence)

        if lh and rh:
            shoulder_mid_x = (ls.x + rs.x) / 2.0
            hip_mid_x = (lh.x + rh.x) / 2.0
            orientation_ratio = (shoulder_mid_x - hip_mid_x) / shoulder_span
            is_turned = abs(orientation_ratio) > 0.35
            conf = min(conf, lh.confidence, rh.confidence)

        return TorsoFeatures(
            orientation_ratio=orientation_ratio,
            tilt_angle_deg=tilt_deg,
            is_turned=is_turned,
            confidence=conf,
            is_valid=True
        )

    def _extract_hand_features(
        self,
        obs: VisionObservation,
        torso: TorsoFeatures,
        t: float
    ) -> HandDynamicsFeatures:
        lw = obs.get_keypoint("left_wrist")
        rw = obs.get_keypoint("right_wrist")
        ls = obs.get_keypoint("left_shoulder")
        rs = obs.get_keypoint("right_shoulder")
        lh = obs.get_keypoint("left_hip")
        rh = obs.get_keypoint("right_hip")

        # Normalizer: shoulder span (or fallback to bounding box width)
        shoulder_span = (
            float(np.hypot(rs.x - ls.x, rs.y - ls.y))
            if (ls and rs) else max(0.05, obs.width_norm * 0.4)
        )
        torso_len = (
            float(abs((lh.y + rh.y) / 2.0 - (ls.y + rs.y) / 2.0))
            if (ls and rs and lh and rh) else max(0.10, obs.height_norm * 0.5)
        )

        # 1. Dual-Hand Convergence Ratio (normalized by shoulder span)
        convergence_ratio = 1.0
        is_converged = False
        if lw and rw:
            wrist_dist = float(np.hypot(lw.x - rw.x, lw.y - rw.y))
            convergence_ratio = wrist_dist / max(1e-4, shoulder_span)
            # Hands cupped/manipulating when distance < 0.35 of shoulder span
            is_converged = convergence_ratio < 0.35

        # 2. Lap / Under-desk excursion (wrist below hip horizontal plane)
        hip_y = (lh.y + rh.y) / 2.0 if (lh and rh) else (obs.bbox_xyxy_norm[1] + 0.65 * obs.height_norm)
        left_below = bool(lw and lw.y > hip_y)
        right_below = bool(rw and rw.y > hip_y)

        # 3. Object proximity (distance to nearest detected item / torso length)
        min_obj_dist = 1.0
        nearest_class = None
        wrists = [w for w in (lw, rw) if w]

        for obj in obs.nearby_objects:
            ox, oy = obj.center_norm
            for w in wrists:
                d = float(np.hypot(w.x - ox, w.y - oy)) / max(1e-4, torso_len)
                if d < min_obj_dist:
                    min_obj_dist = d
                    nearest_class = obj.class_name

        # 4. Detailed MediaPipe Hands features if available
        det_left = self._extract_detailed_hand(obs.left_hand) if obs.left_hand else None
        det_right = self._extract_detailed_hand(obs.right_hand) if obs.right_hand else None

        return HandDynamicsFeatures(
            dual_hand_convergence_ratio=convergence_ratio,
            is_converged=is_converged,
            max_hand_velocity=0.0,  # Computed continuously via buffer
            left_hand_below_lap=left_below,
            right_hand_below_lap=right_below,
            min_object_distance_ratio=min_obj_dist,
            nearest_object_class=nearest_class,
            detailed_left=det_left,
            detailed_right=det_right,
            confidence=1.0
        )

    def _extract_detailed_hand(
        self,
        hand_obs: Optional[HandLandmarksObservation]
    ) -> Optional[DetailedHandFeatures]:
        if not hand_obs or not hand_obs.is_detected:
            return None

        wrist = hand_obs.get("wrist")
        thumb_tip = hand_obs.get("thumb_tip")
        index_tip = hand_obs.get("index_finger_tip")
        index_mcp = hand_obs.get("index_finger_mcp")

        if not wrist or not thumb_tip or not index_tip or not index_mcp:
            return None

        # Palm scale (wrist to index MCP distance)
        palm_size = float(np.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y))
        if palm_size < 1e-4:
            return None

        # Normalized pinch distance
        pinch_dist = float(np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y))
        pinch_ratio = pinch_dist / palm_size
        is_pinching = pinch_ratio < 0.25

        # Finger curl: distance of fingertips to wrist relative to extended finger span
        tips = ["index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
        tip_dists = [
            float(np.hypot(pt.x - wrist.x, pt.y - wrist.y))
            for name in tips if (pt := hand_obs.get(name))
        ]
        mean_curl = 1.0 - (float(np.mean(tip_dists)) / (2.0 * palm_size)) if tip_dists else 0.0

        return DetailedHandFeatures(
            side=hand_obs.side,
            pinch_distance_ratio=pinch_ratio,
            is_pinching=is_pinching,
            mean_finger_curl=float(np.clip(mean_curl, 0.0, 1.0)),
            confidence=hand_obs.confidence,
            is_detected=True
        )

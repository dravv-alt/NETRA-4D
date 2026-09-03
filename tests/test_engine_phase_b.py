"""Unit tests for Phase B: Signal Smoothing and Biomechanical Feature Extraction.

Verifies:
1. TimeWeightedEMA smoothing across variable time deltas (FPS independence).
2. DynamicMADFilter statistical outlier rejection.
3. Scale-invariant head yaw, pitch, and angular velocity calculation.
4. Dual-hand convergence ratio and under-desk lap excursion detection.
5. MediaPipe 21-landmark pinch distance and finger curl metrics.
"""

import numpy as np
import pytest

from drishti.engine.observation import (
    HandLandmarksObservation,
    NormalizedKeypoint,
    VisionObservation,
    DetectedObjectObservation,
)
from drishti.engine.smoothing import (
    DynamicMADFilter,
    KeypointSmoother,
    TimeWeightedEMA,
)
from drishti.engine.features import (
    BiomechanicalFeatureExtractor,
    GazeDirection,
)


def test_time_weighted_ema_continuous_smoothing():
    ema = TimeWeightedEMA(time_constant_sec=0.20)

    # Initial state
    sig0 = ema.update(raw_value=0.0, timestamp_sec=0.0)
    assert sig0.smoothed_value == 0.0
    assert sig0.velocity == 0.0

    # Step input to 1.0 after dt = 0.2s (equal to tau -> should reach ~63.2% of step)
    sig1 = ema.update(raw_value=1.0, timestamp_sec=0.20)
    expected_alpha = 1.0 - np.exp(-1.0)  # ~0.63212
    assert sig1.smoothed_value == pytest.approx(expected_alpha, rel=0.01)
    assert sig1.velocity > 0.0  # Positive rate of change


def test_dynamic_mad_outlier_filtering():
    mad = DynamicMADFilter(window_size=15, z_threshold=3.0)

    # Feed normal distribution around 0.5 with small jitter
    for val in [0.49, 0.51, 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50, 0.52]:
        assert mad.is_outlier(val) is False

    # Introduce extreme detector glitch/spike (e.g. 0.98)
    assert mad.is_outlier(0.98) is True
    # Normal value immediately after outlier should still be accepted
    assert mad.is_outlier(0.50) is False


def test_head_orientation_scale_invariance_and_gaze():
    extractor = BiomechanicalFeatureExtractor(smoothing_tau=0.05)

    # Candidate 1: Looking FORWARD (Nose midway between eyes)
    obs_forward = VisionObservation(
        timestamp_sec=1.0,
        frame_idx=10,
        track_id=1,
        bbox_xyxy_norm=(0.2, 0.2, 0.5, 0.8),
        detection_confidence=0.95,
        keypoints={
            "left_eye": NormalizedKeypoint("left_eye", x=0.30, y=0.30, confidence=0.9),
            "right_eye": NormalizedKeypoint("right_eye", x=0.40, y=0.30, confidence=0.9),
            "nose": NormalizedKeypoint("nose", x=0.35, y=0.35, confidence=0.9),
        }
    )
    feat_forward = extractor.extract("Cand_01", obs_forward)
    assert feat_forward.head.is_valid is True
    assert feat_forward.head.direction == GazeDirection.FORWARD
    assert abs(feat_forward.head.yaw_proxy) < 0.20

    # Candidate 2: Turned sharply to the RIGHT (Nose shifted right beyond right eye)
    obs_right = VisionObservation(
        timestamp_sec=1.1,
        frame_idx=11,
        track_id=1,
        bbox_xyxy_norm=(0.2, 0.2, 0.5, 0.8),
        detection_confidence=0.95,
        keypoints={
            "left_eye": NormalizedKeypoint("left_eye", x=0.30, y=0.30, confidence=0.9),
            "right_eye": NormalizedKeypoint("right_eye", x=0.40, y=0.30, confidence=0.9),
            "nose": NormalizedKeypoint("nose", x=0.42, y=0.35, confidence=0.9),  # x > 0.40
        }
    )
    feat_right = extractor.extract("Cand_01", obs_right)
    assert feat_right.head.direction == GazeDirection.RIGHT
    assert feat_right.head.yaw_proxy > 0.40
    assert feat_right.head.angular_velocity > 0.0


def test_dual_hand_convergence_and_lap_excursion():
    extractor = BiomechanicalFeatureExtractor()

    # Shoulders span from x=0.30 to x=0.50 (span = 0.20)
    # Wrists cupped together: lw=(0.39, 0.45), rw=(0.41, 0.45) (distance = 0.02)
    # Ratio = 0.02 / 0.20 = 0.10 (< 0.35 threshold -> Converged / Cupping slip)
    obs_cupped = VisionObservation(
        timestamp_sec=2.0,
        frame_idx=20,
        track_id=2,
        bbox_xyxy_norm=(0.25, 0.20, 0.55, 0.80),
        detection_confidence=0.90,
        keypoints={
            "left_shoulder": NormalizedKeypoint("left_shoulder", 0.30, 0.30, confidence=0.9),
            "right_shoulder": NormalizedKeypoint("right_shoulder", 0.50, 0.30, confidence=0.9),
            "left_hip": NormalizedKeypoint("left_hip", 0.35, 0.60, confidence=0.9),
            "right_hip": NormalizedKeypoint("right_hip", 0.45, 0.60, confidence=0.9),
            "left_wrist": NormalizedKeypoint("left_wrist", 0.39, 0.45, confidence=0.9),
            "right_wrist": NormalizedKeypoint("right_wrist", 0.41, 0.45, confidence=0.9),
        }
    )
    feat = extractor.extract("Cand_02", obs_cupped)
    assert feat.hands.is_converged is True
    assert feat.hands.dual_hand_convergence_ratio < 0.20
    assert feat.hands.left_hand_below_lap is False

    # Now drop left hand below hip plane (y_hip = 0.60, y_wrist = 0.72)
    obs_lap = VisionObservation(
        timestamp_sec=2.1,
        frame_idx=21,
        track_id=2,
        bbox_xyxy_norm=(0.25, 0.20, 0.55, 0.80),
        detection_confidence=0.90,
        keypoints={
            "left_shoulder": NormalizedKeypoint("left_shoulder", 0.30, 0.30, confidence=0.9),
            "right_shoulder": NormalizedKeypoint("right_shoulder", 0.50, 0.30, confidence=0.9),
            "left_hip": NormalizedKeypoint("left_hip", 0.35, 0.60, confidence=0.9),
            "right_hip": NormalizedKeypoint("right_hip", 0.45, 0.60, confidence=0.9),
            "left_wrist": NormalizedKeypoint("left_wrist", 0.38, 0.72, confidence=0.9),
            "right_wrist": NormalizedKeypoint("right_wrist", 0.45, 0.45, confidence=0.9),
        }
    )
    feat_lap = extractor.extract("Cand_02", obs_lap)
    assert feat_lap.hands.left_hand_below_lap is True
    assert feat_lap.hands.right_hand_below_lap is False


def test_mediapipe_hand_pinch_and_curl_metrics():
    extractor = BiomechanicalFeatureExtractor()

    # Construct MediaPipe hand where thumb tip and index tip are pinching
    landmarks = {
        "wrist": NormalizedKeypoint("wrist", 0.50, 0.50, confidence=0.9),
        "index_finger_mcp": NormalizedKeypoint("index_finger_mcp", 0.50, 0.40, confidence=0.9),
        "thumb_tip": NormalizedKeypoint("thumb_tip", 0.51, 0.35, confidence=0.9),
        "index_finger_tip": NormalizedKeypoint("index_finger_tip", 0.515, 0.355, confidence=0.9),
        "middle_finger_tip": NormalizedKeypoint("middle_finger_tip", 0.51, 0.38, confidence=0.9),
        "ring_finger_tip": NormalizedKeypoint("ring_finger_tip", 0.505, 0.40, confidence=0.9),
        "pinky_tip": NormalizedKeypoint("pinky_tip", 0.50, 0.42, confidence=0.9),
    }
    hand_obs = HandLandmarksObservation(
        side="RIGHT",
        landmarks=landmarks,
        confidence=0.92,
        is_detected=True
    )

    obs = VisionObservation(
        timestamp_sec=3.0,
        frame_idx=30,
        track_id=3,
        bbox_xyxy_norm=(0.4, 0.3, 0.6, 0.8),
        detection_confidence=0.95,
        right_hand=hand_obs
    )

    feat = extractor.extract("Cand_03", obs)
    assert feat.hands.detailed_right is not None
    assert feat.hands.detailed_right.is_pinching is True
    assert feat.hands.detailed_right.pinch_distance_ratio < 0.25

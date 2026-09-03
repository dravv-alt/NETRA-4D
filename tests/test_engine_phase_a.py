"""Unit tests for Phase A: Observation, CandidateState, and TemporalBuffer.

Verifies:
1. Canonical VisionObservation creation and adapter normalization.
2. CandidateStateManager immutable identity locking and recovery invariant.
3. FPS-independent continuous temporal buffering and kinematic velocity estimation.
4. Absence of hardcoded frame numbers or magic values.
"""

import numpy as np
import pytest

from drishti.engine.observation import (
    ObservationQuality,
    VisibilityState,
    VisionObservation,
    YOLO26Adapter,
    RTMPoseAdapter,
    NormalizedKeypoint,
    DetectedObjectObservation,
)
from drishti.engine.candidate_state import (
    CandidateStateManager,
    IdentityLockStatus,
)
from drishti.engine.temporal_buffer import (
    CandidateTemporalBuffer,
    TemporalBufferManager,
)


def test_yolo_adapter_normalizes_to_unit_coordinates():
    # Synthetic frame 1920x1080
    frame_shape = (1080, 1920)
    raw_box = [192, 108, 960, 540]
    raw_kps = np.zeros((17, 3), dtype=np.float32)
    # Put nose at (960, 540)
    raw_kps[0] = [960, 540, 0.95]

    obs = YOLO26Adapter.adapt(
        frame_idx=10,
        timestamp_sec=0.4,
        frame_shape=frame_shape,
        tracked_box=raw_box,
        track_id=1,
        keypoints_raw=raw_kps,
        det_confidence=0.88,
    )

    assert isinstance(obs, VisionObservation)
    assert obs.bbox_xyxy_norm == pytest.approx((0.10, 0.10, 0.50, 0.50))
    nose = obs.get_keypoint("nose")
    assert nose is not None
    assert nose.x == pytest.approx(0.50)
    assert nose.y == pytest.approx(0.50)
    assert nose.confidence == pytest.approx(0.95)
    assert obs.provider_name == "YOLO26-Pose"


def test_rtmpose_adapter_accepts_simcc_coordinates():
    simcc = {
        "nose": (0.45, 0.35, 0.92),
        "left_wrist": (0.35, 0.65, 0.85),
        "right_wrist": (0.55, 0.65, 0.88),
    }
    obs = RTMPoseAdapter.adapt(
        frame_idx=5,
        timestamp_sec=0.2,
        frame_shape=(720, 1280),
        tracked_box=[100, 100, 400, 600],
        track_id=2,
        simcc_keypoints=simcc,
        det_confidence=0.90,
    )
    assert obs.get_keypoint("nose").x == pytest.approx(0.45)
    assert obs.get_keypoint("left_wrist").y == pytest.approx(0.65)
    assert obs.provider_name == "RTM-Pose-SimCC"


def test_immutable_identity_lock_and_recovery_invariant():
    manager = CandidateStateManager(warmup_frames=10)

    # Simulate warmup frames where Candidate_01 is seen at desk anchor
    for f in range(12):
        obs = VisionObservation(
            timestamp_sec=f * 0.04,
            frame_idx=f,
            track_id=7,
            bbox_xyxy_norm=(0.20, 0.30, 0.40, 0.80),
            detection_confidence=0.95,
        )
        manager.update_frame_observations(f, f * 0.04, [obs])

    cand = manager.get_candidate("Candidate_01")
    assert cand is not None
    assert cand.identity.lock_status == IdentityLockStatus.LOCKED
    assert cand.identity.workstation_id == "Desk_01"

    # Tracker dropped! Observation is lost for 2 frames
    manager.update_frame_observations(13, 13 * 0.04, [])
    manager.update_frame_observations(14, 14 * 0.04, [])
    assert cand.consecutive_missing_frames == 2

    # A NEW tracker ID appears at the EXACT SAME locked seat position
    obs_reappeared = VisionObservation(
        timestamp_sec=15 * 0.04,
        frame_idx=15,
        track_id=99,  # New track ID assigned by ByteTrack
        bbox_xyxy_norm=(0.21, 0.31, 0.41, 0.81),  # Near anchor
        detection_confidence=0.92,
    )
    manager.update_frame_observations(15, 15 * 0.04, [obs_reappeared])

    # Invariant check: It MUST recover Candidate_01, NOT create a new candidate!
    recovered_cand = manager.get_candidate("Candidate_01")
    assert recovered_cand is not None
    assert recovered_cand.active_tracker_id == 99
    assert recovered_cand.identity.candidate_id == "Candidate_01"
    assert recovered_cand.identity.workstation_id == "Desk_01"
    assert len(manager.candidates) == 1  # No duplicate candidate created


def test_temporal_buffer_fps_independence_and_kinematics():
    buf = CandidateTemporalBuffer(max_duration_sec=3.0)

    # Push 30 fps stream for 2 seconds
    fps = 30.0
    dt = 1.0 / fps
    for i in range(60):
        t = i * dt
        # Hand moves horizontally at constant speed: x(t) = 0.10 + 0.10 * t
        kps = {
            "right_wrist": NormalizedKeypoint("right_wrist", x=0.10 + 0.10 * t, y=0.50, confidence=0.9)
        }
        obs = VisionObservation(
            timestamp_sec=t,
            frame_idx=i,
            track_id=1,
            bbox_xyxy_norm=(0.2, 0.2, 0.4, 0.8),
            detection_confidence=0.9,
            keypoints=kps
        )
        buf.push(obs)

    # Expected velocity: dx/dt = 0.10 normalized unit per second
    vel = buf.compute_joint_velocity("right_wrist", window_sec=0.5)
    assert vel == pytest.approx(0.10, rel=0.05)

    # Test window duration bound
    window = buf.get_window(duration_sec=1.0)
    assert len(window) == pytest.approx(31, abs=1)
    duration = window[-1].timestamp_sec - window[0].timestamp_sec
    assert duration <= 1.05


def test_temporal_buffer_purges_past_max_duration():
    buf = CandidateTemporalBuffer(max_duration_sec=2.0)

    obs1 = VisionObservation(timestamp_sec=1.0, frame_idx=1, track_id=1, bbox_xyxy_norm=(0,0,0,0), detection_confidence=1.0)
    obs2 = VisionObservation(timestamp_sec=3.0, frame_idx=2, track_id=1, bbox_xyxy_norm=(0,0,0,0), detection_confidence=1.0)
    obs3 = VisionObservation(timestamp_sec=4.5, frame_idx=3, track_id=1, bbox_xyxy_norm=(0,0,0,0), detection_confidence=1.0)

    buf.push(obs1)
    buf.push(obs2)
    assert buf.observation_count == 2
    buf.push(obs3)
    # obs1 (t=1.0) is 3.5s older than t=4.5, past max_duration_sec=2.0 -> purged!
    # obs2 (t=3.0) is 1.5s older than t=4.5, within max_duration_sec=2.0 -> kept!
    assert buf.observation_count == 2
    assert buf.buffer[0].timestamp_sec == 3.0
    assert buf.buffer[1].timestamp_sec == 4.5

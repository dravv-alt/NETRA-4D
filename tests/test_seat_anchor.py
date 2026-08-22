"""Unit tests for Seat Anchoring and Hysteresis."""

import pytest
from drishti.schemas import CameraProfile, SeatProfile, PersonObservation, Keypoint
from drishti.tracking.seat_anchor import SeatAnchorManager

@pytest.fixture
def sample_camera_profile():
    return CameraProfile(
        camera_id="VIDEO_01",
        seats={
            "A01": SeatProfile(
                seat_id="A01",
                polygon=[[0.0, 0.0], [0.4, 0.0], [0.4, 0.5], [0.0, 0.5]]
            ),
            "A02": SeatProfile(
                seat_id="A02",
                polygon=[[0.5, 0.0], [0.9, 0.0], [0.9, 0.5], [0.5, 0.5]]
            )
        }
    )

def test_seat_anchor_locking(sample_camera_profile):
    mgr = SeatAnchorManager(sample_camera_profile, lock_seconds=1.0, min_consecutive_hits=2)
    
    # Obs 1 in seat A01
    obs1 = PersonObservation(
        provider="yolo26",
        camera_id="VIDEO_01",
        video_id="V01",
        frame_index=0,
        timestamp_sec=0.0,
        entity_local_id=1,
        bbox_xyxy_px=(50, 50, 150, 150),
        bbox_xyxy_norm=(0.1, 0.1, 0.3, 0.4),
        detection_confidence=0.9
    )
    s1 = mgr.update(obs1)
    
    # Obs 2 in seat A01 after 1.5s
    obs2 = PersonObservation(
        provider="yolo26",
        camera_id="VIDEO_01",
        video_id="V01",
        frame_index=15,
        timestamp_sec=1.5,
        entity_local_id=1,
        bbox_xyxy_px=(50, 50, 150, 150),
        bbox_xyxy_norm=(0.1, 0.1, 0.3, 0.4),
        detection_confidence=0.9
    )
    s2 = mgr.update(obs2)
    assert s2 == "A01"

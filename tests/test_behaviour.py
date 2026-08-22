"""Unit tests for 4D Spatio-Temporal Behaviour Engine and Risk State Machine."""

import pytest
from drishti.context.camera_profile import CameraContextManager
from drishti.behaviour.engine import BehaviourEngine
from drishti.risk.state_machine import RiskStateMachine
from drishti.schemas import PersonObservation, Keypoint, RiskStateEnum

def test_behaviour_and_risk_flow():
    cm = CameraContextManager()
    engine = BehaviourEngine(cm, history_seconds=12.0)
    risk_machine = RiskStateMachine(watch_threshold=0.35, review_threshold=0.68)

    # Frame 1
    obs1 = PersonObservation(
        provider="yolo26",
        camera_id="VIDEO_01",
        video_id="V01",
        frame_index=0,
        timestamp_sec=0.0,
        entity_local_id=1,
        bbox_xyxy_px=(100, 100, 200, 200),
        bbox_xyxy_norm=(0.2, 0.2, 0.4, 0.4),
        detection_confidence=0.95,
        keypoints_2d={
            "nose": Keypoint(name="nose", x=0.3, y=0.25, confidence=0.9),
            "left_ear": Keypoint(name="left_ear", x=0.25, y=0.25, confidence=0.9),
            "right_ear": Keypoint(name="right_ear", x=0.35, y=0.25, confidence=0.9)
        }
    )

    ev1 = engine.update([obs1])
    snaps1 = risk_machine.update(0.0, ev1, [obs1])
    assert 1 in snaps1
    assert snaps1[1].state == RiskStateEnum.NORMAL

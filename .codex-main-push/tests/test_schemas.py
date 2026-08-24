"""Unit tests for Drishti-4D Pydantic schemas."""

from drishti.schemas import (
    PersonObservation,
    ObjectObservation,
    Keypoint,
    ObjectClassEnum,
)

def test_person_observation_creation():
    kpt = Keypoint(name="nose", x=0.5, y=0.3, confidence=0.95)
    obs = PersonObservation(
        provider="yolo26",
        camera_id="CAM01",
        video_id="VID01",
        frame_index=10,
        timestamp_sec=0.4,
        entity_local_id=1,
        bbox_xyxy_px=(100.0, 100.0, 200.0, 300.0),
        bbox_xyxy_norm=(0.1, 0.1, 0.2, 0.3),
        detection_confidence=0.92,
        keypoints_2d={"nose": kpt}
    )
    assert obs.provider == "yolo26"
    assert obs.entity_local_id == 1
    assert obs.keypoints_2d["nose"].confidence == 0.95

def test_object_observation_creation():
    obj = ObjectObservation(
        camera_id="CAM01",
        video_id="VID01",
        frame_index=12,
        timestamp_sec=0.5,
        object_class=ObjectClassEnum.MOBILE_PHONE,
        confidence=0.88,
        bbox_xyxy_px=(50, 50, 80, 80),
        bbox_xyxy_norm=(0.05, 0.05, 0.08, 0.08),
        associated_seat_id="A02"
    )
    assert obj.object_class == ObjectClassEnum.MOBILE_PHONE
    assert obj.associated_seat_id == "A02"

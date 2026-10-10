from types import SimpleNamespace

import numpy as np

from drishti.privacy import (
    apply_privacy_mask_and_collect_annotations,
    build_adaptive_face_mask,
    gaze_proxy_from_pose,
)


def person(keypoints, *, tracked=True):
    return SimpleNamespace(
        entity_local_id=7,
        bbox_xyxy_px=(10.0, 10.0, 90.0, 150.0),
        keypoints_2d=keypoints,
        metadata={"entity_id_source": "tracker" if tracked else "frame_local_index"},
    )


def point(x, y, confidence=0.9):
    return SimpleNamespace(x=x, y=y, confidence=confidence)


def test_gaze_proxy_is_based_on_pose_geometry_not_a_guess():
    candidate = person({
        "left_eye": point(0.30, 0.20),
        "right_eye": point(0.50, 0.20),
        "nose": point(0.46, 0.25),
    })
    direction, confidence = gaze_proxy_from_pose(candidate, 100, 100)
    assert direction == "RIGHT"
    assert confidence == 0.9


def test_privacy_mask_covers_detected_eye_region():
    candidate = person({
        "left_eye": point(0.30, 0.20),
        "right_eye": point(0.50, 0.20),
        "nose": point(0.40, 0.25),
    })
    mask = build_adaptive_face_mask(100, 100, [candidate])
    assert mask[20, 30] > 0
    assert mask[20, 50] > 0


def test_missing_facial_keypoints_blurs_head_but_marks_gaze_unknown():
    image = np.full((160, 100, 3), 255, dtype=np.uint8)
    candidate = person({})
    masked, annotations = apply_privacy_mask_and_collect_annotations(image, [candidate])
    assert annotations[0].face_region_source == "person_head_region_fallback"
    assert annotations[0].gaze_proxy == "UNKNOWN"
    assert annotations[0].candidate_id == "Candidate 7"
    assert masked.shape == image.shape


def test_dict_person_privacy_masking():
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    raw_person = {
        "track_id": 4,
        "bbox": [20.0, 20.0, 80.0, 140.0],
        "keypoints": np.array([[50.0, 40.0, 0.9], [45.0, 35.0, 0.9], [55.0, 35.0, 0.9]]),
        "confidence": 0.88,
    }
    masked, annotations = apply_privacy_mask_and_collect_annotations(image, [raw_person])
    assert len(annotations) == 1
    assert annotations[0].candidate_id == "Candidate 4"
    assert annotations[0].face_region_source == "pose_facial_keypoints"
    assert masked.shape == image.shape

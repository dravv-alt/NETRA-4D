from types import SimpleNamespace
import pytest

from drishti.cbt.reasoner import AtomicEventType, AutomaticCBTBehaviouralEngine, CBTBehaviouralReasoner
from drishti.cbt.scene import AutomaticWorkstationCalibrator, CBTSceneMap, CandidateWorkstationBinder
from drishti.schemas import Keypoint, ObjectClassEnum, ObjectObservation, PersonObservation


def polygon(x1, y1, x2, y2):
    return {"points": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]}


def scene():
    return CBTSceneMap.model_validate({
        "camera_id": "TEST_CAM", "video_id": "TEST_VIDEO",
        "reference_width": 1000, "reference_height": 800,
        "workstations": [
            {
                "workstation_id": "W1", "seat": polygon(.05, .65, .45, .95),
                "own_monitor": polygon(.10, .10, .35, .30),
                "desk": polygon(.05, .35, .48, .70),
                "keyboard": polygon(.15, .48, .32, .58),
                "mouse": polygon(.33, .48, .43, .60),
                "under_desk": polygon(.05, .70, .48, .98),
                "right_neighbour_id": "W2",
            },
            {
                "workstation_id": "W2", "seat": polygon(.55, .65, .95, .95),
                "own_monitor": polygon(.65, .10, .90, .30),
                "desk": polygon(.52, .35, .95, .70),
                "under_desk": polygon(.52, .70, .95, .98),
                "left_neighbour_id": "W1",
            },
        ],
        "provenance": {
            "source_video": "test.mp4", "source_frame_index": 1,
            "source_frame_sha256": "a" * 64, "calibrated_by": "test",
            "calibrated_at_utc": "2026-08-23T00:00:00Z",
        },
    })


def point(name, x, y, confidence=.95):
    return Keypoint(name=name, x=x, y=y, confidence=confidence)


def person(candidate_id=1, x1=.10, x2=.40, head=0.0, wrist=(.25, .55)):
    eye_center = (x1 + x2) / 2
    return PersonObservation(
        provider="test", camera_id="TEST_CAM", video_id="TEST_VIDEO",
        frame_index=0, timestamp_sec=0, entity_local_id=candidate_id,
        bbox_xyxy_px=(x1 * 1000, 160, x2 * 1000, 640),
        bbox_xyxy_norm=(x1, .20, x2, .80), detection_confidence=.95,
        keypoints_2d={
            "left_eye": point("left_eye", eye_center - .02, .25),
            "right_eye": point("right_eye", eye_center + .02, .25),
            "nose": point("nose", eye_center + head * .02, .28),
            "left_shoulder": point("left_shoulder", x1 + .08, .38),
            "right_shoulder": point("right_shoulder", x2 - .08, .38),
            "left_hip": point("left_hip", x1 + .10, .62),
            "right_hip": point("right_hip", x2 - .10, .62),
            "left_wrist": point("left_wrist", *wrist),
            "right_wrist": point("right_wrist", wrist[0] + .03, wrist[1]),
        },
        metadata={"entity_id_source": "tracker"},
    )


def privacy(direction="FORWARD", confidence=.95):
    return SimpleNamespace(gaze_proxy=direction, gaze_proxy_confidence=confidence)


def object_observation(raw_class, box, associated=1, association=.9):
    object_class = ObjectClassEnum.MOBILE_PHONE if raw_class == "cell phone" else ObjectClassEnum.PAPER_CHIT
    return ObjectObservation(
        camera_id="TEST_CAM", video_id="TEST_VIDEO", frame_index=0, timestamp_sec=0,
        object_class=object_class, confidence=.85, bbox_xyxy_px=tuple(value * 1000 for value in box),
        bbox_xyxy_norm=box, associated_entity_id=associated,
        association_confidence=association, metadata={
            "raw_model_class": raw_class, "association_method": "nearest_visible_wrist",
            "custom_chit_checkpoint": "chit.pt" if raw_class == "chit" else None,
        },
    )


def monitor_observation(box=(.10, .10, .35, .30)):
    return ObjectObservation(
        camera_id="TEST_CAM", video_id="TEST_VIDEO", frame_index=0, timestamp_sec=0,
        object_class=ObjectClassEnum.MONITOR, confidence=.9,
        bbox_xyxy_px=tuple(value * 1000 for value in box), bbox_xyxy_norm=box,
        is_static_equipment=True, metadata={"raw_model_class": "laptop"},
    )


def test_candidate_binding_uses_calibrated_workstation():
    binding = CandidateWorkstationBinder(scene()).bind([person()])[1]
    assert binding.workstation_id == "W1"
    assert binding.method == "foot_or_bottom_anchor_inside_seat"


def test_candidate_device_id_matches_candidate_and_uses_real_monitor_detection():
    reasoner = CBTBehaviouralReasoner(scene())
    reasoner.update(0.0, [person(candidate_id=12)], [monitor_observation()], [privacy()])
    device = reasoner.last_device_bindings[12]
    assert device.device_id == "PC-12"
    assert device.workstation_id == "W1"
    assert device.detected_class == "laptop"
    assert device.method == "candidate_workstation_plus_current_device_detection"
    assert reasoner.frame_snapshot()["candidate_device_bindings"]["12"]["device_id"] == "PC-12"


def test_one_workstation_device_is_not_assigned_to_two_candidates():
    reasoner = CBTBehaviouralReasoner(scene())
    first = person(candidate_id=2)
    second = person(candidate_id=3, x1=.12, x2=.42)
    reasoner.update(0.0, [first, second], [monitor_observation()], [privacy(), privacy()])
    assert len(reasoner.last_device_bindings) == 1
    assert list(reasoner.last_device_bindings.values())[0].device_id in {"PC-2", "PC-3"}


def test_scene_rejects_wrong_runtime_aspect_ratio():
    calibrated = scene()
    calibrated.validate_frame_geometry(500, 400)
    try:
        calibrated.validate_frame_geometry(500, 500)
    except ValueError as exc:
        assert "aspect ratio" in str(exc)
    else:
        raise AssertionError("A workstation map must not silently move to another camera geometry")


def test_default_automatic_calibration_promotes_from_real_stable_observations():
    calibrator = AutomaticWorkstationCalibrator(
        "TEST_CAM", "TEST_VIDEO", 1000, 800, warmup_sec=.2, minimum_observations=3
    )
    for timestamp in (0.0, .1, .2):
        calibrator.observe(timestamp, [person()], [monitor_observation()])
    automatic_scene = calibrator.finalize()
    assert automatic_scene is not None
    assert automatic_scene.calibration_status == "AUTOMATIC"
    assert automatic_scene.workstations[0].own_monitor.center == pytest.approx((.225, .2))

    engine = AutomaticCBTBehaviouralEngine(
        "TEST_CAM", "TEST_VIDEO", 1000, 800, calibration_warmup_sec=.2
    )
    engine.calibrator.minimum_observations = 3
    for timestamp in (0.0, .1, .2):
        engine.update(timestamp, [person()], [monitor_observation()], [privacy()])
    assert engine.active_reasoner is not None
    assert engine.status == "AUTOMATIC_MAP_ACTIVE"


def test_neighbour_screen_sequence_requires_excursion_and_return():
    reasoner = CBTBehaviouralReasoner(
        scene(), baseline_learning_sec=.7, neighbour_persistence_sec=2.0
    )
    candidate = person()
    for index in range(8):
        assert reasoner.update(index / 10, [candidate], [], [privacy()]) == []
    turned = person(head=.8)
    assert reasoner.update(1.0, [turned], [], [privacy("RIGHT")]) == []
    reasoner.update(3.1, [turned], [], [privacy("RIGHT")])
    events = reasoner.update(3.2, [candidate], [], [privacy()])
    assert any(event.event_type == "NEIGHBOUR_SCREEN_OVERWATCH_REVIEW" for event in events)
    event = next(event for event in events if event.event_type == "NEIGHBOUR_SCREEN_OVERWATCH_REVIEW")
    assert event.details["target_workstation"] == "W2"
    assert "RETURN_TO_OWN_SCREEN" in event.details["sequence"]


def test_mouse_zone_is_counter_evidence_for_phone_like_detection():
    reasoner = CBTBehaviouralReasoner(scene(), baseline_learning_sec=0)
    phone_in_mouse_zone = object_observation("cell phone", (.35, .50, .40, .56))
    reasoner.update(0, [person()], [phone_in_mouse_zone], [privacy()])
    types = {item.event_type for item in reasoner.last_atomic_evidence}
    assert AtomicEventType.STATIC_WORKSTATION_OBJECT in types
    assert AtomicEventType.PHONE_VISIBLE in types
    phone = next(item for item in reasoner.last_atomic_evidence if item.event_type == AtomicEventType.PHONE_VISIBLE)
    assert phone.details["normal_device_zone_prior"] == .65
    assert phone.confidence < phone_in_mouse_zone.confidence


def test_phone_sequence_requires_workstation_hand_ownership_and_time():
    reasoner = CBTBehaviouralReasoner(scene(), baseline_learning_sec=0, phone_use_persistence_sec=.5)
    low_hand = person(wrist=(.25, .78))
    phone = object_observation("cell phone", (.22, .74, .28, .82))
    assert reasoner.update(0, [low_hand], [phone], [privacy()]) == []
    events = reasoner.update(.6, [low_hand], [phone], [privacy()])
    assert any(event.event_type == "PHONE_RETRIEVE_USE_CONCEAL_REVIEW" for event in events)
    event = next(event for event in events if event.event_type == "PHONE_RETRIEVE_USE_CONCEAL_REVIEW")
    assert "wrist ownership" in event.source_signals
    assert event.details["important"] == "Review priority is not cheating probability."


def test_chit_transfer_requires_custom_detection_boundary_and_persistence():
    reasoner = CBTBehaviouralReasoner(scene(), baseline_learning_sec=0)
    crossing = person(wrist=(.60, .78))
    chit = object_observation("chit", (.40, .68, .44, .73))
    assert reasoner.update(0, [crossing], [chit], [privacy()]) == []
    events = reasoner.update(.31, [crossing], [chit], [privacy()])
    assert any(event.event_type == "CHIT_TRANSFER_REVIEW" for event in events)
    event = next(event for event in events if event.event_type == "CHIT_TRANSFER_REVIEW")
    assert event.details["target_workstation"] == "W2"
    assert "custom chit checkpoint" in event.source_signals


def test_cross_candidate_graph_distinguishes_repeated_bilateral_attention():
    reasoner = CBTBehaviouralReasoner(
        scene(), baseline_learning_sec=.7, neighbour_persistence_sec=.2,
        repeated_excursions=1,
    )
    first, second = person(), person(2, .60, .90)
    for index in range(8):
        reasoner.update(index / 10, [first, second], [], [privacy(), privacy()])

    def excursion(start, source_first):
        if source_first:
            turned = [person(head=.8), second]
            annotations = [privacy("RIGHT"), privacy()]
        else:
            turned = [first, person(2, .60, .90, head=-.8)]
            annotations = [privacy(), privacy("LEFT")]
        reasoner.update(start, turned, [], annotations)
        reasoner.update(start + .3, turned, [], annotations)
        return reasoner.update(start + .4, [first, second], [], [privacy(), privacy()])

    excursion(1.0, True)
    excursion(4.5, True)
    excursion(8.0, False)
    events = excursion(11.5, False)
    assert any(event.event_type == "COORDINATED_CANDIDATE_INTERACTION_REVIEW" for event in events)
    graph = reasoner.graph_snapshot()
    assert len(graph["edges"]) == 2
    assert all(edge["attention_count"] == 2 for edge in graph["edges"])

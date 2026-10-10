from types import SimpleNamespace

import numpy as np

from drishti.temporal import BaselineTemporalEngine, draw_temporal_review_events


def point(x: float, y: float, confidence: float = 0.9):
    return SimpleNamespace(x=x, y=y, confidence=confidence)


def person(candidate_id: int, box, *, gaze="FORWARD", low_hand=False):
    x1, y1, x2, y2 = box
    keypoints = {
        "left_hip": point(x1 + 0.35 * (x2 - x1), y1 + 0.65 * (y2 - y1)),
        "right_hip": point(x1 + 0.65 * (x2 - x1), y1 + 0.65 * (y2 - y1)),
        "left_wrist": point(x1 + 0.30 * (x2 - x1), y1 + (0.8 if low_hand else 0.45) * (y2 - y1)),
        "right_wrist": point(x1 + 0.70 * (x2 - x1), y1 + 0.45 * (y2 - y1)),
    }
    return SimpleNamespace(
        entity_local_id=candidate_id,
        bbox_xyxy_norm=box,
        bbox_xyxy_px=tuple(value * 1000 for value in box),
        keypoints_2d=keypoints,
        gaze=gaze,
    )


def privacy(gaze: str):
    return SimpleNamespace(gaze_proxy=gaze, gaze_proxy_confidence=0.9)


def detected(raw_class: str, confidence: float, box, associated=None):
    return SimpleNamespace(
        metadata={"raw_model_class": raw_class, "association_method": "test"},
        confidence=confidence,
        bbox_xyxy_norm=box,
        associated_entity_id=associated,
        association_confidence=0.9,
    )


def detected_chit(confidence: float, box, associated=None):
    value = detected("chit", confidence, box, associated)
    value.object_class = SimpleNamespace(value="PAPER_CHIT")
    value.metadata.update({"custom_chit_checkpoint": "real_chit.pt", "inference_view": "candidate_crop:3"})
    return value


def test_phone_event_requires_association_and_persistence():
    engine = BaselineTemporalEngine(phone_confidence=0.35)
    candidate = person(7, (0.1, 0.1, 0.4, 0.9))
    phone = detected("cell phone", 0.8, (0.2, 0.5, 0.25, 0.6), associated=7)
    assert engine.update(0.0, [candidate], [phone], [privacy("FORWARD")]) == []
    events = engine.update(0.31, [candidate], [phone], [privacy("FORWARD")])
    assert [event.event_type for event in events] == ["PHONE_USE_REVIEW"]
    assert events[0].evidence_kind == "MODEL_PLUS_GEOMETRY"
    assert events[0].is_new is True


def test_custom_phone_class_reaches_review_events_with_original_provenance():
    engine = BaselineTemporalEngine(phone_confidence=.25)
    candidate = person(7,(.1,.1,.4,.9))
    phone = detected('phone',.8,(.2,.5,.25,.6),associated=7)
    phone.metadata.update({'model_weight':'phone_candidate.pt','inference_view':'person_context_crop'})
    assert engine.update(0,[candidate],[phone],[privacy('FORWARD')]) == []
    events = engine.update(.31,[candidate],[phone],[privacy('FORWARD')])
    assert events[0].event_type == 'PHONE_USE_REVIEW'
    assert events[0].details['checkpoint'] == 'phone_candidate.pt'
    assert events[0].details['raw_model_class'] == 'phone'


def test_recent_phone_evidence_is_explicitly_not_a_fresh_detection():
    engine = BaselineTemporalEngine(phone_confidence=.25, phone_visibility_hold_sec=1.0)
    candidate = person(7, (.1, .1, .4, .9))
    phone = detected('phone', .8, (.2, .5, .25, .6), associated=7)
    engine.update(0.0, [candidate], [phone], [privacy("FORWARD")])
    fresh = engine.temporal_evidence(0.0)["phone_evidence"]
    assert fresh[0]["state"] == "FRESH_PHONE_DETECTION"
    engine.update(.5, [candidate], [], [privacy("FORWARD")])
    bridged = engine.temporal_evidence(.5)["phone_evidence"]
    assert bridged[0]["state"] == "TEMPORALLY_SUPPORTED_PHONE_CANDIDATE"
    assert "not a fresh phone detection" in bridged[0]["semantics"]
    engine.update(1.1, [candidate], [], [privacy("FORWARD")])
    assert engine.temporal_evidence(1.1)["phone_evidence"] == []


def test_low_hand_is_explicitly_a_proxy_not_chit_detection():
    engine = BaselineTemporalEngine(hand_low_duration_sec=0.5)
    candidate = person(2, (0.1, 0.1, 0.4, 0.9), low_hand=True)
    engine.update(0.0, [candidate], [], [privacy("FORWARD")])
    engine.update(0.25, [candidate], [], [privacy("FORWARD")])
    events = engine.update(0.5, [candidate], [], [privacy("FORWARD")])
    assert events[0].event_type == "LOW_HAND_POSSIBLE_CHIT_REVIEW"
    assert events[0].evidence_kind == "BEHAVIORAL_PROXY"
    assert "No chit or paper object was detected" in events[0].details["important"]


def test_chit_event_requires_explicit_custom_model_class_and_persistence():
    engine = BaselineTemporalEngine()
    candidate = person(3, (0.1, 0.1, 0.4, 0.9))
    chit = detected_chit(0.81, (0.2, 0.5, 0.3, 0.6), associated=3)
    assert engine.update(0.0, [candidate], [chit], [privacy("FORWARD")]) == []
    events = engine.update(0.31, [candidate], [chit], [privacy("FORWARD")])
    assert events[0].event_type == "CHIT_USE_REVIEW"
    assert events[0].details["checkpoint"] == "real_chit.pt"


def test_mutual_gaze_creates_pairwise_interaction_review():
    engine = BaselineTemporalEngine(interaction_duration_sec=0.5)
    left = person(1, (0.10, 0.10, 0.35, 0.90))
    right = person(2, (0.40, 0.10, 0.65, 0.90))
    annotations = [privacy("RIGHT"), privacy("LEFT")]
    engine.update(0.0, [left, right], [], annotations)
    engine.update(0.25, [left, right], [], annotations)
    events = engine.update(0.5, [left, right], [], annotations)
    assert any(event.event_type == "CANDIDATE_INTERACTION_REVIEW" for event in events)


def test_other_screen_event_excludes_associated_owner():
    engine = BaselineTemporalEngine(screen_duration_sec=0.5)
    viewer = person(1, (0.10, 0.10, 0.30, 0.90))
    owner = person(2, (0.45, 0.10, 0.70, 0.90))
    laptop = detected("laptop", 0.85, (0.48, 0.45, 0.58, 0.55), associated=2)
    annotations = [privacy("RIGHT"), privacy("FORWARD")]
    engine.update(0.0, [viewer, owner], [laptop], annotations)
    engine.update(0.25, [viewer, owner], [laptop], annotations)
    events = engine.update(0.5, [viewer, owner], [laptop], annotations)
    screen_events = [event for event in events if event.event_type == "OTHER_SCREEN_LOOK_REVIEW"]
    assert len(screen_events) == 1
    assert screen_events[0].candidate_ids == (1,)


def test_event_overlay_draws_without_mutating_input():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    candidate = person(1, (0.1, 0.1, 0.3, 0.8), low_hand=True)
    engine = BaselineTemporalEngine(hand_low_duration_sec=0.0)
    events = engine.update(0.0, [candidate], [], [privacy("FORWARD")])
    output = draw_temporal_review_events(frame, [candidate], events)
    assert output.sum() > 0
    assert frame.sum() == 0


def test_sustained_evidence_score_increases_but_stays_bounded():
    engine = BaselineTemporalEngine(hand_low_duration_sec=0.5)
    candidate = person(4, (0.1, 0.1, 0.4, 0.9), low_hand=True)
    annotation = [privacy("FORWARD")]
    engine.update(0.0, [candidate], [], annotation)
    engine.update(0.25, [candidate], [], annotation)
    first = engine.update(0.5, [candidate], [], annotation)[0]
    for timestamp in (0.75, 1.0, 1.25, 1.5, 1.75):
        engine.update(timestamp, [candidate], [], annotation)
    sustained = engine.update(2.0, [candidate], [], annotation)[0]
    assert sustained.evidence_score > first.evidence_score
    assert sustained.evidence_score <= 1.0

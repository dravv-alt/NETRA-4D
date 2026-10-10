from types import SimpleNamespace

import numpy as np

from drishti.render.operator_overlay import draw_operator_overlay


def test_operator_overlay_renders_reviewable_entities_without_mutating_input():
    frame = np.zeros((180, 260, 3), dtype=np.uint8)
    person = SimpleNamespace(entity_local_id=3, detection_confidence=.91, bbox_xyxy_px=(30, 40, 110, 160))
    phone = SimpleNamespace(
        object_class=SimpleNamespace(value="PHONE"), confidence=.72,
        bbox_xyxy_px=(65, 95, 83, 125), metadata={"raw_model_class": "cell phone"},
    )
    event = SimpleNamespace(candidate_ids=[3], label="LOW HAND / POSSIBLE CHIT REVIEW [PROXY]", duration_sec=1.2)

    rendered = draw_operator_overlay(frame, [person], [phone], [event])

    assert rendered.shape == frame.shape
    assert np.array_equal(frame, np.zeros_like(frame))
    assert rendered.sum() > 0


def test_operator_overlay_suppresses_generic_furniture_objects():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    chair = SimpleNamespace(
        object_class=SimpleNamespace(value="CHAIR"), confidence=.99,
        bbox_xyxy_px=(20, 65, 80, 90), metadata={"raw_model_class": "chair"},
    )

    # The header is rendered, but the chair's top-left edge must remain black:
    # generic furniture is intentionally absent from the operator surface.
    rendered = draw_operator_overlay(frame, [], [chair], [])
    assert tuple(rendered[65, 20]) == (0, 0, 0)

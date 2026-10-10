import numpy as np
import pytest

from drishti.api.app import LIVE_RENDER_LAYER_DEFAULTS, _validated_live_render_layers, _write_live_render_layers
from drishti.render.evidence_renderer import EvidenceRenderer


def test_live_render_layer_contract_requires_all_boolean_controls():
    layers = dict(LIVE_RENDER_LAYER_DEFAULTS)
    assert _validated_live_render_layers(layers) == layers

    with pytest.raises(Exception):
        _validated_live_render_layers({"poseSkeleton": True})
    with pytest.raises(Exception):
        _validated_live_render_layers({**layers, "objects": 1})

    writes = []

    class FakeFile:
        name = "render_layers.json"

        def with_name(self, name):
            writes.append(("with_name", name))
            return self

        def write_text(self, contents, encoding):
            writes.append(("write_text", contents, encoding))

        def replace(self, target):
            writes.append(("replace", target))

    class FakeDirectory:
        def __truediv__(self, name):
            assert name == "render_layers.json"
            return FakeFile()

    _write_live_render_layers(FakeDirectory(), layers)
    assert any(action[0] == "write_text" for action in writes)


def test_evidence_renderer_honours_server_layer_preferences(monkeypatch):
    renderer = EvidenceRenderer(enable_face_blur=False)
    calls: list[str] = []
    for name in (
        "render_person_boxes", "render_candidate_skeletons", "render_evidence_objects",
        "render_hand_anomalies", "render_gaze_arrows", "render_candidate_pills",
    ):
        monkeypatch.setattr(renderer, name, lambda *args, _name=name, **kwargs: calls.append(_name))

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    renderer.render_frame(
        frame, [], {}, [], set(), "TEST", 0, 0, 25.0, ["review"],
        layers={
            "personBoxes": False, "trackingIds": False, "poseSkeleton": False,
            "objects": False, "seatMapping": False, "attentionRays": False,
            "behaviorEvents": False, "threeDVector": False,
        },
    )
    assert calls == []

    renderer.render_frame(
        frame, [], {}, [], set(), "TEST", 0, 0, 25.0, ["review"],
        layers={
            "personBoxes": True, "trackingIds": True, "poseSkeleton": True,
            "objects": True, "seatMapping": True, "attentionRays": True,
            "behaviorEvents": True, "threeDVector": False,
        },
    )
    assert set(calls) == {
        "render_person_boxes", "render_candidate_skeletons", "render_evidence_objects",
        "render_hand_anomalies", "render_gaze_arrows", "render_candidate_pills",
    }

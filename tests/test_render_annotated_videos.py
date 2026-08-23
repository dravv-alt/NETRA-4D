from types import SimpleNamespace

from scripts.render_annotated_videos import high_confidence_phone_detections


def detection(raw_class: str, confidence: float):
    return SimpleNamespace(metadata={"raw_model_class": raw_class}, confidence=confidence)


def test_phone_stills_require_exact_class_and_high_confidence():
    objects = [
        detection("cell phone", 0.69),
        detection("cell phone", 0.70),
        detection("book", 0.99),
    ]
    selected = high_confidence_phone_detections(objects, 0.70)
    assert len(selected) == 1
    assert selected[0].confidence == 0.70

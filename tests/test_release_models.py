import pytest

from drishti.release.evaluation import evaluate_events
from drishti.release.models import ModelRegistry


def test_registry_blocks_unapproved_role():
    fixture = __import__("pathlib").Path("tests/fixtures/model_registry").resolve()
    registry = ModelRegistry(fixture / "registry.yaml", fixture)
    assert registry.get("pose").path.name == "approved.pt"
    with pytest.raises(PermissionError, match="held-out evaluation"):
        registry.get("experimental")


def test_event_evaluation_reports_real_counts():
    prediction = {"event_type": "PHONE_USE_REVIEW", "candidate_ids": [3], "started_at_sec": 10, "ended_at_sec": 14, "evidence_score": .8}
    label = {"event_type": "PHONE_USE_REVIEW", "candidate_ids": [3], "started_at_sec": 11, "ended_at_sec": 13, "review_status": "CONFIRMED"}
    report = evaluate_events([prediction], [label])
    metrics = report["per_event_type"]["PHONE_USE_REVIEW"]
    assert report["status"] == "EVALUATED"
    assert metrics == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "reviewed_positive_windows": 1}


def test_event_evaluation_refuses_empty_labels():
    assert evaluate_events([], [])["status"] == "INSUFFICIENT_LABELS"

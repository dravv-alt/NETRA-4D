"""Event-level held-out evaluation. It refuses to invent quality metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
    start = max(float(first["started_at_sec"]), float(second["started_at_sec"]))
    end = min(float(first.get("ended_at_sec", first["started_at_sec"])), float(second.get("ended_at_sec", second["started_at_sec"])))
    union_start = min(float(first["started_at_sec"]), float(second["started_at_sec"]))
    union_end = max(float(first.get("ended_at_sec", first["started_at_sec"])), float(second.get("ended_at_sec", second["started_at_sec"])))
    return max(0.0, end - start) / max(1e-9, union_end - union_start)


def evaluate_events(predictions: Iterable[dict[str, Any]], labels: Iterable[dict[str, Any]], *, minimum_overlap: float = 0.25) -> dict[str, Any]:
    """Match predictions to CONFIRMED human labels by type, candidate, and time."""
    # The product runner upserts the same stable event as it persists. Keep its
    # latest observation so an active event is evaluated once, not once/frame.
    latest_predictions: dict[str, dict[str, Any]] = {}
    anonymous_predictions: list[dict[str, Any]] = []
    for row in predictions:
        event_id = row.get("event_id")
        if event_id:
            latest_predictions[str(event_id)] = row
        else:
            anonymous_predictions.append(row)
    predicted = list(latest_predictions.values()) + anonymous_predictions
    labelled = list(labels)
    positive = [row for row in labelled if row.get("review_status") == "CONFIRMED"]
    negatives = [row for row in labelled if row.get("review_status") == "FALSE_POSITIVE"]
    if not labelled:
        return {"status": "INSUFFICIENT_LABELS", "reason": "No human-reviewed labels supplied", "per_event_type": {}}
    by_type: dict[str, dict[str, Any]] = {}
    types = sorted({str(row.get("event_type")) for row in predicted + positive + negatives})
    for event_type in types:
        candidates = [row for row in predicted if row.get("event_type") == event_type]
        truth = [row for row in positive if row.get("event_type") == event_type]
        matched_truth: set[int] = set()
        true_positive = 0
        for candidate in sorted(candidates, key=lambda row: float(row.get("evidence_score", 0)), reverse=True):
            ids = set(candidate.get("candidate_ids", []))
            best = next((index for index, target in enumerate(truth) if index not in matched_truth and ids == set(target.get("candidate_ids", [])) and _overlap(candidate, target) >= minimum_overlap), None)
            if best is not None:
                matched_truth.add(best)
                true_positive += 1
        false_positive = len(candidates) - true_positive
        false_negative = len(truth) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        by_type[event_type] = {"tp": true_positive, "fp": false_positive, "fn": false_negative, "precision": precision, "recall": recall, "f1": f1, "reviewed_positive_windows": len(truth)}
    return {"status": "EVALUATED", "labelled_windows": len(labelled), "minimum_temporal_overlap": minimum_overlap, "per_event_type": by_type}

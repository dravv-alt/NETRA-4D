"""Measure the human-reviewed event queue and suggest bounded threshold changes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.event_repository import EventRepository
from drishti.events import ReviewStatus


def metrics(truth: list[bool], accepted: list[bool]) -> dict[str, float | int]:
    tp = sum(expected and predicted for expected, predicted in zip(truth, accepted))
    fp = sum(not expected and predicted for expected, predicted in zip(truth, accepted))
    fn = sum(expected and not predicted for expected, predicted in zip(truth, accepted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def best_one_sided(records, truth, field: str, keep: str):
    values = sorted({float(value) for record in records if (value := (
        record.duration_sec if field == "duration_sec" else
        record.evidence_score if field == "evidence_score" else record.details.get(field)
    )) is not None})
    best = None
    for threshold in values:
        observed = [
            record.duration_sec if field == "duration_sec" else
            record.evidence_score if field == "evidence_score" else record.details.get(field)
            for record in records
        ]
        accepted = [value is not None and (float(value) >= threshold if keep == "above" else float(value) <= threshold) for value in observed]
        result = {"field": field, "keep": keep, "threshold": threshold, **metrics(truth, accepted)}
        rank = (result["f1"], result["precision"], result["recall"])
        if best is None or rank > best[0]:
            best = (rank, result)
    return best[1] if best else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("outputs/netra4d_events.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("outputs/temporal_calibration.json"))
    args = parser.parse_args()
    repository = EventRepository((ROOT / args.database).resolve())
    reviewed = [event for event in repository.list(limit=500) if event.review_status in {
        ReviewStatus.CONFIRMED, ReviewStatus.FALSE_POSITIVE
    }]
    grouped = defaultdict(list)
    for event in reviewed:
        grouped[event.event_type].append(event)
    report = {
        "reviewed_events": len(reviewed),
        "calibration_scope": (
            "Precision/retention tuning within emitted proposals only. Full recall and lower-than-current "
            "thresholds require frame-level negative annotations and a new replay."
        ),
        "ready": bool(reviewed),
        "event_types": {},
    }
    for event_type, records in sorted(grouped.items()):
        truth = [record.review_status == ReviewStatus.CONFIRMED for record in records]
        report["event_types"][event_type] = {
            "reviewed": len(records),
            "confirmed": sum(truth),
            "false_positive": len(records) - sum(truth),
            "current_queue_metrics": metrics(truth, [True] * len(records)),
            "suggestions": [item for item in (
                best_one_sided(records, truth, "duration_sec", "above"),
                best_one_sided(records, truth, "evidence_score", "above"),
                best_one_sided(records, truth, "normalized_center_distance", "below"),
                best_one_sided(records, truth, "gaze_proxy_confidence", "above"),
            ) if item is not None],
        }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if reviewed else 2


if __name__ == "__main__":
    raise SystemExit(main())

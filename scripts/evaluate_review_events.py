"""Evaluate held-out, human-reviewed NETRA event labels against predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.release.evaluation import evaluate_events


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="review_events.jsonl from a canonical run")
    parser.add_argument("--labels", required=True, help="human-labelled JSONL; never use training labels here")
    parser.add_argument("--minimum-overlap", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.minimum_overlap <= 1:
        raise ValueError("--minimum-overlap must be in (0, 1]")
    report = evaluate_events(read_jsonl(Path(args.predictions)), read_jsonl(Path(args.labels)), minimum_overlap=args.minimum_overlap)
    report.update({"predictions": str(Path(args.predictions)), "labels": str(Path(args.labels)), "split_requirement": "labels must come from camera/session-held-out review data"})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "EVALUATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

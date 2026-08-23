"""Compare two completed NETRA-4D full-video runs without inventing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_run(path: Path) -> dict[str, Any]:
    run_path = path / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {run_path}")
    return json.loads(run_path.read_text(encoding="utf-8"))


def counts_by_asset(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["asset_id"]: item["counts"] for item in run.get("assets", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = load_run(args.baseline.resolve())
    candidate = load_run(args.candidate.resolve())
    before, after = counts_by_asset(baseline), counts_by_asset(candidate)
    report = {
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "same_models": baseline.get("models") == candidate.get("models"),
        "same_privacy_pipeline": baseline.get("privacy_pipeline") == candidate.get("privacy_pipeline"),
        "per_video": {},
    }
    for asset_id in sorted(set(before) | set(after)):
        old, new = before.get(asset_id, {}), after.get(asset_id, {})
        report["per_video"][asset_id] = {
            key: {"baseline": old.get(key), "candidate": new.get(key)}
            for key in sorted(set(old) | set(new))
        }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.resolve().write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Import generated review_events.jsonl into the backend SQLite queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from drishti.event_repository import EventRepository
from drishti.events import EvidenceKind, EventLifecycle, EventRecord, SpecialistStatus


def import_run(run_dir: Path, repository: EventRepository) -> int:
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    repository.delete_pending_run(run_record["run_id"])
    count = 0
    for video_dir in sorted(run_dir.glob("VIDEO_*")):
        events_path = video_dir / "review_events.jsonl"
        if not events_path.is_file() or not (video_dir / "COMPLETE.json").is_file():
            continue
        keyframes = {path.name: path for path in (video_dir / "review_event_frames").glob("*.jpg")} if (video_dir / "review_event_frames").exists() else {}
        for event_index, line in enumerate(events_path.read_text(encoding="utf-8").splitlines()):
            item = json.loads(line)
            candidates = [int(value) for value in item["candidate_ids"]]
            identity = f"{run_record['run_id']}|{video_dir.name}|{event_index}|{item['frame_index']}|{item['event_type']}|{candidates}"
            event_id = str(uuid5(NAMESPACE_URL, identity))
            keyframe = next((path for name, path in keyframes.items() if name.startswith(f"frame_{int(item['frame_index']):06d}_") and item["event_type"].lower() in name), None)
            started = float(item["started_at_sec"])
            ended = float(item["timestamp_sec"])
            repository.upsert(EventRecord(
                event_id=event_id,
                run_id=run_record["run_id"],
                camera_id=video_dir.name,
                video_id=video_dir.name,
                event_type=item["event_type"],
                label=item["label"],
                lifecycle=EventLifecycle.CLOSED,
                evidence_kind=EvidenceKind(item["evidence_kind"]),
                evidence_score=float(item["evidence_score"]),
                candidate_ids=candidates,
                workstation_ids=[value for value in (
                    item.get("details", {}).get("source_workstation"),
                    item.get("details", {}).get("target_workstation"),
                ) if value],
                started_at_sec=started,
                ended_at_sec=ended,
                duration_sec=max(0.0, ended - started),
                source_signals=item.get("source_signals", []),
                keyframe_path=str(keyframe.relative_to(ROOT)) if keyframe else None,
                model_provenance={"source_run": str(run_dir), "models_rerun": run_record.get("models_rerun")},
                reasoning_provenance={"source_run_kind": run_record.get("run_kind")},
                review_priority=item.get("details", {}).get("review_priority", item.get("evidence_score")),
                uncertainty=item.get("details", {}).get("uncertainty"),
                specialist_status=SpecialistStatus.REQUESTED if item.get("details", {}).get("specialist_request") else SpecialistStatus.NOT_REQUIRED,
                specialist_request=item.get("details", {}).get("specialist_request"),
                details={**item.get("details", {}), "source_frame_index": item["frame_index"]},
                created_at_utc=datetime.now(timezone.utc),
            ))
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--database", type=Path, default=Path("outputs/netra4d_events.sqlite"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    database = (ROOT / args.database).resolve()
    count = import_run(run_dir, EventRepository(database))
    print(json.dumps({"database": str(database), "imported": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

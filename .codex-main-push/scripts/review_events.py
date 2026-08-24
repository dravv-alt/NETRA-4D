"""Human-review the generated NETRA-4D event queue; never auto-label evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.event_repository import EventRepository
from drishti.events import ReviewDecision, ReviewStatus


KEYS = {
    ord("c"): ReviewStatus.CONFIRMED,
    ord("f"): ReviewStatus.FALSE_POSITIVE,
    ord("n"): ReviewStatus.NEEDS_CONTEXT,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("outputs/netra4d_events.sqlite"))
    parser.add_argument("--reviewer", required=True, help="Human reviewer name or ID")
    parser.add_argument("--event-type")
    parser.add_argument("--video-id")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    repository = EventRepository((ROOT / args.database).resolve())
    events = repository.list(
        review_status=ReviewStatus.PENDING.value,
        event_type=args.event_type,
        video_id=args.video_id,
        limit=args.limit,
    )
    if args.list_only:
        print(json.dumps([
            {
                "event_id": event.event_id,
                "video_id": event.video_id,
                "event_type": event.event_type,
                "candidate_ids": event.candidate_ids,
                "keyframe_path": event.keyframe_path,
            }
            for event in events
        ], indent=2))
        return 0

    reviewed = 0
    for index, event in enumerate(events, 1):
        if not event.keyframe_path:
            continue
        image_path = (ROOT / event.keyframe_path).resolve()
        try:
            image_path.relative_to((ROOT / "outputs").resolve())
        except ValueError:
            raise RuntimeError(f"Refusing evidence outside outputs: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        title = f"{index}/{len(events)} {event.video_id} {event.event_type} candidates={event.candidate_ids}"
        cv2.putText(image, "C=confirm  F=false  N=needs context  S=skip  Q=quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(title, image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)
        if key == ord("q"):
            break
        status = KEYS.get(key)
        if status is None:
            continue
        repository.review(event.event_id, ReviewDecision(
            status=status,
            reviewed_by=args.reviewer,
            comment="Human keyframe review; inspect temporal clip when NEEDS_CONTEXT.",
        ))
        reviewed += 1
    cv2.destroyAllWindows()
    print(json.dumps({"reviewed_this_session": reviewed, "database_counts": repository.counts()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

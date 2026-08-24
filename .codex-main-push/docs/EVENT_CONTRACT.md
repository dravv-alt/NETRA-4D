# NETRA-4D event contract

Version: `1.0.0`

`src/drishti/events.py` is the backend-facing source of truth. Events are
Pydantic-validated before SQLite persistence or WebSocket delivery.

## Required identity and evidence fields

| Field | Meaning |
|---|---|
| `event_id` | Stable UUID/string identity across lifecycle updates |
| `run_id`, `camera_id`, `video_id` | Source and execution identity |
| `lifecycle` | `OPEN`, `ACTIVE`, or `CLOSED` |
| `candidate_ids` | One or more ByteTrack candidate IDs |
| `workstation_ids` | Calibrated source/target CBT workstation IDs |
| `event_type`, `label` | Machine event key and review-facing label |
| `evidence_kind` | Model evidence, geometry, or explicitly labelled proxy |
| `evidence_score` | Bounded 0-1 proposal/evidence score; not cheating probability |
| `started_at_sec`, `ended_at_sec` | Source-relative media timestamps |
| `started_at_utc`, `ended_at_utc` | Optional wall-clock timestamps for live sources |
| `keyframe_path`, `clip_path` | Evidence paths constrained to `outputs/` by the API |
| `model_provenance` | Weight hashes, model names, and runtime device |
| `reasoning_provenance` | Behavioural engine and workstation-map identity |
| `review_priority` | Evidence ranking score; explicitly not cheating probability |
| `uncertainty` | Remaining perception/geometry uncertainty |
| `specialist_status`, `specialist_request` | Optional 3-D geometry-resolution lifecycle |
| `review_status` | `PENDING`, `CONFIRMED`, `FALSE_POSITIVE`, or `NEEDS_CONTEXT` |
| `reviewed_by`, `reviewed_at_utc`, `review_comment` | Human decision audit data |

## Lifecycle

```text
proposal persists -> OPEN -> evidence clip finishes -> CLOSED
                       |                              |
                       +------ WebSocket updates -----+
```

A human review is independent of event lifecycle. Re-importing or closing an
event cannot overwrite an existing human decision.

## API

```text
POST  /api/events
GET   /api/events
GET   /api/events/{event_id}
PATCH /api/events/{event_id}/review
POST  /api/events/{event_id}/specialist-resolution
GET   /api/events/{event_id}/keyframe
GET   /api/events/{event_id}/clip
GET   /api/candidates/{candidate_id}/events
GET   /api/workstations/{workstation_id}/events
WS    /ws/events
```

SQLite uses WAL mode and stores the validated full JSON payload plus indexed
event fields. `review_audit` records every approve/reject/context decision.

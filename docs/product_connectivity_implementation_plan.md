# NETRA-4D Product Connectivity Implementation Plan

## 1. Objective

Convert the current hybrid frontend/backend application into a connected,
durable evidence-review product while preserving the existing inference,
rendering, event-ingestion, and verified-run workflows.

## 1.1 Single-video scope — non-negotiable

This implementation plan is scoped to exactly one video and its derived
artifacts:

| Field | Canonical value |
|---|---|
| Asset | `VIDEO_03` |
| Source | `input/videos/VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4` |
| Frame range | `0–2199` |
| Frame count | `2200` |
| Resolution | `1280×720` |
| FPS | `25` |
| Duration | `88 seconds` |
| Trusted boxes | `outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl` |
| Frozen reference run | `outputs/unified_pipeline/video03_review_verified_20261003` |
| Raw CVAT XML SHA-256 | `301620001d90d0f2fb06deb3535e98dc67c6c641f7a10e84b5b01c2db1452382` |

All events, evidence media, candidates, workstations, review decisions,
analytics, cases, benchmark records, and application acceptance tests in this
plan must resolve to `VIDEO_03` and frames `0–2199`.

The original approximately 3,200-frame source is provenance only. Frames 2200
and beyond must not be processed, imported, shown as review evidence, or used in
metrics. Other repository videos (`VIDEO_01`, `VIDEO_02`, `VIDEO_04`, and
`VIDEO_05`) are explicitly outside this implementation plan.

This is a Video 3 development and product-integration exercise. It is not a
held-out or generalization evaluation. Existing same-video limitations remain
visible anywhere results are displayed.

### Approved Video 3 lineage for the application

The application may index and present the following Video 3 artifact families,
provided their manifests, hashes, limitations, and original paths are retained:

- immutable manual annotation source:
  `MANUAL_ANNOTATIONS(CVAT)/VIDEO_3_EXAM_HALL_PHONE_USE/annotations.xml`;
- bounded annotation derivative: `datasets/video03_0_2199_derivative/`;
- human-review package:
  `outputs/unified_pipeline/video03_review_verified_20261003/human_review_v1`;
- refined single-phone annotation candidate:
  `datasets/video03_refined_single_phone_20261003/`;
- frozen baseline:
  `outputs/unified_pipeline/video03_review_verified_20261003`;
- fresh/refined GPU diagnostic runs under `outputs/unified_pipeline/video03_*`;
- Video 3 preprocessing ablations under `outputs/unified_pipeline/video03_review_preprocessed_*`;
- Video 3 head-direction diagnostics under `outputs/analysis/video03_head_direction_*`;
- Video 3 specialist, temporal, interaction, and fusion experiments under
  `outputs/experimental/video03_*`;
- the 2,200-frame full evidence replay:
  `outputs/experimental/video03_refined_full_evidence_20261003`;
- Video 3 reports referenced by `docs/EXPERIMENT_LEDGER.md`.

Artifact discovery must not accept a directory merely because its name begins
with `video03`. It must validate the manifest/source identity, frame range,
source hash when available, and relationship to the canonical Video 3 lineage.
Training outputs may be displayed as development provenance but must not be
presented as held-out evaluation.

This plan addresses:

- durable human review decisions;
- event-correct evidence playback;
- explicit loading, empty, error, and live-data states;
- URL-addressable navigation and selection;
- live-session recovery;
- persistent Video 3 camera/session context, candidates, workstations, layouts,
  and cases;
- backend-derived analytics, benchmark data, and health telemetry;
- removal or explicit isolation of synthetic/demo behavior;
- secure server-side AI-query handling;
- reproducible API and database contracts.

## 2. Explicit non-goals

- No repository-wide directory reorganization.
- No ingestion, migration, inference, evaluation, or UI presentation of any
  video other than canonical `VIDEO_03`.
- No model retraining or inference redesign.
- No modification of trusted annotations or historical run outputs.
- No migration of video or image binaries into SQLite.
- No behavioral verdict inferred from a model score.
- No silent replacement of unavailable backend data with demo data.
- No deletion of existing showcase components until replacements are verified.

The repository layout remains frozen. New modules are added inside the current
`src/drishti`, `Frontend_Updated/src`, `tests`, and `docs` boundaries.

## 3. Confirmed starting point

### Reusable backend capabilities

- `EventRepository` persists `review_events`, `review_audit`, and
  `specialist_audit` in SQLite.
- `GET /api/events` returns persisted event records.
- `PATCH /api/events/{event_id}/review` persists a review decision.
- Event keyframes and clips are served by event-specific endpoints.
- Candidate and workstation timelines already exist.
- Verified run manifests can be listed.
- Video 3 live/replay session records can be created, started, stopped, listed,
  and streamed by the existing application pathways.
- Live layer controls and telemetry WebSockets already exist.
- Live-session metadata and controls are currently filesystem-backed.

### Confirmed frontend gaps

- `App.tsx` owns navigation, event mapping, review state, and demo fallback.
- Review decisions update React state but do not call the review API.
- Evidence Review uses preloaded-demo media instead of selected-event media.
- Real event review fields are discarded by `toFrontendEvent()`.
- There is no URL router dependency.
- Demo events, seats, cameras, and health are initial operational state.
- Cases, analytics, camera configuration, and layout changes are not durable.
- Benchmark Lab reads a compile-time snapshot.
- The browser contains a direct Groq fallback.

### Recovery baseline

The pre-change checkpoint is Git commit `1abf837`. It must remain runnable
throughout implementation.

## 4. Product data rules

1. Backend records are authoritative for persisted state.
2. WebSocket messages update the client cache but do not replace persistence.
3. Browser state may hold drafts and UI preferences only.
4. Demo data must never appear on an operational route after a backend error.
5. Missing data is represented as unavailable or unknown, never fabricated.
6. Media remains on disk or object storage; SQLite stores references, hashes,
   provenance, timestamps, and relationships.
7. Review status and operational priority remain separate concepts.
8. `escalated` is a case/workflow action, not an event review status.
9. Existing proxy/evidence language remains unchanged; no cheating probability
   is introduced.

## 5. Canonical review mapping

| Frontend action | Backend `ReviewStatus` | Additional action |
|---|---|---|
| Mark normal | `FALSE_POSITIVE` | None |
| Needs follow-up | `NEEDS_CONTEXT` | Optionally create follow-up task later |
| Confirm evidence | `CONFIRMED` | Does not itself assert misconduct |
| Escalate to case | Preserve current review status | Create or update a case |

The frontend should use backend enum names internally. Human-readable button
labels can remain product language. `PENDING` cannot be submitted as a final
decision because the backend contract already rejects it.

## 6. Target frontend architecture within the existing tree

Add the following modules without moving current components:

```text
Frontend_Updated/src/
  api/
    client.ts
    contracts.ts
    events.ts
    liveSessions.ts
    layouts.ts
    cases.ts
    analytics.ts
    runs.ts
    health.ts
  state/
    AppDataProvider.tsx
    eventStore.ts
    liveSessionStore.ts
    operatorStore.ts
  routing/
    AppRouter.tsx
    routeState.ts
  components/
    feedback/
      BackendUnavailable.tsx
      EmptyState.tsx
      LoadingState.tsx
      RetryPanel.tsx
  demo/
    demoData.ts
    DemoApp.tsx
```

The precise filenames may be adjusted to existing conventions, but the
responsibilities must remain separated:

- transport and error handling;
- API-to-view-model mapping;
- shared server state;
- route state;
- operational UI;
- explicitly labelled demo UI.

## 7. Target backend architecture within the existing tree

Keep `src/drishti/api/app.py` as the application entry point, but move new
business logic into focused modules rather than expanding the route file:

```text
src/drishti/
  api/
    app.py
    contracts.py
    dependencies.py
  repositories/
    product_repository.py
    migration_runner.py
  services/
    case_service.py
    analytics_service.py
    evidence_export_service.py
    live_session_service.py
    query_context_service.py
```

This is code modularization inside the current package, not a workspace-tree
reorganization. Existing import paths and launch commands must remain valid.

## 8. Database migration strategy

Do not replace or destructively rewrite the current SQLite database.

Introduce:

```text
schema_migrations(version, applied_at_utc, checksum)
```

Each migration must be:

- ordered and checksummed;
- transaction-bound;
- safe to rerun;
- tested against an empty database and a copy of the current schema;
- backed up before application in an operational environment.

### Proposed entities

#### `exam_sessions`

- `session_id` primary key
- name, status, scheduled/started/ended timestamps
- source and configuration hashes
- created/updated timestamps

For this plan, the table contains one canonical Video 3 session. Supporting a
second exam/video is not an acceptance criterion.

#### `camera_sources`

- `camera_id` primary key
- session relationship
- display name and source kind
- encrypted/indirect source reference; do not return RTSP credentials
- enabled state and last health timestamp
- configuration provenance

Only the camera/source context represented by Video 3 is populated in this
plan. A generic schema is retained to avoid a destructive future migration, but
no additional video or camera dataset is imported.

#### `live_sessions`

- `live_session_id` primary key
- exam session and camera relationships
- source kind, process state, operator, overlay profile
- filesystem artifact directory
- started/stopped timestamps and failure reason

Every indexed live/replay session used for acceptance must point to the
canonical Video 3 source or its server-rendered derived artifacts.

#### `candidates`

- stable session-scoped `candidate_id`
- external reference or display alias
- no biometric identity claim

#### `workstations`

- workstation identifier and active layout relationship
- normalized geometry and metadata

#### `seat_assignments`

- session, candidate, workstation
- validity interval and provenance

#### `spatial_layouts`

- layout id, version, status (`DRAFT`, `ACTIVE`, `ARCHIVED`)
- geometry JSON, source artifact reference and hash
- reviewer, created/activated timestamps

#### `cases` and `case_events`

- case lifecycle and assigned reviewer
- event membership with immutable linkage timestamps
- summary and disposition separated from event evidence

#### `case_exports`

- package path, SHA-256, generation timestamp
- source case version and manifest
- export status and failure reason

#### `dispatch_actions`

- session/event/case relationship
- action type, operator, timestamp, status, notes

#### `runtime_samples`

- session/camera/provider timestamped measurements
- FPS, latency, GPU/CPU memory, utilization, temperature when available
- nullable fields when the platform cannot measure a value

#### `review_attachments`

- event/case relationship
- media path, media type, hash, provenance, creator and timestamp

Binary media must not be stored as SQLite blobs.

## 9. API contract principles

- Use Pydantic request and response models for every product endpoint.
- Reject or exclude records whose `video_id` is not `VIDEO_03` in the
  Video-3-scoped product endpoints and snapshots.
- Never return arbitrary untyped dictionaries for durable entities.
- Return relative API URLs, not hardcoded `127.0.0.1` URLs.
- Use consistent error payloads with `code`, `message`, and optional details.
- Paginated collections return `items`, `total`, `limit`, and `offset`.
- Mutations return the canonical persisted record.
- Add optimistic-concurrency fields (`updated_at_utc` or version) where two
  operators can edit the same record.
- OpenAPI generation must pass before generated clients are considered valid.

## 10. Implementation phases

### Phase 0 — Freeze behavior and add safety tests

Purpose: protect the current working paths before connection changes.

Tasks:

1. Record the current endpoint inventory and response schemas.
2. Add a canonical Video 3 scope fixture that asserts the source, frame bound,
   annotation hash, and frozen reference-run path.
3. Add an OpenAPI-generation regression test.
4. Add API media-path containment tests.
5. Add a frontend production-build check to the validation script/CI.
6. Add a check that operational components do not import demo data.
7. Preserve existing live-monitor and layer-control tests.

Exit criteria:

- Current tests pass.
- OpenAPI failure is reproduced by a focused test.
- Demo imports are inventoried, not yet removed.
- Any Video 3 record outside frames 0–2199 is rejected by focused tests.

### Phase 1 — Repair API contracts and create one frontend API client

Purpose: create a single reliable transport and mapping boundary.

Backend tasks:

1. Resolve the Pydantic forward reference causing `/openapi.json` to fail.
2. Define typed responses for health, event lists, live sessions, layouts,
   runs, and errors.
3. Make Video 3 the explicit product data filter for event, run, evidence, and
   snapshot reads.
4. Preserve existing endpoint paths unless a versioned replacement is needed.
5. Add CORS configuration through environment/config rather than scattered
   literals.

Frontend tasks:

1. Add `api/client.ts` with:
   - normalized API base URL;
   - JSON and media URL helpers;
   - request timeout and abort support;
   - typed `ApiError`;
   - no silent catch-and-ignore behavior.
2. Add backend contract types and a single `mapEventRecord()` adapter.
3. Map all real event fields:
   - review status/comment/reviewer/timestamp;
   - keyframe and clip availability;
   - backend review priority and uncertainty;
   - lifecycle, evidence kind, provenance, and source signals.
4. Do not fabricate neighbour, board, invigilator, candidate-name, or severity
   values when absent.

Exit criteria:

- `/openapi.json` returns 200.
- All event transport goes through the central client.
- Contract and mapping tests cover complete and sparse records.

### Phase 2 — Complete the human-review vertical slice

Purpose: make one core workflow fully durable before adding new entities.

Tasks:

1. Replace `handleDecision()`'s local-only update with
   `PATCH /api/events/{event_id}/review`.
2. Require an operator identity from an explicit local operator/session field;
   remove the hardcoded reviewer name.
3. Add mutation states:
   - idle;
   - saving;
   - saved;
   - conflict;
   - failed with Retry.
4. Apply optimistic UI only if rollback is implemented.
5. Replace the cached event with the canonical response after saving.
6. Ensure WebSocket updates merge by `event_id` and `updated_at_utc` without
   erasing local in-flight state.
7. Display persisted audit fields after reload.
8. Keep escalation separate and disabled until cases exist.

Exit criteria:

- A review survives refresh and backend restart.
- Another browser receives the updated record.
- All four backend statuses render correctly.
- Failure never presents a decision as saved.

### Phase 3 — Make Evidence Review event-correct

Purpose: ensure the operator judges the selected event's evidence.

Tasks:

1. Remove `/api/preloaded-demo` usage from operational Evidence Review.
2. Load the selected event by ID when entering the route.
3. Use:
   - `/api/events/{event_id}/keyframe`;
   - `/api/events/{event_id}/clip`.
4. Show media independently:
   - keyframe available;
   - clip available;
   - media not captured;
   - media missing/corrupt;
   - backend unavailable.
5. Display source signals and provenance from that same event.
6. Do not synthesize T-1/event/T+1 frames unless the backend exposes
   event-aligned frame endpoints with verified timestamps.
7. Add clear proxy/uncertainty labels.

Exit criteria:

- Network requests reference only the selected event ID.
- Event text, keyframe, clip, and review record agree.
- Missing media produces an explicit state, not unrelated fallback media.

### Phase 4 — Replace in-memory navigation with URL routing

Purpose: preserve page and selection context across refresh and browser history.

Tasks:

1. Add `react-router-dom`.
2. Define routes:

```text
/
/live/:sessionId?
/events
/events/:eventId/review
/candidates/:candidateId?
/cases/:caseId?
/analytics
/system/cameras
/benchmark/:runId?
/demo/*
```

3. Move event, candidate, case, session, and run selection into route params.
4. Encode filters, page, sort, and time range in query parameters.
5. Keep only ephemeral UI state locally.
6. Add not-found and invalid-selection states.

Exit criteria:

- Refresh preserves page and selected entity.
- Back/Forward navigation works.
- A review URL can be bookmarked and opened directly.
- Invalid IDs return a clear 404/empty state.

### Phase 5 — Eliminate silent demo fallback

Purpose: make operational truth explicit.

Tasks:

1. Replace demo initial state with explicit application states:
   `LOADING`, `BACKEND_UNAVAILABLE`, `NO_DATA`, `LIVE_DATA`.
2. Add visible Retry actions for failed loads.
3. Do not keep stale demo records after an API failure.
4. Move showcase-only presets and synthetic components behind `/demo`.
5. Add a persistent `DEMO` banner and prevent demo mutations from reaching the
   operational database.
6. Fail production builds/tests if operational files import `demoData.ts`.

Exit criteria:

- Stopping the backend shows `BACKEND_UNAVAILABLE`.
- An empty database shows `NO_DATA`.
- Demo records appear only under `/demo`.

### Phase 6 — Rehydrate live sessions

Purpose: reconnect the UI to existing server-side live work after reload.

Tasks:

1. Add typed live-session contracts.
2. On `/live`, call `GET /api/live/sessions`.
3. Select only the canonical Video 3 replay/live context for this product
   implementation; do not mix other source videos into the session picker.
4. Reconcile stored status with the actual child-process state.
5. Reattach MJPEG, telemetry WebSocket, controls, and render-layer state for a
   selected active session.
6. Represent orphaned, stopped, failed, and recoverable sessions explicitly.
7. Add heartbeat/last-telemetry timestamp and stale-session detection.
8. Persist live-session metadata to SQLite while retaining filesystem artifact
   records; migrate existing session directories through an idempotent indexer.

Exit criteria:

- Refreshing an active live route reattaches without creating a new session.
- Backend restart reconciles filesystem session records safely.
- A stale session cannot be displayed as running.
- No non-Video-3 session appears in this product flow.

### Phase 7 — Persist sessions, cameras, candidates, workstations, and layouts

Purpose: establish shared operational context for all pages.

Tasks:

1. Apply versioned migrations for the core entities.
2. Seed/index only the Video 3 session, source context, annotations, derived
   evidence paths, and existing human-review records.
3. Add CRUD/read APIs with validation and audit fields.
4. Store camera secrets indirectly; API responses must redact credentials.
5. Change blueprint upload to multipart bytes, not filename-only input.
6. Save layout drafts and activate a specific version explicitly.
7. Make `/api/spatial/active-layout` return the persisted active layout.
8. Remove blank-image and fabricated ten-seat operational fallback.
9. Mark calibration failure or insufficient evidence explicitly.
10. Derive candidate/workstation views from persisted assignments and events.

Exit criteria:

- Saved layout survives reload and backend restart.
- Active-layout reads return the saved version and hash.
- Camera and candidate pages no longer depend on demo constants.
- No calibrator response invents seats after a failed observation.

### Phase 8 — Drive Command Center and Candidates from one snapshot

Purpose: prevent cross-page disagreement.

Tasks:

1. Add a typed operational snapshot endpoint containing:
   - the canonical Video 3 session;
   - its camera/source context and health;
   - active layout and assignments;
   - event counts and recent events;
   - active live sessions;
   - data freshness timestamps.
2. Initialize the shared frontend store from that snapshot.
3. Apply event/live WebSocket deltas after the snapshot version.
4. Remove random surveillance capsules and fixed ingestion counts.
5. Replace dispatch animation with a persisted `dispatch_actions` mutation.

Exit criteria:

- Command Center, Event Queue, and Candidates agree after reload.
- Every displayed operational number has a backend source and timestamp.
- Dispatch state is durable and auditable.

### Phase 9 — Cases and evidence-package export

Purpose: implement escalation as a real workflow.

Tasks:

1. Add case creation from one or more reviewed events.
2. Add case lifecycle, assignee, notes, and event membership APIs.
3. Require explicit operator action; never auto-create a misconduct case from a
   model event.
4. Generate an evidence package server-side containing:
   - case manifest;
   - event records and review audit;
   - referenced media copies or an explicit missing-media entry;
   - model/reasoning provenance;
   - per-file hashes and package hash.
5. Never include unsupported claims such as physical inspection or
   multi-camera confirmation.
6. Store export status and provide a genuine download endpoint.

Exit criteria:

- Cases survive reload.
- Exported hashes are computed from actual files.
- Re-exporting the same frozen case version is reproducible or documents
  timestamp-dependent metadata.

### Phase 10 — Real analytics, benchmarks, and health

Purpose: replace quantitative-looking synthetic panels with traceable values.

Tasks:

1. Add analytics endpoints derived from persisted events/reviews:
   - Video 3 counts by status/type/candidate/workstation;
   - review turnaround time;
   - event duration distributions;
   - candidate-hour denominator only when session/assignment coverage exists.
2. Do not calculate unsupported accuracy metrics.
3. Load Benchmark Lab from `/api/runs` and `/api/runs/latest`.
4. Filter Benchmark Lab to Video 3 runs and preserve each run's limitations and
   provenance.
5. Add runtime sample ingestion and aggregation.
6. Display GPU, temperature, VRAM, FPS, and provider latency only when measured;
   otherwise show `NOT AVAILABLE`.
7. Remove `Math.random()` and fixed quantitative arrays from operational
   analytics.

Exit criteria:

- Every chart can identify its endpoint, filter, denominator, and freshness.
- New Video 3 verified/experimental runs appear without rebuilding frontend
  source.
- Unsupported metrics render as unavailable, not zero.

### Phase 11 — Repair and secure the AI query console

Purpose: ensure the console reasons only over real, authorized backend context.

Tasks:

1. Replace nonexistent `repository.list_events()` with `repository.list()`.
2. Remove the hardcoded fallback event descriptions.
3. Remove browser-side Groq calls and `VITE_GROQ_API_KEY` handling.
4. Keep provider credentials server-side only.
5. Build query context through a service using persisted events, layouts,
   assignments, cases, and explicit time/session filters.
6. Return cited record IDs with the answer.
7. Return an explicit provider-unavailable response rather than fabricated
   context.
8. Add prompt-injection and data-boundary tests before enabling unrestricted
   operator queries.

Exit criteria:

- Built JavaScript contains no provider secret.
- Query answers cite existing event/case IDs.
- Empty context produces an honest no-data response.
- Query context cannot include records from another video.

### Phase 12 — Authentication and operator accountability

Purpose: replace typed names with accountable identities.

This phase can start with a local deployment identity mechanism but must not
pretend to be enterprise authentication.

Tasks:

1. Add operator accounts/roles and authenticated session identity.
2. Derive `reviewed_by`, case assignee, dispatch operator, and export creator
   from the authenticated request—not request text.
3. Add authorization for review, case export, camera configuration, and layout
   activation.
4. Preserve immutable audit records.

Exit criteria:

- The frontend cannot impersonate another reviewer by changing payload text.
- Privileged mutations reject unauthorized callers.

## 11. Test strategy

### Backend unit tests

- Event mapping and review transitions.
- Migration ordering, rollback, and idempotence.
- Repository CRUD and relational constraints.
- Media containment and missing-file behavior.
- Analytics denominators and empty-data behavior.
- Evidence export hashes and manifests.
- Query-context construction without fabricated fallback.
- Video 3 filtering and strict frame-bound enforcement.

### Backend API tests

- OpenAPI generation.
- Review success, 404, validation failure, and concurrency conflict.
- Event-specific keyframe and clip responses.
- Pagination and filtering.
- Live-session reconciliation.
- Layout draft/activation lifecycle.
- Case creation/membership/export lifecycle.
- Redaction of source credentials.

### Frontend tests

- API adapter handles full, sparse, and invalid records.
- Explicit loading/error/empty/live states.
- Review mutation success, failure, retry, and rollback.
- Direct route loading and browser navigation.
- Selected event controls all evidence URLs.
- Live-session reattachment.
- Operational components cannot import demo data.
- No browser-side provider key or direct Groq request.

### End-to-end workflows

1. Load real events → open event → view matching media → review → refresh →
   decision remains.
2. Start live session → navigate away → return → refresh → reattach → stop.
3. Upload blueprint → save draft → activate → restart → active layout remains.
4. Confirm/review event → create case → attach events → export → verify hashes.
5. Stop backend → explicit failure → restart → Retry → data restored.
6. Attempt to load another video ID or frame 2200+ → request is rejected or
   excluded from the Video 3 product view.

## 12. Observability requirements

Every request should have a request ID. Mutations should log:

- authenticated operator;
- entity and operation;
- success/failure;
- timestamp;
- request ID;
- previous/new version where applicable.

Frontend errors must report a safe request ID and Retry action. Do not log RTSP
credentials, provider keys, full authorization tokens, or sensitive media.

## 13. Commit sequence

Use one purpose per commit:

1. `test(api): reproduce and guard OpenAPI generation`
2. `fix(api): restore typed OpenAPI schema generation`
3. `feat(frontend): add typed API client and event adapter`
4. `feat(review): persist human review decisions`
5. `feat(evidence): load media for selected persisted event`
6. `feat(routing): persist page and entity selection in URLs`
7. `fix(frontend): replace silent demo fallback with explicit states`
8. `feat(live): rehydrate and reconcile live sessions`
9. `feat(storage): add versioned product schema migrations`
10. `feat(layouts): persist spatial layouts and camera registry`
11. `feat(snapshot): serve shared operational state`
12. `feat(cases): persist cases and event membership`
13. `feat(exports): generate hashed evidence packages`
14. `feat(analytics): serve persisted operational aggregates`
15. `feat(benchmark): load verified run manifests dynamically`
16. `fix(query): use persisted context and server-only credentials`
17. `chore(demo): isolate showcase data under explicit demo mode`
18. `feat(auth): bind operator actions to authenticated identity`

Each commit must pass the relevant focused tests. Milestone commits must pass
the complete Python suite and frontend production build.

## 14. Recommended release milestones

### Milestone A — Trustworthy review workflow

Phases 0–5. This is the immediate priority and first shippable improvement.

Deliverable: real event list, correct evidence, durable decisions, stable URLs,
and no silent demo fallback.

### Milestone B — Durable operations

Phases 6–8.

Deliverable: recoverable live sessions, persistent cameras/layouts/assignments,
and a consistent Video 3 operational snapshot.

### Milestone C — Case management and reporting

Phases 9–10.

Deliverable: real cases, verifiable evidence exports, backend-derived analytics,
and current run manifests.

### Milestone D — Secure intelligence and accountability

Phases 11–12.

Deliverable: grounded query console and authenticated operator audit.

## 15. Overall definition of done

The connectivity repair is complete only when:

- no operational page silently displays demo records;
- event review decisions survive refresh and restart;
- event evidence belongs to the selected event;
- route state is bookmarkable;
- live sessions reattach deterministically;
- layouts, cameras, sessions, assignments, and cases are durable;
- analytics and health values have real data sources and timestamps;
- case exports contain actual files and computed hashes;
- provider secrets exist only server-side;
- OpenAPI generation and all required tests pass;
- existing inference and verified-run workflows remain runnable;
- every operational record shown by this implementation belongs to Video 3 and
  frames 0–2199;
- the repository tree has not been reorganized.

## 16. First implementation slice

Begin with Phases 0–3 only:

```text
OpenAPI contract repair
        ↓
typed frontend API client
        ↓
complete event mapping
        ↓
persisted human review
        ↓
selected-event keyframe and clip
        ↓
refresh/restart verification
```

Do not begin cases, analytics, or database expansion until this vertical slice
passes its end-to-end test. It resolves the highest-risk product defect: an
operator currently appears to review evidence, but the decision and displayed
media are not reliably tied to the durable event record.

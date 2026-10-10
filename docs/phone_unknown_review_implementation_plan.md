# Unknown phone proposal review: implementation plan

## Existing, frozen inputs

- Baseline: `outputs/unified_pipeline/video03_review_verified_20261003`.
- Phone detections are written by `src/drishti/release/unified_review.py` through
  `CanonicalInferenceService` into `detections.jsonl`.
- `scripts/export_phone_review_queue.py` creates
  `phone_proposals_needs_review.jsonl` and `phone_review_sample.csv`; it marks
  absent-CVAT proposals as review-only, never negative.
- `scripts/evaluate_cvat_boxes.py` is the current confidence-ordered, one-to-one
  IoU>=0.5 partial-label evaluator.  It intentionally reports recall only.
- `scripts/audit_cvat_video.py` is the CVAT XML exporter.  It preserves explicit
  boxes and represents annotation absence as UNKNOWN.

## Existing review assets and schemas

`scripts/build_video03_human_review.py` exact-joins each unknown export row to
`detections.jsonl`, producing a stable proposal ID, raw box/confidence, person and
seat association, crop/model provenance, and status `UNKNOWN`.  Its deterministic
`temporal_clusters`/`spatial_link` grouping is already tested and creates the 22
apparent-object segments in `human_review_v1/segments.jsonl`.  It also produces
representative frames, `review_data.json`, and a hash-protected package.

`scripts/serve_video03_human_review.py` and `scripts/templates/video03_review.html`
provide a resumable local interface.  Decisions are append-only and restricted to
`UNKNOWN`, `CONFIRMED_PHONE`, `NOT_PHONE`, and `AMBIGUOUS`; no endpoint edits CVAT
or starts training.

## Phase 1: canonical cluster export

Add `scripts/cluster_phone_unknowns.py` as a thin CLI over the existing exact
provenance join and `temporal_clusters` functions.  It will write the requested
`phone_unknown_clusters.jsonl`, with deterministic IDs, frame range, duration,
confidence summary, association distribution, representative proposal/frame IDs,
provenance, thresholds, and status `UNKNOWN`.  It will refuse overwrite and verify
the source export hashes.  It does not create labels or metrics.

## Later gated phases (not implemented in Phase 1)

Phase 2 will reuse the package's representative images/preview rather than create
redundant crops.  Phase 3 will use the existing server/UI and add a cluster summary
export only if needed.  Phase 4 requires human decisions.  Only then may a separate,
manually exported refined CVAT artifact be audited.  A refined evaluator must treat
unreviewed as UNKNOWN and AMBIGUOUS as excluded; it may report precision/F1/mAP or
FP/hour only after its denominators are demonstrably exhaustive.

## Tests and risks

Keep the existing clustering, decision-state, and append-only tests.  Add Phase-1
reproducibility and no-overwrite tests.  Primary risks: association/segment identity
is an apparent-object review aid, not physical-phone identity; confirmed missing
phones require a human CVAT edit; same-video metrics remain development diagnostics,
not generalization.

# NETRA-4D release workflow

`scripts/run_product_pipeline.py` is the supported inference entry point for a
local video, webcam, or RTSP source.  It is the only runner permitted for a
new product claim. It uses `configs/model_registry.yaml`, writes privacy-safe
video, raw detection JSONL, review-event JSONL, SQLite review records, and a
completion manifest in one run directory.

The historical `run_unified_pipeline.py`, `run_fused_pipeline.py`, and
`run_situational_pipeline.py` are experimental research runners. They are not
release paths and must not be used for product demonstrations or metric claims.

## Model gate

```powershell
python scripts/validate_models.py --output outputs/model_audits/current.json
```

Only `release_approved: true` registry entries can be loaded by the product
runner. A custom checkpoint must have an immutable training-data manifest,
class map, held-out camera/session evaluation report, and human approval before
that flag changes. `--include-unapproved` audits a checkpoint; it never enables
it for the product.

## Run one source

```powershell
python scripts/run_product_pipeline.py `
  --source path\\to\\exam.mp4 `
  --camera-id CAMERA_01 `
  --require-cuda
```

The output label is review evidence. `PHONE_USE_REVIEW` means a COCO phone box
was associated with a tracked candidate and persisted; proxy events remain
explicitly marked. No event is a cheating verdict.

## Held-out event evaluation

Create a JSONL label file from human review with at least these fields:

```json
{"event_type":"PHONE_USE_REVIEW","candidate_ids":[12],"started_at_sec":31.2,"ended_at_sec":34.5,"review_status":"CONFIRMED","split":"test_camera_02"}
```

Use whole cameras or examination sessions for `test`, never randomly sampled
overlapping frames from the training videos. Then run:

```powershell
python scripts/evaluate_review_events.py `
  --predictions outputs/product_runs/<run>/review_events.jsonl `
  --labels datasets/review_labels/test.jsonl `
  --output outputs/evaluations/<run>.json
```

The report contains TP, FP, FN, precision, recall, and F1 per event type. It
returns `INSUFFICIENT_LABELS` rather than manufacturing a metric when review
labels do not exist.

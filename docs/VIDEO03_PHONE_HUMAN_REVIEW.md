# Video 3 manual phone proposal review

Current package: `outputs/unified_pipeline/video03_review_verified_20261003/human_review_v1`.
Read its `REPORT.md` for all temporal segments, confidence/association distributions,
person failure ranges, 143 visual examples and proposed manual annotation checks.
`PHONE_MISS_DIAGNOSTICS.md` provides the detector-only crop replay for all 27 known
phone-box misses. These are diagnostics, not behavioural-event accuracy.

## Start human review

```powershell
python scripts/serve_video03_human_review.py --package outputs/unified_pipeline/video03_review_verified_20261003/human_review_v1
```

Open http://127.0.0.1:8765. Stop the server with Ctrl+C when finished.
The server is local-only and does not serve the repository outside the review package.

All **849 proposals on 846 frames**, grouped into **22 apparent-object temporal
segments**, start UNKNOWN. The UI provides video/frame seeking, original boxes,
raw confidence, person associations, crop provenance, model hash, stable IDs and
an append-only `manual_decisions.jsonl` ledger. Reviewer ID and explicit inspection
acknowledgement are required. Segment-wide decisions require inspection of every
member; mixed segments must be reviewed proposal by proposal.

Allowed manual states: UNKNOWN, CONFIRMED_PHONE, NOT_PHONE, AMBIGUOUS. A NOT_PHONE
decision applies only to its reviewed proposal, not to a whole frame. UNKNOWN and
AMBIGUOUS are never negative training labels. No training/export endpoint exists.
The server verifies review-data fingerprints before loading previous decisions.

The original 22 CSV samples map to proposal/segment IDs in
`original_samples_mapped.csv`; they do not exhaustively cover the 849 proposals.
All range/frame lists are exported as JSONL/CSV. The preview is a derived H.264
copy of the existing annotated output, not a replacement canonical input.
Overlays contain experimental proxies and best-effort privacy masking from the
existing run; they do not establish object identity, validated behaviour or complete
face anonymization. Keep this examination footage within its authorized review scope.

## Rebuild reproducibly into a new directory

```powershell
python scripts/build_video03_human_review.py --probe-pose --output outputs/unified_pipeline/video03_review_verified_20261003/human_review_v2
python scripts/probe_video03_phone_misses.py --package outputs/unified_pipeline/video03_review_verified_20261003/human_review_v2
```

Existing package directories are refused. The builder validates and hashes the
immutable XML, canonical video and verified source exports. Annotation geometry
is used for post-inference measurement/examples only; never for generating
diagnostic detection crops. The lower-confidence phone replay is isolated and
does not change production thresholds or create training labels.

The current v1 report includes visually inspected recommendations and phone-probe
findings in addition to the reproducible builder's base report. These recommendations
are not corrections applied to ground truth.

Verification: 10 focused tests passed; a headless browser loaded all 22 segments
with all 849 proposals UNKNOWN, submitting zero decisions. Raw XML and source
exports remain unchanged. This work stops at **WAITING_FOR_HUMAN_REVIEW**.

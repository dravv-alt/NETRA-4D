# Video 3 unified review profile

This profile extends the shared inference service with NETRA's own reasoning engines.
It is not the old all-checkpoints experimental runner and is not a cheating classifier.

## Reproduce

```powershell
python scripts/run_unified_pipeline.py --asset VIDEO_03 --profile video03-review --sample-every 1 --allow-experimental-models --evaluate-cvat
```

Use `--output-dir` to select a new run directory. Existing directories are refused.
The source resolves to the canonical 2,200-frame MP4, indices 0–2199, 88 seconds.
Use `--max-frames 8` without `--evaluate-cvat` for a short smoke test. Evaluation
always uses the 2,200-frame annotation denominator and marks shorter runs partial.
Evaluation needs the already-derived `outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl`.
The original CVAT XML is unchanged. Annotations never determine detection crops.

## Actual workflow

1. Bound Video 3 input and fingerprint it.
2. Validate registry files and require explicit experimental-model authorization.
3. YOLO26n pose + ByteTrack produce person observations; YOLO26n supplies raw COCO objects.
4. The 100-cumulative-epoch phone-only checkpoint predicts on the full frame and
   person context crops. Crop proposals persist at most one second; phone boxes
   always require inference on current pixels, not carried-forward detections.
5. Deduplicate objects and retain raw confidence, crop provenance, and model hashes.
6. Feed the same observations to environment profiling, role discovery, candidate
   identity/seat anchoring, semantic zones, five-second buffers, biomechanical
   features, primitive events, persistence/repetition, competing hypotheses, and
   candidate state machines. These are experimental review proxies.
7. Shared temporal review events, privacy masking and overlays produce the evidence
   video, JSONL, and SQLite review records. Behavioural hypotheses stay in the
   reasoning JSONL; the video does not display uncalibrated cheating probabilities.
8. Optional post-inference CVAT recall measurement and export of unknown phone
   proposals for manual review. Absence of an annotation is not a negative label.

## Models and disabled branches

| Component | Selection / status |
| --- | --- |
| Person / pose | `models/weights/yolo26n-pose.pt`; boxes/keypoints only |
| Generic objects | `models/weights/yolo26n.pt`; raw COCO evidence |
| Phone | registry `phone_cvat_video03_100epochs`; **unapproved**, same-video training |
| Multi-HMR / SMPL-X | Disabled; no validated 3D task in this run |
| Legacy phone / cheating / chit / multimodal models | Not selected; lack relevant validation |
| RTMPose / MediaPipe / hand-contour specialists | Disabled; no claimed corroboration |
| Enhancement | Disabled to preserve the detector's evaluated raw-input distribution |

The legacy runner remains available only via `--profile legacy-experimental`.
Its outputs must not be interpreted as validated release behaviour.

## Outputs and correctness boundaries

- `annotated.mp4`: privacy-masked review evidence.
- `detections.jsonl`: raw boxes/keypoints, confidence, associations, model provenance.
- `unified_reasoning.jsonl`: candidate state and explicitly uncalibrated proxies.
- `review_events.jsonl`, `events.sqlite`: shared temporal events and human review status.
- `COMPLETE.json`: bounded-run counts, source/model hashes, selected/disabled branches and runtime.
- `INCOMPLETE.json`: caught failures or interruption; process exits nonzero on failure.
- With evaluation: `metrics.json`, `phone_proposals_needs_review.jsonl`, `phone_review_sample.csv`.

Phone association is per current detector person, never broadcast to every candidate.
Unassociated detections remain available in the raw evidence and review queue.
The reasoning adapter does not register synthetic head-turn events when no measured
temporal pattern exists. The environment warmup now increments and finalizes.
Automatic roles/seats/neighbour mapping, head orientation, and behavioural states
are not independently evaluated against semantic CVAT attributes. Model recall on
this video is not generalization performance; precision, F1, mAP and false positives
per hour require exhaustively reviewed annotation coverage and a held-out session.

## Verified local run: 2026-10-03

Run: `outputs/unified_pipeline/video03_review_verified_20261003`.
Inference completed on CUDA. FFprobe decoded exactly **2,200 frames**, 1280×720,
25 FPS, 88 seconds. Detections and reasoning both cover frames 0–2199.
The final full test suite passed **110 tests** (5 warnings in the retained legacy
Multi-HMR test). Compile and tracked whitespace checks passed. These software
checks are not model-quality measurements. Changes remain local/uncommitted.

| Box metric, IoU ≥ 0.5 | Measured result |
| --- | --- |
| Phone | 473 / 500 = **94.6% recall** |
| Phone CVAT track 3 | 415 / 424 = 97.88% |
| Phone CVAT track 4 | 58 / 76 = 76.32% |
| Phone mean matched IoU | 0.9138 |
| Person, all manual tracks | 2,950 / 4,205 = 70.15% |
| Student track 0 | 1,150 / 2,130 = 53.99% |
| Student track 1 | 1,490 / 1,744 = 85.44% |
| Invigilator box track 2 | 310 / 331 = 93.66% |
| Precision / F1 / mAP / false positives per hour | Unavailable: coverage is not exhaustively labelled |

CVAT phone tracks **3 and 4 represent one physical phone**, per the annotator.
The per-track rows describe visibility segments, not different devices. The
inference pipeline does not claim to have validated physical-phone identity.

| Manual phone visibility window | Matched frames / labelled frames |
| --- | --- |
| 144–439 | 290 / 296 |
| 1290–1389 | 97 / 100 |
| 1552–1579 | 28 / 28 |
| 1664–1739 | 58 / 76 |

Every window contains matched detections; this is not a validated behavioural
event-recall metric or evidence that all phone-use episodes are correctly reviewed.

The run produced 1,323 raw phone proposals. **849 proposals across 846 frames**
without visible Phone annotations are UNKNOWN and need review; the CSV provides
22 representative samples. No negative training examples were created.
Review output contains 1,828 event observations with 19 unique review event IDs:
1,285 phone, 498 low-hand and 45 other-screen proxy observations. Repeated event
observations and IDs must not be treated as proven incident counts.

Elapsed inference was **752.70 seconds / 2.92 processed FPS**, including concurrent
verification activity. This is an offline run, not demonstrated 25-FPS real-time
operation. The reasoning adapter had no visible candidate rows on 449 frames;
role, seat and behavioural attribute accuracy remains unmeasured.
Environment profiling now finalizes at 30 frames, but its heuristic returned
`LARGE_WRITTEN_CLASSROOM` from a median of one detected monitor; this is not an
independently validated environment classification.

Next priorities: review the unknown phone proposals, improve the lower-recall
student/late-phone segment, and validate roles/association and phone events on a
separate camera/session. The raw CVAT XML hash remains
`301620001d90d0f2fb06deb3535e98dc67c6c641f7a10e84b5b01c2db1452382`.

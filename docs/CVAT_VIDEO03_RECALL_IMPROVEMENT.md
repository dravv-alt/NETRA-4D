Video 3 phone recall: verified canonical result
==============================================

The completed canonical pipeline matched **473 / 500 visible Phone boxes =
94.6% recall at IoU ≥0.5**, compared with 23% for full-frame inference from the
same 100-epoch checkpoint. Confidence stayed at **0.25**. This is Phone object
box recall on the same camera/session used for training, not cheating accuracy
or independent held-out model quality.

What changed
------------

The phone-only checkpoint was trained on 384-pixel context crops but was being
evaluated against the entire CCTV frame. The new shared inference helper scans
the full frame at image size 1280 and detector-generated person context crops
at image size 640. It transforms crop boxes back into frame coordinates and
deduplicates overlapping detections while preserving raw confidence values.
Crop locations come only from automatic person detections, never CVAT boxes.

For up to one second after a person detector dropout, the last person crop can
still be searched. Only the spatial proposal is retained: the model must detect
the phone again from current pixels. No previous Phone box is copied forward.
Cached proposals record age and source identity, expire, and clear on timestamp
reset or frame-size change. Phone/person association uses current observations.

The typed Phone provider is wired into `CanonicalInferenceService`, alongside
pose, COCO objects, privacy masking, temporal reasoning, JSONL provenance, SQLite
review events and annotated output. A temporal filter that previously accepted
only `cell phone` now also accepts the trained model's `phone` class, retaining
checkpoint and inference-view provenance. Unapproved model roles require the
explicit experimental flag; production approval remains false.

Measured comparisons
--------------------

| Inference mode | Person proposal source | Phone matches | Recall |
|---|---|---:|---:|
| Full frame, 100-epoch candidate | Not used | 115/500 | 23.0% |
| Person crops, no retention | Cached prior detector output | 436/500 | 87.2% |
| Person crops, no retention, canonical pipeline | Fresh detections | 435/500 | 87.0% |
| Person crops, one-second retention | Cached prior detector output | 478/500 | 95.6% |
| Person crops, one-second retention, canonical pipeline | Fresh detections | **473/500** | **94.6%** |

The cached proposals came from the earlier original-video run; the fresh runs
processed the canonical bounded MP4. Use the fresh end-to-end result as the
headline. In the cached no-retention experiment, 44 of 64 misses occurred when
the Phone was not covered by current person crops; the remaining 20 had no
Phone prediction. This motivated the bounded proposal-retention experiment.

The final run processed all **2,200 frames (0–2199)**. FFprobe decoded exactly
2,200 frames from its annotated MP4 at 25 FPS. Phone track 3 matched 415/424
boxes (97.88%); track 4 matched 58/76 (76.32%). These tracks are two appearances
of the same physical phone, not two independent entities. Mean matched Phone
IoU was 0.914. Person box recall remained 70.15%, an unresolved capability gap.

End-to-end elapsed time was 484.70 seconds, approximately **4.54 FPS** on the
laptop GPU. This includes the complete pipeline and output writes, with some
concurrent diagnostic work. It does not demonstrate 25-FPS real-time processing.
The run emitted 1,285 Phone review-event observations across 13 stable event IDs;
these are repeated heuristic review updates, not confirmed incidents.

Precision and remaining work
----------------------------

The model proposed Phone boxes on 1,319 frames. **849 proposals on 846 frames
had no corresponding visible Phone annotation on that frame.** They remain
UNKNOWN / NEEDS_REVIEW, not automatically declared false positives or negatives.
This is a substantial review burden and a concrete precision concern. Recall
alone does not make this checkpoint ready for unattended use.

The export `phone_proposals_needs_review.jsonl` preserves their source, raw score,
box, crop metadata and model provenance. `phone_review_sample.csv` contains 22
representative candidates selected across three-second windows, with blank
reviewer-decision and notes fields. No negative labels were fabricated.
Review these examples, then use explicitly confirmed false positives as hard
negatives. Full-video precision, F1, mAP and false positives/hour remain
unavailable until annotation/review coverage supports them. An independently
annotated camera/session is still required for model release approval.

Artifacts and reproduction
--------------------------

Final run:
`outputs/product_runs/video03_phone_crop_memory_canonical_20261002/`

- `annotated.mp4`: completed privacy-masked inference video.
- `metrics.json`: full bounded-video box evaluation.
- `COMPLETE.json`: run configuration, frame count, model hashes and experimental status.
- `detections.jsonl`: raw people/object observations and crop provenance.
- `review_events.jsonl`: human-review event observations.
- `phone_proposals_needs_review.jsonl`: 849 uncertain proposals.
- `phone_review_sample.csv`: 22 candidates for review.

Review-event database:
`outputs/video03_cvat_experiment/phone_crop_memory_events.sqlite`.

```powershell
python scripts/run_product_pipeline.py --source input/videos/VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4 --video-id VIDEO_03 --device cuda:0 --require-cuda --max-frames 2200 --phone-role phone_cvat_video03_100epochs --allow-experimental-models --phone-roi-hold-seconds 1 --run-id video03_phone_new_run
python scripts/evaluate_cvat_boxes.py --labels outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl --predictions outputs/product_runs/video03_phone_new_run/detections.jsonl --output outputs/product_runs/video03_phone_new_run/metrics.json --max-frames 2200
```

Verification: 25 focused tests passed, including crop remapping, deduplication
without confidence boosting, proposal expiration/reset, no fabricated Phone
boxes, model approval gating, custom-class temporal events and the Video 3 bound.
Raw annotation and model checkpoint hashes were verified unchanged. The initial
partial canonical run was stopped for the temporal-class fix and is explicitly
marked incomplete; later completed runs preserve their own evidence. All code
changes remain local workspace changes rather than a claimed published release.

Video 3 longer training — 2026-10-02
==================================

Completed 70 additional epochs from the 30-epoch phone candidate, giving 100
cumulative epochs. The old checkpoint and raw CVAT XML hashes were verified
unchanged. This is a new fine-tuning stage from learned weights, with a fresh
AdamW optimizer; the completed prior checkpoint had no optimizer state to resume.

The same 99 positive Phone crops were reused. Both annotation tracks represent
the same physical phone. No new labels or negatives were generated. The first
10 backbone layers stayed frozen; image size 640, batch 8, seed 42, initial
learning rate 0.001, no mosaic or flips, and CUDA were used. Source video remains
bounded to exactly 2,200 frames, indices 0–2199.

Results
-------

| Metric | Initial 30 epochs | 100 cumulative epochs |
|---|---:|---:|
| Full-video explicit Phone box matches | 87/500 | 115/500 |
| Full-video Phone recall, IoU ≥0.5 | 17.4% | 23.0% |
| Mean IoU of matched full-video boxes | 0.698 | 0.776 |
| Training-crop precision | 89.64% | 94.78% |
| Training-crop recall | 79.80% | 95.96% |
| Training-crop mAP@0.5 | 92.76% | 98.83% |
| Training-crop mAP@0.5:0.95 | 72.39% | 93.27% |

The full-video comparison used identical video and label hashes, 2,200 prediction
frames, image size 1280, confidence 0.25 and IoU 0.5. Recall improved by 5.6
percentage points. Track 3 matched 97/424 boxes; track 4 matched 18/76. Both
tracks belong to the same physical phone; these counts are observations, not
distinct phone entities. Full-video precision/F1/mAP remain unavailable without
confirmed exhaustive annotation coverage. Training-fit checks reuse training
crops, and full-video evaluation shares their camera/session/phone. Neither is
independent held-out accuracy.

The model still misses 385/500 explicit Phone observations. Additional epochs
helped modestly, while crop fit is already high. The next useful experiment is
person-crop inference using detector-generated proposals, followed by separately
reviewed negative examples and an independently annotated camera/session.
The model remains an experimental registry role, `phone_cvat_video03_100epochs`;
production model approvals were not changed.

Artifacts
---------

Run directory: `outputs/training/video03_phone_continued_100epochs_20261002`.

- `fit/weights/last.pt`: candidate evaluated on the full video.
- `fit/weights/best.pt`: checkpoint used by the final training-fit validator.
- `training_manifest.json`: checkpoint lineage, optimizer mode, config, hashes,
  versions and before/after recall.
- `full_video_evaluation/metrics.json`: completed 2,200-frame evaluation.
- `full_video_evaluation/detections.jsonl`: raw bounded-video predictions.
- `fit/results.csv`: per-epoch training history.

Candidate SHA-256:
`12c26f587f3ee6784330e864c3426543cb95a087f4480a5a02d70d10a95c283d`.

Reproduce this continuation with a new output directory:

```powershell
python scripts/continue_cvat_phone.py --previous-run outputs/training/video03_phone_bootstrap_20261002_v2 --additional-epochs 70 --output outputs/training/video03_phone_continuation_new_run
```

The script performs training and bounded-video evaluation sequentially, refuses
to overwrite a run, checks source hashes, and records failures explicitly.

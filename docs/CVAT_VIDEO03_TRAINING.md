Video 3 phone bootstrap training — 2026-10-02
============================================

Training completed for 30 epochs using COCO-pretrained YOLO26n. The dataset
contains 99 explicit positive Phone crops, sampled every five frames from 500
visible phone-box observations. Tracks 3 and 4 retain their CVAT IDs and share
the user-confirmed physical entity `phone_01`. No unannotated frames were used
as negative training examples. This trains the `phone` object class only;
Student/Invigilator detection and head/gaze/hand attributes were not retrained.

The first run stopped before its first epoch because sandboxed Windows named
pipes denied the label loader. The approved rerun completed successfully in
`outputs/training/video03_phone_bootstrap_20261002_v2`.

Checkpoint and evidence
-----------------------

- Candidate: `outputs/training/video03_phone_bootstrap_20261002_v2/fit/weights/last.pt`.
- SHA-256: `2ef775eaaa0241adde8c863a5ff8080f8a8593fbbbe50605b05e65b6cff9e87e`.
- Provenance and config: `training_manifest.json` in the same run folder.
- YOLO labels and frame/crop/entity metadata: `dataset/data.yaml` and `dataset/samples.json`.
- Full 2,200-frame inference: `full_video_evaluation/detections.jsonl`.
- Full video metrics: `full_video_evaluation/metrics.json`.
- Experimental registry role: `phone_cvat_video03_bootstrap`, release approval false.

The training-fit check on the SAME 99 crops produced precision 89.64%, recall
79.80%, mAP@0.5 92.76%, and mAP@0.5:0.95 72.39%. Ultralytics' final validator
evaluated `best.pt`; the full-video experiment evaluated `last.pt`. These are
training-fit metrics, not held-out validation accuracy. YAML deliberately uses
the training images for the fit check because no independent session exists.

On all 2,200 full video frames, at image size 1280, confidence 0.25 and IoU 0.5,
the candidate matched **87 / 500 = 17.4%** of explicit visible Phone boxes.
Track 3: 80/424; track 4: 7/76. Mean matched IoU was 0.698. This compares with
the earlier canonical 640-image-size baseline's 0/500 result; inference image
size differs, so this is not a controlled architecture-only comparison.
The earlier sampled baseline at 1280 also matched 0/20 labelled Phone boxes.

The candidate's full-video pass took 44.35 seconds. It contains detector-only
inference, decoding, serialization and evaluation, not the entire canonical
pose/privacy/event/rendering pipeline. Precision, F1, mAP and false positives
per hour for the full video remain unavailable because exhaustive annotation
coverage is not confirmed. Unmatched proposals are unknown rather than false
positives. Training and video evaluation share the same camera/session/physical
phone; these results cannot establish generalization to another examination.

The model learned the crops but still misses most full-frame phone observations.
It is registered as an experimental candidate and is not automatically promoted
into the production model roles. Useful next work is to measure person-crop
inference using detector proposals (not ground-truth locations), add separately
reviewed negative examples, and evaluate on an independently annotated session.

Permanent Video 3 processing bound
---------------------------------

The working source is
`input/videos/VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4`, verified
at exactly **2,200 frames (indices 0–2199), 25 FPS, 88 seconds**. The original
is preserved in
`dataset_gdrive/source_archive/VIDEO_03_mobile_phone_exam_hall_01_original_3283_frames.mkv`.
Neither raw CVAT export nor the archived original was overwritten.

Both manifests select the bounded local file. The shared materializer refuses
an original-source fallback if it is missing. Canonical/CVAT runners redirect
known Video 3 paths and cap requested frame counts at 2,200. Unified, fused and
situational direct runners also redirect known Video 3 paths. Other videos and
live sources are unaffected. Old output artifacts are historical evidence and
are not rewritten. Source archival does not truncate the raw XML metadata;
derived exports apply the working bound and exclude unknown labels.

Reproduce training and evaluation
---------------------------------

From the repository root with the installed GPU Python environment:

```powershell
python scripts/train_cvat_phone.py --output outputs/training/video03_phone_new_run --epochs 30
python scripts/evaluate_trained_phone.py --model outputs/training/video03_phone_new_run/fit/weights/last.pt --output outputs/training/video03_phone_new_run/full_video_evaluation
```

The scripts refuse to overwrite run folders. Windows may require the same
approved execution outside the sandbox for the label loader's named pipe.
Training format and API follow the [Ultralytics detection dataset documentation](https://docs.ultralytics.com/datasets/detect/)
and [training documentation](https://docs.ultralytics.com/modes/train/).

Verification: training completed, a fresh candidate loaded for 2,200-frame
inference, raw annotation hash stayed unchanged, the shared dataset loader
returned the bounded source, and 11 focused tests passed. No head/gaze/hand
classifier, independent held-out evaluation, or production promotion is implied.

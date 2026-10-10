Video 3 CVAT experiment
======================

The source has 3,283 decoded frames at 25 FPS and 1280×720. The experiment
processes exactly 2,200 frames: zero-based indices **0–2199**, or 88 seconds.
OpenCV reports a 3,282-frame metadata estimate; FFprobe's decoded count is 3,283.
The original video and raw XML/ZIP are retained unchanged.
The original 3,283-frame source is now archived at
`dataset_gdrive/source_archive/VIDEO_03_mobile_phone_exam_hall_01_original_3283_frames.mkv`.
The sole Video 3 source under `input/videos` is the bounded MP4. Both dataset
manifests, the shared Drive materializer, canonical runner, CVAT workflow and
direct unified/fused/situational runners resolve known Video 3 inputs to that MP4.
Canonical requested frame counts above 2,200 are capped; a missing bounded file
fails rather than downloading the longer source. Other videos retain their lengths.

The CVAT 1.1 interpolation export has five manual tracks: two Students, one
Invigilator and two Phone tracks. Per the user's identity clarification, both
Phone tracks represent **one physical phone**, appearing repeatedly. Track count
is not physical-object count. It contains explicit boxes through frame 2130,
whose last Student box is an outside marker; the final visible box is at 2129.
Frames 2131–2200 lack exported annotations despite the stated reviewed range.
Missing labels are UNKNOWN. No interpolation or extrapolation is applied.

Student attributes follow the schema embedded in the export, including `back`,
`own_screen`, `phone`, `unknown` and `invigilator`. They remain attributes rather
than detection classes. The raw export is hashed before and after processing.

Current outputs
---------------

- `input/videos/VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4`: bounded source copy.
- `outputs/video03_cvat_experiment/dataset/audit.json`: geometry, attributes and track audit.
- `outputs/video03_cvat_experiment/dataset/trusted_overlay.mp4`: explicit CVAT boxes.
- `outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl`: all exported box records, including outside markers.
- `outputs/video03_cvat_experiment/dataset/coco.json`: 86 sampled images, 189 visible boxes; attributes and track IDs retained.
- `outputs/product_runs/video03_cvat_2200_20261002/`: canonical model overlay, detections, events and run manifest.
- `outputs/video03_cvat_experiment/metrics.json`: full-run recall against explicit labels.
- `outputs/video03_cvat_experiment/phone_comparison_resolution/comparison.json`: sampled model/resolution/tiling experiment.

Interpretation
--------------

Visible box counts are Student 3,874, Invigilator 331, Phone 500. There are
219 keyframes and seven outside markers. The checks found no invalid geometry,
out-of-bounds boxes, duplicate overlaps above 0.9 IoU, missing/invalid schema
attributes or large jump/size-change proposals. Identity switches cannot be
established from these checks alone.

Phone track 3 has gaps at 441–1289 and 1391–1551, both following explicit outside
markers. These are expected outside intervals, not automatically corrected gaps.
Tracks 3 and 4 map to the same physical entity, `phone_01`, in the derived
`dataset/entity_mapping.json`. Its four explicitly visible intervals are
144–439, 1290–1389, 1552–1579 and 1664–1739 (inclusive). These are annotation
visibility episodes, not four different phones or four validated behavioural
events. The identity mapping comes from the user's clarification, not model
re-identification; the raw CVAT track IDs are retained.
Review the missing exported tail at 2131–2200 before treating it as ground truth.

Person evaluation merges Student and Invigilator because the deployed pose model
has a generic person class. Matches use confidence-ordered one-to-one IoU ≥0.5.
CVAT track IDs and model track IDs are separate namespaces; ID equality is never
used for matching. Missing prediction frames remain in the recall denominator.

Precision, F1, mAP and false positives/hour are unavailable until exhaustive
annotation coverage is confirmed. Unmatched predictions are UNKNOWN. Event and
attribute evaluation require corresponding predictions and human event windows.
This is one-session diagnosis; do not randomly split adjacent frames into train
and test or call these results held-out camera/session validation. The sampled
COCO export preserves partial labels; ordinary COCO training/evaluation would
incorrectly turn missing objects into background without a coverage policy.

Reproduce the complete workflow
-------------------------------

Run from the repository root with the installed Python/CUDA environment and
FFmpeg/FFprobe on PATH. The command refuses to overwrite an existing run:

```powershell
python scripts/run_cvat_experiment.py --video input/videos/VIDEO_03_mobile_phone_exam_hall_01.mkv --annotations 'MANUAL_ANNOTATIONS(CVAT)/VIDEO_3_EXAM_HALL_PHONE_USE/annotations.xml' --max-frames 2200 --run-id video03_cvat_new_experiment
```

The orchestrator creates derived outputs, runs the same canonical inference
service, evaluates recall and compares phone checkpoints. The registry's model
release approvals are unchanged by this diagnostic experiment. Failed steps are
recorded in `experiment.json`. No processing beyond the requested bound or
automatic semantic annotation changes are performed.

Sampled phone findings
----------------------

Twenty one-second sampled frames contain a visible manually labelled Phone.
At confidence 0.25 and IoU 0.5, all five configurations matched 0/20 boxes:
COCO full-frame at 640 and 1280, COCO full-frame plus four tiles at 640, and
the custom phone checkpoint full-frame at 640 and 1280. COCO produced zero
phone proposals. The specialist produced 27 proposals at 640 and 95 at 1280,
but none overlapped a labelled Phone sufficiently. These proposal counts are
not false-positive counts because annotation completeness is unconfirmed.

At frame 175, the specialist's 640 prediction is approximately
`[195.87, 528.94, 306.51, 679.03]` at confidence 0.617, whereas the manual Phone
is `[1043.90, 548.89, 1090.40, 593.34]`. This is a localization failure in that
example. Increasing image size alone did not recover sampled phone recall.
Use representative camera/session annotations and inspect the specialist's
training/class mapping before choosing retraining or replacement. Threshold
calibration requires a separate validation session, not this diagnostic sample.

Completed full-run results
--------------------------

The canonical CUDA run completed all 2,200 frames. FFprobe verified the inference
overlay, annotation overlay and bounded source each contain exactly 2,200 frames.
At IoU 0.5, generic Person recall was **2,966 / 4,205 = 70.54%**. Student track 0:
1,168/2,130 = 54.84%; Student track 1: 1,489/1,744 = 85.38%; Invigilator track 2:
309/331 = 93.35%. Mean IoU of matched Person boxes was 0.782.

Phone recall was **0 / 500 = 0%**. Both Phone tracks were missed: 0/424 and 0/76.
The baseline cannot support a measured phone-review capability on this session.
The specialist's sampled localization failures warrant training-data and
checkpoint scrutiny rather than declaring it a working replacement.

The run recorded 802.38 seconds for 2,200 frames, or approximately **2.74 FPS**
end to end (364.72 ms/frame). This includes capture, inference, rendering and
output writes; sampled comparison and export work ran concurrently. It is not
an isolated model-latency benchmark and does not demonstrate 25-FPS live use.
The run's 519 temporal event observations and five new-event observations are
heuristic outputs, not validated event detections or five confirmed incidents.

Next steps: review the missing CVAT tail; inspect missed Student boxes; inspect
the custom checkpoint's training labels and phone proposal locations; gather
independent annotated camera/session clips before retraining, calibrating and
reporting precision or event-level performance.

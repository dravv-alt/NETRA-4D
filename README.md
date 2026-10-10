# NETRA-4D

Privacy-aware examination-video evidence pipeline built around real computer-vision model inference. NETRA-4D generates auditable annotated videos, raw per-frame detections, high-confidence phone-review frames, and run manifests.

The repository deliberately distinguishes **measured outputs** from planned or experimental capabilities. It does not create simulated boxes, synthetic SMPL-X joints, seeded incidents, fabricated benchmark scores, or inferred malpractice verdicts.

## Product release path

Use [`scripts/run_product_pipeline.py`](scripts/run_product_pipeline.py) for
all new local-video, webcam, and RTSP product runs. It is the single supported
pipeline: registry-approved perception models -> privacy-safe evidence ->
review events -> SQLite/API-compatible records -> manifest. Validate the model
class maps first with `python scripts/validate_models.py`. The historical
unified, fused, and situational runners are experimental and are not release
paths. See [RELEASE_WORKFLOW.md](docs/RELEASE_WORKFLOW.md) for the held-out
evaluation contract; it intentionally reports no accuracy metric before human
labels exist.

## What runs today

- Official Ultralytics `yolo26n-pose.pt`: person boxes and 17 COCO keypoints.
- Official Ultralytics `yolo26n.pt`: raw COCO object detections for new runs.
- ByteTrack candidate IDs when tracking is enabled; otherwise IDs are visibly marked as frame-local.
- Low-confidence phone proposals from candidate crops and optional overlapping tiles, wrist/person association, IoU deduplication, and temporal confirmation.
- A bounded live webcam/RTSP runner that drops stale frames and records FPS, latency, a privacy-safe stream, event keyframes, and clips.
- A stable SQLite/FastAPI event and human-review backend with REST, WebSocket updates, media serving, and candidate timelines.
- A workstation-relative CBT Peak reasoner with candidate baselines, own/neighbor monitor semantics, normal-device counter-evidence, four sequence behaviours, uncertainty routing, and cross-candidate graph state.
- A one-to-one candidate/device binding: tracked candidate `N` is mapped to the assigned workstation screen as `PC-N`, with current detector or scene-map provenance and association confidence.
- Privacy overlay sequence: pose keypoints -> 2-D gaze proxy -> face/head blur -> candidate/gaze annotation. `GAZE PROXY` is head-direction from eye/nose geometry, not eye/iris tracking.
- Five user-supplied Google Drive videos, materialized one at a time for real decoding and then removed from temporary staging.
- One output directory per source video, with an annotated MP4, JSONL evidence, hashes, summary, and completion marker.

## Honest limitations

- A detected `cell phone` is evidence for review, not proof of phone usage, cheating, or other misconduct.
- No accuracy, precision, recall, or F1 is claimed because reviewed ground-truth labels are not currently available.
- Multi-HMR/SMPL-X integration code is available, but there is no completed 3-D run yet. It requires a compatible CUDA/Python environment and the official Multi-HMR checkpoint.
- A complete privacy + YOLO26 five-video CUDA run is verified locally from downloaded Kaggle output. Kaggle allocated a Tesla P100 for that completed job, not the requested T4; the recorded runtime metadata is preserved rather than relabelled. Never use a folder containing `INCOMPLETE.txt` as result evidence.
- Pairwise interaction and gaze-to-neighbour-screen proposals are implemented as explicitly labelled temporal proxies, not learned cheating verdicts.
- The real chit detector adapter, labeling tool, and strict training runner are implemented, but no trained chit checkpoint is claimed: the merged repository contains no genuine chit labels or checkpoint. Low-hand events remain proxies until humans label real frames and train/validate the custom model.

## Pipeline

```text
Google Drive video (one at a time)
        v
Decoded source frames + source hash/provenance
        v
YOLO26 pose -----------> people + COCO keypoints + candidate tracking
YOLO26 object detection -> raw object boxes/classes/confidence
        v
Gaze proxy -> face/head blur -> post-blur candidate annotation
        v
Annotated MP4 + detections.jsonl + phone-review stills + COMPLETE.json
        v
Kaggle kernel output / private Kaggle dataset / optional Google Drive upload
```

The runner writes phone-review stills only for the exact raw COCO class `cell phone` at confidence >= `0.70`. Those stills remain inside their own `VIDEO_XX/phone_detection_frames/` directory.

## Repository map

```text
ARCHITECTURE.md                         end-to-end components and data flow
drishti4d/configs/datasets.json       five supplied Drive assets
models/weights/                       official YOLO weights (ignored by Git)
models_smplx_v1_1.zip                 supplied SMPL-X archive (870 MB)
src/drishti/providers/                real YOLO providers
src/drishti/privacy.py                privacy/gaze-proxy layer
scripts/run_product_pipeline.py       supported batch/live product inference path
scripts/validate_models.py            model file/class-map release gate
scripts/evaluate_review_events.py     held-out human-review metric report
scripts/render_annotated_videos.py    historical five-Drive-video renderer
scripts/run_live_webcam.py            browser-capture compatibility adapter
scripts/review_events.py              human-only event review workflow
scripts/analyze_reviewed_events.py    reviewed threshold calibration report
scripts/label_chit_examples.py        human YOLO chit annotation over Drive videos
scripts/train_chit_detector.py        strict real custom detector training
scripts/import_review_events.py       deterministic SQLite review-queue import
scripts/calibrate_cbt_workstations.py real-frame human workstation calibration
scripts/auto_calibrate_from_evidence.py automatic map from verified real detections
scripts/preview_workstation_map.py     calibration review image from a real frame
src/drishti/cbt/                      primary workstation-relative reasoner
scripts/annotate_baseline_events.py   fast temporal replay over verified evidence
scripts/run_kaggle_gpu_pipeline.py    submit -> wait -> download -> validate -> publish
scripts/benchmark_detectors.py        same-frame YOLO11n vs YOLO26n comparison
scripts/run_multihmr_smplx.py         strict official Multi-HMR runner
scripts/upload_outputs_to_gdrive.py   resumable Drive uploader
scripts/compress_output_videos.py     verified CRF transcoding without implicit deletion
docs/PIPELINE_AND_WORKING_STATUS.md   detailed audit/status document
tests/                                deterministic unit tests only
```

## Setup

```powershell
python -m pip install -r requirements.txt
python scripts/fetch_official_yolo_weights.py
python -m pytest tests -q -p no:cacheprovider
```

The first command that uses a missing official YOLO weight downloads it from Ultralytics. Model files and credentials are ignored by Git.

## Run the complete annotated-video pipeline

Use a T4-or-newer GPU runtime for the full run:

```powershell
python scripts/render_annotated_videos.py `
  --device cuda:0 `
  --require-cuda `
  --tracker bytetrack.yaml `
  --phone-confidence 0.70 `
  --run-id privacy_yolo26_t4_v1
```

## Live webcam/RTSP pipeline

For the integrated demo, start MediaMTX (when installed), the FastAPI backend,
and the React frontend together:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_live_demo.ps1
```

Open `http://127.0.0.1:5173`. The monitor provides three real source paths:

- **Start Webcam** runs camera `0` through the backend and renders annotated MJPEG immediately.
- **Upload & Stream** processes the selected video losslessly, frame by frame, while annotated frames, telemetry, events, keyframes, and clips arrive in the UI.
- Enter an `rtsp://` URL and press Enter to process a MediaMTX/network stream.

The same page preloads three integrity-validated **Previous Sampled Runs** from
`OUTPUT/netra4d_final_20260823`. Each card plays the real annotated MP4 and
shows measured frame/person/object/event counts, downloadable JSONL evidence,
high-confidence snapshots, and the original source reference. VIDEO_01-03 also
have hash-verified, browser-compatible unannotated previews in
`OUTPUT/showcase_inputs`, selectable with the **Source Input / Annotated Output**
toggle. The selected annotated outputs are served through H.264 browser previews
in `OUTPUT/showcase_outputs`; the integrity-validated original MP4s remain
unchanged in their run directories. If a live/upload
session fails or does not publish its first frame within ten seconds, the player
automatically switches to the selected verified output instead of going blank.

MediaMTX accepts RTSP publishers at `rtsp://127.0.0.1:8554/<path>` and exposes
its local control API on port `9997`. For example, an installed FFmpeg can
publish a dataset video with:

```powershell
ffmpeg -re -i input.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/exam
```

Uploaded files use decoder back-pressure: every source frame is processed and
timestamps are derived from its frame index and FPS. Webcam/RTSP sources instead
discard stale queued frames when inference falls behind so the displayed feed
remains live. Per-frame state is atomically published under
`OUTPUT/live_sessions/<session-id>/`; durable video, metrics, JSON evidence,
snapshots, and clips are written under `OUTPUT/live_runs/<run-id>/`.

```powershell
python scripts/run_live_webcam.py `
  --source 0 `
  --device cuda:0 `
  --require-cuda `
  --record-stream
```

CBT Peak automatically warms up a workstation map from real person/monitor
observations by default. Drawing boxes is not required. A human-reviewed map is
an optional override:

```powershell
python scripts/calibrate_cbt_workstations.py --asset VIDEO_03 `
  --camera-id CAMERA_04 --calibrated-by <reviewer>
python scripts/run_live_webcam.py --source 0 --device cuda:0 --require-cuda `
  --workstation-map drishti4d/configs/workstations/VIDEO_03.yaml `
  --workstation-mode human --require-workstation-map
```

See [CBT Peak Engine](docs/CBT_PEAK_ENGINE.md). Automatic maps are labelled
`AUTOMATIC`; human maps are explicit overrides. No fixed workstation polygons
are assumed or copied between cameras.

The live-source queue capacity is two frames. Each run writes `metrics.json`,
`live_annotated.mp4`, and per-event keyframes/clips under
`outputs/live_runs/<run-id>/`.

## Review and tune the 128 baseline events

The verified replay is loaded into `outputs/netra4d_events.sqlite` as 128
`PENDING` records: 78 low-hand proxies, 45 other-screen proxies, and 5
interaction proxies. No true/false decision is fabricated.

```powershell
python scripts/review_events.py --reviewer <your-name>
python scripts/analyze_reviewed_events.py
python -m uvicorn drishti.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

The analyzer remains `ready: false` until a human marks events `CONFIRMED` or
`FALSE_POSITIVE`. See [the stable event contract](docs/EVENT_CONTRACT.md).

## Real chit/paper dataset and detector

```powershell
python scripts/label_chit_examples.py --asset VIDEO_01
python scripts/train_chit_detector.py datasets/chit_reviewed/dataset.yaml --device cuda:0
```

Only explicitly human-boxed examples or explicitly marked negatives are saved.
The full renderer accepts the resulting checkpoint through `--chit-model`.

Every successful run has this structure:

```text
outputs/full_video_runs/<run-id>/
  run.json
  VIDEO_01/ ... VIDEO_05/
    VIDEO_XX_annotated.mp4
    detections.jsonl
    phone_detection_frames/       # only when qualifying detections exist
    summary.json
    COMPLETE.json
```

`COMPLETE.json` is the upload gate. `INCOMPLETE.txt` means the run failed or was interrupted and must not be published as evidence.

## Kaggle GPU: repeatable full-run lifecycle

The orchestration script uses the local `kaggle.json` only through environment variables; it never prints, commits, or submits the key. It submits a private CUDA-required kernel, waits for completion, downloads actual output, validates all five completion markers, then creates/versions the private results dataset.

```powershell
python scripts/run_kaggle_gpu_pipeline.py `
  --run-id kaggle_gpu_privacy_yolo26_t4_v1
```

Kaggle may allocate a different GPU than the UI selection. Always use the device recorded in `run.json`; the existing completed run records a Tesla P100. The entrypoint rejects unavailable CUDA and never silently falls back to CPU.

To compare two validated complete runs:

```powershell
python scripts/compare_verified_runs.py `
  outputs/full_video_runs/<baseline-run> `
  outputs/kaggle_gpu_downloads/<new-run>/<new-run> `
  --output outputs/run_comparisons/<new-run>.json
```

## Measured YOLO benchmark

Run both object models on identical real frames:

```powershell
python scripts/benchmark_detectors.py `
  --asset VIDEO_02 `
  --max-frames 50 `
  --sample-every 10 `
  --warmup-frames 1 `
  --device cuda:0 `
  --require-cuda
```

The current bounded CPU evidence run used five VIDEO_02 frames (0, 100, 200, 300, 400), excluding one warm-up frame per model:

| Model | Mean inference | Raw detections | Raw phone detections |
|---|---:|---:|---:|
| YOLO11n | 44.91 ms | 27 | 0 |
| YOLO26n | 44.72 ms | 27 | 0 |

See [the full benchmark evidence](outputs/benchmarks/video02_yolo11_vs_yolo26_cpu_warm.json). This is a runtime/count comparison, not an accuracy benchmark.

## Multi-HMR and SMPL-X

The supplied `models_smplx_v1_1.zip` contains real SMPL-X assets. Verify the archive without extracting it:

```powershell
python scripts/prepare_smplx_assets.py --verify-only
```

For actual 3-D reconstruction, create a compatible CUDA/Python environment for the official Multi-HMR dependency set, then use the pinned bootstrap and runner. The runner samples real frames from a Drive asset, supplies `SMPLX_NEUTRAL.npz`, invokes the official `demo.py`, and stores provenance.

```powershell
python scripts/bootstrap_multihmr_repo.py
python scripts/run_multihmr_smplx.py `
  --repo third_party/multi-hmr `
  --asset VIDEO_02 `
  --require-cuda
```

This 3-D stage is intentionally separate from the privacy-safe annotated video until it is executed and validated on the target T4 runtime.

## Upload a verified run to Google Drive

The uploader preserves the per-video directory layout, uses resumable uploads, and writes a receipt with returned Google Drive file IDs. Target folder: `1vEfaikd1G9_dDr0-fbDm01w20Mk0TI18`.

```powershell
python scripts/upload_outputs_to_gdrive.py `
  outputs/full_video_runs/<run-id> `
  --auth oauth `
  --root-name NETRA-4D-<run-id>
```

For Colab, use `--auth colab`. An upload is verified only when a receipt appears under `outputs/upload_receipts/`.

## Compress completed videos

AV1 CRF 28 is the archival default. It preserves the original resolution and
usually saves substantial space, but AV1 encoding is CPU-bound on Tesla T4 and
P100 GPUs. The command writes a second file and verifies resolution, duration,
video readability, and size reduction before producing a receipt:

```powershell
python scripts/compress_output_videos.py `
  outputs/kaggle_gpu_downloads/yolo26n_full_t4_v1/netra4d_outputs/yolo26n_full_t4_v1 `
  --codec av1 --crf 28 --preset 8 --limit 1
```

After visually checking the one-video trial, omit `--limit` for all five. Add
`--delete-source-after-verify` only when the verified originals should actually
be removed. HEVC (`--codec h265 --crf 24 --preset medium`) is faster and more
widely practical; H.264 is the most compatible.

## Current evidence

The current retained evidence is the complete Kaggle CUDA run at `outputs/kaggle_gpu_downloads/yolo26n_full_t4_v1/netra4d_outputs/yolo26n_full_t4_v1`:

- 5 videos and 15,346 timeline frames, each in its own `VIDEO_XX` directory.
- YOLO26n object detection, YOLO26n pose, ByteTrack IDs, privacy masks, and gaze-proxy overlays.
- All five directories contain `COMPLETE.json` and an annotated MP4.
- 0 detections met the exact `cell phone >= 0.70` evidence gate, so 0 phone-review stills were written.
- Runtime metadata records CUDA on `Tesla P100-PCIE-16GB`; it is not represented as a T4 result.

The CPU-light temporal baseline can be added without rerunning YOLO:

```powershell
python scripts/annotate_baseline_events.py `
  outputs/kaggle_gpu_downloads/yolo26n_full_t4_v1/netra4d_outputs/yolo26n_full_t4_v1 `
  --run-id baseline_events_full_v1
```

The verified local replay processed all 15,346 frames and emitted 128 review
events. Labels explicitly distinguish phone model evidence from low-hand,
candidate-interaction, and other-screen behavioral proxies.

Read [PIPELINE_AND_WORKING_STATUS.md](docs/PIPELINE_AND_WORKING_STATUS.md) for file hashes, known limitations, previous Kaggle publication status, and the experimental DigitalNegatives Watcher merge.

## Video 3 unified review experiment

The bounded Video 3 profile uses shared registry inference and NETRA's candidate,
zone, temporal and state-machine components. Its trained phone model is explicitly
experimental, not release-approved. Run inference plus CVAT recall evaluation with:

```powershell
python scripts/run_unified_pipeline.py --asset VIDEO_03 --profile video03-review --allow-experimental-models --evaluate-cvat
```

See [VIDEO03_UNIFIED_REVIEW.md](docs/VIDEO03_UNIFIED_REVIEW.md) for the actual model
selection, disabled branches, outputs, and evaluation limitations.

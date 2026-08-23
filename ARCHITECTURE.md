# NETRA-4D architecture

Last updated: 2026-08-23

## Purpose

NETRA-4D is a privacy-aware examination-video evidence pipeline. It runs real
computer-vision models, preserves raw detections and provenance, produces one
annotated video directory per source video, and stores only qualifying review
frames. It does not manufacture bounding boxes, incidents, accuracy scores, or
3-D results.

The system produces review evidence. It does not automatically prove cheating.

## System overview

```text
Google Drive source manifest
          |
          v
One-video temporary materialization + source validation
          |
          v
Original decoded frame
          |
          +-----------------------+
          |                       |
          v                       v
YOLO26n object detector    YOLO26n-pose detector
          |                       |
          |                person boxes + keypoints
          |                       |
          |                 ByteTrack candidate IDs
          |                       |
          +-----------+-----------+
                      v
          raw detections.jsonl + temporal evidence
                      |
          +-----------+------------------+
          |                              |
          v                              v
cheap event proposals             privacy-safe renderer
          |                       gaze proxy -> face blur
          |                       -> ID/evidence overlay
          v                              |
flagged time windows                     v
          |                       annotated MP4
          v
Multi-HMR + SMPL-X
only for ambiguous flagged windows
          |
          v
3-D corroboration evidence, when actually executed

Per-video result folder
  -> COMPLETE.json upload gate
  -> AV1/CRF verified archival copy
  -> Kaggle output/private dataset
  -> optional Google Drive upload
```

The production reasoning target is now workstation-relative CBT context. See
[CBT_PEAK_ENGINE.md](docs/CBT_PEAK_ENGINE.md). Generic camera-relative rules
remain active during automatic-map warm-up and as the safe fallback when
stable workstation geometry cannot be recovered.

## Architectural principles

1. **Inference precedes privacy rendering.** Models receive the original frame.
   Face blur is applied only to the rendered evidence video.
2. **Raw evidence and presentation are separate.** JSONL stores model outputs;
   overlays never become the only record of a detection.
3. **No simulated fallback.** Missing CUDA, weights, input, or decoding produces
   an explicit failure or diagnostic frame, not fake boxes or keypoints.
4. **One source, one result directory.** Videos, review frames, summaries,
   hashes, and completion state remain isolated by `VIDEO_XX`.
5. **Heavy 3-D inference is conditional.** Multi-HMR/SMPL-X belongs on flagged
   windows at low sampling frequency, not on every frame of every video.
6. **Detection is not a verdict.** A phone box or behavioral proxy creates
   review evidence, not an automatic malpractice conclusion.
7. **Runtime truth comes from manifests.** The accelerator recorded in
   `run.json` is authoritative. A UI-selected T4 must not overwrite a recorded
   P100 allocation.

## Processing stages

### 1. Dataset ingestion

The source registry is `drishti4d/configs/datasets.json`. Google Drive assets
are downloaded in chunks and materialized one at a time because OpenCV requires
a seekable input for dependable MKV decoding. The ingest layer validates file
size and rejects empty files or HTML permission pages.

Source videos are temporary inputs. They are not committed to Git and are
removed from temporary staging after processing.

Primary implementation:

- `src/drishti/data/google_drive.py`
- `src/drishti/video/reader.py`
- `drishti4d/configs/datasets.json`

### 2. Person, pose, object, and candidate tracking

YOLO26n-pose produces model-derived person boxes and 17 COCO keypoints.
YOLO26n produces raw COCO object classes, boxes, and confidence values.
ByteTrack assigns temporal candidate identifiers when enabled. A non-tracked ID
must be labelled frame-local rather than presented as a persistent identity.

Primary implementation:

- `src/drishti/providers/yolo26_provider.py`
- `src/drishti/providers/object_provider.py`
- `models/weights/yolo26n-pose.pt`
- `models/weights/yolo26n.pt`

### 3. Temporal screening and event proposals

The fast path samples detection and pose and maintains evidence by stable
candidate ID. `src/drishti/temporal.py` now implements the first CPU-only
baseline: phone/person association, low-hand review, pairwise mutual-direction
review, and laptop/head-direction review with temporal persistence. Every
geometric inference is labelled as a proxy. The engine can run during primary
inference or replay the verified JSONL through
`scripts/annotate_baseline_events.py` without rerunning YOLO.

The primary path is `src/drishti/cbt/`: semantic workstation maps,
candidate binding, candidate-specific baselines, atomic evidence, competing
hypotheses, counter-evidence, four CBT sequences, cross-candidate graph state,
and uncertainty-aware specialist requests. By default it derives a per-camera
map from stable real person/monitor observations. A human-drawn map is an
optional override; the original temporal engine remains active during warm-up
and as fallback.

Each successful candidate/workstation binding also creates a one-to-one device
binding. Candidate `N` receives device ID `PC-N`; a workstation screen cannot
be assigned to two candidates in the same frame. The record distinguishes a
current `laptop`/`monitor` detection from a calibrated or automatic scene-map
screen region and preserves the corresponding confidence and method. These
bindings are emitted in frame snapshots, event evidence, graph nodes, and the
privacy-safe video overlay.

Current evidence status:

| Capability | Evidence available today | Status |
|---|---|---|
| Phone present | Exact YOLO `cell phone` box and confidence | Working |
| Phone use | Phone-to-person/wrist association | Experimental proxy |
| Chit/paper use | Custom checkpoint adapter plus hand-below-desk/lap behavior | Adapter/training implemented; no trained checkpoint yet, so current events are proxies |
| Candidate interaction | ByteTrack pair, proximity, mutual head-direction and duration | Working review proxy; not a learned classifier |
| Looking at another screen | YOLO laptop, non-owner candidate direction and duration | Working review proxy; not iris gaze or proof of copying |

The merged experimental Watcher subsystem contains temporal rules, risk
accumulation, and SQLite persistence. Its fixed seat polygons apply only to its
original 1280x720 sample camera and must not be reused across the five dataset
videos without camera-specific calibration.

Experimental location: `DI1_39_DigitalNegatives/watcher`.

### 4. Conditional Multi-HMR and SMPL-X

Multi-HMR estimates multi-person 3-D body state and uses SMPL-X assets. This is
an optional corroboration stage for ambiguous flagged windows, ideally sampled
at 0.5-1 FPS. It is too expensive for the real-time full-frame fast path.

The repository contains a strict official runner and the supplied SMPL-X model
archive, but no completed 3-D output is currently claimed.

Primary implementation:

- `scripts/bootstrap_multihmr_repo.py`
- `scripts/prepare_smplx_assets.py`
- `scripts/run_multihmr_smplx.py`
- `models_smplx_v1_1.zip`

### 5. Privacy-safe rendering

The required render order is:

```text
original frame
  -> compute pose and gaze proxy
  -> create face/head region from model keypoints
  -> apply blur to the face/head region
  -> draw gaze proxy over the blurred region
  -> draw candidate ID, person, object, and evidence annotations
```

The current gaze annotation is a 2-D head-direction proxy derived from
eye/nose geometry. It is not iris-level eye tracking. Missing or weak facial
keypoints produce `UNKNOWN` rather than a fabricated direction.

Primary implementation:

- `src/drishti/privacy.py`
- `scripts/render_annotated_videos.py`

### 5a. Real-time and backend event path

```text
webcam / RTSP -> capture thread -> bounded queue (2, drop oldest)
 -> YOLO26 pose + ByteTrack -> YOLO26 objects/crops
 -> temporal engine -> privacy renderer -> annotated stream
 -> OPEN/CLOSED events -> SQLite -> REST/WebSocket
 -> per-event keyframe and pre/post-roll clip
```

`scripts/run_live_webcam.py` measures processed FPS, dropped frames, mean
latency, and P95 latency. `src/drishti/events.py` defines the stable event
contract; `src/drishti/event_repository.py` persists it and preserves human
review decisions across re-imports. `src/drishti/api/app.py` provides live
WebSocket events, review mutation, media serving, and per-candidate timelines.
No frontend files are required or modified by this backend path.

### 6. Evidence and completion contract

Every completed run follows this layout:

```text
outputs/<runtime-or-run-id>/
  run.json
  VIDEO_01/
    VIDEO_01_annotated.mp4
    VIDEO_01_annotated.av1.mp4       optional compressed copy
    detections.jsonl
    phone_detection_frames/          only qualifying frames
    summary.json
    COMPLETE.json
  VIDEO_02/
  ...
  VIDEO_05/
  compression_av1_crf28.json         optional compression receipt
```

`COMPLETE.json` is the publication gate. `INCOMPLETE.txt` means the run cannot
be treated as completed evidence. High-confidence phone frames are written only
for the exact `cell phone` class meeting the configured threshold, currently
0.70. An empty review-frame directory is a valid negative result.

### 7. Compute and orchestration

The preferred execution environment is a Kaggle CUDA runtime. The lifecycle is:

```text
local source bundle
  -> private Kaggle notebook submission
  -> CUDA-required execution
  -> one-video-at-a-time processing
  -> Kaggle notebook output
  -> local download
  -> five-folder completion validation
  -> optional private Kaggle dataset version
```

Credentials are loaded from `kaggle.json` into the Kaggle CLI environment and
must never be printed, committed, or bundled with notebook source.

Primary implementation:

- `scripts/submit_kaggle_gpu_job.py`
- `scripts/run_kaggle_gpu_pipeline.py`
- `scripts/publish_results_to_kaggle.py`

### 8. Compression and storage

Annotated videos are initially written with OpenCV MPEG-4 for reliable frame
production. `scripts/compress_output_videos.py` creates separate AV1, HEVC, or
H.264 CRF copies and validates codec readability, resolution, duration, and
size reduction before writing a receipt.

AV1 CRF 28/preset 8 is the archival setting. T4 and P100 GPUs cannot
hardware-encode AV1, so this stage uses CPU on those runtimes. Original videos
must remain until a human has visually checked the compressed copies. No
automatic deletion is part of the normal pipeline.

### 9. Google Drive publication

`scripts/upload_outputs_to_gdrive.py` preserves the local per-video hierarchy,
uses resumable uploads, updates matching files in the same parent, and writes a
local receipt containing returned Drive file IDs. A Drive upload is not claimed
until that receipt exists.

## Current verified result

The retained five-video run is:

`outputs/kaggle_gpu_downloads/yolo26n_full_t4_v1/netra4d_outputs/yolo26n_full_t4_v1`

It contains five completed annotated videos and five verified AV1 CRF 28
copies. The run processed 15,346 timeline frames. Its runtime manifest records
CUDA on `Tesla P100-PCIE-16GB`; it is not a verified T4 run. No phone detection
met the 0.70 evidence threshold, so no high-confidence phone still was created.

## Interfaces and ownership

| Layer | Input | Output | Owner |
|---|---|---|---|
| Ingestion | Drive asset ID | Validated local video stream | `src/drishti/data` |
| Detection | Original frame | Raw object/person/keypoint evidence | `src/drishti/providers` |
| Tracking | Person detections | Candidate track IDs | renderer/ByteTrack |
| Temporal reasoning | Track histories | Review events/windows | primary screener; Watcher experimental |
| 3-D corroboration | Flagged original frames | Multi-HMR/SMPL-X evidence | `scripts/run_multihmr_smplx.py` |
| Privacy rendering | Frame + evidence | Privacy-safe annotated frame | `src/drishti/privacy.py` |
| Encoding | Annotated frames | MP4/AV1 video | renderer/compressor scripts |
| Publication | Completed run folder | Kaggle/Drive artifact + receipt | publication scripts |
| Presentation | Verified manifests/results | Review UI/API | `src/drishti/api`, `frontend` |

## Verification commands

```powershell
python -m pytest tests -q -p no:cacheprovider
python -m compileall -q src scripts DI1_39_DigitalNegatives/watcher/watcher
python scripts/prepare_smplx_assets.py --verify-only
python scripts/render_annotated_videos.py --device cuda:0 --require-cuda
```

Related detailed status: `docs/PIPELINE_AND_WORKING_STATUS.md` and
`docs/CHEATING_DETECTION_CAPABILITY_MATRIX.md`.

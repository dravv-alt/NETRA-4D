# NETRA-4D pipeline and working-status handoff

Last verified: 2026-08-22

## 1. What the project does

NETRA-4D processes examination-hall video with real computer-vision models. It
creates annotated videos containing model-produced person boxes, pose
keypoints, and raw object detections. It preserves model confidence and source
provenance so a result can be audited later.

The project does **not** currently claim that it can automatically prove
malpractice. A phone detection means that the object model detected the COCO
class `cell phone`; it does not by itself prove phone usage or misconduct.

## 2. End-to-end pipeline

```text
Five public Google Drive videos
        |
        v
Chunked one-video temporary staging
        |
        +--> SHA-256, byte count, FPS, dimensions, frame count
        |
        v
YOLO26 pose model --------> person boxes + 17 COCO keypoints
        |
YOLO26 object model ------> raw object class + confidence + box
        |
        v
Per-frame overlay + JSONL evidence
        |
        +--> annotated MP4 for every source timeline position
        +--> phone still only for exact `cell phone` confidence >= 0.70
        +--> summary.json + COMPLETE.json + hashes
        |
        v
Private Kaggle dataset + private `netra-4d` notebook
```

OpenCV requires a seekable video for reliable MKV decoding. Therefore Drive
input is downloaded in chunks to an OS/workspace temporary directory one video
at a time, processed, and deleted. This is temporary staging, not zero-copy
network streaming, and source videos are never committed to Git.

## 3. Primary verified components

### Drive dataset ingestion

- Manifest: `drishti4d/configs/datasets.json`
- Implementation: `src/drishti/data/google_drive.py`
- Contains exactly the five user-supplied Drive IDs, filenames, and expected
  byte counts.
- Rejects zero-byte downloads, HTML permission pages, and size mismatches.

### Real model providers

- Pose: `src/drishti/providers/yolo26_provider.py`
- Objects: `src/drishti/providers/object_provider.py`
- Pose weight: `yolo26n-pose.pt`
- Object weight: `yolo26n.pt`
- No fallback boxes, seeded detections, or simulated keypoints exist in these
  providers.
- Without a tracker, person identifiers are explicitly frame-local rather than
  falsely presented as durable identities.

### Sampled verification runner

- `scripts/run_verified_inference.py`
- Runs real inference on selected frames for fast smoke testing.
- Writes raw detections, model hashes, runtime/device facts, measured latency,
  and preview overlays.

### Full annotated-video runner

- `scripts/render_annotated_videos.py`
- Executes both models on every decodable frame.
- Produces one directory per video:

```text
VIDEO_XX/
  VIDEO_XX_annotated.mp4
  detections.jsonl
  summary.json
  COMPLETE.json
  phone_detection_frames/   # created only if confidence >= 0.70
```

- An output folder is upload-ready only after `COMPLETE.json` exists.
- Privacy stage order: pose facial keypoints -> 2-D gaze proxy -> face/head blur
  -> candidate and gaze-proxy annotation drawn on the blurred mask. This is not
  iris-level gaze tracking; missing/weak eye/nose keypoints produce `UNKNOWN`.
- If a source position is corrupted, the runner writes a red diagnostic frame
  saying that decoding failed and executes no inference for that position. It
  never duplicates a neighbouring scene frame or fabricates detections.

## 4. Actual completed run

Retained run directory:
`outputs/kaggle_gpu_downloads/yolo26n_full_t4_v1/netra4d_outputs/yolo26n_full_t4_v1`

| Video | Timeline frames | Frames with inference | Decode failures | Annotated MP4 bytes |
|---|---:|---:|---:|---:|
| VIDEO_01 | 2,210 | 2,198 | 12 | 154,824,999 |
| VIDEO_02 | 1,145 | 1,145 | 0 | 50,032,543 |
| VIDEO_03 | 3,282 | 3,282 | 0 | 99,706,257 |
| VIDEO_04 | 5,304 | 5,304 | 0 | 190,626,503 |
| VIDEO_05 | 3,405 | 3,405 | 0 | 115,534,392 |

Measured total: **5 videos and 15,346 timeline frames**.

All five MP4 files were reopened after rendering and their frame counts matched
the expected timeline counts. All five folders contain `COMPLETE.json`; none
contains `INCOMPLETE.txt`.

No raw `cell phone` detection reached the configured high-confidence threshold
of 0.70. Consequently, **zero phone still images were saved**. This is a real
negative result, not a missing-output fallback.

The run used PyTorch 2.7.1+cu118 with CUDA on a `Tesla P100-PCIE-16GB` allocated
by Kaggle. It must not be described as a T4 result merely because T4 was
selected in the UI.

## 5. Kaggle storage and notebook

- Private dataset:
  `https://www.kaggle.com/datasets/dravvalt/netra-4d-verified-results`
- Private notebook:
  `https://www.kaggle.com/code/dravvalt/netra-4d`
- Notebook version 2 reached Kaggle status `complete`.
- Kaggle readback listed all five annotated videos, five JSONL detection logs,
  five summaries, and five completion markers.
- Publishing implementation: `scripts/publish_results_to_kaggle.py`.
- GPU-job bundling implementation: `scripts/submit_kaggle_gpu_job.py`.

`kaggle.json` is ignored by Git. Its key is loaded only into the Kaggle CLI
process environment and is never printed or uploaded.

## 6. Google Drive output upload

- Implementation: `scripts/upload_outputs_to_gdrive.py`
- Default destination folder ID:
  `1vEfaikd1G9_dDr0-fbDm01w20Mk0TI18`
- Preserves the local directory tree, uses resumable media uploads, updates a
  same-named file in the same parent instead of creating duplicates, and writes
  a receipt containing Drive file IDs.
- Code and tests are working, but the completed 855 MB run has not been claimed
  as uploaded to Google Drive because this workspace still lacks Google OAuth
  write credentials/a connected upload tool.

## 7. Merged DigitalNegatives Watcher subsystem

Location: `DI1_39_DigitalNegatives/watcher`

The incoming subsystem adds genuine YOLOv8 detection/pose, video enhancement,
keypoint smoothing, tracking, seat association, temporal features, object-hand
fusion, configurable rules, heuristic risk accumulation, alerts, and SQLite
persistence.

It is currently labelled **experimental**, because:

- seat polygons are fixed calibration for one incoming sample camera;
- risk and alert thresholds are hand-configured heuristics;
- no reviewed ground-truth evaluation proves their precision or recall;
- historical DB/log/image outputs were supplied but were not reproduced by the
  verified pipeline.

During integration, chair-as-human test tracking was removed, `book` stopped
being treated as a chit proxy, phone fusion was restricted to the exact phone
class, and phone-rule confidence was raised from 0.15 to 0.70.

## 8. Frontend and API

- API: `src/drishti/api/app.py`
- Server launcher: `scripts/run_server.py`
- Frontend: `frontend/`

The frontend is intended to display verified run metadata and outputs. It must
not display prefilled incidents, fake accuracy values, or simulated risk as if
they came from the real video run.

## 9. What is working now

- Five-file Drive manifest and temporary chunked ingestion.
- Real YOLO pose and object inference.
- Full annotated MP4 production.
- Raw per-frame JSONL evidence.
- High-confidence-only phone-still gate.
- Model/source/output hashing and completion markers.
- Strict CUDA requirement option for Colab/Kaggle.
- Complete five-video CUDA execution and completion-marker validation.
- Private Kaggle dataset upload and completed notebook.
- Credential-safe Kaggle publisher.
- Resumable Google Drive upload code.
- Experimental Watcher source, enhancement, temporal/risk, and SQLite modules.
- Primary CPU-only temporal baseline with honest review-proxy overlays, JSONL events, and first-event keyframes.
- Evidence-replay renderer that adds the baseline to verified outputs without rerunning YOLO.

## 10. What is not yet proven or production-ready

- Automatic malpractice classification.
- Accuracy, precision, recall, F1, or false-positive rate.
- Validated phone *usage* classification; current evidence is phone presence plus an experimental hand/person association.
- A trained chit/paper detector; hand-below-desk is only a concealment proxy.
- Validated pairwise candidate-interaction classification; the current implementation is a geometric review proxy.
- Validated gaze-to-neighbour-screen attribution; current implementation combines a YOLO laptop with a 2-D head-direction proxy.
- Multi-HMR/SMPL-X 3D reconstruction.
- Cross-camera identity or biometrics.
- General seat mapping across all five videos.
- Validated Watcher alert/risk thresholds.
- Successful upload receipt for the target Google Drive folder.
- A verified T4 run; the completed Kaggle allocation recorded a P100.

## 11. Verification commands

```powershell
python -m pytest tests -q -p no:cacheprovider
python -m compileall -q src scripts DI1_39_DigitalNegatives/watcher/watcher
python scripts/run_verified_inference.py --max-frames 3
python scripts/render_annotated_videos.py --device auto --phone-confidence 0.70
```

To guarantee GPU execution in Colab/Kaggle:

```powershell
python scripts/render_annotated_videos.py --device cuda:0 --require-cuda
```

The command stops if CUDA is unavailable; it never silently replaces GPU
execution with CPU or simulated output.

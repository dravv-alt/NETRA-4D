# NETRA-4D experiment ledger

This is the repository index for reproducible experiments and engineering
milestones.  It records what question was asked, where the immutable inputs and
write-once outputs live, what was measured, and what must **not** be inferred
from the result.  It is deliberately an evidence ledger, not a claim ledger.

## Evidence policy

- Video and model binaries are excluded from Git by design.  Their paths,
  hashes, manifests, and commands are recorded below.
- The original Video 3 CVAT XML is immutable.  Its SHA-256 is
  `301620001d90d0f2fb06deb3535e98dc67c6c641f7a10e84b5b01c2db1452382`.
- A missing CVAT phone box is `UNKNOWN`, never an automatic negative label.
- Video 3 is a development/diagnostic session.  Same-video training or
  calibration results are not held-out generalization results.
- No entry establishes a cheating verdict or behavioural-event accuracy unless
  it explicitly names compatible human ground truth.

## Dataset lineage

| ID | Purpose | Inputs / artifact | Status |
| --- | --- | --- | --- |
| D0 | Immutable manual source | `MANUAL_ANNOTATIONS(CVAT)/VIDEO_3_EXAM_HALL_PHONE_USE/annotations.xml` | Raw CVAT source; never modify automatically. |
| D1 | Bounded Video 3 annotation derivative | `datasets/video03_0_2199_derivative/` | Frames 0–2199 only; derived manifest records the source relationship. |
| D2 | Review-refined annotation iterations | `datasets/video03_refined_from_human_review_*` | Review-derived artifacts; preserve as distinct versions. |
| D3 | One-physical-phone refinement | `datasets/video03_refined_single_phone_20261003/` | Tracks are visibility segments for one physical phone, not two devices. |

## Completed experiments and runs

| ID | Hypothesis / question | Reproducible implementation | Run / result | Boundary |
| --- | --- | --- | --- | --- |
| E0 | Can a single canonical Video 3 review pipeline preserve raw evidence and explicit provenance? | `scripts/run_unified_pipeline.py`, `src/drishti/release/`, `configs/model_registry.yaml` | `outputs/unified_pipeline/video03_review_verified_20261003`; 2,200 frames, CUDA; phone recall 473/500 (94.6%), mean matched IoU 0.9138; person recall 2950/4205 (70.15%). | Phone model was trained on Video 3; precision/F1/mAP/FP-hour unavailable before exhaustive review. |
| E1 | Can unknown phone detections be reviewed without treating missing annotation as a false positive? | `scripts/cluster_phone_unknowns.py`, `build_video03_human_review.py`, `serve_video03_human_review.py` | 1,323 raw proposals; 849 non-CVAT proposals across 846 frames grouped into 22 temporal clusters. | Review outcomes define future ground truth; they are not training labels. |
| E2 | Does full-frame plus person-context phone inference produce auditable evidence? | `src/drishti/providers/phone_review_provider.py`, `phone_crop.py`, canonical release service | Evidence preserves crop, association, checkpoint, and frame provenance. | Detection evidence only; not phone use or cheating. |
| E3 | Does universal adaptive preprocessing improve the canonical detector? | `src/drishti/preprocessing/`, `configs/preprocessing_video03_comparison_fast.yaml` | `video03_review_preprocessed_fast_gpu_20261004_run2`: phone recall 95.8% vs 94.6%; person recall 67.85% vs 70.15%; runtime +19.2%. | Conditional phone specialist only; not a universal preprocessing stage. |
| E4 | Is the current head-direction proxy a coordinate/sign issue? | `scripts/evaluate_head_direction_proxy.py`, `export_head_direction_intermediates.py`, `run_head_direction_diagnostic.py` | Baseline accuracy 5.40%, macro F1 4.54% on 1,853 matched samples. Sign inversions did not repair it; x/y semantic-swap diagnostic reached 42.15% accuracy but only 16.27% macro F1. | Likely semantic/reference and proxy limitations; do not calibrate/retrain from this result alone. |
| E5 | Can specialist work be scheduled by sustained system quality rather than every frame? | `src/drishti/engine/adaptive_escalation.py`, `configs/specialist_escalation.yaml`, `src/drishti/experimental/video03_full.py` | 180-frame smoke: 285 observations, 11 CUDA RTMPose invocations; first escalation at frame 60, cooldown at frame 71. | Scheduling evidence only; no behavioural accuracy claim. |
| E6 | Can an LSTM operate without inventing a behavioural target? | `src/drishti/experimental/temporal_lstm.py`, `scripts/smoke_temporal_lstm.py` | Offline smoke produced 154 evidence samples and 80 contiguous windows; output dimension 12. | Untrained, offline next-observable-evidence boundary; not canonical inference or an accuracy result. |
| E7 | Can classroom geometry be represented without fabricating real-world physics? | `src/drishti/engine/contextual_physics.py` | `outputs/unified_pipeline/video03_contextual_physics_smoke_20261006`; emits `IMAGE_PLANE_ONLY` and `UNKNOWN_UNCALIBRATED` absent human calibration. | No metric or physical-world claim without calibration. |
| E8 | Can an experimental full evidence pipeline retain optional component provenance? | `scripts/run_video03_full_experimental.py`, `docs/full_pipeline_activation_results_20261003.md` | 2,200-frame augmentation replay recorded sampled RTMPose/hands, interaction, temporal, hypotheses, state and fusion outputs. | 3D remains `PENDING_ADAPTER_VALIDATION`; no hand/interaction/behaviour accuracy labels exist. |

## Current engineering hypotheses

| ID | Hypothesis | Test required | Status |
| --- | --- | --- | --- |
| H1 | RTSP must use a bounded live worker so display latency cannot grow without bound. | Fixed local source → MediaMTX → live worker GPU benchmark with processed FPS, dropped frames, p50/p95 latency, GPU utilisation and VRAM. | Planned; frontend runtime dependency must be repaired first. |
| H2 | RTMPose can add useful pose evidence only when its crop/frame transform and track association are retained. | A0/A1 matched-track ablation with agreement, missing-keypoint rate and jitter. | Experimental; CUDA smoke exists, contribution unmeasured. |
| H3 | A better head direction representation needs explicit subject-relative annotation semantics and held-out validation. | Reproducible instrumentation replay, semantic convention confirmation, then a held-out session. | Instrumentation work exists; no calibration or retraining approved. |
| H4 | Temporal fusion may help next-observable-evidence prediction, not misconduct inference. | Persistence baseline vs trained LSTM on a held-out session. | Offline boundary only. |

## Commands and reports

- Canonical Video 3 profile: see `docs/VIDEO03_UNIFIED_REVIEW.md`.
- Phone-review workflow: see `docs/VIDEO03_PHONE_HUMAN_REVIEW.md`.
- Full optional-component capability gate: see `docs/full_pipeline_activation_plan.md`.
- Head-direction findings: see `docs/head_direction_diagnostic.md`.
- Preprocessing ablation: see `docs/video03_adaptive_preprocessing_ablation.md`.
- Specialist scheduling and LSTM boundary: see `docs/adaptive_specialist_escalation.md`.

## Change-control rule

Each source feature, experiment harness, annotation artifact, or documentation
report is committed independently.  A commit must name its hypothesis/purpose,
tests or run evidence, and limitations.  Generated media remains in the
write-once `outputs/` or `OUTPUT/` trees and is referenced by this ledger rather
than committed as opaque binaries.

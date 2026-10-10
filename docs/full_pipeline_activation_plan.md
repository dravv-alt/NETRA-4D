# DrishtiAI full behaviour-aware pipeline activation plan

Status: **inventory and experiment design only**. This document is the required
pre-implementation gate for `video03-full-experimental`. No optional model is
enabled by this plan, no ground truth is edited, and the verified baseline run is
not overwritten.

Audit date: 2026-10-03  
Repository HEAD: `f93dcc84a9cdca7be7ba5b844119611b9899b8b6`  
Working tree: dirty; the commit is recorded for provenance, but current local
files include uncommitted pipeline/data work. Every experiment must record both
the HEAD and a dirty-tree state/fingerprint.

## Safety and reference boundary

The reference is:

```text
profile: video03-review
source: input/videos/VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4
frames: 0–2199 (2,200 frames)
resolution: 1280×720
fps: 25
duration: 88 seconds
labels: outputs/video03_cvat_experiment/dataset/trusted_boxes.jsonl
raw CVAT XML SHA-256: 301620001d90d0f2fb06deb3535e98dc67c6c641f7a10e84b5b01c2db1452382
reference run: outputs/unified_pipeline/video03_review_verified_20261003
```

The XML is immutable. CVAT absence is UNKNOWN, never a negative. The two CVAT
phone tracks (3 and 4) are visibility segments of one physical phone. Video 3 is
a same-session development diagnostic: the phone checkpoint was trained from this
video and cannot support held-out generalization claims.

The reference remains runnable with:

```powershell
python scripts/run_unified_pipeline.py --asset VIDEO_03 --profile video03-review --allow-experimental-models --output-dir <new-run>
```

The optional path must fail closed on missing/incompatible models, use a new output
directory, preserve raw outputs, and label all extra evidence as experimental,
uncalibrated review evidence. No path may emit a cheating probability or a binary
cheating result.

## Inventory of components

| Component | Current implementation | Checkpoint / dependency | Input | Output currently available | Integration point | Validation requirement | Evaluation metric | Compute cost | Status / blocker |
|---|---|---|---|---|---|---|---|---|---|
| Baseline person pose | `YOLO26PoseProvider` with Ultralytics tracking; Video 3 unified adapter | `models/weights/yolo26n-pose.pt`, SHA-256 `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` | Original BGR frame | Person box, 17 COCO keypoints, confidence, detector-local track ID | `CanonicalInferenceService` → `UnifiedReviewEngine` | Existing trusted run and complete-frame checks | Person box IoU recall is available; precision/F1/mAP are not because negatives are incomplete. Track coverage, fragmentation and ID-switch audit required | CUDA; reference run included pose in 2.92 processed FPS end-to-end | **Reference / working**. Student-0 recall is only 53.99%; no automatic GT correction |
| Generic objects | `ObjectDisambiguationProvider` | `models/weights/yolo26n.pt`, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Original frame + current person observations | COCO object boxes, raw class/confidence, nearest-person association | Canonical service | Class map and raw provenance already checked; exhaustive object labels required for precision | Class-specific recall/IoU where labels exist; association accuracy only after reviewed samples | CUDA; runs every frame in baseline | **Reference / working as raw object evidence**. Not a phone-use or behaviour classifier |
| Video-03 phone detector | `PhoneReviewProvider`, full frame + current/held detector-person crops; one-second ROI proposal memory | `outputs/training/video03_phone_continued_100epochs_20261002/fit/weights/last.pt`, SHA-256 `12c26f587f3ee6784330e864c3426543cb95a087f4480a5a02d70d10a95c283d`; class map exactly `{phone}` | Original frame + detector-produced person boxes | `MOBILE_PHONE` box, raw confidence, crop view/region/age, source person association | Canonical service → temporal phone review events | Independent camera/session and human review of unknown proposals; no training on unknown queue | Current same-video diagnostic: 473/500 = 94.6% IoU≥.5, mean matched IoU .9138. Precision/F1/mAP/FP-hour unavailable | Dominant reference cost; full 2,200-frame run 752.70 sec | **Reference experimental evidence**; registry remains `release_approved: false` |
| RTMPose | `RTMPoseONNXRunner` + `RTMPoseAdapter` in `src/drishti/engine/specialists.py` | `models/weights/rtmpose_m/rtmpose_m.onnx`, SHA-256 `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`; ONNX Runtime 1.21.0 | Candidate crop BGR | 17 SimCC keypoints normalized to the crop; confidence proxy | Existing Stage-2 `EscalationGate` callback only; no full-frame/person-track comparison output | Map crop coordinates back to frame; associate to the same YOLO track; preserve both outputs; validate temporal alignment and crop transform | Keypoint agreement/error, missing-keypoint rate, temporal jitter, raw-only vs RTMPose-only vs fused person coverage; no “better” claim without labels | CPUExecutionProvider only (`['CPUExecutionProvider']`); likely slower than CUDA baseline; benchmark per crop/frame | **Available but not activated in Video 3 profile**. No `pose_comparison.jsonl`; no person-level ground-truth keypoints |
| MediaPipe hands | `MediaPipeHandRunner` using official Tasks Hand Landmarker | `models/weights/hand_landmarker.task`, SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`; MediaPipe 1.0.1 available | Wrist/hand crop BGR | 21 landmarks, handedness side, confidence, palm centre, z coordinate | Existing Stage-2 escalation callback; `VisionObservation`/feature classes can consume detailed hand observations | Implement left/right crop transforms, explicit UNKNOWN when side/association is ambiguous, and frame-level hand evidence output | No hand labels exist. First metric is reviewed-sample coverage/association agreement and runtime; precision/recall only after hand labels | CPU Tasks runtime; targeted windows only | **Available but unactivated**. Current callback can influence legacy suspicion scoring; must be separated into evidence-only output |
| Hand geometry/features | `BiomechanicalFeatureExtractor` has wrist/lap/object distance and optional detailed-hand feature extraction | No extra checkpoint beyond MediaPipe task | Canonical observation + optional hand landmarks + objects | Ratios such as hand convergence, below-lap, nearest-object class | `PrimitiveEventDetector`, temporal engine | Test coordinate semantics and missing-hand propagation; avoid default confidence 1.0 when no hand model evidence | Reviewed hand-evidence agreement; temporal stability and unknown rate | Low after hand inference | **Partial implementation**. No `hand_evidence.jsonl`; current default features can look more certain than evidence |
| Interaction graph | `src/drishti/cbt/reasoner.py` has workstation/candidate graph snapshots; object provider has nearest-person association | No new checkpoint; camera scene calibration required | Tracked people, objects, keypoints, candidate/seat map | Legacy graph/event snapshots; not emitted by `video03-review` | Would be a new evidence adapter after identity/zones | Define relation schema; explicit UNKNOWN for ambiguous association; no hard-coded Video 3 seats; reviewed pair/object samples required | Relation coverage/association agreement and temporal duration; event precision/recall only after temporal labels | Low/medium deterministic cost | **Partial/legacy only**. No `interaction_graph.jsonl` in reference run; fixed seat polygons cannot transfer across cameras |
| Temporal evidence | `TemporalBufferManager`, `PrimitiveEventDetector`, `PersistenceDetector`, `RepetitionDetector` | No checkpoint | Canonical observations, object relations, timestamps | Primitive events, persistence/repetition patterns, review events | `UnifiedReviewEngine` and baseline temporal engine | Emit explicit per-track features: persistence, transitions, duration, frequency, spacing, occlusion; eliminate synthetic events | Event/window metrics only with temporal episode labels; otherwise coverage/qualitative review | Low/medium CPU | **Partial working layer**. Reference reasoning contains proxies but not the requested formal evidence record |
| Competing hypotheses | `CompetingHypothesisEngine` | No checkpoint | Temporal pattern + context flags | H1–H5 distribution, supporting/counter evidence, uncertainty proxy | `UnifiedReviewEngine` / state machine | Replace ambiguous “suspicion” naming in experiment outputs; preserve missing/conflicting evidence and provenance | Human agreement with hypothesis annotations; calibration only for an explicitly defined internal score, never cheating probability | Low CPU | **Implemented but not independently validated**. Existing score is heuristic and uncalibrated |
| Behavioural state machine | `CandidateStateMachine` | No checkpoint | `EvidenceEvaluationResult` | NORMAL/OBSERVING/SUSPICIOUS/INVESTIGATING/REVIEW/HIGH_PRIORITY_REVIEW/INSUFFICIENT_EVIDENCE/COOLDOWN | `UnifiedReviewEngine` | Add explicit evidence gates for requested states and prevent single-cue transitions; retain transitions/reasons | State-transition agreement on temporal annotations; duration/coverage; no cheating accuracy | Low CPU | **Implemented, but current states/thresholds are proxy policy**. Must not be presented as misconduct classification |
| Evidence fusion | `EvidenceArbitrator` and hypothesis engine exist; no canonical vector/JSONL contract | No checkpoint | Pose, hand, object, interaction, temporal, occlusion and contradictory evidence | Legacy arbitrated state / heuristic score | New `EvidenceFusion` adapter before state machine | Define schema with present/missing/conflicting evidence, component versions and status; no scalar “risk” | Evidence completeness, conflict detection, human-review agreement; no behavioural accuracy without labels | Low CPU | **Not complete for requested output**. No `evidence_fusion.jsonl`; `EvidenceArbitrator` uses heuristic confidence/state |
| Multi-HMR / SMPL-X | Multi-HMR model loader and standalone runners; checkpoint loads with current Python/Torch | `models/multiHMR/multiHMR_672_S.pt`, SHA-256 `60124549867146ae460045f68f5ae9bb9d1f0cfac3aa5261d69c3db1d2247b5b`; SMPL-X neutral/male/female assets present | Full frame or sampled image | Mesh/joints/translation from model; current code renders overlays or returns specialist measurements | Legacy unified/fused runner and escalation scaffold; not consumed by verified profile | Run representative flagged cases only; preserve association to YOLO tracks; quality/failed-inference fields; test current CUDA memory and forward pass | 3D/2D agreement, reconstruction quality, runtime/GPU memory, reviewed ambiguity resolution; no “improvement” without labelled ambiguity set | Heavy CUDA; legacy path reserves 2.2–2.5 GB scratch and is unsuitable for blind full-video use on 6 GB GPU | **Checkpoint/loadable, integration incomplete**. `trimesh` and `pyrender` are absent; no `3d_pose_evidence.jsonl`; no measured Video 3 output |
| Legacy custom phone | Old `yolo26m_phone` checkpoint | `models/weights/yolo26m_phone.pt`, SHA-256 `af69bbdbdcf4595dcec24bd6e072895ba9e4e083d47cf1e42b1203b32d641e52`; classes `6`, `phone`, `undefined` | Candidate crop/full frame in legacy runner | Raw custom boxes | Legacy unified/fused/situational runners | Class semantics and held-out evaluation; do not mix with Video-03 phone-only model | Per-class metrics on independent labels; same-video fit not sufficient | CUDA, large YOLO model | **Quarantined**. Not selected by reference profile |
| Chit detector | Legacy detector/provider paths | `models/weights/yolo26m_chit.pt`, SHA-256 `48731d1a9364578664107e00d735bb833af678312b006c68ca340ea29ba4b150`; class `chit-paper` | Candidate/hand crops | Raw paper proposal | Legacy escalation/object provider | Exhaustive paper/chit box labels + reviewed hard negatives; association to hand | Box recall/precision/mAP only with coverage; event metrics need episodes | CUDA, large YOLO model | **Quarantined**. No trusted Video 3 chit labels |
| Cheating posture detector | Legacy custom model | `models/weights/yolo26m_cheating.pt`, SHA-256 `3443f5af236eae98f355d3f435f0ddeedcf91abd2967e4b0c1faa5fe9e0693c9`; classes `person`, `students_cheating`, `students_not_cheating` | Candidate crop | Raw posture classes | Legacy unified runner | Independent posture labels; explicitly separate posture from misconduct | Per-class detection/classification metrics on held-out sessions; never cheating accuracy | CUDA, large YOLO model | **Blocked/quarantined**. Must not feed misconduct score |
| Multimodal exam detector | Legacy custom model | `models/weights/yolo26m_exam_multimodal.pt`, SHA-256 `3aae60707b7a71c4449630850824725efca20ef98728c68e35f515279b668891`; classes `chit-paper`, `hand-signs`, `head_movement`, `mobilephone` | Candidate crop | Raw class proposals | Legacy unified runner | Class-specific held-out labels and class-map audit | Per-class box metrics; no behavioural verdict | CUDA, large YOLO model | **Blocked/quarantined**. Labels mix observable objects/movements but are not behaviour ground truth |
| Enhancement | `PreprocessingPipeline` exists | `configs/preprocessing.yaml` | Frame/crop | Enhanced image and quality report | Legacy paths | A/B experiment must keep raw input as control and record transform config/hash | Detector recall/IoU and runtime delta on same labelled windows; never assume enhancement helps | CPU/GPU preprocessing cost | **Disabled in reference**. Any use changes input distribution and needs its own ablation |

## Dependency and checkpoint checks performed

The current environment reported:

```text
Python: local python 3.12
Torch: 2.4.0+cu124; CUDA available; CUDA 12.4
Ultralytics: 8.4.126
ONNX Runtime: 1.21.0; providers AzureExecutionProvider, CPUExecutionProvider
MediaPipe: 1.0.1
SMPL-X: importable
trimesh: unavailable
pyrender: unavailable
```

The RTMPose runner initialized successfully, but only CPU ONNX execution is
available. The MediaPipe hand landmarker initialized successfully. The Multi-HMR
checkpoint loaded into its native `third_party/multi-hmr` `Model` with 48 missing
and 5 unexpected state-dict keys under `strict=False`; this is a load diagnostic,
not a completed forward/video validation. The checkpoint is not an Ultralytics
checkpoint and must not be loaded through `YOLO`.

The repository contains `models/weights/rtmpose_m/rtmpose_m.onnx`,
`models/weights/hand_landmarker.task`, `models/multiHMR/multiHMR_672_S.pt`, and
SMPL-X assets. It does not contain a supported temporal/behaviour dataset for
head direction, gaze target, hand activity, paper/chit use, or interaction.

## Required output contracts

Each experimental run must create a new directory and include:

```text
manifest.json
config.json
source_provenance.json
model_provenance.json
runtime.json
metrics.json              # null/unavailable with an explicit reason when needed
baseline_reference.json   # A0 comparison, never overwritten
pose_comparison.jsonl     # when RTMPose is enabled
hand_evidence.jsonl       # when hands are enabled
interaction_graph.jsonl   # when interaction graph is enabled
3d_pose_evidence.jsonl    # when 3D is enabled
temporal_evidence.jsonl
hypotheses.jsonl
state_transitions.jsonl
evidence_fusion.jsonl     # when fusion is enabled
review_queue.jsonl
annotated.mp4             # optional derived visualization
```

Every record and manifest must carry source hash, raw CVAT hash/reference label
hash, checkpoint hash, configuration hash, UTC timestamp, git HEAD, dirty-tree
indicator/fingerprint, frame range, enabled/disabled components, runtime and GPU
memory. Run directories are write-once; an existing directory is an error.

## Controlled ablations

The implementation should use one runner and one provider contract. A run declares
the following profiles; optional components never silently turn on:

| Run | Enabled components | Primary question | Required output/metrics |
|---|---|---|---|
| A0 | `video03-review` baseline | Reference behaviour and cost | Existing baseline measurements, hashes, runtime |
| A1 | A0 + RTMPose | Does an independent pose source disagree/usefully cover misses? | `pose_comparison.jsonl`; agreement, missing rate, jitter, raw-vs-retained coverage, runtime |
| A2 | A0 + MediaPipe hands | What hand landmarks/object distances are observable? | `hand_evidence.jsonl`; reviewed coverage/association and runtime; no hand accuracy claim |
| A3 | A0 + interaction graph | Can object/near-neighbour relations be represented reproducibly? | `interaction_graph.jsonl`; reviewed relation agreement, unknown rate, duration |
| A4 | A0 + sampled Multi-HMR/SMPL-X | Does 3D add evidence on selected ambiguous windows? | `3d_pose_evidence.jsonl`; 2D/3D agreement, quality, runtime/GPU memory |
| A5 | A0 + formal temporal evidence | Are persistence, transitions and repetition reproducible? | `temporal_evidence.jsonl`; feature continuity and labelled episode metrics when available |
| A6 | A0 + evidence vector/fusion | Can present/missing/conflicting evidence be audited? | `evidence_fusion.jsonl`; completeness/conflict tests and human audit |
| A7 | A0 + all independently enabled experimental branches | What is the combined cost/evidence profile? | All contracts plus per-component timing; not an accuracy claim |

For each A1–A7, compare against the same A0 source, frame range and reference
labels. Do not compare runs that changed thresholds, crops, frame sampling, or
source files without recording that as a confounder. The experiment runner must
support a component failure as an explicit `FAILED` or `DEGRADED` manifest, not a
successful run with swallowed exceptions.

## Implementation order after this gate

1. Add a shared experiment manifest/configuration and provider capability checks.
2. Implement A1 RTMPose comparison on a small deterministic frame sample before
   full-video execution; map crop coordinates and preserve same-track identity.
3. Implement A2 hand evidence on the same sample; all ambiguous left/right/object
   relations become UNKNOWN.
4. Implement A3 relation records using existing object/seat geometry, with no
   fixed coordinates copied from another camera.
5. Formalize A5 temporal records and A6 evidence vectors before connecting them to
   state transitions.
6. Add A4 sampled 3D only after representative windows and GPU-memory checks pass.
7. Run A1–A7 as separate new directories, evaluate available box/track metrics,
   and export qualitative/manual-review queues where labels do not exist.
8. Add the full evaluation/held-out-session split before any release approval.

No component is “better” until an ablation shows a measurable contribution on the
same task and a suitable held-out evaluation. Until then, every output is evidence
for human review, not a misconduct classification.

## Annotation extension plan

Do not modify Video 3 automatically. For future sessions, extend CVAT in separate
reviewed exports with:

| Layer | Required labels/attributes |
|---|---|
| Persons | Student, Invigilator, stable track identity, visible box, occluded/partially visible, camera/session ID |
| Objects | Phone, paper/chit, own screen, neighbour screen, keyboard/mouse, other object; visible/occluded/truncated |
| Head | `front`, `left`, `right`, `up`, `down`, `back`, `unknown`; annotator confidence/visibility |
| Gaze target | own paper/screen, neighbour, neighbour paper/screen, invigilator, board, entrance, none, unknown |
| Hands | left/right visibility, writing, holding phone, holding chit, hands hidden, reaching neighbour, normal, unknown |
| Relations | student→object `near/holding/reaching-toward`; student→student `near/attention-aligned/reaching-toward`; relation confidence and UNKNOWN |
| Episodes | start/end frame, episode type, evidence quality, occlusion intervals, reviewer status |

Temporal labels must be window-level, not just isolated frames. Use camera/session
splits, never random overlapping frames. Include hard negatives: phones visible but
not held, laptops/remote controls, paper/books, hands below desks during normal
writing, head turns to board/invigilator, occlusion, glare, border truncation,
empty seats, track fragmentation and unknown/ambiguous examples. Keep an UNKNOWN
state distinct from NOT_PRESENT and do not infer negatives from missing boxes.

Evaluation protocol: box IoU/mAP only with exhaustive frame coverage for the target
class; tracking coverage/ID switches/fragmentation per session; pose agreement and
keypoint error only where keypoint labels exist; hand/object relation agreement on
reviewed samples; temporal episode recall/precision only with explicit episode
ground truth; no reasoning/behaviour accuracy until behavioural hypotheses are
human-annotated. Reserve at least one complete camera/session as held out and do
not train any component on it before evaluation.

## Generalization split

Video 3 remains the development/evaluation diagnostic for existing phone boxes, with
the clear same-video-training limitation. A second CCTV session must be designated
held out before model training. Run A0–A7 on it with no retraining or threshold
selection on that session. Report development and held-out results separately,
including box/track/pose/association/runtime metrics and unknown/manual-review
rates. No aggregate number may merge the sessions without the split shown.


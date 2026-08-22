# DRISHTI-4D — Finalized 3-Person Work Distribution & Benchmarking Plan

## Purpose

This document freezes the independent implementation responsibilities for the three team members working on **Drishti-4D**, while ensuring that all three pipelines evaluate the same behaviours, use the same dataset protocol, produce comparable outputs, and can be benchmarked fairly.

The three independent branches are:

- **Anuj:** YOLO26 Pose + Tracking + 2D Behaviour Pipeline
- **Om:** RTMPose + Object Detection + 2D Behaviour Pipeline
- **Dravvya:** Multi-HMR + SMPL-X + 3D Behaviour Pipeline

All three branches must independently process the same raw examination videos and attempt to classify all relevant behavioural categories from the Problem Statement so that pose representation, temporal behaviour quality, runtime, robustness, and false-alert behaviour can be compared fairly.

---

## 1. Overall Architecture

```text
                         SAME RAW DATASET
                               │
                  SAME ANNOTATION PROTOCOL
                               │
                  SAME CAMERA CONFIGURATION
                               │
                    SAME BEHAVIOUR LABELS
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼

           ANUJ               OM             DRAVVYA

       YOLO26 POSE         RTMPOSE         MULTI-HMR
          +                   +                +
       TRACKING             OBJECT            SMPL-X
          +                DETECTION            +
       2D BEHAVIOUR          +             3D BEHAVIOUR
                           2D BEHAVIOUR

              │                │                │
              ▼                ▼                ▼

       anuj_events.jsonl om_events.jsonl dravvya_events.jsonl

              └────────────────┼────────────────┘
                               ▼
                        BENCHMARK ENGINE
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
          ACCURACY         TEMPORALITY        SPEED
          ROBUSTNESS       FALSE ALERTS       VRAM
          COVERAGE         EVENT IoU          LATENCY
                               │
                               ▼
                     FINAL HYBRID DRISHTI
```

---

## 2. Common Contract — Nobody Changes This Independently

All three must use the exact same behaviour names and definitions.

### B0 — `NORMAL`
Includes:
- normal writing
- thinking
- brief glance
- looking around momentarily
- stretching
- posture adjustment
- head scratching
- stationery handling
- looking toward board
- looking toward invigilator

### B1 — `REPEATED_SIDEWARD_GLANCE`
Student repeatedly looks toward one side or a neighbouring student over a temporal window. A single brief glance must not count.

### B2 — `SUSTAINED_EXCESSIVE_HEAD_TURN`
Head orientation remains substantially displaced for an abnormal duration, or repeated head-turn activity becomes excessive.

### B3 — `TORSO_ROTATION_TOWARD_NEIGHBOUR`
Student rotates the torso/body toward a neighbouring seat.

### B4 — `UNUSUAL_HAND_ACTIVITY`
Examples:
- repeated unusual hand movement
- hand extending beyond expected writing area
- possible desk or seat-boundary crossing
- possible hand-to-neighbour movement
- repeated non-writing hand behaviour

Normal writing must not trigger this class.

### B5 — `POSSIBLE_NONVERBAL_INTERACTION`
Temporal interaction between students using head/body/hand evidence, potentially reciprocal:

```text
A → B
+
B → A
```

Never output “communication confirmed.” Use “possible reciprocal/non-verbal interaction; review recommended.”

### B6 — `POSSIBLE_OBJECT_INTERACTION`
All three pipelines attempt to detect behaviour associated with handling, passing, receiving, or interacting with an object.

---

## 3. Object Taxonomy

Om owns explicit object detection:

```text
O1 MOBILE_PHONE
O2 PAPER_CHIT
O3 OTHER_PROHIBITED_OBJECT
```

`PAPER_CHIT` and generic prohibited-object classes should only be scored if the dataset provides enough usable examples or labels.

---

## 4. UNKNOWN / Abstention Contract

Each pipeline must be able to independently mark reliability by body component:

```text
HEAD     = GOOD / POOR / UNKNOWN
TORSO    = GOOD / POOR / UNKNOWN
HANDS    = GOOD / POOR / UNKNOWN
TRACK    = GOOD / POOR / UNKNOWN
CAMERA   = GOOD / DEGRADED
```

A missing or unreliable head estimate must not automatically destroy valid torso or hand evidence.

---

## 5. Shared Dataset Structure

```text
dataset/
├── raw/
│   ├── *.mkv
│   └── *.mp4
├── annotations/
│   ├── behaviour_events.csv
│   ├── tracking_gt.csv
│   ├── pose_subset/
│   └── object_gt.csv
├── camera_configs/
│   ├── CAM01.yaml
│   ├── CAM02.yaml
│   └── ...
└── benchmark_protocol.yaml
```

No person consumes another person's predictions during the independent benchmark phase.

---

## 6. Shared Camera Context

Every pipeline uses the same camera calibration:

```yaml
camera_id: CAM01

seats:
  A01:
    polygon: [...]
  A02:
    polygon: [...]
  A03:
    polygon: [...]

neighbors:
  A01: [A02]
  A02: [A01, A03]

zones:
  board:
    polygon: [...]
  invigilator:
    polygon: [...]
  entrance:
    polygon: [...]

blind_spots:
  - polygon: [...]
```

This allows the system to distinguish `head → board` from `head → neighbour A04`.

---

## 7. Shared Temporal Protocol

All three pipelines maintain approximately a **12-second rolling behavioural window** and reason using:

```text
current movement
+
duration
+
repetition
+
direction
+
neighbour relation
+
counter-evidence
+
quality
```

Behaviour must not be classified purely frame-by-frame.

---

# 8. ANUJ — YOLO26 Pose + Tracking Pipeline

## Frozen Model

Use:

```text
yolo26n-pose.pt
```

Do not silently replace YOLO26 Pose with YOLO11, YOLOv8, or another pose model.

### Responsibilities

1. `.mkv` / `.mp4` streaming video ingestion.
2. YOLO26 Pose inference.
3. Save person bounding boxes, 17 body keypoints, per-keypoint confidence, pose confidence and timestamps.
4. Benchmark **ByteTrack vs BoT-SORT**.
5. Measure:
   - IDF1
   - ID switches
   - fragmentation
   - track recovery
   - lost-track duration
   - seat stability
   - FPS
6. Map `track_id → seat_id`.
7. Derive 2D head features using nose/eyes/ears/shoulders.
8. Derive 2D torso features using shoulders/hips.
9. Derive 2D hand features using shoulder/elbow/wrist motion.
10. Maintain neighbour interaction relations `A → B`, `B → A`.
11. Apply counter-evidence for board/invigilator/brief-normal movements.
12. Independently classify every common behaviour:
   - `NORMAL`
   - `REPEATED_SIDEWARD_GLANCE`
   - `SUSTAINED_EXCESSIVE_HEAD_TURN`
   - `TORSO_ROTATION_TOWARD_NEIGHBOUR`
   - `UNUSUAL_HAND_ACTIVITY`
   - `POSSIBLE_NONVERBAL_INTERACTION`
   - `POSSIBLE_OBJECT_INTERACTION`
   - `UNKNOWN`

### Deliverables

```text
outputs/anuj/
├── raw_pose.jsonl
├── tracks.jsonl
├── behaviour_events.jsonl
├── tracked_pose_video.mp4
├── runtime.json
├── metrics.json
└── error_cases/
```

### Definition of Done

```text
RAW VIDEO
→ YOLO26 POSE
→ TRACK
→ SEAT
→ 2D FEATURES
→ ALL COMMON BEHAVIOURS
→ EVENT TIMELINES
→ JSONL
```

Plus a **ByteTrack vs BoT-SORT comparison report**.

---

# 9. OM — RTMPose + Object Detection Pipeline

Om's branch must independently run from raw video.

```text
VIDEO
 ↓
PERSON DETECTION
 ↓
RTMPOSE
 ↓
LOCAL TEMPORAL ASSOCIATION
 ↓
2D BEHAVIOUR
 ↓
OBJECT DETECTION
 ↓
OBJECT ASSOCIATION
 ↓
EVENTS
```

### Responsibilities

1. Run RTMPose on the raw videos.
2. Save 2D landmarks, coordinates, keypoint confidence, bboxes, timestamps and local person IDs.
3. Implement a local identity/association mechanism sufficient for temporal analysis.
4. Map local tracks to seat IDs using the shared camera configuration.
5. Derive head direction, duration, repetition and target zone.
6. Derive torso orientation and neighbour-directed rotation.
7. Derive wrist trajectory, velocity, acceleration and boundary crossing.
8. Independently classify all common behaviours:
   - `NORMAL`
   - `REPEATED_SIDEWARD_GLANCE`
   - `SUSTAINED_EXCESSIVE_HEAD_TURN`
   - `TORSO_ROTATION_TOWARD_NEIGHBOUR`
   - `UNUSUAL_HAND_ACTIVITY`
   - `POSSIBLE_NONVERBAL_INTERACTION`
   - `POSSIBLE_OBJECT_INTERACTION`
   - `UNKNOWN`
9. Object detection priority:
   - Mobile phone first
   - Paper/chit if supported
   - Other prohibited objects only if supported
10. Object-to-student/seat attribution using:
   - object bbox
   - seat overlap
   - person bbox
   - wrist proximity
   - temporal persistence
11. Maintain reciprocal interaction evidence and counter-evidence.

### Example Object Output

```json
{
  "object": "phone",
  "confidence": 0.92,
  "associated_seat": "A03",
  "association_confidence": 0.81
}
```

### Deliverables

```text
outputs/om/
├── raw_rtmpose.jsonl
├── local_tracks.jsonl
├── object_detections.jsonl
├── object_associations.jsonl
├── behaviour_events.jsonl
├── annotated_video.mp4
├── runtime.json
├── metrics.json
└── error_cases/
```

### Definition of Done

```text
RAW VIDEO
→ RTMPOSE
→ LOCAL TEMPORAL ASSOCIATION
→ 2D FEATURES
→ ALL COMMON BEHAVIOURS
→ OBJECT DETECTION
→ OBJECT ATTRIBUTION
→ EVENT TIMELINES
→ JSONL
```

---

# 10. DRAVVYA — Multi-HMR + SMPL-X Pipeline

This branch must not be only a mesh-visualization demo.

Its technical question is:

> Can explicit 3D human geometry improve behavioural inference under difficult camera angles, perspective distortion and ambiguous 2D posture?

### Responsibilities

1. Run Multi-HMR independently on raw frames/video.
2. Recover multiple reconstructed humans.
3. Extract usable SMPL-X information such as:
   - 3D joints
   - global orientation
   - body pose
   - head orientation
   - torso orientation
   - arms/wrists
   - person translation
   - projection information
   - fit/confidence quality
4. Maintain temporal person association independently.
5. Use projected bbox overlap, spatial location, seat anchoring and temporal continuity.
6. Calculate 3D head yaw/pitch/roll.
7. Calculate 3D torso yaw/pitch/roll.
8. Derive 3D hand/arm trajectory and neighbour-directed movement.
9. Maintain a 3D interaction graph:
   - A head → B
   - A torso → B
   - B head → A
   - B torso → A
10. Independently classify all common behaviours:
   - `NORMAL`
   - `REPEATED_SIDEWARD_GLANCE`
   - `SUSTAINED_EXCESSIVE_HEAD_TURN`
   - `TORSO_ROTATION_TOWARD_NEIGHBOUR`
   - `UNUSUAL_HAND_ACTIVITY`
   - `POSSIBLE_NONVERBAL_INTERACTION`
   - `POSSIBLE_OBJECT_INTERACTION`
   - `UNKNOWN`
11. Behaviour-specific quality:
   - `HEAD_3D = GOOD/POOR/UNKNOWN`
   - `TORSO_3D = GOOD/POOR/UNKNOWN`
   - `HAND_3D = GOOD/POOR/UNKNOWN`
12. Never fabricate precise 3D posture when fitting quality is poor.

Dravvya is not required to identify a phone as a phone. Explicit object identity belongs to Om; Dravvya must detect the corresponding interaction behaviour where possible.

### Deliverables

```text
outputs/dravvya/
├── raw_multihmr.jsonl
├── smplx_parameters/
├── temporal_tracks.jsonl
├── geometry_3d.jsonl
├── behaviour_events.jsonl
├── mesh_overlay.mp4
├── runtime.json
├── metrics.json
└── error_cases/
```

### Definition of Done

```text
RAW VIDEO
→ MULTI-HMR
→ SMPL-X
→ TEMPORAL ASSOCIATION
→ 3D HEAD / TORSO / HANDS
→ ALL COMMON BEHAVIOURS
→ 3D INTERACTION
→ EVENT TIMELINES
→ JSONL
```

---

# 11. Benchmark A — Controlled Representation Benchmark

Purpose:

> Which pose representation provides the best behavioural information?

Hold constant:

- same videos
- same timestamps
- same camera zones
- same seat maps
- same behaviour definitions
- same temporal window
- same hard-negative clips
- same validation/test split
- same evaluation code

Compare:

```text
YOLO26 Pose
vs
RTMPose
vs
Multi-HMR + SMPL-X
```

---

# 12. Benchmark B — Native End-to-End Benchmark

Each person may optimize their complete pipeline.

Compare:

> Which full pipeline gives the best deployability trade-off?

Include:

- temporal identity quality
- behaviour accuracy
- robustness
- coverage
- false alerts
- temporal precision
- runtime

---

# 13. Pose Accuracy Benchmark

Create a small manually annotated subset spanning:

- different videos
- near/far students
- camera-angle variation
- occlusion
- different poses
- varying lighting

Evaluate:

### PCK — Percentage of Correct Keypoints
How often predicted landmarks are sufficiently close to manually annotated ground truth.

### OKS — Object Keypoint Similarity
Use if annotation format supports it.

Also report:

```text
head landmark availability
shoulder availability
wrist availability
hip availability
```

---

# 14. Tracking / Temporal Association Benchmark

Report:

```text
IDF1
ID switches
fragmentation
track coverage
lost-track duration
seat-assignment accuracy
```

Anuj additionally performs the dedicated:

```text
ByteTrack vs BoT-SORT
```

benchmark.

---

# 15. Behaviour Classification Benchmark

For every behaviour:

| Behaviour | Precision | Recall | F1 |
|---|---:|---:|---:|
| Repeated side glance |  |  |  |
| Sustained head turn |  |  |  |
| Torso toward neighbour |  |  |  |
| Unusual hand activity |  |  |  |
| Possible non-verbal interaction |  |  |  |
| Possible object interaction |  |  |  |

Primary aggregate metric:

## Macro-F1

Accuracy alone must not be the primary metric.

---

# 16. Temporal Behaviour Benchmark

Evaluate not only event class but event timing.

Example:

```text
Ground truth:
41.2 → 49.8 sec

Prediction:
42.1 → 50.4 sec
```

Measure:

- Temporal IoU
- event onset error
- event end error
- duration error

---

# 17. Hard-Negative Benchmark

Create a common negative set:

```text
normal writing
normal glance
stretch
posture adjustment
head scratch
board look
invigilator look
stationery movement
```

Measure:

- False Positive Rate
- false review events / student-hour
- alerts / student-hour

---

# 18. Alert Workload Benchmark

All pipelines report:

```text
alerts / student-hour
review events / 100 students / hour
false alerts / student-hour
mean event duration
```

Also measure:

```text
original footage requiring review
vs
Drishti-prioritized footage
```

Only report measured review-time reduction.

---

# 19. UNKNOWN / Coverage Benchmark

Always report accuracy together with coverage.

Required metrics:

```text
UNKNOWN rate
valid-analysis coverage
head coverage
torso coverage
hand coverage
```

This prevents a system from appearing artificially good by abstaining too often.

---

# 20. Runtime Benchmark

Use the same:

- hardware
- video
- resolution
- sampling policy

Measure:

```text
mean inference ms/frame
P50 inference latency
P95 inference latency
effective FPS
peak GPU VRAM
peak RAM
startup/model-load time
```

Report separately:

```text
pure model inference
vs
full end-to-end pipeline
```

---

# 21. Robustness Matrix

All three pipelines must be tested under the same conditions:

| Condition | YOLO26 Pose | RTMPose | Multi-HMR + SMPL-X |
|---|---|---|---|
| Normal |  |  |  |
| Low light |  |  |  |
| Blur |  |  |  |
| Partial occlusion |  |  |  |
| Distant student |  |  |  |
| Crowded scene |  |  |  |
| Oblique camera |  |  |  |
| Track loss |  |  |  |
| Lighting change |  |  |  |

Compare:

- behaviour F1
- coverage
- latency
- UNKNOWN rate

---

# 22. Shared Event Output Schema

All three must write the same event structure:

```json
{
  "schema_version": "1.0",
  "pipeline": "YOLO26 | RTMPOSE | SMPLX",
  "video_id": "VIDEO_01",
  "camera_id": "CAM01",
  "entity": {
    "local_track_id": "T17",
    "seat_id": "A03"
  },
  "event": {
    "behaviour": "REPEATED_SIDEWARD_GLANCE",
    "start_sec": 41.2,
    "end_sec": 49.8,
    "occurrences": 4,
    "target": "A04"
  },
  "scores": {
    "behaviour_confidence": 0.86,
    "quality": 0.91
  },
  "quality": {
    "head": "GOOD",
    "torso": "GOOD",
    "hands": "POOR",
    "track": "GOOD"
  },
  "evidence": [
    "4 rightward head turns",
    "3 directed toward A04"
  ]
}
```

---

# 23. Shared Risk States

All three use:

```text
NORMAL
WATCH
REVIEW
UNKNOWN
```

Two benchmark modes:

## Fixed-Threshold Benchmark
Same thresholds for all three.

Purpose: compare representations fairly.

## Calibrated Benchmark
Each pipeline tunes thresholds on the same validation set, then runs once on the untouched test set.

Purpose: compare each pipeline at its best defensible operating point.

---

# 24. Data Split Rule

Do **not** use random-frame splitting.

Invalid:

```text
Video X frame 100 → train
Video X frame 101 → test
```

Use complete videos/sessions:

```text
whole videos/sessions → DEVELOPMENT
different videos/sessions → VALIDATION
unseen videos/sessions → TEST
```

Where possible, include at least one camera/session holdout benchmark.

Freeze the exact split only after the forensic dataset audit.

---

# 25. Final Comparison Table

| Metric | YOLO26 Pose | RTMPose | Multi-HMR + SMPL-X |
|---|---:|---:|---:|
| Pose PCK |  |  |  |
| Head coverage |  |  |  |
| Wrist coverage |  |  |  |
| Tracking/association IDF1 |  |  |  |
| Repeated glance F1 |  |  |  |
| Head-turn F1 |  |  |  |
| Torso-turn F1 |  |  |  |
| Hand-activity F1 |  |  |  |
| Interaction F1 |  |  |  |
| **Macro-F1** |  |  |  |
| Event tIoU |  |  |  |
| False alerts/student-hour |  |  |  |
| UNKNOWN rate |  |  |  |
| Valid-analysis coverage |  |  |  |
| Low-light F1 |  |  |  |
| Occlusion F1 |  |  |  |
| Oblique-angle F1 |  |  |  |
| Mean latency |  |  |  |
| P95 latency |  |  |  |
| Effective FPS |  |  |  |
| Peak VRAM |  |  |  |

Om additionally reports:

| Object Metric | RTMPose + Detector |
|---|---:|
| Phone precision |  |
| Phone recall |  |
| Phone F1 / mAP |  |
| Phone-to-seat association accuracy |  |
| Paper/chit metric if supported |  |

---

# 26. Hypotheses to Test — Not Results

Do not claim these until measured:

```text
YOLO26 Pose
→ expected fastest
→ expected high coverage
→ likely suitable primary real-time layer

RTMPose
→ may provide stronger 2D landmark precision
→ may improve head/hand refinement
→ likely more expensive than YOLO26 Pose

Multi-HMR + SMPL-X
→ may provide stronger explicit 3D orientation
→ may help ambiguous 2D geometry
→ likely highest compute cost
```

The benchmark must determine whether these hypotheses are actually true.

---

# 27. Potential Final Hybrid Architecture

If measurements justify it:

```text
                    YOLO26 POSE
               FAST PRIMARY LAYER
                       │
             confidence sufficient?
                  /           \
                YES            NO
                 │              │
                 ▼              ▼
              CONTINUE       RTMPOSE
                                 │
                         still ambiguous /
                         3D geometry useful?
                                 │
                                 ▼
                         MULTI-HMR + SMPL-X
```

Om's object-detection branch operates alongside the behavioural pipeline.

Potential fusion:

```text
camera context
+
YOLO26 evidence
+
RTMPose evidence
+
SMPL-X evidence
+
object evidence
+
interaction
+
counter-evidence
+
quality
        ↓
calibrated temporal risk
        ↓
prioritized review event
        ↓
evidence timeline
        ↓
human invigilator
```

---

# 28. Non-Negotiable Team Rules

Nobody independently changes:

- dataset split
- behaviour names
- behaviour definitions
- camera zones
- annotation schema
- temporal window definition
- benchmark metrics
- event output schema
- evaluation logic

without synchronizing the change across all three branches.

Allowed to differ:

- model internals
- feature extraction method
- local tracking implementation
- confidence estimation
- pipeline optimization
- calibrated thresholds during calibrated benchmark mode

The systems may differ technically, but the **question being evaluated must remain identical**.

---

# 29. Final Ownership Summary

| Person | Core Model | Main Responsibilities |
|---|---|---|
| **Anuj** | **YOLO26 Pose** | Pose, ByteTrack vs BoT-SORT, seat anchoring, 2D head/torso/hand behaviour, interaction, event timelines |
| **Om** | **RTMPose** | RTMPose pose pipeline, local temporal association, all common behaviours, phone/object detection, object attribution |
| **Dravvya** | **Multi-HMR + SMPL-X** | 3D reconstruction, temporal association, 3D head/torso/hand behaviour, 3D interaction, event timelines |

---

# 30. Final Goal

At the end, the team should be able to answer quantitatively:

- Which representation gives the best behavioural accuracy?
- Which model handles head, torso, hand and interaction behaviours best?
- Which model produces the fewest false alerts?
- Which model maintains the highest usable coverage?
- Which model is most robust to occlusion, low light and oblique camera views?
- What is the latency, FPS and VRAM cost of each approach?
- When is RTMPose refinement worth using?
- When is Multi-HMR + SMPL-X 3D reasoning worth its computational cost?
- Can the three branches be fused into an adaptive specialist architecture that is more efficient and reliable than running every model on every frame?

That comparison is the core technical evidence behind the final **Drishti-4D** architecture.

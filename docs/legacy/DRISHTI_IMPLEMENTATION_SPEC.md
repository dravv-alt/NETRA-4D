# Drishti-4D — Implementation & Technical Execution Specification

> **Project:** DrishtiAI Hackathon — Problem Statement #1  
> **Problem:** AI-Based Human Behaviour Monitoring Using Pose Estimation and Gesture Analysis  
> **Document purpose:** A handoff-ready implementation specification for Codex/engineering execution.  
> **Status:** Implementation baseline, not a claim of achieved performance.  
> **Last updated:** 2026-08-13

---

## 0. Executive Intent

Drishti-4D is not a frame-level “cheating classifier.” It is a **privacy-conscious, multi-person, temporal behavioural monitoring system** for examination halls.

The system must:

1. Detect and track multiple seated students from surveillance video.
2. Extract skeletal pose landmarks and movement patterns.
3. Detect and summarize sustained behaviours such as repeated sideward glances, excessive head turning, torso rotation toward neighbours, unusual hand movements, and possible non-verbal communication.
4. Detect feasible prohibited objects such as mobile phones and, if the data supports it, paper/chits or other unwanted objects.
5. Produce **behavioural risk scores**, **explainable event summaries**, evidence clips, confidence/quality information, and timelines.
6. Preserve human oversight: the system flags **review events**, never declares a student guilty of cheating.
7. Explicitly handle poor visibility, occlusion, tracking failure, camera blind spots, low light, and uncertainty through an **UNKNOWN / ABSTAIN** state rather than forcing a prediction.
8. Be structured so that the hackathon prototype can evolve into a real multi-camera deployment.

The official PS specifically emphasizes real-time multi-person monitoring, pose/gesture analysis, explainable alerts, event logs, privacy, robustness, false-positive reduction, long-duration tracking, deployment scale, and a hackathon extension based on **sustained behavioural risk rather than isolated movements**.

---

# 1. Non-Negotiable Product Contracts

## 1.1 PS Contract

Every explicit PS objective must have a mapped subsystem and a measurable test.

| PS requirement | Drishti subsystem | Evidence of completion |
|---|---|---|
| Detect multiple seated students | Primary pose/person perception | annotated video + detection metrics |
| Track students continuously | Multi-object tracker + seat anchoring | persistent IDs + ID metrics |
| Extract pose landmarks | YOLO pose baseline | per-track 17-keypoint observations |
| Head/body/hand behaviour | Temporal behaviour engine | labelled event detection |
| Mobile phone / chit / object | Object branch | object detections + event fusion |
| Real-time alerts | Event/risk engine | live or streamed event output |
| Explainable reasons | Deterministic summary engine | reason codes + evidence timeline |
| Behaviour analytics | Event store/dashboard API | JSONL/DB logs + timelines |
| Privacy | Anonymous track/seat identity | no face recognition / no name mapping |
| Robustness | Quality + abstention + specialist routing | challenge-specific test suite |
| Scale | decoupled workers + bounded state | throughput/latency benchmark |

## 1.2 Performance Contract

Every metric shown in a demo or presentation must be one of:

- `TARGET` — desired acceptance threshold, not yet achieved.
- `MEASURED` — produced by a reproducible evaluation run.
- `NOT MEASURED` — explicitly unresolved.

Never present targets as results.

## 1.3 Failure Contract

Every major failure mode must define:

```text
failure mode
→ detection signal
→ mitigation
→ fallback
→ metric
→ reproducible test
```

Example:

```text
OCCLUSION
→ low visible-keypoint ratio
→ temporal history + optional specialist
→ UNKNOWN if still unreliable
→ behaviour F1 vs. occlusion severity
→ natural + synthetic occlusion test
```

## 1.4 Deployment Contract

Prototype code must be modular enough to migrate from:

```text
OpenCV + Python + Ultralytics
```

toward:

```text
RTSP/GStreamer/DeepStream
→ batched GPU inference
→ tracking
→ Drishti behaviour engine
→ event bus
→ monitoring backend
```

Do not prematurely implement DeepStream/TensorRT in the hackathon baseline, but do not tightly couple behavioural logic to OpenCV either.

---

# 2. System Architecture

```text
                           CAMERA / VIDEO
                                │
                                ▼
                  ┌─────────────────────────┐
                  │ CAMERA HEALTH LAYER     │
                  │ fps / blur / brightness │
                  │ frozen feed / jitter    │
                  │ timestamps / coverage   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ PRIMARY PERCEPTION      │
                  │ YOLO26n-Pose            │
                  │ person + 17 keypoints   │
                  └────────────┬────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ MULTI-PERSON TRACKING     │
                 │ BoT-SORT / ByteTrack      │
                 │ + anonymous Seat Anchor   │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ QUALITY / UNCERTAINTY     │
                 │ keypoint visibility       │
                 │ track stability           │
                 │ image quality             │
                 │ occlusion                 │
                 └───────┬──────────┬────────┘
                         │          │
                  sufficient      uncertain
                         │          │
                         │          ▼
                         │   SPECIALIST ROUTER
                         │   ├─ RTMPose
                         │   ├─ MediaPipe Face
                         │   ├─ MediaPipe Hands
                         │   └─ Multi-HMR / 3-D [late]
                         │
                         └──────────────┬──────────────
                                        ▼
                           TEMPORAL TRACK STATE
                  ┌────────────────────────────────┐
                  │ head-direction features        │
                  │ torso-orientation features     │
                  │ wrist / hand motion            │
                  │ pose velocity / acceleration   │
                  │ seat + neighbour geometry      │
                  │ repetitions / persistence      │
                  │ counter-evidence               │
                  │ perception quality             │
                  └──────────────┬─────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
      BEHAVIOUR EVIDENCE                     OBJECT EVIDENCE
      repeated glance                        phone
      sustained head turn                    paper/chit [if feasible]
      neighbour torso turn                   other prohibited object
      unusual wrist/hand motion
      possible signalling
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
                         EVIDENCE FUSION
                                 │
                                 ▼
                     BEHAVIOURAL RISK ENGINE
                  temporal accumulation + decay
                  confidence / quality weighting
                  persistence + repetition
                  counter-evidence
                  hysteresis
                  UNKNOWN / ABSTAIN
                                 │
                                 ▼
                         EXPLAINABLE EVENT
                  seat/track + timestamp + reason
                  score + confidence + evidence clip
                                 │
                                 ▼
                        INVIGILATOR REVIEW
```

---

# 3. Development Strategy

## 3.1 Build one dependency layer at a time

Required order:

```text
P0 dataset audit
→ P1 pose + tracking baseline
→ P2 track-centric temporal representation
→ P3 seat anchoring + neighbour geometry
→ P4 behaviour feature extraction
→ P5 temporal evidence + risk engine
→ P6 explainable events
→ P7 object branch
→ P8 uncertainty specialists
→ P9 robustness evaluation
→ P10 dashboard / live integration
→ P11 deployment optimization
```

Do **not** implement all specialist models before the primary baseline is evaluated.

## 3.2 Gate-based execution

Each phase must produce artifacts and pass an acceptance gate before the next major phase.

Example:

```text
P1 PASS requires:
- video decoded correctly
- multiple students detected
- persistent track IDs visible
- pose keypoints serialized
- annotated video produced
- runtime statistics recorded
```

---

# 4. Repository Layout

Start lean but keep production boundaries.

```text
drishti4d/
│
├── configs/
│   ├── baseline.yaml
│   ├── behaviour.yaml
│   └── seat_map.example.yaml
│
├── data/
│   ├── raw/                 # gitignored; actual videos/dataset
│   ├── annotations/
│   ├── metadata/
│   └── processed/
│
├── outputs/
│   ├── audits/
│   ├── tracks/
│   ├── events/
│   ├── videos/
│   ├── metrics/
│   └── logs/
│
├── scripts/
│   ├── audit_dataset.py
│   ├── run_pose_tracking.py
│   ├── calibrate_seats.py
│   ├── run_behaviour.py
│   ├── evaluate.py
│   └── benchmark.py
│
├── src/drishti/
│   ├── __init__.py
│   ├── schemas.py
│   │
│   ├── video/
│   │   ├── reader.py
│   │   ├── stream.py
│   │   └── health.py
│   │
│   ├── perception/
│   │   ├── pose_tracker.py
│   │   ├── pose_quality.py
│   │   └── object_detector.py
│   │
│   ├── tracking/
│   │   ├── seat_anchor.py
│   │   └── track_quality.py
│   │
│   ├── behaviour/
│   │   ├── buffer.py
│   │   ├── features.py
│   │   ├── head.py
│   │   ├── torso.py
│   │   ├── hands.py
│   │   ├── interaction.py
│   │   ├── evidence.py
│   │   └── risk.py
│   │
│   ├── specialists/
│   │   ├── router.py
│   │   ├── mediapipe_face.py
│   │   ├── mediapipe_hands.py
│   │   ├── rtmpose.py
│   │   └── multihmr.py
│   │
│   ├── events/
│   │   ├── builder.py
│   │   ├── summarizer.py
│   │   └── store.py
│   │
│   ├── evaluation/
│   │   ├── behaviour_metrics.py
│   │   ├── tracking_metrics.py
│   │   ├── runtime_metrics.py
│   │   └── robustness.py
│   │
│   └── visualization/
│       ├── overlay.py
│       └── timeline.py
│
├── tests/
│   ├── test_schemas.py
│   ├── test_geometry.py
│   ├── test_buffer.py
│   ├── test_risk.py
│   └── test_event_builder.py
│
├── requirements-core.txt
├── requirements-specialists.txt
├── README.md
└── DRISHTI_IMPLEMENTATION_SPEC.md
```

### Immediate implementation subset

Codex should initially create only the files required for **P0–P2**, plus interfaces/stubs only where necessary.

Avoid a repository full of empty abstractions.

---

# 5. Dependency Strategy

## 5.1 Core environment

Core prototype dependencies:

```text
ultralytics
opencv-python
numpy
scipy
pandas
pydantic
pyyaml
rich
pytest
```

Optional for evaluation:

```text
scikit-learn
motmetrics    # only if it fits the chosen tracking evaluation protocol
```

Do not hard-pin arbitrary versions without checking the target machine. Record exact installed versions after the environment is verified:
x
```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

Save to:

```text
outputs/audits/environment.txt
```

## 5.2 Specialist environments

Do **not** force all specialists into the core environment.

### MediaPipe

Can live in the core environment if dependency resolution remains clean.

### MMPose / RTMPose

Prefer a separate environment or optional extra. MMPose supports pretrained 2-D human pose inference and includes RTMPose in its model zoo/inference ecosystem.

### Multi-HMR

Keep isolated. The official repository currently documents its own environment and was tested with Python 3.9/CUDA 12.1. It should be treated as an optional late-stage specialist, not a baseline dependency.

The SMPL-X model is separately licensed/downloaded and must not be silently bundled.

---

# 6. Dataset Forensic Audit — P0

We do not know the exact dataset structure in advance. The implementation must discover it rather than assume folders or annotation formats.

## 6.1 Required CLI

```bash
python scripts/audit_dataset.py \
  --root /path/to/dataset \
  --output outputs/audits/dataset_audit.json
```

## 6.2 Discover

Recursively enumerate:

- videos: `.mp4 .avi .mov .mkv .mpeg .mpg .webm`
- images: `.jpg .jpeg .png .bmp`
- annotation candidates: `.json .jsonl .csv .xml .txt .yaml .yml`
- README / label map / metadata files

## 6.3 For each video record

```json
{
  "video_id": "...",
  "path": "...",
  "readable": true,
  "codec": "...",
  "width": 1920,
  "height": 1080,
  "fps": 25.0,
  "frame_count": 12345,
  "duration_sec": 493.8,
  "size_bytes": 0,
  "sample_brightness": 0.0,
  "sample_blur_score": 0.0
}
```

Do not load full videos into RAM.

For quality statistics, sample a bounded number of frames uniformly, e.g. 20–50 per video.

## 6.4 Dataset report must answer

- How many videos?
- Total duration?
- Resolution distribution?
- FPS distribution?
- Corrupt/unreadable media?
- Annotation files present?
- Label names?
- Are labels frame-level, clip-level, track-level, or event-level?
- Are normal behaviours included?
- Are suspicious behaviours included?
- Are phone/chit labels included?
- Are camera/session identities available?
- Are there separate train/validation/test splits?

## 6.5 Behaviour coverage table

Generate:

```text
outputs/audits/behaviour_coverage.csv
```

Suggested schema:

```text
behaviour,label_exists,n_examples,n_videos,annotation_type,notes
side_glance,...
sustained_head_turn,...
torso_turn,...
unusual_hand_motion,...
nonverbal_signal,...
phone,...
paper_chit,...
normal,...
```

If the dataset does not support a behaviour, mark it `NOT SUPPORTED BY CURRENT DATASET` rather than inventing coverage.

## 6.6 Split integrity

Do not randomly split frames from the same video across train/test.

Preferred split unit:

1. exam session
2. camera/video
3. subject grouping if available and ethically appropriate

Goal: prevent temporal and scene leakage.

---

# 7. Core Data Schemas

Use typed schemas so downstream modules do not depend on Ultralytics-specific result objects.

## 7.1 Keypoint

```python
class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float
```

Coordinates should be stored in both pixel and normalized form when useful.

## 7.2 FrameObservation

```python
class FrameObservation(BaseModel):
    camera_id: str
    video_id: str
    frame_index: int
    timestamp_sec: float
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_confidence: float
    keypoints: dict[str, Keypoint]
    pose_quality: float
    seat_id: str | None = None
```

## 7.3 TrackState

```python
class TrackState(BaseModel):
    track_id: int
    seat_id: str | None
    first_seen_sec: float
    last_seen_sec: float
    track_quality: float
    visible_keypoint_ratio: float
    history_size: int
    status: str
```

## 7.4 BehaviourEvidence

```python
class BehaviourEvidence(BaseModel):
    evidence_type: str
    track_id: int
    seat_id: str | None
    start_sec: float
    end_sec: float
    strength: float          # 0..1
    quality: float           # 0..1
    direction: str | None
    target_seat_id: str | None
    reason_codes: list[str]
```

## 7.5 RiskState

```python
class RiskState(BaseModel):
    track_id: int
    seat_id: str | None
    timestamp_sec: float
    raw_evidence: float
    quality: float
    persistence: float
    counter_evidence: float
    risk_score: float
    state: str               # NORMAL | WATCH | REVIEW | UNKNOWN
```

## 7.6 ReviewEvent

```python
class ReviewEvent(BaseModel):
    event_id: str
    camera_id: str
    video_id: str
    track_id: int
    seat_id: str | None
    start_sec: float
    end_sec: float
    peak_risk: float
    mean_quality: float
    evidence: list[BehaviourEvidence]
    summary: str
    evidence_clip_path: str | None
    human_status: str        # UNREVIEWED | DISMISSED | CONFIRMED_RELEVANT
```

`CONFIRMED_RELEVANT` means an invigilator considered the event relevant. Do not model `CHEATING_CONFIRMED` unless the institution explicitly defines a separate human administrative workflow.

---

# 8. P1 — YOLO Pose + Multi-Person Tracking

## 8.1 Primary model

Use the current Ultralytics pose model baseline:

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
```

YOLO pose provides a practical 17-keypoint body representation suitable for the baseline. The exact model variant can later be benchmarked against larger variants if the GPU budget allows.

## 8.2 Tracking

Ultralytics currently supports pose-model tracking and multiple tracker backends including BoT-SORT and ByteTrack.

Initial benchmark:

```text
A: botsort.yaml, ReID disabled
B: bytetrack.yaml
```

Choose using measured:

- IDF1 when ground truth exists
- ID switches
- track fragmentation
- track continuity through partial occlusion
- throughput

Do not choose a tracker solely because it is “more advanced.”

### Privacy note

If BoT-SORT appearance ReID is later enabled, do not persist appearance embeddings. ReID is not face recognition, but the privacy model should still minimize retained appearance information.

## 8.3 Tracking output

Serialize one JSONL record per tracked person per analysed frame:

```text
outputs/tracks/<video_id>.jsonl
```

Example:

```json
{"timestamp_sec":12.44,"frame_index":311,"track_id":7,"bbox":[...],"keypoints":{...},"pose_quality":0.86}
```

## 8.4 Overlay

Annotated video must display:

- bbox
- skeleton
- anonymous track ID
- seat ID if assigned
- pose quality
- system state (`NORMAL/WATCH/REVIEW/UNKNOWN`) only after behaviour engine exists

## 8.5 P1 acceptance gate

Required:

- [ ] reads representative dataset video
- [ ] multiple students detected
- [ ] track IDs persist across consecutive frames
- [ ] 17-keypoint observations extracted
- [ ] timestamps based on video time, not loop count alone
- [ ] JSONL output produced
- [ ] annotated video produced
- [ ] FPS + latency + GPU memory recorded
- [ ] at least 3 representative videos inspected before P1 is considered stable

---

# 9. P2 — Track-Centric Temporal Buffer

Behaviour cannot be inferred from independent frames.

Create a bounded history per track:

```python
TrackBuffer[
    track_id
] -> deque[FrameObservation]
```

## 9.1 Time-based window, not frame-count assumptions

Configuration example:

```yaml
temporal:
  history_seconds: 12.0
  minimum_observation_seconds: 1.0
  max_gap_seconds: 1.5
```

Frame sampling rate may vary. All temporal features must use timestamps.

## 9.2 Missing observations

Do not linearly interpolate long gaps by default.

Rules:

```text
gap <= short_gap_threshold
→ allow local smoothing/interpolation for geometric features

gap > threshold
→ mark evidence continuity broken
→ reduce track quality
```

## 9.3 Normalization

Raw pixel displacement is not comparable across students.

Normalize spatial features by a robust local body scale such as:

```text
shoulder width
or
bbox diagonal
or
shoulder-to-hip torso length
```

The chosen scale must handle missing keypoints gracefully.

---

# 10. P3 — Seat Anchoring and Neighbour Geometry

Seat anchoring is a central Drishti-specific improvement because examination participants are usually seated in known or semi-fixed positions.

## 10.1 Seat map

Initial implementation: manual normalized polygons.

```yaml
camera_id: CAM01
seats:
  - seat_id: A01
    polygon: [[0.05,0.20],[0.18,0.20],[0.18,0.48],[0.05,0.48]]
  - seat_id: A02
    polygon: [[0.20,0.20],[0.33,0.20],[0.33,0.48],[0.20,0.48]]
```

Coordinates are normalized `[0,1]` so maps survive resolution changes.

## 10.2 Assigning track → seat

Use a stable body point, preferably hip midpoint, otherwise bbox bottom-center.

Assignment logic:

1. Find seat polygon containing the anchor point.
2. Require persistence for `N` observations / `T` seconds before locking.
3. Maintain hysteresis so short detector jitter does not change seat assignment.
4. If a track consistently leaves its seat polygon, mark `SEAT_UNCERTAIN` rather than instant reassignment.

## 10.3 Neighbour graph

Represent seats as a graph:

```python
neighbors = {
    "A02": {"left": "A01", "right": "A03", "front": "B02", "back": None}
}
```

Do not infer “suspicious” from generic left/right alone when the seat map allows a more meaningful feature:

```text
head direction toward neighbouring seat
```

## 10.4 Camera readiness use

Seat maps also enable:

- seat coverage measurement
- blind-spot reporting
- per-seat pose quality baselines
- camera-angle-specific thresholds

---

# 11. P4 — Pose Quality and Uncertainty

## 11.1 Visible keypoint ratio

For required keypoints `K`:

```text
visible_ratio = visible(K, conf >= threshold) / len(K)
```

Use behaviour-specific keypoint sets.

Example:

```text
head behaviour:
  nose, eyes, ears, shoulders

torso behaviour:
  shoulders, hips

hand behaviour:
  elbows, wrists, shoulders
```

## 11.2 Pose quality

Start with a deterministic score:

```text
pose_quality =
  w1 * mean_required_keypoint_confidence
+ w2 * visible_required_keypoint_ratio
+ w3 * bbox_confidence
```

Normalize to `[0,1]`.

Weights are configuration values and must later be calibrated.

## 11.3 Track quality

Signals:

- age of track
- recent observation coverage
- ID discontinuities
- bbox jumps
- seat consistency
- pose quality trend

## 11.4 Image quality

Low-cost signals:

### Blur

Variance of Laplacian on grayscale image or student crop.

### Brightness

Mean + percentile distribution, not mean only.

### Frozen feed

Compare frame hashes / low pixel difference over time while timestamps advance.

### Global camera vibration

Optional later: estimate global optical flow / feature motion. Large coherent background motion should reduce confidence or trigger camera-health warning.

## 11.5 Aggregate perception quality

Do not let one extremely bad factor be hidden by averaging all others.

Recommended starting form:

```text
Q = min(
    pose_quality_adjusted,
    track_quality_adjusted,
    image_quality_adjusted
)
```

or a conservative geometric mean.

Benchmark both.

## 11.6 UNKNOWN / ABSTAIN

Initial configurable rule:

```text
if Q < unknown_quality_threshold:
    state = UNKNOWN
```

UNKNOWN is not NORMAL.

It means:

> insufficient evidence to classify this interval reliably.

Measure the **coverage** of the system:

```text
coverage = fraction of time not UNKNOWN
```

A useful system should improve precision on covered intervals without abstaining on everything.

---

# 12. P4 — Behaviour Features

The first production baseline should be **interpretable geometric + temporal features**, not an opaque end-to-end classifier.

The dataset may later justify supervised temporal models.

## 12.1 Neutral/reference pose

Absolute image geometry changes across seats/cameras.

Maintain a robust neutral reference for each seat/track from high-quality, low-motion intervals.

Possible baseline window:

```text
median of first stable 20–60 seconds
```

Do not blindly assume the first seconds are neutral; require stable posture and high quality.

Store:

- neutral head proxy
- neutral shoulder width
- neutral shoulder line angle
- neutral torso scale
- neutral wrist resting regions

## 12.2 Head direction — baseline

Do not claim precise gaze estimation from distant CCTV.

Primary output should initially be categorical or relative:

```text
LEFT
CENTER
RIGHT
UP/DOWN [optional]
UNKNOWN
```

### Baseline features

When visible:

- nose position relative to eye/ear midpoint
- left/right eye/ear confidence asymmetry
- nose displacement from shoulder midpoint
- head width relative to neutral reference
- head centroid velocity
- persistence of directional change

Compute a `head_turn_score ∈ [0,1]` and direction.

If facial landmarks are insufficient, do not force a direction.

### Behaviour evidence

```text
SIDE_GLANCE
```

requires direction change away from neutral for a configurable minimum duration.

```text
REPEATED_SIDE_GLANCE
```

requires multiple separated side-glance episodes in a rolling window.

Use **time**, not frame counts.

## 12.3 Torso rotation

2-D pose cannot directly recover true 3-D torso yaw, so represent this as an orientation proxy unless a specialist resolves it.

Features:

- shoulder-line angle
- apparent shoulder width / neutral shoulder width
- shoulder midpoint displacement relative to hip midpoint
- nose/upper-body displacement relative to torso center
- left/right arm visibility asymmetry
- temporal persistence

Output:

```text
torso_turn_direction: LEFT | CENTER | RIGHT | UNKNOWN
torso_turn_strength: 0..1
```

Do not output fake degrees of yaw unless validated against a model that actually estimates them.

## 12.4 Wrist / hand motion

For each visible wrist:

```text
velocity = distance(p_t, p_prev) / dt
acceleration = (v_t - v_prev) / dt
```

Normalize displacement by body scale.

Use robust smoothing before derivatives.

Potential features:

- wrist speed
- repeated oscillation
- hand raised away from desk
- hand crosses seat boundary
- hand approaches neighbouring seat region
- hand disappears under desk after object detection
- repeated hand-to-face/body motion

`UNUSUAL_HAND_MOTION` should be contextual and temporal; simple movement is not suspicious.

## 12.5 Interaction geometry

For seat `S` and neighbour `N`:

Evidence increases when multiple independent signals point toward `N`:

```text
head → N
+ torso → N
+ wrist trajectory → boundary with N
+ temporal persistence
```

## 12.6 Reciprocal interaction [later]

Possible signalling evidence may include a time-correlated response:

```text
Seat A02 gesture toward A03
then
Seat A03 head/gesture response within Δt
```

This must remain a review signal, not a cheating conclusion.

---

# 13. P5 — Behaviour Evidence Engine

Raw features must be converted into discrete evidence episodes.

## 13.1 Evidence lifecycle

```text
INACTIVE
→ CANDIDATE
→ ACTIVE
→ ENDED
```

Example head-turn event:

```text
head turn score exceeds onset threshold
for minimum onset duration
→ ACTIVE

score drops below lower offset threshold
for minimum release duration
→ ENDED
```

Use hysteresis to avoid rapid event flicker.

## 13.2 Suggested initial evidence types

```text
SIDE_GLANCE
REPEATED_SIDE_GLANCE
SUSTAINED_HEAD_TURN
TORSO_TURN_TOWARD_NEIGHBOUR
SUSTAINED_TORSO_ROTATION
UNUSUAL_HAND_MOTION
HAND_CROSSES_SEAT_BOUNDARY
POSSIBLE_NONVERBAL_SIGNAL
PHONE_VISIBLE
PAPER_CHIT_VISIBLE
OBJECT_INTERACTION
```

These names are engineering labels, not accusations.

---

# 14. Counter-Evidence

False-positive control is a first-class subsystem.

Potential counter-evidence:

- movement duration is very brief
- isolated event with no repetition
- rapid return to stable writing posture
- whole-body stretch signature rather than neighbour-directed orientation
- direction is toward known board/invigilator zone if such zones are configured
- no neighbour-directed alignment
- motion occurs while pose quality is unstable → prefer UNKNOWN, not suspicious

Do not subtract counter-evidence simply because an object is absent. Absence of a phone does not prove a head turn is normal.

Counter-evidence should reduce risk only when it has an interpretable basis.

---

# 15. P5 — Risk Engine

The hackathon extension goal is implemented here.

## 15.1 Signal vector

At time `t`:

```text
H_t = head behaviour evidence
B_t = body/torso evidence
G_t = hand/gesture evidence
I_t = neighbour interaction evidence
O_t = prohibited-object evidence
Q_t = perception quality
C_t = counter-evidence
```

All normalized to `[0,1]` where practical.

## 15.2 Raw evidence

Only average active/available signals.

```text
E_t = weighted_mean(active evidence signals)
```

Avoid treating missing evidence as zero when it is actually unobservable.

## 15.3 Persistence

Example smooth persistence factor:

```text
P_t = 1 - exp(-duration / tau)
```

Repetition can be added as:

```text
RPT = min(1, n_recent_events / repetition_target)
```

## 15.4 Quality-aware instantaneous contribution

```text
S_t = Q_t * E_t * f(P_t, RPT)
```

## 15.5 Risk memory / decay

Starting baseline:

```text
risk_t = clamp(
    alpha * risk_(t-1)
    + (1-alpha) * S_t
    - counter_weight * C_t,
    0,
    1
)
```

Use time-corrected decay if analysed FPS is variable.

A more robust implementation uses exponential decay based on elapsed seconds:

```text
decay = exp(-dt / risk_tau_seconds)
risk_t = decay * risk_prev + (1 - decay) * new_evidence
```

## 15.6 State machine

Initial development thresholds only:

```text
UNKNOWN  : quality < Q_min
NORMAL   : risk < T_watch
WATCH    : T_watch <= risk < T_review
REVIEW   : risk >= T_review AND event constraints satisfied
```

Example starting config:

```yaml
risk:
  unknown_quality_threshold: 0.45
  watch_threshold: 0.35
  review_threshold: 0.68
  risk_tau_seconds: 12.0
  review_min_duration_seconds: 2.0
```

**These are starting parameters, not ground truth.** Tune on validation data.

## 15.7 Review gate

A high numerical risk alone should not necessarily create an event.

Require at least one interpretable rule such as:

```text
repeated head event
or
sustained neighbour-directed torso event
or
strong object evidence
or
multi-signal interaction evidence
```

This makes alerts easier to defend.

---

# 16. P6 — Explainable Event Generation

Use deterministic templates first. Do not add an LLM dependency to the critical path.

Example event:

```json
{
  "event_id": "CAM01_A02_0042",
  "seat_id": "A02",
  "start_sec": 702.4,
  "end_sec": 731.2,
  "peak_risk": 0.81,
  "mean_quality": 0.90,
  "evidence": [
    "RIGHT_HEAD_TURN_X4",
    "HEAD_DIRECTED_TO_A03_X3",
    "TORSO_DIRECTED_TO_A03_6.8S"
  ],
  "summary": "Seat A02 repeatedly turned toward neighbouring seat A03 four times over 28.8 seconds; torso rotation toward A03 persisted for 6.8 seconds. Review recommended.",
  "decision": "UNREVIEWED"
}
```

## 16.1 Required event output

Each review event contains:

- camera ID
- anonymous seat/track ID
- start/end time
- duration
- peak/mean risk
- perception quality
- evidence list
- counter-evidence list if relevant
- summary
- evidence clip reference
- overlay snapshot/reference
- human review status

## 16.2 Clip extraction

Save a configurable context window:

```yaml
events:
  pre_roll_seconds: 5
  post_roll_seconds: 5
```

For production, use a ring buffer rather than keeping entire streams in RAM.

---

# 17. P7 — Object Detection Branch

## 17.1 Mobile phone

A pretrained object detector may provide a useful baseline for mobile phone detection.

Requirements:

- crop/scene detection benchmark
- assess size/resolution limits in CCTV footage
- correlate phone detection with specific student track/seat
- require temporal persistence or multiple detections before strong evidence

## 17.2 Paper / chit

Do not assume a generic pretrained model reliably detects small chits.

Decision tree:

```text
current dataset contains labelled paper/chit?
├─ yes → train/finetune a dedicated detector
└─ no  → mark as planned/unsupported by current data
```

Do not fabricate an accuracy claim.

## 17.3 Track association

Associate object to student using:

- object center inside/near student bbox
- hand-object proximity
- seat polygon membership
- temporal continuity

`PHONE_VISIBLE` should remain object evidence. `PHONE_INTERACTION` requires spatial/temporal association with the student.

---

# 18. P8 — Specialist Routing

Specialists are **conditional refiners**, not models that run on every student every frame.

## 18.1 Router inputs

Possible triggers:

```text
low required-keypoint confidence
high behavioural ambiguity
head-direction conflict
wrist evidence near alert threshold
track quality high but pose quality low
sufficient crop resolution for specialist
```

## 18.2 RTMPose

Use as a higher-quality 2-D body pose specialist where the primary 17-keypoint estimate is insufficient.

Do not invoke when the crop is extremely occluded or too small; more compute cannot recover absent visual evidence.

## 18.3 MediaPipe Face Landmarker

Use only for sufficiently resolved face crops.

The MediaPipe Tasks API supports IMAGE, VIDEO, and LIVE_STREAM modes. For video mode, timestamps must monotonically increase.

Important privacy rule:

```text
Face Landmarker is used for geometric landmarks/head orientation only.
No face recognition.
No identity embeddings.
No biometric name matching.
```

## 18.4 MediaPipe Hand Landmarker

Use on a person/hand ROI when wrist-level evidence is ambiguous and the crop contains enough pixels.

Again, VIDEO/LIVE_STREAM execution is timestamped and should not be instantiated per frame.

## 18.5 Multi-HMR / SMPL-X

Late-stage, selective 3-D specialist.

Use only if experiments show meaningful improvement on ambiguous torso/3-D posture cases.

Treat as a separate service/environment because it is significantly heavier and has its own dependency/model setup.

Do not block the hackathon baseline on Multi-HMR.

---

# 19. Camera Readiness and Health

Real-world deployment must report when the camera setup is inadequate.

## 19.1 Pre-exam readiness report

Generate per camera:

```text
CAM01
✓ stream readable
✓ stable FPS
✓ 46/48 configured seats covered
✓ acceptable global brightness
✓ acceptable sharpness
⚠ A11 partially occluded
⚠ D01 face/head resolution below preferred range
Coverage score: 91.7%
```

## 19.2 Health events

Separate system-health events from behaviour events.

Examples:

```text
CAMERA_DISCONNECTED
LOW_FPS
FROZEN_STREAM
SEVERE_BLUR
SEVERE_DARKNESS
GLOBAL_CAMERA_MOVEMENT
GPU_INFERENCE_FAILURE
DECODER_FAILURE
```

Never convert a camera failure into a behavioural alert.

---

# 20. Real-Time Pipeline Design

## 20.1 Decouple decode FPS from inference FPS

A 30 FPS stream does not require all expensive models at 30 FPS.

Prototype target:

```text
decode: native stream FPS
primary inference: configurable 10–15 FPS target
specialists: event-driven / selective
behaviour features: timestamp based
```

Tracking and temporal logic must account for skipped frames.

## 20.2 Latency measurement

Record:

```text
decode_ms
pose_track_ms
feature_ms
specialist_ms
risk_ms
overlay_ms
end_to_end_ms
```

Report p50, p95, and worst observed latency.

## 20.3 Memory

Use bounded deques per track.

Do not retain full frames for every track.

Use a global compressed ring buffer only for event pre-roll extraction if needed.

---

# 21. Evaluation Protocol

## 21.1 Hard targets

These are **project targets** until measured:

| Metric | Initial target |
|---|---:|
| visible-student detection/pose recall | >= 90% |
| tracking IDF1 | >= 85% |
| behaviour Macro-F1 | >= 80% |
| review-event precision | >= 90% |
| false-positive rate on labelled event windows | <= 10% |
| analysed FPS per camera on demo hardware | >= 10 FPS |
| pipeline event latency | < 1 s target |

Do not claim these as achieved until evaluation artifacts exist.

## 21.2 Better deployment metric

Once temporal labels are available, add:

```text
false alerts per student-hour
```

This is more operationally meaningful than a frame-level false-positive rate.

## 21.3 Detection

When ground truth exists:

- precision
- recall
- F1
- performance by student scale/bbox size

## 21.4 Tracking

Preferred:

- IDF1
- ID switches
- track fragmentation
- optionally HOTA if the annotation/evaluation stack supports it

## 21.5 Behaviour

Use event/window-level evaluation.

Report:

- Macro-F1
- per-class precision
- per-class recall
- per-class F1
- confusion matrix

Avoid reporting only accuracy on an imbalanced dataset.

## 21.6 Alerts

Evaluate generated review events:

- event precision
- event recall
- false alerts per student-hour
- median alert delay
- duplicate-event rate

## 21.7 Abstention

Report:

```text
coverage
precision on covered intervals
recall on covered intervals
error vs. quality threshold
```

The goal is not to maximize UNKNOWN.

## 21.8 Runtime

Per hardware profile:

- analysed FPS/camera
- end-to-end p50/p95 latency
- GPU utilization
- GPU memory
- CPU utilization
- dropped-frame rate

---

# 22. Robustness Test Matrix

Every major PS technology challenge needs a test bucket.

| Challenge | Test / stratum | Expected system response |
|---|---|---|
| normal vs suspicious | hard-negative clips | high alert precision |
| occlusion | natural + synthetic occlusion | degrade quality / abstain safely |
| low light | low-brightness clips | quality warning, stable if observable |
| glare | overexposed regions | quality drop / UNKNOWN where necessary |
| low resolution | downsampled clips | quantify performance degradation |
| motion blur | synthetic/natural blur | quality detection + reduced confidence |
| camera angle | per-camera evaluation | seat-calibrated features |
| tracking failure | crossing/partial blockage | quantify ID switches |
| blind spots | seat coverage audit | mark unobservable seats |
| behaviour diversity | multiple people/sessions | per-session metrics |
| long duration | full-length playback | bounded memory + stable IDs |
| scale | multi-stream replay | throughput benchmark |

## 22.1 Synthetic robustness transforms

Allowed only as additional stress tests, not a replacement for natural examples:

- brightness reduction
- contrast shift
- Gaussian/motion blur
- compression degradation
- partial masks/occlusion
- downsampling

Keep the original test split untouched; label transformed versions separately.

---

# 23. Real-World Deployment Architecture

## 23.1 Small hall

```text
2–4 RTSP cameras
      │
      ▼
edge/central GPU node
      │
      ├─ stream workers
      ├─ pose + tracking
      ├─ behaviour engine
      ├─ local event store
      └─ dashboard API
```

## 23.2 Medium/large hall

```text
CAM01 ─┐
CAM02 ─┤
CAM03 ─┤
...    ├──► ingest/batching layer
CAM08 ─┘          │
                  ▼
             GPU inference
                  │
                  ▼
              trackers
                  │
                  ▼
       Drishti behaviour workers
                  │
                  ▼
              event bus
          ┌───────┴────────┐
          ▼                ▼
      event store       dashboard
```

## 23.3 Production migration path

After the Python prototype is validated:

```text
OpenCV ingest
→ GStreamer / NVIDIA DeepStream
→ ONNX/TensorRT where beneficial
→ batched multi-stream inference
→ persistent event services
```

NVIDIA DeepStream is designed for streaming video analytics and includes inference/tracking pipeline components. TensorRT is NVIDIA's inference optimization/runtime stack for deploying optimized engines on NVIDIA GPUs.

Do not migrate until model correctness and evaluation are stable.

---

# 24. Network and Hardware Failure Strategy

## 24.1 Stream worker

Each camera worker should expose:

```text
CONNECTED
RECONNECTING
DEGRADED
OFFLINE
```

Use exponential reconnect backoff and timestamp every interruption.

## 24.2 Edge buffer [deployment]

Maintain short local retention for:

- recent frames for evidence clips
- unsent event metadata

When central connectivity is lost:

```text
continue local inference if hardware permits
queue events locally
resynchronize when connection returns
```

## 24.3 GPU failure

On inference exception/OOM:

- log health event
- do not silently skip behaviour output
- optionally reduce input resolution / inference rate according to a controlled degradation policy
- mark monitoring coverage degraded

---

# 25. Privacy and Ethics Implementation

Hard constraints:

1. No facial identity recognition.
2. No mapping anonymous track IDs to student names in the core behavioural engine.
3. Face landmarks, if used, are geometric evidence only.
4. Do not persist face embeddings.
5. Store only the minimum required evidence clips/logs.
6. Make retention duration configurable.
7. Human review remains final.
8. UI language must say `Review Event`, not `Cheating Detected`.
9. UNKNOWN must be visible rather than hidden.

Preferred identifier:

```text
CAM02-SEAT-B17
```

not a biometric identity.

---

# 26. Configuration

Example `configs/baseline.yaml`:

```yaml
project:
  name: drishti4d
  output_dir: outputs

video:
  analysis_fps: 12
  resize_width: null
  max_duration_sec: null

pose:
  model: yolo26n-pose.pt
  confidence: 0.25
  iou: 0.50
  device: auto

tracking:
  backend: botsort.yaml
  use_seat_anchor: true
  seat_lock_seconds: 2.0
  seat_release_seconds: 3.0

quality:
  keypoint_confidence: 0.35
  unknown_quality_threshold: 0.45
  min_head_visible_ratio: 0.60
  min_torso_visible_ratio: 0.75
  min_hand_visible_ratio: 0.50

temporal:
  history_seconds: 12.0
  max_gap_seconds: 1.5
  smoothing_seconds: 0.35

head:
  candidate_min_seconds: 0.35
  sustained_min_seconds: 1.5
  repeated_window_seconds: 20.0
  repeated_count: 3

torso:
  candidate_min_seconds: 0.75
  sustained_min_seconds: 2.0

hands:
  motion_threshold: 0.12
  candidate_min_seconds: 0.5

risk:
  watch_threshold: 0.35
  review_threshold: 0.68
  risk_tau_seconds: 12.0
  review_min_duration_seconds: 2.0
  counter_weight: 0.25

events:
  merge_gap_seconds: 2.0
  pre_roll_seconds: 5.0
  post_roll_seconds: 5.0
  save_clips: true

specialists:
  enabled: false
  mediapipe_face: false
  mediapipe_hands: false
  rtmpose: false
  multihmr: false
```

All numerical values above are **initial engineering defaults for calibration**, not scientifically validated thresholds.

---

# 27. Required CLIs

## Dataset audit

```bash
python scripts/audit_dataset.py \
  --root <DATASET_ROOT>
```

## Pose + tracking

```bash
python scripts/run_pose_tracking.py \
  --source <VIDEO_PATH> \
  --config configs/baseline.yaml
```

Expected outputs:

```text
outputs/tracks/<video>.jsonl
outputs/videos/<video>_pose_track.mp4
outputs/metrics/<video>_runtime.json
```

## Seat calibration

```bash
python scripts/calibrate_seats.py \
  --source <VIDEO_PATH> \
  --output configs/seat_map.<camera>.yaml
```

Initial implementation can be a simple OpenCV polygon-click UI or accept a hand-authored YAML file. Do not overbuild a web UI here.

## Behaviour

```bash
python scripts/run_behaviour.py \
  --tracks outputs/tracks/<video>.jsonl \
  --seat-map configs/seat_map.<camera>.yaml \
  --config configs/behaviour.yaml
```

## End-to-end

Once stable:

```bash
python scripts/run_behaviour.py \
  --source <VIDEO_PATH> \
  --seat-map <SEAT_MAP> \
  --config configs/baseline.yaml \
  --save-overlay
```

---

# 28. Logging and Reproducibility

Every run should create a unique run directory:

```text
outputs/runs/YYYYMMDD-HHMMSS_<short-name>/
```

Store:

```text
config.yaml
command.txt
environment.txt
summary.json
runtime.json
tracks.jsonl
events.jsonl
annotated.mp4 [optional]
```

Record:

- git commit hash if repository is initialized
- model checkpoint name
- hardware/GPU
- dependency versions
- dataset/video IDs
- random seeds when training/evaluation uses randomness

---

# 29. Tests

## 29.1 Unit tests

### Geometry

- normalized distance is scale-consistent
- seat polygon containment
- neighbour lookup
- angle/feature functions handle missing keypoints

### Temporal buffer

- time-window eviction
- gap handling
- no negative timestamps
- duplicate frame handling

### Risk engine

Test explicitly:

```text
isolated short glance
→ no REVIEW

repeated neighbour-directed turns
→ risk rises

return to stable posture
→ risk decays

quality collapses
→ UNKNOWN

quality recovers
→ exits UNKNOWN according to hysteresis
```

### Event builder

- event start/end correct
- nearby evidence merged correctly
- duplicate alerts suppressed
- summary contains only observed evidence

## 29.2 Integration test

Use one short representative video fixture and verify:

```text
video → pose → tracks → features → risk → events → overlay
```

Do not include the entire dataset in tests.

---

# 30. Immediate Codex Execution Plan

This is the part Codex should execute first.

## TASK 0 — Bootstrap

Create:

```text
requirements-core.txt
configs/baseline.yaml
src/drishti/schemas.py
outputs/.gitkeep
```

Add a minimal README with commands.

### Gate

```bash
python -m pytest -q
```

must run even if only schema tests exist initially.

---

## TASK 1 — Dataset auditor

Implement `scripts/audit_dataset.py`.

Requirements:

- recursive media discovery
- OpenCV probe
- bounded frame sampling
- codec/resolution/FPS/duration
- brightness + blur estimate
- annotation-file discovery
- JSON report
- human-readable console summary
- no full-video memory load

### Output

```text
outputs/audits/dataset_audit.json
outputs/audits/dataset_audit.csv
```

### Stop condition

After this task, inspect the actual dataset output before making assumptions about labels or training.

---

## TASK 2 — Pose + tracking baseline

Implement `src/drishti/perception/pose_tracker.py` and `scripts/run_pose_tracking.py`.

Requirements:

- `YOLO("yolo26n-pose.pt")`
- configurable tracker
- timestamp-aware processing
- extract bbox, track ID, 17 keypoints/confidence
- convert result to Drishti schema
- JSONL serialization
- annotated MP4
- runtime timings
- do not initialize the model per frame

### First run

Use **one representative dataset video only**.

### Gate

Manually inspect the overlay and verify:

- IDs mostly stable
- seated students detected
- skeletons plausible
- runtime acceptable enough to continue

Then run 2–3 more representative videos.

---

## TASK 3 — Track buffer + quality

Implement:

```text
behaviour/buffer.py
perception/pose_quality.py
tracking/track_quality.py
```

Create tests before behaviour heuristics.

### Gate

A track can be replayed from JSONL and produce stable per-second quality values.

---

## TASK 4 — Seat anchoring

Implement manual seat map loading and track-to-seat assignment.

Do not build automatic seat detection yet.

### Gate

Annotated video shows stable anonymous seat IDs for seated students.

---

## TASK 5 — First behaviour: repeated head turn

Implement exactly one end-to-end behaviour first:

```text
head turn proxy
→ event episode
→ repetition counter
→ risk contribution
→ explanation
```

Do not implement hand, torso, objects simultaneously.

### Required demo

```text
short isolated turn → no REVIEW
repeated sustained turns → WATCH / REVIEW
bad pose quality → UNKNOWN
```

Only after this behaviour is correct should the next behaviour be added.

---

# 31. After the First Behaviour Works

Implementation order:

```text
1. repeated/sustained head turn
2. torso turn toward neighbour
3. hand/wrist motion
4. multi-signal neighbour interaction
5. phone branch
6. paper/chit if dataset supports it
7. counter-evidence tuning
8. specialist router
9. robustness suite
10. dashboard/API
```

---

# 32. What NOT To Do Yet

Codex must not:

- build the full dashboard before event logic works
- train a large end-to-end behavioural classifier without first auditing labels
- use random frame-level train/test splits
- treat each frame as an independent behaviour decision
- hard-code one dataset folder layout
- claim head-pose degrees from weak 2-D geometry
- force a prediction when keypoints are missing
- label alerts as “cheating detected”
- use face recognition
- add Multi-HMR to the critical path
- run MediaPipe/MMPose specialists on every person every frame
- store all decoded frames in memory
- present target metrics as measured metrics
- silently skip camera/model failures

---

# 33. Definition of Done — Hackathon

The hackathon implementation is considered technically complete only if the demo can show:

- [ ] multiple seated students detected
- [ ] persistent anonymous tracking
- [ ] pose landmarks
- [ ] stable track/seat identities
- [ ] at least head + torso + hand behavioural evidence
- [ ] at least mobile-phone object evidence if visually feasible
- [ ] temporal persistence/repetition rather than frame-only flags
- [ ] behavioural risk score
- [ ] `NORMAL / WATCH / REVIEW / UNKNOWN` states
- [ ] deterministic explainable event summary
- [ ] evidence timeline/clip
- [ ] camera/pose quality indicator
- [ ] measured latency/FPS
- [ ] evaluation report with clearly marked achieved metrics
- [ ] false-positive examples and mitigation demonstrated
- [ ] privacy statement and no identity recognition
- [ ] architecture path to multi-camera deployment

---

# 34. Definition of Done — Beyond Hackathon Pilot

For a real small-hall pilot:

- multi-camera RTSP ingest
- camera readiness test
- per-camera seat map/calibration
- long-duration tracking test
- event logs across a full exam session
- false alerts per student-hour measured
- alert review workflow
- configurable retention policy
- network reconnection and event buffering
- health monitoring
- robustness validation in real lighting/occlusion conditions
- documented hardware throughput
- institution-specific threshold calibration
- human-in-the-loop validation protocol

For larger deployments, add stream batching, GPU scheduling, centralized event services, production observability, and potentially DeepStream/TensorRT optimization.

---

# 35. Key Engineering Principle

Whenever a technological challenge cannot be fully “solved,” Drishti must do one of four things:

```text
DETECT the limitation
MEASURE its impact
MITIGATE when possible
ABSTAIN / EXPOSE when not reliable
```

Examples:

```text
blind spot
→ report seat unobservable

severe occlusion
→ UNKNOWN

low-resolution face
→ do not fake head yaw

tracking instability
→ lower quality + prevent strong review event

network failure
→ health event + local buffer
```

This is the standard for a deployment-oriented behavioural monitoring system.

---

# 36. Technical References Verified for This Specification

These are implementation references, not sources of project performance claims.

1. **Ultralytics — Multi-Object Tracking**  
   https://docs.ultralytics.com/modes/track  
   Current documentation covers YOLO26 tracking, pose-model tracking, BoT-SORT, ByteTrack, `persist=True`, tracker configs, optional BoT-SORT ReID, and camera-motion compensation.

2. **MMPose — Inference with Existing Models**  
   https://mmpose.readthedocs.io/en/latest/user_guides/inference.html  
   MMPose provides image/video inference and RTMPose model support through its inference/model ecosystem.

3. **MediaPipe Hand Landmarker — Python API**  
   https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker  
   Supports image, video, and live-stream workflows with timestamped video/live processing.

4. **MediaPipe Face Landmarker — Python API**  
   https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker  
   Supports video and live-stream landmark inference; use for geometric refinement only, not identity recognition.

5. **NAVER Multi-HMR — Official Repository**  
   https://github.com/naver/multi-hmr  
   Multi-person whole-body 3-D mesh recovery; treat as an optional isolated specialist. The repository documents its environment and separate SMPL-X asset requirements.

6. **NVIDIA DeepStream Developer Guide**  
   https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html  
   Production-oriented streaming video analytics framework with multi-stream inference/tracking components.

7. **NVIDIA TensorRT Documentation**  
   https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html  
   NVIDIA inference optimization/runtime stack for deployment on NVIDIA GPUs.

---

# 37. Source-of-Truth Note

The problem-statement requirements that drive this specification come from the uploaded document:

```text
Problem Statement DrishtiAI Hackathon.pdf
Problem Statement #1: AI-Based Human Behaviour Monitoring Using Pose Estimation and Gesture Analysis
```

The document explicitly defines objectives, risk factors, technology challenges, explainability/human-oversight requirements, the hackathon extension goal, and example deployment hardware/scales.

The engineering thresholds, algorithms, phased architecture, uncertainty state, seat anchoring, temporal evidence design, and deployment interfaces in this specification are **our implementation design** intended to satisfy and exceed those requirements. They are not claims that the PS itself mandated those exact algorithms or numerical thresholds.

---

# 38. First Codex Instruction

Paste this file into the repository and instruct Codex:

> Implement **TASK 0 and TASK 1 only** from `DRISHTI_IMPLEMENTATION_SPEC.md`. Do not implement behaviour detection, specialists, dashboard, model training, or deployment optimization yet. Keep the dataset path configurable. Produce the exact audit artifacts described in the spec, add tests where applicable, and stop after reporting the dataset audit results and any discovered annotation schema. Do not assume labels that are not present in the dataset.

After inspecting the audit output, proceed to TASK 2.

---

**End of specification.**

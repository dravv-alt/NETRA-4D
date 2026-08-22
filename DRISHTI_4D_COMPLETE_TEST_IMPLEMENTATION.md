# DRISHTI-4D — Complete Behaviour, Context & Test Implementation Specification

> **Project:** DrishtiAI Hackathon — Problem Statement #1  
> **System:** AI-Based Human Behaviour Monitoring Using Pose Estimation and Gesture Analysis  
> **Purpose:** Single source-of-truth test implementation document for YOLO26 Pose, RTMPose, and Multi-HMR + SMPL-X pipelines  
> **Status:** Engineering/Test Specification — thresholds are initial defaults until calibrated  
> **Core rule:** A movement is **evidence**, not a misconduct conclusion. Context, persistence, repetition, quality, counter-evidence, and interaction determine the final state.

---

# 0. Why This Document Exists

Drishti-4D must not behave like:

```text
HEAD_TURN_RIGHT
→ SUSPICIOUS
```

The system must instead reason like:

```text
HEAD_TURN_RIGHT
        │
        ▼
WHAT IS THE TARGET CONTEXT?
        │
        ├── BOARD
        │      └── legitimate / counter-evidence
        │
        ├── INVIGILATOR
        │      └── legitimate / counter-evidence
        │
        ├── ENTRANCE / CLOCK
        │      └── usually legitimate / weak evidence
        │
        ├── NEIGHBOUR_SEAT
        │      └── behavioural evidence
        │
        ├── BLIND_SPOT
        │      └── confidence reduction / possible UNKNOWN
        │
        └── UNKNOWN_CONTEXT
               └── do not over-claim
```

A **brief isolated movement** may remain `NORMAL`.

Risk should increase only when one or more of the following are observed:

- repetition,
- sustained duration,
- neighbour-directed orientation,
- torso + head alignment,
- hand/arm evidence,
- reciprocal student-to-student response,
- object interaction,
- high-quality persistent tracking,
- lack of legitimate-context counter-evidence.

All three independent pipelines must implement and test the same behavioural contract:

1. **Anuj:** YOLO26 Pose + tracking
2. **Om:** RTMPose + object detection
3. **Dravvya:** Multi-HMR + SMPL-X

---

# 1. Final Shared Behaviour Taxonomy

All three pipelines must classify the same labels.

## B0 — `NORMAL`

Examples:

- normal writing,
- thinking,
- brief glance,
- normal look-around,
- stretching,
- posture adjustment,
- head scratching,
- stationery handling,
- looking at board,
- looking at invigilator,
- short legitimate body adjustment.

## B1 — `REPEATED_SIDEWARD_GLANCE`

Multiple separated head-orientation episodes within a rolling temporal window, especially when repeatedly directed toward the same neighbouring seat.

## B2 — `SUSTAINED_EXCESSIVE_HEAD_TURN`

A prolonged or excessive head turn relative to the student's neutral orientation.

## B3 — `TORSO_ROTATION_TOWARD_NEIGHBOUR`

Torso/body orientation becomes persistently aligned toward a neighbouring seat.

## B4 — `UNUSUAL_HAND_ACTIVITY`

Examples:

- repeated non-writing hand movement,
- hand crossing normal seat/desk boundary,
- arm extension toward neighbour,
- unusual wrist trajectory,
- repeated interaction outside expected writing region.

## B5 — `POSSIBLE_NONVERBAL_INTERACTION`

Temporal relationship between two students, such as:

```text
A → B
then
B → A
```

using head, torso, hand, or gesture evidence.

Never output `COMMUNICATION CONFIRMED`. Use `POSSIBLE_NONVERBAL_INTERACTION — review recommended`.

## B6 — `POSSIBLE_OBJECT_INTERACTION`

Behaviour indicating handling, receiving, passing, hiding, or interacting with a potentially relevant object.

Object identity is a separate task owned primarily by Om.

---

# 2. Object Taxonomy

Explicit object identification:

```text
O1 MOBILE_PHONE
O2 PAPER_CHIT
O3 OTHER_PROHIBITED_OBJECT
```

`PAPER_CHIT` and `OTHER_PROHIBITED_OBJECT` become scored classes only when the dataset contains sufficient usable examples/annotations.

---

# 3. Final System States

Every pipeline must produce one of:

```text
NORMAL
WATCH
REVIEW
UNKNOWN
```

These states are **not direct aliases for individual movements**.

---

# 4. State Contract — Context-Aware Behaviour

## 4.1 `NORMAL`

A movement remains `NORMAL` when one or more of these apply:

- isolated/brief movement,
- expected writing motion,
- stretch/posture adjustment,
- direction is toward `BOARD`,
- direction is toward `INVIGILATOR`,
- direction is toward another known legitimate zone,
- movement returns rapidly to neutral,
- no repetition,
- no neighbour-directed multi-signal support,
- no reciprocal interaction,
- no object evidence.

### Required example

```text
Student turns head right
→ target = BOARD
→ duration = 1.8 sec
→ no torso turn
→ no hand event
→ no repetition

FINAL = NORMAL
```

### Hard failure

If a normal board look generates `REVIEW`, the test fails.

## 4.2 `WATCH`

`WATCH` means behaviour is worth monitoring, but evidence is not strong enough for human review.

Typical triggers:

- repeated neighbour-directed glance,
- mildly sustained neighbour-directed head turn,
- weak torso alignment,
- unusual hand activity without corroboration,
- possible interaction but insufficient reciprocal evidence,
- moderate risk with good quality,
- ambiguous context with some relevant evidence.

Example:

```text
Student A03
head → A04
3 occurrences in 18 sec
torso = mostly neutral
hands = normal
quality = GOOD

FINAL = WATCH
```

A repeated board look should normally **not** become WATCH unless additional abnormal evidence exists.

## 4.3 `REVIEW`

`REVIEW` requires interpretable, sustained, or multi-signal evidence.

At least one strong condition should exist:

```text
repeated neighbour-directed head movement
+
persistence
```

or:

```text
head → neighbour
+
torso → same neighbour
```

or:

```text
A → B
+
B → A within temporal response window
```

or:

```text
strong phone/object association
+
student interaction
```

Example:

```text
A03 head → A04 ×4
+
A03 torso → A04 for 5.8 sec
+
A04 responds toward A03 after 1.4 sec
+
quality = GOOD

FINAL = REVIEW
```

`REVIEW` means human inspection is recommended. It does not mean cheating is confirmed.

## 4.4 `UNKNOWN`

`UNKNOWN` means the system does not have sufficient visual/perceptual evidence to classify reliably.

Triggers:

- severe occlusion,
- blind spot,
- head too small,
- critical keypoints missing,
- unstable track identity,
- major motion blur,
- poor lighting,
- tracking gap,
- camera health degraded,
- conflicting signals with insufficient quality.

Use behaviour-specific quality:

```text
HEAD     = GOOD / POOR / UNKNOWN
TORSO    = GOOD / POOR / UNKNOWN
HANDS    = GOOD / POOR / UNKNOWN
TRACK    = GOOD / POOR / UNKNOWN
CAMERA   = GOOD / DEGRADED
```

---

# 5. Semantic Camera Context — Mandatory

Every camera must have a configuration profile.

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
  A03: [A02]

zones:
  board:
    polygon: [...]

  invigilator_left:
    polygon: [...]

  invigilator_right:
    polygon: [...]

  entrance:
    polygon: [...]

  clock:
    polygon: [...]

blind_spots:
  - polygon: [...]
```

Coordinates should preferably be normalized `[0,1]`.

---

# 6. Context Resolution Contract

A movement must be represented as:

```text
MOVEMENT
+
TARGET_CONTEXT
```

Examples:

```text
HEAD_TURN_RIGHT → BOARD
HEAD_TURN_LEFT  → INVIGILATOR
HEAD_TURN_RIGHT → SEAT_A04
TORSO_RIGHT     → SEAT_A04
HAND_EXTENSION  → SEAT_A04_BOUNDARY
```

Not merely:

```text
HEAD_TURN_RIGHT
TORSO_RIGHT
HAND_RIGHT
```

---

# 7. Common Context Types

All three pipelines must use:

```text
BOARD
INVIGILATOR
NEIGHBOUR_SEAT
ENTRANCE
CLOCK
OWN_DESK
OWN_SEAT
BLIND_SPOT
UNKNOWN_CONTEXT
```

Additional legitimate zones may be added only through shared camera configuration.

---

# 8. Context-Aware Counter-Evidence

Counter-evidence reduces risk only when it has an interpretable reason.

## Legitimate contextual evidence

```text
head → board
head → invigilator
head → clock
brief look → entrance
hand motion → own writing region
symmetric stretch
posture correction
rapid return to neutral writing posture
```

## Not valid counter-evidence

Absence of one signal does not automatically cancel another. For example, `phone not detected` does not prove a head turn is normal.

---

# 9. Common Temporal Window

All three pipelines use approximately:

```yaml
temporal:
  history_seconds: 12.0
  max_gap_seconds: 1.5
```

Longer rolling aggregation may additionally be used for repetition counting.

All calculations must use timestamps, not frame counts.

---

# 10. Neutral Behaviour Baseline

Do not automatically assume the first seconds of a video are neutral.

Select stable baseline intervals where:

```text
pose quality = high
track quality = high
motion = low
seat assignment = stable
```

Use these intervals to estimate:

- neutral head orientation,
- neutral torso orientation,
- neutral shoulder width,
- neutral wrist regions,
- normal body scale.

---

# 11. Behaviour-Specific Context Rules

## 11.1 Repeated Sideward Glance

### Positive evidence

```text
head repeatedly leaves neutral
+
direction repeatedly targets same NEIGHBOUR_SEAT
+
episodes separated in time
```

### Counter-evidence

```text
target = BOARD
target = INVIGILATOR
target = CLOCK
single brief glance
normal look-around pattern
```

### Required tests

| Test | Expected |
|---|---|
| One 0.5 sec neighbour glance | NORMAL |
| One 2 sec board glance | NORMAL |
| Three board glances during instruction | NORMAL |
| Four glances toward same neighbour | WATCH or REVIEW depending calibration |
| Four neighbour glances + torso alignment | REVIEW |
| Head visibility poor | HEAD UNKNOWN |

## 11.2 Sustained Excessive Head Turn

### Positive evidence

```text
head orientation displaced from neutral
+
duration exceeds calibrated threshold
+
target = NEIGHBOUR_SEAT
```

### Counter-evidence

```text
target = BOARD
target = INVIGILATOR
legitimate hall context
```

### Required tests

| Test | Expected |
|---|---|
| 2 sec head turn toward board | NORMAL |
| 4 sec look at invigilator | NORMAL / no REVIEW |
| Sustained neighbour turn with no repetition | WATCH |
| Sustained neighbour turn + torso same direction | REVIEW |
| Severe blur | UNKNOWN if unreliable |

## 11.3 Torso Rotation Toward Neighbour

### Positive evidence

```text
torso orientation toward neighbour
+
persistence
```

Stronger:

```text
head + torso → same neighbour
```

### Counter-evidence

```text
stretching
posture adjustment
reaching within own desk
turning toward invigilator
```

### Required tests

| Test | Expected |
|---|---|
| Whole-body stretch | NORMAL |
| Brief posture correction | NORMAL |
| Torso toward board/invigilator | no REVIEW |
| Sustained torso toward neighbour | WATCH |
| Head + torso toward neighbour | REVIEW candidate |

## 11.4 Unusual Hand Activity

### Positive evidence

```text
high wrist movement
+
non-writing trajectory
+
seat/desk boundary crossing
+
persistence/repetition
```

or:

```text
hand → neighbour region
```

### Counter-evidence

```text
normal writing
eraser use
page turning
stationery handling
head scratch
adjusting clothes
```

### Required tests

| Test | Expected |
|---|---|
| Normal writing | NORMAL |
| Page turn | NORMAL |
| Head scratch | NORMAL |
| Brief stationery movement | NORMAL |
| Repeated hand movement toward neighbour boundary | WATCH |
| Hand movement + reciprocal response | REVIEW candidate |
| Wrist keypoints missing | HANDS UNKNOWN |

## 11.5 Possible Non-Verbal Interaction

Required relation graph:

```text
A → B
B → A
```

Signals may include head, torso and hands.

### Counter-evidence

```text
both students independently looking at board
both reacting to invigilator
global classroom event
```

### Required tests

| Test | Expected |
|---|---|
| A and B both look at board | NORMAL |
| A looks at B once, no response | NORMAL/WATCH |
| A repeatedly looks at B, B never responds | WATCH |
| A→B then B→A repeatedly | REVIEW candidate |
| Camera context cannot distinguish target | reduce confidence |

## 11.6 Possible Object Interaction

All three pipelines may identify:

```text
hand/object-like interaction
passing motion
receiving motion
hiding motion
```

Om additionally identifies:

```text
MOBILE_PHONE
PAPER_CHIT
OTHER_PROHIBITED_OBJECT
```

Example association:

```text
phone visible near A03
+
hand proximity
+
seat membership
+
temporal persistence
→ PHONE_ASSOCIATED_WITH_A03
```

### Required tests

| Test | Expected |
|---|---|
| Pen/stationery in own desk | NORMAL |
| Page turn | NORMAL |
| Phone detection with no student association | object evidence only |
| Phone associated with A03 wrist/seat | strong object evidence |
| Object passed A03→A04 | REVIEW candidate |
| Tiny/occluded object | UNKNOWN/weak evidence |

---

# 12. Hard-Negative Test Set — Mandatory

Create:

```text
evaluation/
└── hard_negatives/
    ├── normal_writing/
    ├── brief_glance/
    ├── board_look/
    ├── invigilator_look/
    ├── normal_look_around/
    ├── stretching/
    ├── posture_adjustment/
    ├── head_scratch/
    ├── page_turn/
    ├── stationery_handling/
    └── legitimate_hall_response/
```

Each clip should contain:

```text
video_id
camera_id
start_sec
end_sec
expected_state = NORMAL
behaviour_note
target_context
```

### Hard failure rule

If a hard-negative clip creates a `REVIEW` event without valid independent evidence, the test fails.

---

# 13. Positive Test Set — Mandatory

Create:

```text
evaluation/
└── positives/
    ├── repeated_neighbour_glance/
    ├── sustained_neighbour_head_turn/
    ├── neighbour_torso_turn/
    ├── unusual_hand_neighbour_motion/
    ├── reciprocal_interaction/
    ├── phone_interaction/
    └── object_exchange/
```

Every event annotation should include:

```text
start_sec
end_sec
seat_id
target_seat_id
behaviour
quality_notes
```

---

# 14. Ambiguous Test Set

Create:

```text
evaluation/
└── ambiguous/
    ├── partial_occlusion/
    ├── low_light/
    ├── low_resolution_head/
    ├── blind_spot/
    ├── tracking_instability/
    └── unknown_context/
```

Expected behaviour:

```text
reduce quality
or
produce component-specific UNKNOWN
```

The system must not force `NORMAL` or `REVIEW`.

---

# 15. Shared Event State Machine

```text
INACTIVE
   ↓
CANDIDATE
   ↓
ACTIVE
   ↓
ENDED
```

Use hysteresis (`onset threshold > offset threshold`) to prevent event flicker.

---

# 16. Shared Risk Inputs

At timestamp `t`:

```text
H_t = head evidence
B_t = torso/body evidence
G_t = hand/gesture evidence
I_t = interaction evidence
O_t = object evidence
Q_t = perception quality
C_t = counter-evidence
CTX_t = semantic context
```

---

# 17. Context-Aware Risk Logic

Never use:

```text
risk = head_turn_score
```

Conceptual form:

```text
contextual_evidence =
    evidence
    × context_relevance
    × quality
```

Example context relevance should later be calibrated:

```text
BOARD            → low / legitimate
INVIGILATOR      → low / legitimate
CLOCK            → low / legitimate
NEIGHBOUR_SEAT   → high relevance
UNKNOWN_CONTEXT  → uncertain
BLIND_SPOT       → do not infer strongly
```

Then temporal risk:

```text
risk_t =
decayed_previous_risk
+
new_contextual_evidence
-
interpretable_counter_evidence
```

All weights and thresholds must remain configurable and later calibrated.

---

# 18. Event Merging

Do not generate separate alerts for one repeated pattern.

Instead of:

```text
HEAD EVENT 1
HEAD EVENT 2
HEAD EVENT 3
HEAD EVENT 4
```

produce:

```text
REPEATED_NEIGHBOUR_ORIENTATION
Seat: A03
Target: A04
Occurrences: 4
Start: 23.1
End: 44.6
Peak risk: 0.79
```

---

# 19. Explainable Event Contract

Example review event:

```json
{
  "event_id": "CAM01_A03_0042",
  "camera_id": "CAM01",
  "seat_id": "A03",
  "state": "REVIEW",
  "start_sec": 702.4,
  "end_sec": 731.2,
  "peak_risk": 0.81,
  "mean_quality": 0.90,
  "evidence": [
    "HEAD_TURN_RIGHT_X4",
    "HEAD_TARGET_A04_X3",
    "TORSO_TARGET_A04_5.8S"
  ],
  "counter_evidence": [],
  "summary": "Seat A03 repeatedly oriented toward neighbouring seat A04 four times; torso orientation toward A04 persisted for 5.8 seconds. Review recommended."
}
```

Legitimate-context example:

```json
{
  "seat_id": "A03",
  "state": "NORMAL",
  "evidence": ["HEAD_TURN_RIGHT"],
  "counter_evidence": ["TARGET_BOARD", "ISOLATED_EVENT"],
  "summary": "Brief head turn toward the configured board zone; no review event generated."
}
```

---

# 20. Common Event Output Schema for All 3 Pipelines

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
    "state": "WATCH",
    "start_sec": 41.2,
    "end_sec": 49.8,
    "occurrences": 4
  },
  "context": {
    "target_type": "NEIGHBOUR_SEAT",
    "target_id": "A04"
  },
  "scores": {
    "behaviour_confidence": 0.86,
    "quality": 0.91,
    "risk": 0.57
  },
  "quality": {
    "head": "GOOD",
    "torso": "GOOD",
    "hands": "POOR",
    "track": "GOOD",
    "camera": "GOOD"
  },
  "evidence": [
    "4 rightward head turns",
    "3 directed toward A04"
  ],
  "counter_evidence": []
}
```

---

# 21. Pipeline-Specific Implementation Contract

## 21.1 Anuj — YOLO26 Pose + Tracking

Frozen model:

```text
yolo26n-pose.pt
```

Responsibilities:

- raw `.mkv/.mp4` streaming,
- YOLO26 Pose,
- 17 body keypoints,
- ByteTrack vs BoT-SORT,
- persistent anonymous identity,
- seat anchoring,
- semantic target context,
- head/torso/wrist features,
- all common behaviours,
- temporal events,
- context-aware counter-evidence,
- common output schema,
- runtime metrics.

Required behavioural test:

```text
YOLO detects head turn
→ map direction to context
→ board/invigilator suppresses suspicion
→ repeated neighbour direction increases evidence
```

## 21.2 Om — RTMPose + Object Detection

Responsibilities:

- raw video,
- person detection,
- RTMPose,
- local temporal association,
- seat anchoring,
- semantic context,
- head/torso/wrist features,
- all common behaviours,
- mobile phone detection,
- paper/chit if supported,
- object→seat/student association,
- context-aware counter-evidence,
- common output schema,
- runtime metrics.

Required object test:

```text
phone bbox
+
student bbox
+
seat polygon
+
wrist proximity
+
persistence
→ associated object evidence
```

## 21.3 Dravvya — Multi-HMR + SMPL-X

Responsibilities:

- raw video,
- Multi-HMR,
- SMPL-X,
- multiple reconstructed humans,
- temporal identity association,
- seat anchoring,
- semantic context,
- 3D head yaw/pitch/roll,
- 3D torso yaw/pitch/roll,
- 3D arms/wrist trajectories,
- all common behaviours,
- 3D interaction graph,
- behaviour-specific quality/UNKNOWN,
- common output schema,
- runtime metrics.

Critical requirement:

```text
head_yaw = +30°
```

is not enough. It must resolve to a semantic target such as:

```text
+30° → BOARD
```

or:

```text
+30° → NEIGHBOUR_A04
```

before affecting behavioural risk.

---

# 22. Controlled Representation Benchmark

Hold constant:

- same videos,
- same timestamps,
- same camera zones,
- same seat map,
- same annotations,
- same behaviour definitions,
- same temporal window,
- same hard negatives,
- same validation/test split,
- same evaluation code.

Compare:

```text
YOLO26 Pose
vs
RTMPose
vs
Multi-HMR + SMPL-X
```

---

# 23. Behaviour Metrics

For each behaviour report precision, recall and F1.

| Behaviour | YOLO26 | RTMPose | SMPL-X |
|---|---:|---:|---:|
| Repeated side glance F1 | | | |
| Sustained head turn F1 | | | |
| Torso→neighbour F1 | | | |
| Unusual hand activity F1 | | | |
| Nonverbal interaction F1 | | | |
| Object interaction F1 | | | |
| Macro-F1 | | | |

---

# 24. Context Accuracy Metrics

Add explicit context-resolution metrics:

```text
target-context accuracy
board-look false REVIEW rate
invigilator-look false REVIEW rate
neighbour-target accuracy
```

Target-context classes:

```text
BOARD
INVIGILATOR
NEIGHBOUR_SEAT
ENTRANCE
CLOCK
UNKNOWN_CONTEXT
```

---

# 25. Temporal Metrics

Measure:

```text
Temporal IoU
onset error
end error
duration error
```

---

# 26. Tracking / Association Metrics

All pipelines report where applicable:

```text
IDF1
ID switches
fragmentation
track coverage
lost-track duration
seat-assignment accuracy
```

Anuj additionally benchmarks:

```text
ByteTrack vs BoT-SORT
```

---

# 27. Hard-Negative Metrics

Mandatory:

```text
False Positive Rate
false REVIEW events / student-hour
false WATCH events / student-hour
board-look false REVIEW count
invigilator-look false REVIEW count
normal-writing false REVIEW count
```

---

# 28. Alert Workload Metrics

All pipelines report:

```text
alerts / student-hour
review events / 100 students / hour
false alerts / student-hour
mean event duration
duplicate-event rate
```

Also measure:

```text
original footage duration
vs
Drishti-prioritized review duration
```

Only report actual measured reduction.

---

# 29. Coverage / Abstention Metrics

Report:

```text
UNKNOWN rate
valid-analysis coverage
head coverage
torso coverage
hand coverage
camera-context coverage
```

A system cannot appear better simply by abstaining on difficult cases.

---

# 30. Runtime Metrics

Same hardware, video, resolution and sampling policy.

Measure:

```text
model load time
mean inference ms/frame
P50 latency
P95 latency
effective FPS
peak GPU VRAM
peak RAM
end-to-end FPS
```

Separate pure model inference from full pipeline runtime.

---

# 31. Robustness Test Matrix

| Condition | Expected Behaviour |
|---|---|
| Low light | lower quality / UNKNOWN if needed |
| Blur | lower quality |
| Partial occlusion | component-specific UNKNOWN |
| Distant student | reduced head/hand coverage |
| Crowding | tracking/association stress |
| Oblique camera | context-aware seat-relative features |
| Track loss | monitoring gap / quality reduction |
| Lighting transition | camera-health event if severe |
| Blind spot | explicitly unobservable |
| Camera vibration | health warning / quality reduction |

For transformed stress tests, retain original test data separately.

---

# 32. Camera Readiness Test

Before behaviour analysis, each camera should produce a readiness summary such as:

```text
CAM01

✓ stream readable
✓ FPS stable
✓ board zone configured
✓ invigilator zones configured
✓ seat map configured
✓ neighbour graph configured
✓ expected student scale acceptable

⚠ A11 partially occluded
⚠ D01 head resolution weak

coverage: 91.7%
```

Do not silently run context-aware behaviour if semantic camera context is missing.

---

# 33. Monitoring Gaps

Generate explicit system events:

```text
MONITORING_GAP
CAMERA_DISCONNECTED
DECODER_FAILURE
TRACK_UNRELIABLE
SEVERE_BLUR
SEVERE_DARKNESS
```

Never treat unavailable footage as `NORMAL`.

---

# 34. Dataset Split Rules

Do not randomly split neighbouring frames.

Use:

```text
whole video/session → DEVELOPMENT
different video/session → VALIDATION
unseen video/session → TEST
```

Where possible include a camera/session holdout.

Thresholds must be tuned only on validation data.

---

# 35. Fixed vs Calibrated Benchmark

## Fixed Threshold Mode

Same thresholds for all three pipelines. Purpose: compare representations fairly.

## Calibrated Mode

Each pipeline calibrates on the same validation set, then evaluates once on untouched test data. Purpose: compare each pipeline at its best defensible operating point.

---

# 36. Required Unit Tests

## `test_context.py`

```text
head → board
→ legitimate context

head → invigilator
→ legitimate context

head → neighbour
→ relevant context

head → blind spot
→ low confidence / UNKNOWN context
```

## `test_head_behaviour.py`

```text
single brief glance
→ NORMAL

repeated board glance
→ no REVIEW

repeated neighbour glance
→ WATCH/REVIEW based on threshold

low head quality
→ HEAD UNKNOWN
```

## `test_torso_behaviour.py`

```text
stretch
→ NORMAL

posture adjustment
→ NORMAL

sustained torso neighbour
→ WATCH

head+torso same neighbour
→ REVIEW candidate
```

## `test_hand_behaviour.py`

```text
writing
→ NORMAL

page turn
→ NORMAL

head scratch
→ NORMAL

repeated neighbour-directed hand
→ WATCH/REVIEW
```

## `test_interaction.py`

```text
A+B look at board
→ NORMAL

A→B only once
→ no REVIEW

A→B then B→A repeatedly
→ REVIEW candidate
```

## `test_unknown.py`

```text
head unavailable
→ HEAD UNKNOWN

track unstable
→ strong review blocked

blind spot
→ monitoring unavailable / UNKNOWN
```

---

# 37. Required Integration Tests

## IT-01 — Board Look

Expected:

```text
movement detected = YES
target context = BOARD
counter-evidence = TARGET_BOARD
state = NORMAL
REVIEW generated = NO
```

## IT-02 — Invigilator Look

```text
movement detected = YES
target = INVIGILATOR
state = NORMAL
REVIEW = NO
```

## IT-03 — Brief Neighbour Glance

Recommended hard-negative-safe expectation:

```text
target = NEIGHBOUR_SEAT
single short episode
state = NORMAL
REVIEW = NO
```

## IT-04 — Repeated Neighbour Glance

```text
target = same neighbour
repetition >= calibrated threshold
state = WATCH
```

## IT-05 — Repeated Head + Torso Alignment

```text
head → A04
torso → A04
persistent/repeated
quality good
state = REVIEW
```

## IT-06 — Reciprocal Interaction

```text
A03 → A04
A04 → A03
within Δt
repeated
state = REVIEW candidate
```

## IT-07 — Normal Writing

```text
wrist motion detected
target = OWN_DESK
state = NORMAL
```

## IT-08 — Hand to Neighbour Boundary

```text
hand trajectory crosses boundary
target = neighbour
state = WATCH
```

Additional reciprocal/object evidence may raise to REVIEW.

## IT-09 — Phone Association

```text
phone detected
associated with A03
temporal persistence
state = WATCH/REVIEW according to calibrated object policy
```

## IT-10 — Severe Occlusion

```text
quality reduced
component UNKNOWN
strong review blocked
```

---

# 38. Acceptance Criteria

A pipeline passes context-aware behaviour testing only if:

- [ ] board looks do not generate `REVIEW`
- [ ] invigilator looks do not generate `REVIEW`
- [ ] normal writing does not generate `REVIEW`
- [ ] stretching does not generate `REVIEW`
- [ ] brief isolated head movements do not generate `REVIEW`
- [ ] neighbour-directed repetition increases evidence
- [ ] head+torso alignment is stronger than head-only evidence
- [ ] reciprocal interaction is stronger than one-way evidence
- [ ] bad visibility produces `UNKNOWN` where appropriate
- [ ] missing context does not produce an overconfident alert
- [ ] events contain target context
- [ ] events contain evidence and counter-evidence
- [ ] temporal start/end are recorded
- [ ] duplicate events are merged
- [ ] human-review wording is preserved

---

# 39. Final Comparison Table

| Metric | YOLO26 Pose | RTMPose | Multi-HMR + SMPL-X |
|---|---:|---:|---:|
| Pose/keypoint metric | | | |
| Head coverage | | | |
| Wrist coverage | | | |
| Tracking/association IDF1 | | | |
| Seat assignment accuracy | | | |
| Context target accuracy | | | |
| Board-look false REVIEW rate | | | |
| Invigilator-look false REVIEW rate | | | |
| Repeated glance F1 | | | |
| Sustained head-turn F1 | | | |
| Torso-turn F1 | | | |
| Hand-activity F1 | | | |
| Interaction F1 | | | |
| Object-interaction F1 | | | |
| Macro-F1 | | | |
| Temporal IoU | | | |
| False alerts/student-hour | | | |
| UNKNOWN rate | | | |
| Valid-analysis coverage | | | |
| Low-light F1 | | | |
| Occlusion F1 | | | |
| Oblique-angle F1 | | | |
| Mean latency | | | |
| P95 latency | | | |
| Effective FPS | | | |
| Peak VRAM | | | |

Om additionally reports:

| Object Metric | RTMPose + Detector |
|---|---:|
| Phone precision | |
| Phone recall | |
| Phone F1 / mAP | |
| Phone-to-seat association accuracy | |
| Paper/chit metric if supported | |

---

# 40. Kaggle Test Execution Order — Step by Step

## STEP 0 — Dataset Audit

Collect:

```text
video count
file size
codec
resolution
FPS
duration
frame count
decode status
```

Do not run full model inference yet.

## STEP 1 — Contact Sheets / Camera Inspection

For each video sample a small number of frames and inspect:

```text
camera angle
board location
invigilator location
seat visibility
student scale
occlusion
lighting
```

## STEP 2 — Create Camera Config

Manually define:

```text
seat polygons
neighbours
board
invigilator
entrance
blind spots
```

## STEP 3 — Build Hard-Negative Annotation Sheet

Annotate:

```text
board looks
invigilator looks
normal writing
brief glances
stretch
posture adjustment
```

## STEP 4 — Run Each Independent Pose Pipeline

```text
Anuj     → YOLO26 Pose
Om       → RTMPose
Dravvya  → Multi-HMR + SMPL-X
```

## STEP 5 — Produce Common Geometry/Event Format

Every pipeline writes the shared schema.

## STEP 6 — Run Behaviour Tests

Start with:

```text
NORMAL board look
vs
repeated neighbour glance
```

Only after those pass add:

```text
torso
hands
interaction
objects
```

## STEP 7 — Calibrate Thresholds

Use validation clips only.

## STEP 8 — Run Test Set Once

Produce final metrics.

## STEP 9 — Robustness Tests

Run:

```text
low light
blur
occlusion
downsample
camera/session holdout
```

## STEP 10 — Build Final Benchmark Report

Do not cherry-pick individual model successes.

---

# 41. Minimum Test Annotation CSV

Recommended schema:

```csv
video_id,camera_id,seat_id,target_seat_id,start_sec,end_sec,behaviour,target_context,expected_state,quality_note
VIDEO01,CAM01,A03,,10.0,12.0,NORMAL,BOARD,NORMAL,good
VIDEO01,CAM01,A03,A04,30.0,45.0,REPEATED_SIDEWARD_GLANCE,NEIGHBOUR_SEAT,WATCH,good
VIDEO01,CAM01,A03,A04,50.0,60.0,TORSO_ROTATION_TOWARD_NEIGHBOUR,NEIGHBOUR_SEAT,REVIEW,good
```

---

# 42. Behaviour Test Result CSV

Recommended:

```csv
pipeline,video_id,seat_id,test_id,expected_state,predicted_state,behaviour,target_context,start_error,end_error,quality,pass
YOLO26,VIDEO01,A03,IT-01,NORMAL,NORMAL,NORMAL,BOARD,0.1,0.2,0.92,true
RTMPOSE,VIDEO01,A03,IT-01,NORMAL,NORMAL,NORMAL,BOARD,0.1,0.1,0.94,true
SMPLX,VIDEO01,A03,IT-01,NORMAL,NORMAL,NORMAL,BOARD,0.2,0.2,0.88,true
```

---

# 43. Non-Negotiable Failure Rules

The implementation fails if:

```text
board look → REVIEW
invigilator look → REVIEW
normal writing → REVIEW
stretch → REVIEW
poor visibility → confident REVIEW
blind spot → NORMAL without warning
missing track → fabricated behaviour
```

It also fails if a configured semantic camera context exists but the event output does not record target context.

---

# 44. Final Engineering Principle

For every observed movement:

```text
OBSERVE
   ↓
LOCALIZE PERSON
   ↓
ASSIGN SEAT
   ↓
ESTIMATE MOVEMENT
   ↓
RESOLVE TARGET CONTEXT
   ↓
CHECK QUALITY
   ↓
CHECK TEMPORAL PERSISTENCE
   ↓
CHECK REPETITION
   ↓
CHECK SUPPORTING SIGNALS
   ↓
CHECK COUNTER-EVIDENCE
   ↓
CHECK RECIPROCAL INTERACTION
   ↓
NORMAL / WATCH / REVIEW / UNKNOWN
```

Not:

```text
MOVEMENT
→ SUSPICIOUS
```

---

# 45. Definition of Test Completion

Drishti-4D's behaviour layer is test-ready only when all three pipelines can be run on the same test annotations and produce:

```text
pose/geometry output
+
temporal identity
+
seat assignment
+
semantic target context
+
behaviour event
+
quality
+
counter-evidence
+
state
+
event timeline
+
runtime metrics
```

and all results can populate the common benchmark tables in this document.

---

# 46. Source Documents Consolidated

This specification consolidates and extends:

```text
DRISHTI_IMPLEMENTATION_SPEC(1).md
DRISHTI_4D_FINAL_WORK_DISTRIBUTION(1).md
Problem Statement DrishtiAI Hackathon.pdf
DRISHTI-Improvements.pdf
```

Key retained ideas:

- YOLO26 Pose baseline,
- RTMPose independent pipeline,
- Multi-HMR + SMPL-X independent 3D pipeline,
- common behaviour taxonomy,
- seat anchoring,
- temporal behaviour,
- semantic camera zones,
- hard-negative natural-behaviour tests,
- counter-evidence,
- behaviour-specific UNKNOWN,
- neighbour interaction,
- object attribution,
- calibrated risk,
- event merging,
- false-alert workload metrics,
- robustness evaluation,
- human review rather than cheating accusation.

---

# 47. Immediate Next Step

Do **not** implement this entire document at once.

Current execution sequence:

```text
1. Complete dataset forensic audit
2. Generate representative contact sheets
3. Define camera semantic zones
4. Create initial hard-negative annotations
5. Run first model baseline
6. Test BOARD LOOK vs NEIGHBOUR LOOK
7. Only then proceed to full behaviour implementation
```

This document is the complete final test contract, but implementation remains one verified step at a time.

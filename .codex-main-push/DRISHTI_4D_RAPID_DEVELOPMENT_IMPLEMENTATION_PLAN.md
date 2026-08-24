# DRISHTI-4D — Rapid Development & Prototyping Implementation Plan

> **Project:** DrishtiAI Hackathon — Problem Statement #1  
> **Mode:** RAPID DEVELOPMENT / PROTOTYPING  
> **Execution philosophy:** **BUILD → RUN → BREAK → DEBUG → FIX → CONTINUE**  
> **Purpose:** Handoff-ready implementation plan for an executor model.  
> **Important:** This is an engineering plan, not a claim that any metric or model result has already been achieved.

---

# 0. EXECUTOR INSTRUCTIONS — READ FIRST

1. **Development is the first priority.** Do not spend the first cycle building publication-grade evaluation, exhaustive tests, or perfect threshold calibration.
2. **Do not build a monolith.** Video ingest, YOLO26, RTMPose, Multi-HMR/SMPL-X, object detection, tracking, context, behaviour, risk, storage and dashboard must remain replaceable modules.
3. **Heavy GPU providers run independently.** Persist their outputs and replay them through the shared behaviour engine. A single Kaggle T4 does not need to hold every model simultaneously.
4. **Never rerun expensive inference just because downstream logic changed.** Cache provider outputs in JSONL/NPZ artifacts.
5. **`timestamp_sec` is the temporal source of truth.** Do not infer behaviour duration from nominal FPS or frame count.
6. Use only these final states: `NORMAL`, `WATCH`, `REVIEW`, `UNKNOWN`. `UNKNOWN` is not `NORMAL`.
7. Use the frozen behaviour taxonomy:
   - `REPEATED_SIDEWARD_GLANCE`
   - `SUSTAINED_EXCESSIVE_HEAD_TURN`
   - `TORSO_ROTATION_TOWARD_NEIGHBOUR`
   - `UNUSUAL_HAND_ACTIVITY`
   - `POSSIBLE_NONVERBAL_INTERACTION`
   - `POSSIBLE_OBJECT_INTERACTION`
8. UI and API language must say **Review Event / Behavioural Risk / Observed Evidence**, never “Cheating Detected”.
9. No face recognition, student-name mapping, biometric identity matching, or persistent face embeddings.
10. Never fake functionality. If a provider/module is unavailable, expose it as unavailable or use an explicitly named replay/mock adapter only to keep integration moving.
11. Do not rewrite a working subsystem without an actual reason.
12. During rapid development, use smoke checks only: import/build, one representative video, expected artifacts, schema compatibility.
13. Stop only for genuine blockers such as a missing licensed model asset, impossible dependency conflict, unavailable credential, or missing hardware. Routine coding failures must be debugged and fixed.
14. A raw `PHONE` detection is never a final alert. It must pass phone/monitor/mouse disambiguation, association, temporal/context and uncertainty logic.

---

# 1. PS PRODUCT CONTRACT

The completed prototype must cover:

- multi-person seated-student detection;
- anonymous continuous tracking;
- pose/skeletal landmarks;
- head, torso and hand movement analysis;
- possible non-verbal interaction;
- feasible mobile phone / paper / unwanted-object detection;
- real-time or near-real-time review events;
- explainable reasons, confidence/quality and event timelines;
- behavioural analytics/event logs;
- privacy-conscious monitoring;
- normal-vs-suspicious context handling;
- occlusion, low light, glare, blur, low resolution and blind spots;
- head-pose uncertainty;
- identity switching / track loss;
- camera/network/hardware failures;
- long-duration video;
- multiple cameras and compute constraints;
- seating/camera configurability across institutions;
- **behavioural risk based on sustained patterns instead of isolated motion**;
- human review as the final decision.

---

# 2. TARGET ARCHITECTURE

```text
VIDEO / RTSP
    ↓
VIDEO SOURCE
    ↓
CAMERA HEALTH
    ↓
PERCEPTION PROVIDERS
 ┌─────────────┬─────────────┬──────────────────┐
 ↓             ↓             ↓                  ↓
YOLO26      RTMPose      Multi-HMR+SMPL-X    Objects
+ Track     + Local ID      + 3D ID
 └─────────────┴─────────────┴──────────────────┘
                    ↓
          COMMON OBSERVATION SCHEMA
                    ↓
              SEAT ANCHORING
                    ↓
          SEMANTIC CAMERA CONTEXT
                    ↓
          TEMPORAL STATE PER ENTITY
       ┌────────────┼────────────┐
       ↓            ↓            ↓
      HEAD        TORSO         HAND
       └──────┬─────┴─────┬──────┘
              ↓           ↓
         INTERACTION   OBJECT EVIDENCE
              └──────┬────┘
                     ↓
             QUALITY / UNKNOWN
                     ↓
              COUNTER-EVIDENCE
                     ↓
              BEHAVIOURAL RISK
                     ↓
                EVENT MERGING
                     ↓
          EXPLAINABLE REVIEW EVENT
              ┌──────┴──────┐
              ↓             ↓
          EVENT STORE    DASHBOARD
```

---

# 3. FROZEN TEAM OWNERSHIP

## Anuj — YOLO26 Pose + Tracking
Own:
- `yolo26n-pose.pt` initial fast baseline;
- multi-person pose;
- ByteTrack / BoT-SORT integration;
- persistent local tracks;
- 2D head/torso/wrist primitives;
- provider output serialization;
- runtime artifacts.

**Do not silently replace YOLO26 with YOLOv8/YOLO11.**

## Om — RTMPose + Object Detection
Own:
- person detection required by the RTMPose path;
- RTMPose;
- independent local temporal association;
- 2D pose primitives;
- mobile-phone detection;
- paper/chit only if feasible;
- object→person / object→seat association;
- provider artifacts.

## Dravvya — Multi-HMR + SMPL-X
Own:
- Multi-HMR;
- SMPL-X assets/environment;
- multi-person 3D observations;
- temporal 3D association;
- 3D head/torso/arm-wrist geometry;
- 3D interaction evidence;
- provider artifacts.
Heavy inference should use Kaggle.

## Shared
- schemas;
- seat anchoring;
- camera profiles;
- semantic context;
- temporal buffers;
- behaviour engine;
- counter-evidence;
- risk;
- events;
- storage/API/dashboard;
- camera health;
- RTSP/DeepStream.

---

# 4. REPOSITORY STRUCTURE

```text
drishti4d/
├── README.md
├── configs/
│   ├── app.yaml
│   ├── behaviour.yaml
│   ├── providers.yaml
│   └── cameras/CAM12.yaml
├── artifacts/.gitkeep
├── src/drishti/
│   ├── schemas/{observation,evidence,event,health}.py
│   ├── video/{source,file_source,rtsp_source,timestamps}.py
│   ├── health/camera_health.py
│   ├── providers/{base,replay,yolo26_provider,rtmpose_provider,object_provider,smplx_provider}.py
│   ├── tracking/{entities,seat_anchor}.py
│   ├── context/{camera_profile,zones,neighbour_graph,target_resolver}.py
│   ├── temporal/{buffer,smoothing}.py
│   ├── behaviour/{head,torso,hands,interaction,objects,counter_evidence,engine}.py
│   ├── risk/{scorer,state_machine}.py
│   ├── events/{merger,builder,clipper,summaries}.py
│   ├── storage/{event_store,artifact_store}.py
│   └── api/app.py
├── frontend/
├── scripts/{run_video,run_provider,replay_provider,run_server}.py
└── deployment/{docker,deepstream}/
```

Create modules only when the corresponding phase starts; avoid a repository full of empty abstractions.

---

# 5. SHARED SCHEMAS — PHASE 0A

Implement these **before provider branches merge**.

## `PersonObservation`
Required fields:
```text
schema_version
provider
camera_id
video_id
frame_index
timestamp_sec
entity_local_id
bbox_xyxy_px
bbox_xyxy_norm
detection_confidence
keypoints_2d           # nullable
joints_3d              # nullable
body_parameters        # nullable/reference
quality
metadata
```

## `ObjectObservation`
```text
camera_id
video_id
frame_index
timestamp_sec
object_class
confidence
bbox_xyxy_px
associated_entity_id
associated_seat_id
association_confidence
```

## `BehaviourEvidence`
```text
evidence_id
camera_id
timestamp_sec
entity_id
seat_id
evidence_type
target_type
target_id
strength
quality
start_sec
end_sec
reason_codes
source_provider
```

## `RiskSnapshot`
```text
camera_id
timestamp_sec
entity_id
seat_id
risk_score
state
evidence_ids
counter_evidence_ids
quality
```

## `ReviewEvent`
```text
event_id
camera_id
entity_id
seat_id
start_sec
end_sec
behaviour
state
target_type
target_id
peak_risk
mean_quality
evidence
counter_evidence
summary
clip_path
snapshot_path
review_status
```

---

# 6. PHASE 0B — BOOTSTRAP THE CONTROL PLANE

### Tasks
1. Create repository/package.
2. Implement the shared schemas.
3. Implement `VideoSource` abstraction.
4. Implement working `FileVideoSource`.
5. Create `RTSPVideoSource` interface/stub but do not block on RTSP.
6. Implement `PerceptionProvider` interface:
   ```python
   start(...)
   process_frame(...)
   close(...)
   ```
7. Implement `ReplayProvider` for persisted JSONL/NPZ observations.
8. Add an in-process observation→behaviour→risk→event pipeline.
9. Create CLI:
   ```bash
   python scripts/run_video.py --source <video> --camera CAM12 --provider replay
   ```
10. Create artifact/run directories.

### Exit condition
A video or persisted observation stream can travel:
```text
source → provider/replay → behaviour pipeline → event JSON
```

---

# 7. PHASE 1 — FIRST WORKING VERTICAL SLICE

### Goal
Get one real end-to-end behaviour path working immediately.

Preferred first provider: YOLO26 if available; otherwise replay persisted observations.

Implement:
```text
pose
→ track
→ seat
→ head direction
→ semantic target
→ temporal history
→ counter-evidence
→ risk
→ event
```

### Mandatory context distinction
```text
HEAD_TURN → OWN_MONITOR     = legitimate context
HEAD_TURN → BOARD           = legitimate where configured
HEAD_TURN → INVIGILATOR     = legitimate context
HEAD_TURN → NEIGHBOUR_SEAT  = behaviourally relevant
```

### Exit condition
One actual exam video produces:
```text
track + seat + head movement + target context
+ NORMAL/WATCH/REVIEW/UNKNOWN + event JSON
```

Move forward immediately.


# 8. PHASE 2 — CAMERA PROFILE + SEAT INTELLIGENCE

The PS requires adaptation to different seating layouts and camera placements. Therefore do not hard-code global pixel directions.

## 8.1 Camera profile

Example:

```yaml
camera_id: CAM12

resolution:
  width: 1280
  height: 720

seats:
  A01:
    polygon: [...]
  A02:
    polygon: [...]

neighbors:
  A01:
    left: null
    right: A02
  A02:
    left: A01
    right: A03

zones:
  own_monitor_regions: [...]
  board: null
  invigilator: [...]
  entrance: [...]
  room_front: [...]
  clock: null

blind_spots:
  - polygon: [...]
```

Only configure zones that actually exist in that camera.

## 8.2 Seat anchoring

Input:
```text
tracked person bbox / body anchor / projected person position
```

Output:
```text
track_17 → A04
```

Use temporal hysteresis so one noisy frame does not reassign a seat.

## 8.3 Person ≠ seat

Required for seat exchange:
```text
person track P03 ≠ seat A04
```

A person may leave; another may later occupy the same seat.

### Exit condition
Overlay shows:
```text
track ID + seat ID + neighbours + semantic zones
```

---

# 9. PHASE 3 — ANUJ: YOLO26 POSE + TRACKING PROVIDER

## Goal
Primary fast 2D provider.

## Tasks
1. Load configurable YOLO26 Pose checkpoint. Initial:
   ```text
   yolo26n-pose.pt
   ```
2. Stream frames; never load complete video into RAM.
3. Extract:
   - person bbox;
   - 17 keypoints;
   - per-keypoint confidence;
   - detection confidence;
   - `timestamp_sec`.
4. Add tracker support:
   ```text
   ByteTrack
   BoT-SORT
   ```
   Choose whichever behaves better during real footage development; do not block on a formal benchmark.
5. Serialize every tracked observation to:
   ```text
   artifacts/<video>/yolo26/observations.jsonl
   ```
6. Save runtime:
   ```text
   total frames
   wall time
   effective FPS
   mean inference time
   peak VRAM when available
   ```
7. Compute provider-side primitives only:
   - head landmarks;
   - shoulder/hip geometry;
   - elbow/wrist geometry.
8. Do **not** make final misconduct decisions inside the provider.

### Exit condition
Provider runs standalone and its saved output can be replayed without loading YOLO26 again.

---

# 10. PHASE 4 — OM: RTMPOSE + OBJECT PROVIDER

## 10.1 RTMPose
Use a supported MMPose RTMPose configuration.

Pipeline:
```text
frame
→ person detector / person boxes
→ RTMPose
→ local temporal association
→ common observation
```

Persist:
```text
artifacts/<video>/rtmpose/observations.jsonl
```

Include:
```text
2D keypoints
confidence
bbox
timestamp
local entity ID
```

## 10.2 Independent temporal association
Om must not require Anuj's tracker.

Use:
```text
bbox IoU
center displacement
seat prior
temporal continuity
```

## 10.3 Object detector
Priority:
```text
1. MOBILE_PHONE
2. PAPER_CHIT only if visually/model-feasible
3. other unwanted objects only if support exists
```

## 10.4 Object attribution
Do not stop at “phone detected”.

Associate using:
```text
object bbox
+ person bbox
+ seat polygon
+ wrist proximity
+ temporal persistence
```

Return:
```text
associated_entity_id
associated_seat_id
association_confidence
```

Persist:
```text
artifacts/<video>/objects/detections.jsonl
```

### Exit condition
RTMPose and object outputs can run/replay independently of Anuj.

---

# 10A. OBJECT DISAMBIGUATION — PHONE VS MONITOR VS MOUSE

This subsection is **mandatory** for the object pipeline. Low-resolution CBT CCTV can make switched-off monitors, computer mice, phones, keyboards and dark workstation regions visually similar. Therefore a raw object-detector class/confidence must never be treated as the final decision.

Required flow:

```text
OBJECT DETECTOR
    ↓
PHONE_CANDIDATE / MOUSE / MONITOR / OTHER
    ↓
STATIC WORKSTATION FILTER
    ↓
SEAT + REGION CONTEXT
    ↓
PERSON / HAND / LAP ASSOCIATION
    ↓
TEMPORAL OBJECT BEHAVIOUR
    ↓
OPTIONAL SECOND-STAGE VERIFIER
    ↓
PHONE / MOUSE / MONITOR / OTHER / UNCERTAIN_OBJECT
    ↓
POSSIBLE_OBJECT_INTERACTION evidence
    ↓
SHARED RISK ENGINE
```

## 10A.1 Required classes / states

Support:
```text
PHONE
MOUSE
MONITOR
OTHER
UNCERTAIN_OBJECT
```
If the base detector cannot reliably classify all of them, allow `PHONE_CANDIDATE` and resolve it downstream. Never force an ambiguous low-quality object into `PHONE`.

## 10A.2 Per-seat workstation geometry

Extend camera configuration:

```yaml
seats:
  A03:
    seat_polygon: [...]
    monitor_polygon: [...]
    keyboard_region: [...]
    mouse_region: [...]
    own_desk_region: [...]
    lap_region: [...]
```

Use normalized coordinates where practical. Only configure regions actually visible in that camera.

Mandatory priors:
```text
high monitor-region overlap → MONITOR prior ↑ / phone evidence ↓
inside mouse region          → MOUSE prior ↑
inside lap region            → PHONE prior ↑
outside normal equipment     → attention ↑, but never classify from location alone
```

## 10A.3 Static workstation memory

Exploit fixed CBT cameras. Maintain per-seat stable equipment regions using verified camera config and/or a stable initialization period. Track:

```text
object center trajectory
bbox trajectory
position variance
persistence
region occupancy
movement relative to wrist/person
appearance/disappearance
```

Typical monitor evidence:
```text
fixed location + monitor-region overlap + long persistence + weak hand correlation
```

Typical mouse evidence:
```text
inside mouse region + small localized motion + nearby operating wrist + stays within workstation
```

Typical handheld-phone evidence:
```text
appears/disappears + moves across desk/hand/lap/body regions + travels with wrist/hand
```

Staticness alone is never enough because a phone may rest on the desk.

## 10A.4 Person / seat association

Every candidate must be associated before it contributes strong evidence. Compute:

```text
inside person bbox?
inside seat polygon?
inside own desk?
inside lap?
nearest entity?
nearest seat?
```

Output:
```text
associated_entity_id
associated_seat_id
association_confidence
```

No reliable association → object evidence remains weak or `UNKNOWN`.

## 10A.5 Wrist / hand association

Use pose outputs to compute temporal distances to left/right wrists and hand regions. Useful evidence includes:

```text
object moves with wrist
object remains near one/both hands
object follows hand into lap/body region
```

Mouse pattern:
```text
wrist operates around a mostly local desk-region object
```

Phone pattern:
```text
object and wrist trajectories correlate and/or object changes workstation region
```

Do not require hand proximity as a hard rule because a phone may be placed on the desk.

## 10A.6 Explicit monitor suppression

```text
PHONE_CANDIDATE
+ high monitor overlap
+ long-term static position
+ no meaningful wrist correlation
→ MONITOR / suppress phone evidence
```

Never use colour as the main criterion. `black rectangle = phone` is invalid.

## 10A.7 Explicit mouse suppression

```text
PHONE_CANDIDATE
+ inside mouse region
+ localized low-amplitude movement
+ wrist operates nearby
+ candidate remains in workstation region
→ MOUSE / suppress phone evidence
```

Stronger phone evidence:
```text
outside normal mouse region
+ lap/body/hand association
+ wrist/object motion correlation
+ cross-region movement or appearance/disappearance
```

## 10A.8 Hard-negative mining

Rapid development must harvest real false positives from target CCTV:

```text
monitor → phone
mouse → phone
keyboard/stationery → phone
dark desk patch → phone
```

Save crop + metadata:
```text
video_id, camera_id, timestamp_sec, seat_id, predicted_class, confidence, verified_label, crop_path
```

Fine-tune/retrain with target-domain labels such as:
```text
PHONE
MONITOR
MOUSE
OTHER_WORKSTATION_OBJECT
```

Prefer actual exam-CCTV hard negatives over generic internet negatives.

## 10A.9 Optional second-stage verifier

If context still leaves too many false positives, add a replaceable crop classifier:

```text
candidate crop + local context → PHONE / MOUSE / MONITOR / OTHER
```

Do not block the initial build on this if geometry + temporal context already resolves most cases.

## 10A.10 ROI reprocessing for tiny objects

Optionally re-run object analysis on:
```text
person ROI
seat/workstation ROI
lap ROI
```
This improves effective model focus but does not create missing visual detail. Never treat upscaled pixels as new evidence.

## 10A.11 Final object evidence

Raw detector confidence is not final phone confidence. Fuse:

```text
detector evidence
+ seat/person association
+ lap evidence
+ wrist proximity
+ wrist/object motion correlation
+ region transitions
- monitor overlap
- mouse-region prior
- long-term workstation staticness
- poor-quality penalty
```

All weights/thresholds must be config-driven.

## 10A.12 `UNCERTAIN_OBJECT`

When quality is poor or evidence conflicts:
```text
resolved_class = UNCERTAIN_OBJECT
phone_evidence = LOW
```
Never escalate directly to `REVIEW`.

## 10A.13 Object detection cannot directly create REVIEW

Incorrect:
```text
PHONE confidence > threshold → REVIEW
```

Required:
```text
raw candidate → disambiguation → association → temporal/context evidence
→ POSSIBLE_OBJECT_INTERACTION → shared risk engine
→ NORMAL / WATCH / REVIEW / UNKNOWN
```


# 11. PHASE 5 — DRAVVYA: MULTI-HMR + SMPL-X PROVIDER

## Goal
Produce useful 3D evidence from real CCTV.

## 11.1 Environment
Run the heavy provider on Kaggle in an isolated environment.

Do not modify/destroy the notebook's base PyTorch environment if Multi-HMR requires older dependencies.

## 11.2 SMPL-X assets
SMPL-X model files are externally licensed/downloaded assets.

Expected local pattern:
```text
models/smplx/SMPLX_NEUTRAL.npz
```

Rules:
- never commit licensed model files;
- add asset directories to `.gitignore`;
- fail clearly if asset is missing.

## 11.3 First development target
Use the manageable selected clip:
```text
VIDEO_05 — 03.CCTV Mobile Usage.mkv
```

Start with short windows, then expand once inference works.

## 11.4 Persist 3D output
For each person/time save what the model actually provides/usefully derives:
```text
timestamp
projected 2D joints
3D joints
SMPL-X/body parameters
global/body orientation
camera-space translation
projected bbox
fit / inference quality
```

Artifacts:
```text
artifacts/VIDEO_05/smplx/observations.jsonl
artifacts/VIDEO_05/smplx/joints_3d.npz
artifacts/VIDEO_05/smplx/body_params.npz
artifacts/VIDEO_05/smplx/runtime.json
```

## 11.5 Temporal association
Multi-HMR is primarily frame/image inference; create temporal identity association using:
```text
projected bbox overlap
2D person position
seat prior
camera-space location if stable
temporal continuity
```

## 11.6 3D primitives
Derive:
```text
head orientation
torso orientation
arm/wrist trajectory
body-relative hand position
```

Important:
- hidden legs under desks are inferred, not observed;
- do not treat occluded reconstructed geometry as certain;
- expose quality and allow `UNKNOWN`.

### Exit condition
A short CCTV interval yields:
```text
stable entity + 3D joints/mesh parameters
+ usable head/torso/arm evidence
+ persisted artifacts
```

---

# 12. PHASE 6 — TEMPORAL STATE ENGINE

Shared across providers.

## Tasks
1. Maintain a bounded history:
   ```text
   entity_id → deque(timestamped observations)
   ```
2. Initial configurable history:
   ```text
   ~12 seconds
   ```
3. Use time, not frame counts.
4. Short missing gaps:
   ```text
   keep state but reduce confidence
   ```
5. Long gaps:
   ```text
   break evidence continuity
   ```
6. Normalize movement by local body scale:
   ```text
   bbox diagonal / shoulder width / torso length
   ```

### Exit condition
Replay of saved provider output can produce stable time-window features.

---

# 13. PHASE 7 — HEAD BEHAVIOUR

## Inputs
```text
relative head orientation
target direction
duration
episode count
return-to-neutral
quality
```

## Final behaviours
```text
REPEATED_SIDEWARD_GLANCE
SUSTAINED_EXCESSIVE_HEAD_TURN
```

## Context-aware logic
Not:
```text
HEAD_TURN_RIGHT → suspicious
```

Instead:
```text
HEAD_TURN_RIGHT → OWN_MONITOR      → counter-evidence
HEAD_TURN_RIGHT → INVIGILATOR      → counter-evidence
HEAD_TURN_RIGHT → BOARD            → counter-evidence where configured
HEAD_TURN_RIGHT → SEAT_A04         → behavioural evidence
HEAD_TURN_RIGHT → BLIND_SPOT       → uncertainty
HEAD_TURN_RIGHT → UNKNOWN_CONTEXT  → weak/uncertain evidence
```

Suppress isolated normal movement:
```text
brief look-around
thinking movement
single short glance
```

---

# 14. PHASE 8 — TORSO BEHAVIOUR

2D providers:
```text
shoulder/hip geometry
apparent shoulder width
body-center displacement
```

SMPL-X:
```text
3D body orientation
```

Final behaviour:
```text
TORSO_ROTATION_TOWARD_NEIGHBOUR
```

Strong combination:
```text
head → same neighbour
+ torso → same neighbour
+ persistence
```

Counter-evidence:
```text
stretch
posture correction
own-desk reach
turn toward invigilator
```

---

# 15. PHASE 9 — HAND / GESTURE BEHAVIOUR

Signals:
```text
wrist velocity
wrist acceleration
arm extension
seat-boundary crossing
desk-region crossing
neighbour-directed motion
repetition
```

Final behaviour:
```text
UNUSUAL_HAND_ACTIVITY
```

Natural counter-evidence:
```text
writing
page turning
stationery handling
head scratching
clothing adjustment
```

Generic movement must never automatically become suspicious.

---

# 16. PHASE 10 — INTERACTION ENGINE

Maintain directed entity relations:
```text
A → B
B → A
```

Possible signals:
```text
head A → B
torso A → B
hand A → B
```

Look for temporal response:
```text
A → B
then
B → A within Δt
```

Final label:
```text
POSSIBLE_NONVERBAL_INTERACTION
```

Suppress shared legitimate context:
```text
both look toward invigilator
both react to room front
global classroom disturbance
```

---

# 17. PHASE 11 — QUALITY + UNKNOWN

Required component quality:
```text
HEAD    = GOOD / POOR / UNKNOWN
TORSO   = GOOD / POOR / UNKNOWN
HANDS   = GOOD / POOR / UNKNOWN
TRACK   = GOOD / POOR / UNKNOWN
CAMERA  = GOOD / DEGRADED / OFFLINE
```

## Rules
- Occluded head → head quality falls; torso may remain usable.
- Low-resolution head → do not output fake precise orientation.
- Tracking instability → block strong attribution.
- Blind spot → `MONITORING_GAP`, not `NORMAL`.
- Missing evidence is not zero evidence.
- A provider disagreement may reduce confidence rather than forcing a result.

---

# 18. PHASE 12 — CAMERA HEALTH

Implement low-cost signals:
```text
stream connected
actual FPS
decode errors / dropped frames
brightness
severe darkness
blur
frozen feed
global camera movement/vibration proxy
```

Health events:
```text
CAMERA_DISCONNECTED
LOW_FPS
FROZEN_STREAM
SEVERE_BLUR
SEVERE_DARKNESS
DECODER_FAILURE
GLOBAL_CAMERA_MOVEMENT
MONITORING_GAP
```

Never convert camera failure into a student behaviour alert.

---

# 19. PHASE 13 — COUNTER-EVIDENCE ENGINE

Required reason codes:
```text
TARGET_OWN_MONITOR
TARGET_BOARD
TARGET_INVIGILATOR
TARGET_CLOCK
BRIEF_ISOLATED_MOVEMENT
RETURNED_TO_WRITING
STRETCH_PATTERN
OWN_DESK_HAND_ACTIVITY
GLOBAL_CLASSROOM_RESPONSE
POOR_QUALITY
```

Important:
```text
phone not detected
```
is **not** proof that a head/torso event is normal.

---

# 20. PHASE 14 — BEHAVIOURAL RISK ENGINE

Inputs:
```text
H = head evidence
T = torso evidence
G = hand evidence
I = interaction evidence
O = object evidence
Q = quality
C = counter-evidence
```

Risk depends on:
```text
strength
+ persistence
+ repetition
+ context
+ cross-signal agreement
+ reciprocal interaction
+ object evidence
- counter-evidence
```

Use temporal memory/decay.

Final state machine:
```text
NORMAL
WATCH
REVIEW
UNKNOWN
```

Thresholds are config-driven engineering defaults during rapid development.

Do not claim they are calibrated unless they actually are.


# 21. PHASE 15 — EVENT MERGING + EXPLAINABILITY

Do not emit one alert per frame.

Bad:
```text
head alert
head alert
head alert
head alert
```

Good:
```text
Seat A03
Repeated orientation toward A04
4 occurrences
23.1 → 44.6 sec
```

Each review event must contain:
```text
camera
anonymous entity
seat
start/end time
behaviour
target
peak risk
quality
supporting evidence
counter-evidence
reason codes
snapshot
evidence clip
review status
```

Generate explanations deterministically from observed evidence.

Do not add an LLM to the critical path during rapid prototyping.

---

# 22. PHASE 16 — EVENT STORAGE

Rapid prototype recommendation:
```text
SQLite or PostgreSQL
+
filesystem/object-style artifact directory
```

Minimum logical tables:
```text
cameras
events
event_evidence
camera_health_events
runs
```

Provider arrays/large outputs remain in JSONL/NPZ/files.

Do not unnecessarily duplicate complete raw videos.

---

# 23. PHASE 17 — API + DASHBOARD

The dashboard may be built while specialist providers are still being completed, once the event API/schema is stable.

## Required UI

### Camera view
Show:
```text
anonymous track
seat ID
current state
camera health
```

### Review queue
Priority:
```text
REVIEW
WATCH
```

### Event card
Show:
```text
seat
time
observed behaviour
target
behavioural risk
quality/confidence
reason
clip/snapshot
```

### Timeline
Show:
```text
NORMAL
WATCH
REVIEW
UNKNOWN
MONITORING_GAP
```

### Camera readiness
Show:
```text
online/offline
FPS
quality
configured seats
blind spots / degraded monitoring
```

Never display:
```text
Cheating probability
```

Use:
```text
Behavioural Risk
Review Priority
```

---

# 24. PHASE 18 — RTSP REAL-TIME INGEST

After file-video vertical flow works, implement `RTSPVideoSource`.

Required:
```text
reconnect
exponential/controlled backoff
timestamps
health state
dropped/decode error reporting
```

One stream reconnect must not freeze the whole application.

---

# 25. PHASE 19 — DEEPSTREAM

DeepStream is optional deployment acceleration, not the behaviour engine.

Use for:
```text
RTSP ingest
hardware decode
multi-stream batching
GPU inference integration where practical
tracking integration where practical
```

Boundary must remain:
```text
DeepStream output
→ Drishti Observation Schema
→ Drishti Context/Behaviour/Risk
```

If DeepStream blocks rapid development, keep the file/OpenCV/GStreamer route working and continue.

---

# 26. PHASE 20 — GPU RESOURCE STRATEGY

Do **not** load all heavy models together on one T4.

Recommended execution:

```text
RUN A
YOLO26 → save → process exits

RUN B
RTMPose → save → process exits

RUN C
Object model → save → process exits

RUN D
Multi-HMR + SMPL-X → save → process exits
```

Then replay artifacts through the CPU/lightweight shared engine.

A likely final specialist design:

```text
YOLO26 = fast primary

if 2D evidence uncertain:
    RTMPose refinement

if important event needs 3D geometry:
    Multi-HMR + SMPL-X
```

Do not force SMPL-X on every candidate at every frame unless actual throughput proves it feasible.

---

# 27. PHASE 21 — LONG-VIDEO HANDLING

Current footage includes long recordings.

Never:
```text
load entire video into RAM
extract every frame to PNG
keep unbounded per-person history
```

Use streaming and bounded buffers.

For heavy providers:
```text
coarse/primary processing
→ candidate time window
→ denser specialist inference
```

Persist the result.

---

# 28. COMPLETE PS CHALLENGE COVERAGE MATRIX

The executor must preserve handling for every row.

| Problem / Challenge | Rapid Prototype Implementation |
|---|---|
| Natural movement false positives | semantic context + counter-evidence + temporal persistence |
| Missed suspicious signals | multi-signal fusion + alternate/specialist providers |
| Occlusion | component-specific quality + `UNKNOWN` |
| Poor lighting | camera quality + confidence degradation |
| Glare / overexposure | quality degradation / `UNKNOWN` |
| Low resolution | quality gating; do not invent precise pose |
| Motion blur | blur signal + reduced confidence |
| Camera blind spots | camera profile + `MONITORING_GAP` |
| Camera placement variation | normalized per-camera zones/configuration |
| Seating variation | configurable seat polygons + neighbour graph |
| Head-pose estimation error | quality + context + alternative providers |
| Tracking identity switches | tracker + seat anchor + association quality |
| Track loss | temporal gap state / reduced confidence |
| Multiple students | multi-person providers + per-entity state |
| Student interaction | directed relation graph `A→B`, `B→A` |
| Mobile phone | detector + phone/monitor/mouse disambiguation + seat/hand/lap association + temporal context |
| Paper/chit | detect only if feasible; otherwise hand/object interaction evidence |
| False alert fatigue | temporal aggregation + event merging + counter-evidence |
| Long examinations | streaming + bounded memory + persisted artifacts |
| Behaviour diversity | neutral/reference baseline + configurable thresholds |
| Multiple cameras | independent camera workers + common event layer |
| Network failure | reconnect + health event + monitoring gap |
| Hardware/model failure | provider/camera state `READY/DEGRADED/FAILED` |
| Privacy | anonymous tracks/seats; no identity recognition |
| Explainability | reason codes + evidence + timeline |
| Human oversight | `REVIEW`, never automatic guilt |
| Cross-institution generalization | camera profiles + config-driven geometry |
| Dataset limitations | do not fabricate unsupported labels/performance |
| Compute constraints | provider isolation + caching + selective specialists |
| Storage constraints | compact artifacts + event clips, avoid frame duplication |

---

# 29. PHASE 22 — RAPID ROBUSTNESS FIX PASS

This is **not** a research benchmark phase.

Run the actual product against representative scenarios:

```text
phone-use clips
seat exchange
crowded reception
paper interaction
candidate talking
8 FPS / 480p footage
12 FPS footage
2+ hour recordings
occluded cubicles
```

Workflow:
```text
run
→ inspect obvious wrong behaviour
→ trace responsible module
→ patch
→ rerun
→ move on
```

Do not build a massive evaluation framework here.

---

# 30. PHASE 23 — RAPID PERFORMANCE PASS

After functionality exists, collect obvious bottleneck numbers:
```text
decode FPS
provider FPS
end-to-end FPS
provider latency
behaviour-engine latency
event latency
GPU VRAM
CPU/RAM
```

Optimize only the largest actual bottleneck.

Possible actions:
```text
lower analysis FPS
crop person ROIs
batch inference
cache provider features
run specialists selectively
reduce overlay cost
ONNX/TensorRT later
DeepStream multi-stream batching
```

---

# 31. PHASE 24 — PROVIDER ADAPTER MERGE

The shared engine must accept:

```text
YOLO26 observation
RTMPose observation
SMPL-X observation
```

through the same `PersonObservation` abstraction.

Commands should support:
```bash
--provider yolo26
--provider rtmpose
--provider smplx
--provider replay
```

Do not write three separate applications.

---

# 32. PHASE 25 — HYBRID / SPECIALIST FUSION

Only after providers work individually.

Possible logic:
```text
YOLO26 quality good
→ use primary evidence

YOLO26 wrist/head ambiguous
→ obtain RTMPose evidence

2D orientation ambiguous and review-worthy
→ obtain SMPL-X evidence
```

Fuse using provider quality/confidence and evidence consistency.

Do not blindly average incompatible signals.

---

# 33. CONFIG-FIRST RULE

All changeable behaviour parameters must live in config, not scattered code.

Include:
```text
analysis FPS
keypoint/quality thresholds
history seconds
max temporal gap
head onset/sustain thresholds
repetition window/count
torso thresholds
hand-motion thresholds
interaction Δt
risk decay
WATCH threshold
REVIEW threshold
event merge gap
clip pre/post roll
camera health thresholds
```

Rapid prototype values may be approximate. Centralization allows quick tuning.

---

# 34. REQUIRED COMMANDS

## Run provider
```bash
python scripts/run_provider.py \
  --source <video> \
  --camera CAM12 \
  --provider yolo26 \
  --output artifacts/VIDEO_05/yolo26
```

## Replay cached output
```bash
python scripts/replay_provider.py \
  --observations artifacts/VIDEO_05/yolo26/observations.jsonl \
  --camera-config configs/cameras/CAM12.yaml
```

## End-to-end file pipeline
```bash
python scripts/run_video.py \
  --source <video> \
  --camera CAM12 \
  --provider yolo26 \
  --save-overlay \
  --save-events
```

## API/backend
```bash
python scripts/run_server.py
```

---

# 35. FAILURE ISOLATION

## Provider
Each provider exposes:
```text
READY
DEGRADED
FAILED
```
plus a reason.

A failed optional provider must not crash an unrelated replay/primary-provider run.

## Camera
Each camera owns independent health:
```text
CAM01 ONLINE
CAM02 DEGRADED
CAM03 OFFLINE
```

One camera failure must not terminate others.

---

# 36. PRIVACY / STORAGE DEFAULTS

During rapid prototype:
- no face recognition;
- no names;
- no identity embeddings;
- no face crops persisted unless a debugging need explicitly requires temporary local output;
- event clips only where useful;
- retention configurable;
- anonymous IDs throughout the core engine;
- avoid duplicating multi-GB source videos.

---

# 37. LEARNED TEMPORAL MODEL POLICY

ST-GCN / PoseWatch / Transformer-style temporal classifiers may be used as **additional evidence providers**.

Do not make the entire system depend on a learned temporal classifier before suitable labelled sequences exist.

Architecture:
```text
Temporal Features
    ├── deterministic/interpretable behaviour logic
    └── learned temporal model
             ↓
       additional evidence
```

If learned model fails/unavailable, core behaviour flow still works.

---

# 38. IMMEDIATE TEAM EXECUTION ORDER

## Step 1 — One integrator freezes schemas + folders
No provider-specific object may leak into shared behaviour code.

## Step 2 — Build shared vertical skeleton
```text
VideoSource → Provider adapter → Seat → Context
→ Temporal → Head → Risk → Event JSON
```

## Step 3 — Providers in parallel
**Anuj**
```text
YOLO26 Pose → tracker → artifact
```

**Om**
```text
RTMPose → local association
Object detector → association → artifacts
```

**Dravvya**
```text
Multi-HMR → SMPL-X → temporal 3D association → artifacts
```

## Step 4 — Shared torso + hand + interaction modules

## Step 5 — Quality / `UNKNOWN` / monitoring gaps

## Step 6 — Counter-evidence + full semantic context

## Step 7 — Object attribution + PHONE/MOUSE/MONITOR disambiguation into shared evidence

## Step 8 — Risk + event merging + evidence clips

## Step 9 — API + dashboard

## Step 10 — RTSP

## Step 11 — DeepStream only if it accelerates the already-working pipeline

## Step 12 — Run all dataset scenarios; fix failures

## Step 13 — Performance pass

## Step 14 — Hybrid specialist routing

---

# 39. DEFINITION OF RAPID PROTOTYPE COMPLETE

The development phase is complete when the prototype can demonstrate:

- [ ] file video input;
- [ ] working RTSP adapter/input;
- [ ] multi-person detection;
- [ ] anonymous tracking;
- [ ] pose landmarks;
- [ ] seat anchoring;
- [ ] semantic camera zones;
- [ ] YOLO26 provider;
- [ ] RTMPose provider;
- [ ] Multi-HMR + SMPL-X provider;
- [ ] object provider;
- [ ] phone vs monitor vs mouse disambiguation;
- [ ] per-seat monitor/mouse/lap regions;
- [ ] hard-negative mining for workstation false positives;
- [ ] `UNCERTAIN_OBJECT` abstention;
- [ ] head behaviour;
- [ ] torso behaviour;
- [ ] hand behaviour;
- [ ] possible reciprocal interaction;
- [ ] possible object interaction;
- [ ] `NORMAL / WATCH / REVIEW / UNKNOWN`;
- [ ] component-specific quality;
- [ ] camera health;
- [ ] `MONITORING_GAP`;
- [ ] behavioural risk;
- [ ] context-aware counter-evidence;
- [ ] event merging;
- [ ] deterministic explainable reasons;
- [ ] evidence snapshot/clip;
- [ ] event persistence;
- [ ] dashboard review queue;
- [ ] long-video streaming without loading everything;
- [ ] provider output persistence/replay;
- [ ] provider/camera failures exposed rather than hidden;
- [ ] privacy-conscious anonymous identifiers;
- [ ] no “cheating detected” product language.

---

# 40. DEFER UNTIL AFTER THE PRODUCT WORKS

Do **not** block the rapid build on:

```text
perfect threshold calibration
publication-grade metrics
exhaustive annotations
huge automated test suite
large ST-GCN training project
TensorRT conversion for every model
Kubernetes
Kafka
distributed microservices
production IAM
perfect cross-camera ReID
perfect chit detector
full institutional privacy/legal workflow
```

Preserve interfaces so these can be added later.

---

# 41. EXECUTOR REPORT FORMAT AFTER EACH PHASE

Return concise engineering status:

```text
PHASE:
STATUS: COMPLETE / PARTIAL / BLOCKED

IMPLEMENTED:
- ...

FILES CHANGED:
- ...

RUN COMMAND:
- ...

ACTUAL RESULT:
- ...

KNOWN BREAKAGES:
- ...

NEXT PHASE:
- ...
```

Do not claim success unless the relevant command has actually been run.

---

# 42. SOURCE-OF-TRUTH PRIORITY

If instructions conflict:

```text
1. Official DrishtiAI Problem Statement #1
2. This Rapid Development Plan
3. Frozen schemas and behaviour vocabulary
4. Repository's current working interfaces
5. Provider-specific preferences
```

Do not silently drop a PS requirement because implementation is inconvenient.

---

# 43. FINAL ENGINEERING PRINCIPLE

Drishti must never reduce behaviour to:

```text
movement → suspicious
```

Required reasoning:

```text
person
→ anonymous track
→ seat
→ movement
→ target context
→ temporal persistence
→ repetition
→ supporting head/torso/hand signals
→ reciprocal interaction
→ object evidence
→ quality
→ counter-evidence
→ behavioural risk
→ NORMAL / WATCH / REVIEW / UNKNOWN
→ explainable human-review event
```

---

# 44. FINAL RAPID-DEVELOPMENT PRINCIPLE

```text
BUILD
↓
RUN
↓
BREAK
↓
TRACE FAILURE
↓
FIX
↓
RUN AGAIN
↓
INTEGRATE
↓
MOVE FORWARD
```

The goal of this phase is not to prove every design choice before coding.

The goal is to produce a complete, modular, explainable, privacy-conscious prototype that covers the Problem Statement, survives the major real-world failure modes, uses Kaggle for heavyweight GPU inference where necessary, and can later be hardened with calibration, benchmarking and optimization.

---

# 45. REFERENCE BASIS

This plan consolidates the requirements and architecture from:

- `Problem Statement DrishtiAI Hackathon.pdf`
- `DRISHTI_IMPLEMENTATION_SPEC.md`
- `DRISHTI_4D_FINAL_WORK_DISTRIBUTION.md`
- `DRISHTI_4D_COMPLETE_TEST_IMPLEMENTATION.md`
- `DRISHTI-Improvements.pdf`

Provider installation/integration must follow the relevant current official documentation and actual repository versions instead of assuming old package commands.

---

**END OF RAPID DEVELOPMENT & PROTOTYPING PLAN**

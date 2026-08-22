# DRISHTI — Parallel Object Disambiguation Track Prompt

> **Use this file in a separate coding chat.** This chat owns only the low-resolution CCTV object-disambiguation workstream while the master Drishti implementation continues elsewhere. The output must integrate through existing shared schemas; do not redesign unrelated modules.

# 0. ROLE

You are the executor for the **PHONE vs MONITOR vs MOUSE** parallel track.

Development mode:
```text
BUILD → RUN → BREAK → DEBUG → FIX → CONTINUE
```

Goal: deliver a drop-in subsystem that prevents switched-off monitors, mice, keyboards, stationery and dark workstation regions from becoming false phone alerts while preserving real phone-in-hand/lap evidence.

# 1. NON-NEGOTIABLE CONTRACT

Final system states remain:
```text
NORMAL / WATCH / REVIEW / UNKNOWN
```

Final object behaviour remains:
```text
POSSIBLE_OBJECT_INTERACTION
```

Never emit `REVIEW` directly from detector confidence. Never use “CHEATING”.

# 2. EXPECTED INPUTS

Consume existing/common equivalents of:

```text
ObjectObservation:
  camera_id
  video_id
  frame_index
  timestamp_sec
  object_class
  confidence
  bbox_xyxy_px

Person/Pose Context:
  entity_id
  seat_id
  person bbox
  left/right wrist coordinates
  pose quality
  track quality

Camera/Seat Context:
  seat_polygon
  monitor_polygon
  keyboard_region
  mouse_region
  own_desk_region
  lap_region
  blind_spots
```

# 3. REQUIRED OUTPUT

Return an object interpretation containing at least:

```text
camera_id
timestamp_sec
entity_id
seat_id
candidate_class
resolved_class: PHONE | MOUSE | MONITOR | OTHER | UNCERTAIN_OBJECT
detector_confidence
object_quality
monitor_overlap
mouse_region_overlap
lap_region_overlap
left/right wrist distance or proximity
wrist_object_motion_correlation
staticness
phone_evidence
reason_codes
```

Then convert supported phone evidence into a shared `BehaviourEvidence` for:
```text
POSSIBLE_OBJECT_INTERACTION
```
The master risk engine decides `NORMAL/WATCH/REVIEW/UNKNOWN`.

# 4. PHASE O1 — INSPECT BEFORE EDITING

Inspect the current repository and identify:
1. current object detector implementation;
2. shared observation/evidence schemas;
3. camera-config format;
4. seat/track/pose output;
5. current path that turns object detection into risk/alerts.

Reuse existing interfaces. Do not duplicate architecture.

Report only:
```text
CURRENT FLOW
INTEGRATION POINT
FILES TO CHANGE
FALSE-POSITIVE PATH
```
Then implement.

# 5. PHASE O2 — OBJECT RESOLVER

Create or extend the closest existing module, preferably equivalent to:
```text
src/drishti/behaviour/objects.py
```

Core API concept:
```python
resolve_object_candidate(candidate, person_context, seat_context, temporal_context, quality_context)
```

Keep it deterministic/config-driven initially.

# 6. PHASE O3 — WORKSTATION REGIONS

Extend per-seat config compatibly:

```yaml
A03:
  seat_polygon: [...]
  monitor_polygon: [...]
  keyboard_region: [...]
  mouse_region: [...]
  own_desk_region: [...]
  lap_region: [...]
```

Implement normalized geometry helpers for:
```text
point-in-polygon
bbox/region overlap
IoU
candidate-center containment
region distance
```

Never make colour-based rules.

# 7. PHASE O4 — MONITOR SUPPRESSION

Implement contextual prior:

```text
candidate overlaps monitor region
+ persistent at stable location
+ no meaningful wrist correlation
→ MONITOR prior ↑
→ PHONE evidence ↓
```

Reason codes:
```text
MONITOR_REGION_OVERLAP
LONG_TERM_STATIC_EQUIPMENT
NO_HAND_ASSOCIATION
```

Do not hard-reject from a single noisy frame if context quality is poor.

# 8. PHASE O5 — MOUSE SUPPRESSION

Implement:

```text
candidate inside mouse region
+ localized low-amplitude motion
+ operating wrist nearby
+ candidate remains in workstation region
→ MOUSE prior ↑
→ PHONE evidence ↓
```

Reason codes:
```text
INSIDE_MOUSE_REGION
LOCALIZED_WORKSTATION_MOTION
MOUSE_LIKE_HAND_PATTERN
```

Do not classify from shape alone.

# 9. PHASE O6 — PHONE-SUPPORTING CONTEXT

Increase phone evidence only through multi-cue support such as:

```text
outside normal equipment zones
inside lap/body region
moves with wrist
moves desk → hand → lap/body
appears/disappears
person/seat association strong
both hands converge around candidate
```

Possible reason codes:
```text
LAP_REGION_ASSOCIATION
WRIST_PROXIMITY
WRIST_OBJECT_MOTION_CORRELATION
OBJECT_REGION_TRANSITION
HANDHELD_TRAJECTORY
```

No single cue means phone.

# 10. PHASE O7 — TEMPORAL OBJECT HISTORY

Maintain bounded per-object/per-seat history using `timestamp_sec`:

```text
bbox center/size
region
seat/entity
left/right wrist distance
quality
```

Derive:
```text
position variance
movement amplitude
region transitions
wrist correlation
appearance/disappearance
persistence
```

Never use fixed frame-count assumptions for duration.

# 11. PHASE O8 — STATIC WORKSTATION MAP

Support stable equipment memory from verified camera profiles and/or a stable initialization window.

Do not blindly learn the first object as normal equipment if the video starts after a phone is already present.

Use stable equipment evidence as a prior, not absolute truth.

# 12. PHASE O9 — UNCERTAIN OBJECT

When:
```text
resolution poor
candidate occluded
detector classes close
context conflicts
track unstable
```
return:
```text
resolved_class = UNCERTAIN_OBJECT
phone_evidence = LOW
```
No direct `REVIEW`.

# 13. PHASE O10 — HARD-NEGATIVE HARVESTER

Build a utility that saves target-domain false positives:

```text
artifacts/hard_negatives/
  monitor/
  mouse/
  keyboard_or_other/
  manifest.csv
```

Manifest:
```text
video_id
camera_id
timestamp_sec
seat_id
predicted_class
confidence
verified_label
crop_path
```

Prioritize:
```text
monitor→phone
mouse→phone
keyboard→phone
dark-region→phone
```

These crops are for future fine-tuning/retraining. Do not fabricate labels automatically; allow manual verification.

# 14. PHASE O11 — OPTIONAL SECOND-STAGE VERIFIER

Only if needed after context logic.

Provide a replaceable interface:
```text
candidate crop + local context → PHONE / MOUSE / MONITOR / OTHER
```

Do not make the master pipeline depend on this verifier.

# 15. PHASE O12 — ROI REPROCESSING

Optionally support detection/classification on:
```text
PERSON_ROI
SEAT_ROI
LAP_ROI
```

Record which source produced a candidate. Do not claim upscaling recovered detail that was absent in the source.

# 16. CONFIG-DRIVEN SCORING

Conceptual score:

```text
phone_score =
    detector evidence
  + lap evidence
  + seat/person association
  + wrist proximity
  + wrist/object motion correlation
  + cross-region movement
  - monitor overlap
  - mouse-region prior
  - long-term equipment staticness
  - poor-quality penalty
```

All weights and thresholds belong in config, e.g.:

```yaml
object_disambiguation:
  monitor_overlap_penalty: ...
  mouse_region_penalty: ...
  lap_region_bonus: ...
  wrist_proximity_bonus: ...
  wrist_motion_correlation_bonus: ...
  static_equipment_penalty: ...
  poor_quality_penalty: ...
  uncertain_margin: ...
```

Do not claim these values are calibrated unless actual calibration was performed.

# 17. RAPID SCENARIOS TO RUN

No large test framework. During development run/replay these cases:

```text
turned-off monitor falsely detected as phone
mouse falsely detected as phone
actual phone in hand
actual phone in lap
phone resting on desk
low-resolution ambiguous object
partially occluded object
object near keyboard/mouse
```

For each:
```text
run → inspect → patch → rerun → continue
```

# 18. DO NOT ASSUME

Never assume:
```text
black = phone
rectangular = phone
static = monitor
moving = phone
wrist nearby = phone
detector confidence = final phone confidence
no phone detection = normal behaviour
```

# 19. GIT / PARALLEL-INTEGRATION DISCIPLINE

Use a dedicated branch if branches are used, e.g.:
```text
feature/object-disambiguation
```

Do not edit unrelated dashboard, Multi-HMR, RTMPose or YOLO26 modules except their adapter interface if required.

Before handoff provide:
```text
commit / changed files
new config keys
schema changes, if unavoidable
exact import path
exact integration call
expected inputs
expected outputs
artifact paths
known limitations
unresolved false positives
```

# 20. STATUS FORMAT

After each phase return:

```text
PHASE:
STATUS: COMPLETE / PARTIAL / BLOCKED

IMPLEMENTED:
- ...

FILES CHANGED:
- ...

COMMAND RUN:
- ...

ACTUAL RESULT:
- ...

FALSE POSITIVE OBSERVED:
- ...

FIX APPLIED:
- ...

INTEGRATION IMPACT:
- ...

NEXT:
- ...
```

# 21. DEFINITION OF DONE FOR THIS PARALLEL TRACK

- [ ] raw phone detections no longer directly create alerts;
- [ ] monitor regions configurable per camera/seat;
- [ ] mouse regions configurable per camera/seat;
- [ ] lap/desk regions supported;
- [ ] object→person/seat association works;
- [ ] wrist proximity available;
- [ ] temporal object history implemented;
- [ ] static workstation evidence implemented;
- [ ] `PHONE/MOUSE/MONITOR/OTHER/UNCERTAIN_OBJECT` resolution exists;
- [ ] monitor false positives can be suppressed;
- [ ] mouse false positives can be suppressed;
- [ ] real phone-in-hand/lap can still produce positive evidence;
- [ ] hard-negative harvesting exists;
- [ ] optional verifier is implemented or cleanly deferred;
- [ ] master handoff emits `POSSIBLE_OBJECT_INTERACTION`, not direct `REVIEW`;
- [ ] all scoring parameters are config-driven;
- [ ] integration handoff is documented.

**END OF PARALLEL OBJECT DISAMBIGUATION TRACK PROMPT**

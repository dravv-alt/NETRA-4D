# DRISHTI — Parallel Kaggle Multi-HMR + SMPL-X Development Track Prompt

> **Use this file in a second chat while the master rapid-development plan is executed in the main chat.**  
> This chat owns the heavyweight **Dravvya / Multi-HMR + SMPL-X provider** and its Kaggle GPU execution.  
> The goal is to make the 3D provider operational fast, persist outputs, and hand them back to the main Drishti application through the frozen common schemas.

---

# 0. EXECUTION MODE

Use:

```text
BUILD → RUN → BREAK → DEBUG → FIX → CONTINUE
```

Do **not** return to slow research-style notebook development where every tiny cell waits for human approval.

You may use multiple Kaggle cells or scripts in one implementation pass when the dependency path is clear.

Stop only for a real blocker such as:
- missing licensed SMPL-X model asset;
- unrecoverable dependency conflict;
- unavailable GPU/runtime;
- missing required repository/input file.

Routine installation/runtime failures must be diagnosed and fixed automatically.

---

# 1. THIS CHAT'S SCOPE

Own only:

```text
Multi-HMR environment
SMPL-X asset hookup
single-image inference
short-video inference
multi-person 3D output
3D temporal association
3D head orientation
3D torso orientation
3D wrist/arm geometry
provider quality
artifact persistence
Drishti provider adapter
```

Do **not** redesign:
- master risk engine;
- dashboard;
- RTMPose branch;
- YOLO26 branch;
- event database;
- global behaviour taxonomy.

Integrate with those through shared schemas/artifacts.

---

# 2. CURRENT VERIFIED STARTING STATE

Kaggle runtime already verified:

```text
Python (base notebook): 3.12.13
PyTorch: 2.10.0+cu128
CUDA available: True
GPU: Tesla T4
VRAM: ~14.56 GB
Disk free during preflight: ~19.49 GB
GitHub access: PASS
```

Available environment tools:

```text
python3.9   : unavailable directly
python3.10  : /usr/bin/python3.10
mamba       : /usr/local/bin/mamba
uv          : /usr/local/bin/uv
```

Official Multi-HMR repository was cloned at:

```text
/kaggle/working/multi-hmr
```

Frozen repo commit:

```text
651fb411e1cbcc626aaa5f38805ecab9cc891f7a
```

Repo requirements observed:

```text
Python 3.9 environment expected by official config
PyTorch 2.0.1 era dependency stack
SMPL-X package/assets required
```

Do **not** install these dependencies into the Kaggle base notebook environment.

SMPL-X licensed asset was **not present** at the last check:

```text
/kaggle/working/multi-hmr/models/smplx/SMPLX_NEUTRAL.npz
```

Do not fabricate, bundle, or redistribute that asset. If missing, clearly instruct the user where it must be mounted/copied after they obtain it legitimately.

---

# 3. CURRENT VIDEO TARGET

Use only this first development video until the provider works:

```text
VIDEO_05
03.CCTV Mobile Usage.mkv
/kaggle/input/datasets/dravvalt/drishti-dataset/03.CCTV Mobile Usage.mkv
```

Verified metadata:

```text
1280x720
~12.083 FPS metadata
~281.83 sec / 4.70 min
mpeg4
```

A representative feasibility frame was already extracted around:

```text
requested: 140.910 sec
actual decoded timestamp: 140.567 sec
```

Observed suitability:

```text
HEAD       GOOD
TORSO      GOOD
ARMS       MODERATE
HANDS      POOR–MODERATE
LOWER BODY heavily desk-occluded
upper-body SMPL-X feasibility: strong enough for first attempt
```

Do not waste time choosing another video unless inference proves this frame unusable.

---

# 4. ARCHITECTURE CONTRACT WITH MAIN CHAT

The main application must never import Multi-HMR-specific internal objects.

This provider must ultimately emit Drishti-compatible observations:

```text
PersonObservation:
  schema_version
  provider = "smplx" / "multi_hmr"
  camera_id
  video_id
  frame_index
  timestamp_sec
  entity_local_id
  bbox_xyxy_px
  bbox_xyxy_norm
  detection_confidence / provider confidence
  keypoints_2d
  joints_3d
  body_parameters or artifact reference
  quality
  metadata
```

Provider-specific arrays may be stored externally in `.npz`, but the JSONL observation must point to them cleanly.

---

# 5. ARTIFACT CONTRACT

Persist expensive inference so downstream work does not rerun the GPU model.

Required output structure:

```text
artifacts/
└── VIDEO_05/
    └── smplx/
        ├── observations.jsonl
        ├── joints_3d.npz
        ├── body_params.npz
        ├── runtime.json
        ├── overlays/
        └── debug/
```

`runtime.json` should include where measurable:

```text
frames processed
sampling strategy / requested analysis FPS
wall time
mean inference time
p50/p95 if easy to collect
effective FPS
peak VRAM
model/checkpoint identifiers
repo commit
environment identifier
```

---

# 6. PHASE S1 — ISOLATED ENVIRONMENT

Create an isolated environment using available `mamba` unless a demonstrably better working path exists.

Recommended environment path:

```text
/kaggle/working/envs/multihmr_py39
```

Requirements:
- Python 3.9;
- keep base notebook environment untouched;
- install only provider dependencies inside isolated env;
- record exact package versions after success;
- do not blindly downgrade system CUDA drivers;
- use environment's Python via subprocess/scripts if switching notebook kernel is inconvenient.

Exit condition:

```text
isolated Python works
required imports work
PyTorch sees T4
Multi-HMR imports without crashing
```

---

# 7. PHASE S2 — SMPL-X ASSET GATE

Check:

```text
models/smplx/SMPLX_NEUTRAL.npz
```

If missing:
1. stop only this asset-dependent step;
2. clearly tell user exact expected path;
3. continue all environment/repository work that does not require the file;
4. do not generate fake placeholder model parameters.

Once asset exists, verify that Multi-HMR/SMPL-X loader accepts it.

---

# 8. PHASE S3 — SINGLE-FRAME INFERENCE

Run the already selected VIDEO_05 frame.

Goal is not perfect lower-body reconstruction.

Verify:

```text
person detected?
SMPL-X body instantiated?
2D projected joints available?
3D joints available?
mesh/body parameters available?
upper-body projection plausible?
GPU memory stable?
```

Save overlay/debug outputs.

If lower body is implausible because the desk hides it, preserve the output but mark lower-body quality as poor/occluded instead of treating inferred geometry as observed truth.

---

# 9. PHASE S4 — SHORT-WINDOW VIDEO INFERENCE

After one frame works, process a short timestamp window around the selected moment.

Do not immediately run all 4.7 minutes.

Recommended development sequence:

```text
5–10 sec
→ 20–30 sec
→ longer interval only after stability
```

Use actual decoded timestamps.

Sampling can be lower than source FPS initially if needed for T4 throughput.

Record sampled timestamps explicitly.

---

# 10. PHASE S5 — TEMPORAL PERSON ASSOCIATION

Frame-level Multi-HMR outputs need temporal identity.

Implement local association using available combinations of:

```text
2D bbox IoU
bbox center displacement
projected joint similarity
seat prior
camera-space position
body-shape continuity if useful
previous-track continuity
```

Do not implement biometric face recognition.

Entity IDs remain anonymous.

Output:

```text
entity_local_id
track_quality
association_confidence
```

If ambiguous:

```text
TRACK = POOR / UNKNOWN
```

Do not force an ID.

---

# 11. PHASE S6 — 3D HEAD ORIENTATION

Derive a stable head-orientation representation from reliable SMPL-X/Multi-HMR outputs.

Expose:

```text
yaw
pitch
roll
or an equivalent rotation representation converted to interpretable angles
HEAD_3D quality
```

Important:
- document coordinate convention;
- camera-relative direction must be explicit;
- unwrap/smooth angle discontinuities;
- never call a head turn suspicious here.

This provider emits geometry, not final behaviour decisions.

---

# 12. PHASE S7 — 3D TORSO ORIENTATION

Derive torso/body orientation.

Expose:

```text
torso yaw/pitch/roll or equivalent
torso direction vector
TORSO_3D quality
```

Desk-occluded pelvis/lower body must reduce confidence where relevant.

---

# 13. PHASE S8 — 3D WRIST / ARM FEATURES

Extract:

```text
left/right wrist 3D trajectory
elbow trajectory
arm extension
relative hand-to-torso geometry
relative hand-to-seat/desk geometry when camera mapping exists
HAND_3D quality
```

Do not claim finger-level certainty if image quality does not support it.

---

# 14. PHASE S9 — TEMPORAL SMOOTHING

Use timestamp-aware smoothing.

Avoid fixed-frame-window assumptions because videos vary in FPS.

Maintain raw and smoothed values separately.

Do not smooth across long tracking gaps.

---

# 15. PHASE S10 — QUALITY MODEL

At minimum output:

```text
HEAD_3D   GOOD / POOR / UNKNOWN
TORSO_3D  GOOD / POOR / UNKNOWN
HAND_3D   GOOD / POOR / UNKNOWN
TRACK     GOOD / POOR / UNKNOWN
```

Quality can consider:

```text
visibility / projection plausibility
model confidence if available
joint stability
occlusion
person scale
association stability
```

Missing evidence must not become zero movement.

---

# 16. PHASE S11 — DRISHTI PROVIDER ADAPTER

Create the repository adapter that converts persisted Multi-HMR/SMPL-X results into the common `PersonObservation` schema.

Recommended shape:

```text
src/drishti/providers/smplx_provider.py
```

Support two modes if practical:

```text
LIVE/INFERENCE MODE
  input frames → isolated provider process → observations

REPLAY MODE
  observations.jsonl / npz → shared Drishti engine
```

The main chat must be able to develop head/torso/risk logic using replay without loading Multi-HMR.

---

# 17. PHASE S12 — HANDOFF FEATURES FOR SHARED BEHAVIOUR ENGINE

Provide neutral geometry/features, e.g.:

```text
head orientation
head target ray/direction if derivable
torso orientation
left/right wrist trajectories
arm extension
person camera-space location
quality states
```

Do not hard-code final `REVIEW` decisions into the provider.

Shared behaviour engine will resolve:

```text
head → own monitor vs neighbour
repeated turns
torso corroboration
hand interaction
reciprocal interaction
risk
```

---

# 18. GPU RESOURCE RULES

The T4 is a remote inference worker, not a reason to make the entire app GPU-resident.

Rules:
1. run Multi-HMR separately from YOLO26/RTMPose where memory is tight;
2. persist results;
3. let the process exit to release VRAM;
4. rerun only when upstream reconstruction code/model changes;
5. downstream behaviour changes use replay artifacts;
6. never repeatedly infer a 4.7-minute clip just to tune a risk threshold.

---

# 19. EXPANSION AFTER VIDEO_05 WORKS

Only after the provider works on VIDEO_05, expand to useful challenge clips:

```text
VIDEO_07 — multiple people / paper interaction / stronger occlusion
VIDEO_08 — 640x480, 8 FPS, low-resolution robustness
VIDEO_02 — long multi-person/seat-exchange stress
```

Do not start by running the two multi-hour videos densely.

---

# 20. FAILURE HANDLING

Examples:

```text
no person reconstructed
→ provider observation absent / UNKNOWN, not NORMAL

mesh obviously inconsistent
→ quality POOR

track association ambiguous
→ TRACK UNKNOWN

head joints unstable
→ HEAD_3D POOR/UNKNOWN

GPU OOM
→ lower sample rate / batch / resolution where supported; process sequentially
```

Never hide failures by emitting plausible-looking dummy arrays.

---

# 21. INTEGRATION HANDOFF TO MAIN CHAT

After every useful milestone, provide:

```text
COMMIT / CHANGED FILES
KAGGLE COMMAND(S)
ENVIRONMENT PATH
MODEL/CHECKPOINT
SMPL-X ASSET PATH EXPECTED
INPUT VIDEO/TIME WINDOW
ARTIFACT PATHS
SCHEMA EXAMPLE
FEATURES AVAILABLE
QUALITY FIELDS
RUNTIME/VRAM OBSERVED
KNOWN BREAKAGES
NEXT INTEGRATION STEP
```

The main executor must be able to integrate from this report without reading this chat's entire history.

---

# 22. EXECUTOR REPORT FORMAT

```text
PHASE:
STATUS: COMPLETE / PARTIAL / BLOCKED

IMPLEMENTED:
- ...

FILES CHANGED:
- ...

COMMANDS RUN:
- ...

ACTUAL RESULT:
- ...

ARTIFACTS CREATED:
- ...

GPU / RUNTIME:
- ...

KNOWN BREAKAGES:
- ...

MAIN-CHAT INTEGRATION IMPACT:
- ...

NEXT:
- ...
```

Do not claim success without actual execution.

---

# 23. RAPID TRACK DEFINITION OF DONE

- [ ] isolated Multi-HMR environment works on Kaggle;
- [ ] required SMPL-X asset path is correctly wired;
- [ ] VIDEO_05 single-frame inference works;
- [ ] short-window inference works;
- [ ] outputs are persisted;
- [ ] temporal anonymous person association exists;
- [ ] 3D head orientation is exposed;
- [ ] 3D torso orientation is exposed;
- [ ] wrist/arm trajectories are exposed;
- [ ] component quality / UNKNOWN exists;
- [ ] common Drishti `PersonObservation` adapter exists;
- [ ] replay mode works without GPU model reload;
- [ ] provider-specific failures do not masquerade as normal behaviour;
- [ ] exact handoff instructions to the main rapid-development chat are documented.

---

# 24. FINAL RULE

This parallel chat exists to prevent the main product from waiting on heavyweight 3D inference.

The working relationship is:

```text
MAIN CHAT
builds complete Drishti product continuously
        ↑
        │ persisted/common observations
        │
PARALLEL KAGGLE CHAT
gets Multi-HMR + SMPL-X working and improves its provider
```

Both tracks must converge through frozen schemas and artifacts, not through copy-pasted notebook state.

**END OF PARALLEL KAGGLE MULTI-HMR + SMPL-X TRACK PROMPT**

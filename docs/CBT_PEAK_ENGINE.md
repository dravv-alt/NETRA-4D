# CBT Peak Behavioural Engine

Status: implemented primary reasoner with automatic model-assisted workstation
discovery by default. Human calibration is an optional accuracy override.

## Execution path

```text
real frame
 -> YOLO26 pose and ByteTrack
 -> YOLO26 objects plus candidate crops/optional tiles
 -> automatic real-observation workstation map or optional human map
 -> candidate-workstation binding
 -> candidate-specific opening baseline
 -> atomic workstation-relative evidence
 -> counter-evidence and competing hypotheses
 -> four sequence reasoners
 -> cross-candidate graph
 -> uncertainty gate
 -> explainable review event or specialist request
 -> SQLite/REST/WebSocket/human review
```

The implementation is in `src/drishti/cbt/`. With no map supplied,
`AutomaticCBTBehaviouralEngine` observes real person and monitor detections
during warm-up and activates workstation-relative reasoning once stable
pairings exist. Until then, and when stable geometry cannot be recovered, the
real baseline temporal engine continues to run. No random or fixed boxes are
used.

## Four implemented sequences

1. Neighbour-screen overwatch: own-screen baseline, disengagement, directed
   neighbour-monitor attention, persistence, return, and repetition.
2. Phone retrieval/use/conceal: under-desk hand context, exact YOLO phone,
   wrist ownership, persistence, disappearance, and hand return.
3. Chit transfer: explicit custom-checkpoint chit evidence plus a hand crossing
   a calibrated workstation boundary. Low hand alone never becomes a chit.
4. Coordinated interaction: repeated bidirectional workstation-directed
   attention accumulated as cross-candidate graph edges.

## Candidate to PC mapping

The strongest candidate bound to a workstation owns its screen for that frame.
Candidate ID `N` maps to device ID `PC-N` and the overlay displays
`PC-N <-> CN`. The JSON record includes workstation ID, confidence, method,
detected class, and screen box. A current real laptop/monitor detection is
preferred; otherwise the scene-map monitor region is used and explicitly
labelled as such. One screen is never assigned to two candidates.

## Counter-evidence

Calibrated keyboard, mouse, monitor, and desk regions are explicit normal CBT
context. Phone-like detections inside those regions are down-weighted and
treated as uncertain; repeated stationary equipment becomes counter-evidence.
The system does not rename the detector class or fabricate an alternative box.

## Hypotheses and lifecycle

Each candidate maintains evidence for phone use, chit transfer, neighbour
screen attention, interaction, normal activity, and uncertainty. Internal
states progress through `NORMAL`, `OBSERVATION`, `CANDIDATE`, `WATCH`, and
either `REVIEW` or `SPECIALIST_REVIEW`. The external score is review priority,
not cheating probability.

## 3-D specialist boundary

Weak but relevant geometry produces a structured specialist request. The API
accepts real RTMPose, Multi-HMR/SMPL-X, or human geometry resolutions and
updates evidence/uncertainty while preserving provenance. The specialist is an
uncertainty resolver, not a cheating classifier. No specialist result is
created unless that provider actually executes.

## Default automatic activation

```powershell
python scripts/run_live_webcam.py --source 0 --device cuda:0 --require-cuda

python scripts/render_annotated_videos.py --device cuda:0 --require-cuda `
  --asset VIDEO_03 --run-id video03_cbt_auto_v1
```

Automatic mode is the default. It saves the recovered scene as
`automatic_workstation_map.yaml` in that run/video directory and labels it
`calibration_status: AUTOMATIC`. This is model-derived geometry, not a claim of
human validation.

## Optional human selection

```powershell
python scripts/calibrate_cbt_workstations.py --asset VIDEO_03 `
  --camera-id CAMERA_04 --calibrated-by <reviewer>

python scripts/render_annotated_videos.py --device cuda:0 --require-cuda `
  --workstation-mode human --require-workstation-map `
  --asset VIDEO_03 --run-id video03_cbt_human_v1

python scripts/run_live_webcam.py --source 0 --device cuda:0 --require-cuda `
  --workstation-map drishti4d/configs/workstations/VIDEO_03.yaml `
  --workstation-mode human --require-workstation-map
```

Use `--workstation-mode off` to disable workstation reasoning explicitly.

Every calibration records the real source frame index, SHA-256, reviewer,
timestamp, reference resolution, and normalized polygons.

## What is deliberately not claimed

- Automatic maps are estimates from real detector observations and are not
  described as human-validated.
- No workstation map is copied from the old Watcher seat polygons.
- The existing 128 baseline events have not been retrospectively relabelled as
  CBT Peak results.
- ST-GCN/GNN is not required for this deterministic evidence graph and remains
  a future learned verifier.
- Precision/recall remains unavailable until the event queue is human-reviewed.

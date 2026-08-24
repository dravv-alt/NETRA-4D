# NETRA-4D Presentation Read-Through

Use this document as a pre-presentation briefing. The safest presentation style is to explain the system as an **evidence and human-review platform**, not as an automatic cheating-verdict machine.

## 1. Thirty-second introduction

NETRA-4D is a privacy-aware examination-surveillance pipeline that converts a webcam, uploaded video, or RTSP stream into annotated video and time-aligned review evidence. It detects and tracks candidates and visible objects, masks faces, estimates a two-dimensional head-direction proxy, maps candidates to assigned computer screens, observes candidate-to-candidate interaction, and accumulates evidence when suspicious behavior is sustained over time. Every flagged event is written with an exact source timestamp, confidence, candidate/workstation context, JSON provenance, and a review snapshot or clip. A human reviewer makes the final decision.

## 2. The problem we are addressing

Reviewing examination CCTV manually is slow and inconsistent. A reviewer may need to search hours of footage for a short incident, while a single-frame detector generates too many weak or misleading alerts. Examination labs also contain hard negatives: a mouse can resemble a phone, normal typing can place hands below the desk edge, candidates naturally glance at timers, and people or screens can be occluded.

Our goal was therefore not just object detection. We needed to answer five questions:

1. Who is the observation associated with?
2. Which workstation and PC belong to that candidate?
3. Did the signal persist over time?
4. What exact frame and source timestamp support it?
5. Can a reviewer audit, accept, reject, or escalate the evidence?

## 3. What we built

The implemented system has four connected layers:

```text
Webcam / uploaded video / RTSP through MediaMTX
                         ↓
Pose + object inference + ByteTrack candidate identity
                         ↓
Privacy mask + head-direction proxy + candidate/PC context
                         ↓
Temporal evidence engine + snapshots + JSON/SQLite event records
                         ↓
FastAPI REST/WebSocket/MJPEG backend
                         ↓
React command center, live monitor, event review, analytics, and reports
```

The important point is that this is one coherent data path. The frontend does not need to invent boxes or events when real backend data exists.

## 4. Computer-vision pipeline

### Person pose and tracking

- YOLO26 pose detects people and 17 COCO keypoints.
- ByteTrack maintains candidate track IDs across frames.
- A track ID is an evidence identifier, not facial recognition or a biometric identity.
- Candidate annotations are added after privacy masking so the face remains obscured.

### Object detection

- YOLO26 object inference preserves the raw class, bounding box, and confidence.
- The pipeline looks for relevant objects such as cell phones, laptops/monitors, and contextual hard negatives.
- A detected phone is evidence for review; it does not prove that the candidate used it.
- The current low-hand/possible-chit event is a behavioral proxy. We do not claim a trained chit detector because no validated custom chit checkpoint is present.

### Gaze and screen context

- The system estimates a 2-D head-direction proxy using facial pose keypoints.
- It is not eye or iris tracking.
- Candidate-to-PC association uses candidate and monitor geometry.
- Assigned screens are labelled `PC-<candidate ID>`.
- Looking toward another screen is evaluated relative to candidate/workstation geometry and over time.

### Candidate interaction

- Pairwise candidate distance, mutual orientation, and temporal duration are combined.
- The output is a human-interaction review proposal, not a learned misconduct classification.
- Cross-candidate relations can be represented as graph edges in the contextual reasoning layer.

## 5. Temporal evidence engine

A single unusual frame is noisy. NETRA-4D therefore keeps temporal state for each candidate and event type.

- An event begins from a model or geometry signal.
- Duration and persistence are accumulated using source timestamps.
- Evidence confidence rises while the behavior remains sustained.
- The temporal boost is bounded so confidence cannot grow without limit.
- Activation, threshold crossing, and evidence capture are written at exact video timestamps.
- Uploaded videos use `frame index / source FPS`, preventing slow CPU inference from changing the reported video time.

This is one of the key differences between the system and a normal object-detection demo.

## 6. Privacy and evidence integrity

- Face/head masking is applied before annotated output is displayed or saved.
- Source metadata, model checkpoint hashes, runtime device, and output hashes are stored in the run manifest.
- A run is accepted only when all five video assets contain completion markers and their annotated-video sizes match the manifest.
- Incomplete or corrupted copies are rejected by the API.
- High-confidence snapshots include frame index, exact timestamp, confidence, and event label in the filename.
- Human review decisions are stored separately from model evidence.

## 7. Real-time modes

### Browser webcam

The frontend requests webcam permission using the browser's `getUserMedia` API, but arming the camera does not itself start inference. An invigilator enters an ID and explicitly presses **Invigilator Start Monitoring**. Only then are fresh JPEG frames sent to a bounded backend watcher for pose, object, privacy, candidate/PC, and temporal processing; annotated MJPEG and WebSocket telemetry begin at the same point.

When the invigilator presses **Stop Monitoring**, the browser camera remains armed but frame publication and inference pause. Those paused frames are neither computed nor written as evidence. A later **Resume Monitoring** opens a new active computation window on the same live feed. `monitoring_control.jsonl` records `ARMED`, `MONITORING_STARTED`, `MONITORING_PAUSED`, and `MONITORING_RESUMED` with UTC time and invigilator ID.

For a genuinely live camera, old buffered frames are discarded if inference falls behind. This keeps the displayed result current instead of showing an increasingly delayed queue.

### Uploaded video

An uploaded file is treated as evidence, so decoder back-pressure is used. Every source frame is processed in order, even when inference is slower than the original FPS. Annotated frames and telemetry appear as soon as they are computed.

### RTSP and MediaMTX

MediaMTX listens for RTSP publishers on port `8554`. The frontend can start backend processing from an address such as:

```text
rtsp://127.0.0.1:8554/exam
```

MediaMTX also exposes its local control API on port `9997`.

## 8. Frontend layout

### Command Center

Provides the operational overview: examination state, live alerts, seat/workstation context, event priority, and navigation into evidence review.

### Live Monitor

This is the main demo surface.

- Start Browser Webcam
- Upload & Stream
- RTSP input through MediaMTX
- Annotated MJPEG playback
- Frame index, latency, effective FPS, tracked candidates, objects, and sustained evidence confidence
- Automatic fallback to a verified previous run when a new live source does not produce a frame within ten seconds
- Three previous sampled runs with source/annotated comparison, measured metrics, JSON evidence, and captured snapshots

### Event Queue

Shows pending evidence items ordered for human attention. It separates observation, review, and escalation rather than presenting every model signal as a violation.

### Evidence Review

Displays the event timeline, evidence stack, source signals, confidence, persistence, contextual counter-evidence, clips, and human decision controls.

### Candidates

Organizes track-relative candidate history and workstation context. Candidate IDs are tracking identifiers, not facial identity claims.

### Cases and Audit

Groups reviewed evidence and maintains adjudication/audit context.

### Analytics

Shows temporal behavior, candidate-to-PC assignments, head-direction evidence, candidate interaction state, and actual backend pipeline/session health. Irrelevant anime.js LiDAR and acoustic spectrum demonstrations were removed.

### AI and Cameras

Surfaces ingestion, camera configuration, spatial/context rules, and system health.

### Validated Run

Shows the five real CCTV source files in `netra4d_final_20260823` and their measured frames, source durations, observations, temporal event types, event-involved candidate track IDs, evidence stills, decode integrity, and runtime. The earlier eight-row prototype benchmark with unsupported F1/T4/ground-truth claims was removed. Without reviewed ground truth, this page is proof of pipeline execution, not an accuracy benchmark.

### Get Report

The top-header action downloads a formatted technical research PDF generated from the validated run, measured results, event distribution, methodology, evidence images, architecture, and limitations.

## 9. Measured proof of work

Validated run: `netra4d_final_20260823`

- Five videos processed
- 15,346 frames represented in the run
- 856.75 seconds of source footage
- 19 event-involved candidate track IDs across the five independent video sessions
- 10 candidate track IDs mapped to workstation PCs in the final per-video context graphs
- 29 temporal review-event records
- Four temporal event types and 177.6 seconds of summed event persistence
- Nine high-confidence event snapshots
- 39,481 privacy-mask operations and 12,337 known gaze-proxy observations
- Tesla P100 GPU recorded in runtime provenance
- Zero raw cell-phone stills at the strict COCO `cell phone >= 0.70` rule

Candidate counts above are session-local ByteTrack identifiers involved in events, not a biometric count of distinct people. IDs can fragment after long occlusion, so this distinction matters when presenting the numbers.

That last result should be presented honestly. It demonstrates that the pipeline does not fabricate phone detections merely to make the demo look successful.

## 10. Major challenges and how we solved them

### Challenge: single-frame false positives

**Problem:** A hand, mouse, reflection, or normal glance can look suspicious in one frame.

**Solution:** We added candidate-relative geometry, temporal persistence, bounded confidence accumulation, explicit counter-evidence, and human review labels.

### Challenge: candidate and screen association

**Problem:** A detected laptop does not automatically tell us who owns it.

**Solution:** We implemented workstation scene maps and automatic calibration. Candidates are assigned one-to-one to a workstation and displayed with associated `PC-<ID>` labels where the evidence supports a binding.

### Challenge: occlusion and unstable identities

**Problem:** Desks, chairs, other candidates, and camera angles hide bodies and objects.

**Solution:** ByteTrack preserves identity across temporary detection gaps; pose and object signals are combined; workstation context reduces reliance on a single box; and uncertainty remains explicit when evidence is incomplete.

### Challenge: exact timestamps when inference is slow

**Problem:** Using processing wall time would report incorrect timestamps for uploaded videos.

**Solution:** Uploaded-video timestamps come from frame index and source FPS. Live sources use elapsed capture time.

### Challenge: processing every uploaded frame without making webcam output stale

**Problem:** A single queue policy cannot serve both evidence files and live feeds.

**Solution:** Uploaded videos use lossless decoder back-pressure. Webcam and RTSP feeds use a two-frame bounded queue and drop the oldest stale frame.

### Challenge: browser playback compatibility

**Problem:** OpenCV produced MPEG-4 Part 2 annotated MP4s, which are not consistently supported by browsers.

**Solution:** Integrity-validated originals remain untouched, while separate H.264 browser previews are generated with fast-start metadata.

### Challenge: backend webcam camera access

**Problem:** Python/OpenCV camera `0` failed even though the browser could access the user's webcam.

**Solution:** The frontend now uses browser webcam permission and publishes JPEG frames to the backend inference watcher.

### Challenge: keeping a demo useful when live inference fails

**Problem:** Model initialization, device permissions, or transport errors can otherwise leave a blank screen.

**Solution:** The interface preloads three validated source/output pairs. It automatically switches to the selected annotated proof-of-work run after a ten-second first-frame timeout.

### Challenge: preserving scientific honesty

**Problem:** Hackathon pressure can encourage unsupported accuracy, gaze, 3-D, or cheating claims.

**Solution:** We distinguish raw detections, geometry proxies, temporal evidence, and human conclusions. We removed irrelevant/fabricated 3-D LiDAR and acoustic-animation claims and explicitly document missing ground truth and custom checkpoints.

## 11. Recommended live demonstration sequence

1. Open `http://127.0.0.1:5173`.
2. Show the Command Center and explain human-review workflow.
3. Open Live Monitor.
4. Show the three Previous Sampled Runs.
5. Toggle Source Input and Annotated Output for the same video.
6. Open a high-confidence evidence snapshot and JSON evidence log.
7. Upload a short video and explain lossless per-frame processing.
8. If appropriate, click Arm Browser Webcam and grant permission.
9. Point out that the source is **ARMED** and no frames are being computed yet.
10. Enter the invigilator ID, then click **Invigilator Start Monitoring**.
11. Point out frame index, latency, candidate count, object count, and temporal evidence.
12. Click **Invigilator Stop Monitoring** and show the **PAUSED** state: the live feed remains armed, but no further frames are computed or logged.
13. Resume Monitoring and show the invigilator control record.
14. Show Event Queue and Evidence Review.
15. Open Analytics and explain candidate-to-PC context and pipeline status.
16. Click Get Report and show the generated PDF.

## 12. What not to claim

- Do not claim cheating-detection accuracy without labeled ground truth.
- Do not call head direction eye tracking.
- Do not claim a trained chit detector.
- Do not say a phone detection proves phone usage.
- Do not claim the validated Kaggle run used a T4; its manifest records a Tesla P100.
- Do not claim Multi-HMR/SMPL-X 3-D output was validated in the final run.
- Do not describe proxy events as final disciplinary decisions.

## 13. Likely judge questions

### “How is this more than YOLO?”

YOLO provides frame-level pose and objects. NETRA-4D adds tracking identity, privacy masking, workstation mapping, candidate/PC context, temporal persistence, confidence accumulation, evidence capture, exact timestamps, backend persistence, live delivery, and human adjudication.

### “Does it process every frame?”

Uploaded evidence videos: yes, using lossless back-pressure. Live webcam/RTSP: the newest frame is prioritized and stale frames can be dropped because displaying old frames would no longer be real time.

### “How do you prevent false accusations?”

Outputs are labelled review evidence, proxy types are explicit, confidence requires sustained behavior, counter-evidence and uncertainty are retained, privacy is applied, and a human reviewer makes the final decision.

### “What happens if the live demo fails?”

The frontend automatically plays one of three integrity-validated previous runs, with genuine input/output comparison, measured metrics, JSON logs, and evidence snapshots.

### “What would you do next?”

Collect and adjudicate a representative dataset; train and validate a genuine chit detector; calibrate cameras per examination hall; evaluate precision/recall and event-level time-to-detection; test cross-camera re-identification under privacy constraints; and deploy GPU inference with monitored latency and drift.

## 14. Closing statement

NETRA-4D demonstrates an end-to-end, privacy-aware evidence pipeline rather than an isolated detection model. Its main contribution is converting real per-frame observations into temporally sustained, candidate/workstation-aware, auditable evidence that reaches a reviewer in real time while keeping uncertainty and scientific limitations visible.

# Watcher

Privacy-preserving, skeleton-first AI monitoring system for examination halls.

Watcher continuously observes seated students through camera feeds, converts video into anonymous pose keypoints, analyses temporal behaviour patterns, and produces explainable alerts when sustained suspicious behaviour is detected.

## Core Philosophy

> **Watcher does not accuse. It ranks attention.**

- **Skeleton-first**: Works on keypoints, not raw pixels or face recognition
- **Seat-anchored**: Stable identity via seat location, not biometric tracking
- **Temporal**: Evaluates behaviour over time windows, not single frames
- **Rules-first**: Transparent, configurable rules before learned models
- **Risk score**: 0-100 ranking, not binary verdict
- **Human-in-the-loop**: Evidence and recommendations only, final decisions by invigilators

## Project Structure

```
watcher/
├── configs/
│   ├── source.yaml      # Video source configuration
│   ├── detector.yaml    # Person/object detection settings
│   ├── pose.yaml        # Pose estimation settings
│   ├── rules.yaml       # Behaviour rule thresholds
│   └── seats.yaml       # Seat calibration and hall model
├── data/
│   ├── videos/          # Place test videos here
│   └── samples/
├── outputs/
│   ├── debug/           # Debug frames and overlays
│   ├── overlays/        # Annotated output videos
│   ├── alerts/          # Alert cards and evidence
│   └── evidence/        # Blurred clips and snapshots
├── scripts/
│   ├── check_source.py  # Phase 1: Validate video source
│   ├── run_detection.py # Phase 2: Person/object detection
│   ├── run_tracking.py  # Phase 3: Tracking + SeatTrack
│   ├── run_pose.py      # Phase 4: Pose estimation
│   └── run_watcher.py   # Full pipeline
├── watcher/
│   ├── __init__.py
│   ├── source.py        # Video source abstraction
│   ├── detector.py      # Person/object detection
│   ├── tracker.py       # ByteTrack + SeatTrack binding
│   ├── pose.py          # Pose estimation + smoothing
│   ├── features.py      # Head/hand feature extraction
│   ├── rules.py         # Rule engine + adaptive baseline
│   ├── risk.py          # Risk score accumulator
│   ├── alerts.py        # Alert generation + cards
│   ├── db.py            # SQLite logging
│   └── utils.py         # Utilities
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd watcher
pip install -r requirements.txt
```

### 2. Add Test Video

Place a test video at `data/videos/sample.mp4` or update `configs/source.yaml` with your video path.

### 3. Run Phase 1 Validation

```bash
python scripts/check_source.py
```

This will:
- Open the configured video source
- Print source metadata (type, resolution, FPS, frame count)
- Read 100 frames
- Save a test frame to `outputs/debug/frame_0001.jpg`

### 4. Verify Output

Check that:
- Source opens successfully
- Metadata is printed correctly
- Test frame is saved and visible

## Configuration

### Source Configuration (`configs/source.yaml`)

```yaml
source:
  active: "file://data/videos/sample.mp4"
  # Or: webcam://0, rtsp://..., http://...
```

Supported source types:
- `file://path/to/video.mp4` - Local video file
- `webcam://0` - Webcam device (0, 1, etc.)
- `rtsp://camera-url` - RTSP stream
- `http://stream-url` - HTTP/MJPEG stream

## Development Phases

| Phase | Description | Script |
|-------|-------------|--------|
| 1 | Source abstraction & validation | `check_source.py` |
| 2 | Person & object detection | `run_detection.py` |
| 3 | Tracking + SeatTrack binding | `run_tracking.py` |
| 4 | Pose estimation + keypoints | `run_pose.py` |
| 5 | Head orientation + hand features | *(TBD)* |
| 6 | Object-hand fusion + temporal window | *(TBD)* |
| 7 | Rule engine + adaptive baseline | *(TBD)* |
| 8 | Risk score + alerts + SQLite | *(TBD)* |
| 9 | Invigilator dashboard | *(TBD)* |
| 10 | Source compatibility + demo | *(TBD)* |

Each phase requires human validation before proceeding.

## Privacy & Ethics

- No face recognition or biometric storage
- No student identity matching
- Local processing only
- Evidence clips blurred where feasible
- Configurable retention limits
- Seat ID is primary audit reference (not student identity)

## License

Proprietary - Watcher Project
"""NETRA-4D event, review, media, and verified-run API."""

from __future__ import annotations

import asyncio
import cv2
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from drishti.event_repository import EventRepository
from drishti.events import EventRecord, ReviewDecision, SpecialistResolution

BASE_DIR = Path(__file__).resolve().parents[3]
MANIFEST_PATH = BASE_DIR / "drishti4d" / "configs" / "datasets.json"
VERIFIED_OUTPUTS = BASE_DIR / "outputs" / "verified"
FINAL_OUTPUTS = BASE_DIR / "OUTPUT"
SHOWCASE_INPUTS = FINAL_OUTPUTS / "showcase_inputs"
SHOWCASE_OUTPUTS = FINAL_OUTPUTS / "showcase_outputs"
PRELOADED_DEMO = BASE_DIR / "PRELOADED_IMG_VID_DEMO"
RESEARCH_REPORT = FINAL_OUTPUTS / "NETRA_4D_Research_Report.pdf"
OUTPUTS_ROOT = (BASE_DIR / "outputs").resolve()
DATABASE_PATH = Path(os.environ.get("NETRA4D_EVENT_DB", OUTPUTS_ROOT / "netra4d_events.sqlite"))
repository = EventRepository(DATABASE_PATH)

app = FastAPI(title="NETRA-4D Event API", description="Verified evidence and human review decisions.", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "http://127.0.0.1:5173", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_files() -> list[Path]:
    candidates = []
    for root in (FINAL_OUTPUTS, VERIFIED_OUTPUTS):
        if root.exists():
            candidates.extend(root.glob("*/run.json"))
    valid = []
    for path in candidates:
        try:
            record = _json(path)
            assets = record.get("assets", [])
            if len(assets) != 5:
                continue
            if all(
                (path.parent / item["asset_id"] / "COMPLETE.json").is_file()
                and not (path.parent / item["asset_id"] / "INCOMPLETE.txt").exists()
                and (path.parent / item["asset_id"] / item["outputs"]["annotated_video"]).stat().st_size
                    == item["integrity"]["annotated_video_bytes"]
                for item in assets
            ):
                valid.append(path)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(valid, key=lambda path: path.stat().st_mtime, reverse=True)


def _verified_run(run_id: str) -> tuple[Path, dict[str, Any]]:
    for path in _run_files():
        if path.parent.name == run_id:
            return path.parent, _json(path)
    raise HTTPException(status_code=404, detail="Verified run not found")


def _verified_asset(run_id: str, asset_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir, run = _verified_run(run_id)
    asset = next((item for item in run.get("assets", []) if item.get("asset_id") == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="Verified asset not found")
    return run_dir / asset_id, asset


def _event_or_404(event_id: str) -> EventRecord:
    event = repository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def _media(event: EventRecord, field: str) -> FileResponse:
    relative = getattr(event, field)
    if not relative:
        raise HTTPException(status_code=404, detail=f"Event has no {field}")
    path = (BASE_DIR / relative).resolve()
    if not any(path.is_relative_to(root) for root in (OUTPUTS_ROOT, FINAL_OUTPUTS.resolve())):
        raise HTTPException(status_code=403, detail="Media path is outside output roots")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(path)


@app.get("/api/health")
def health():
    runs = _run_files()
    return {"status": "ready", "version": "3.0.0", "verified_runs": len(runs), "latest_run_id": runs[0].parent.name if runs else None, "events": repository.counts(), "seeded_or_mock_events": 0}


@app.get("/api/datasets")
def datasets():
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset manifest not found")
    return _json(MANIFEST_PATH)


@app.get("/api/runs")
def runs():
    return [{"run_id": path.parent.name, "summary": _json(path).get("summary", {})} for path in _run_files()]


@app.get("/api/runs/latest")
def latest_run():
    files = _run_files()
    if not files:
        raise HTTPException(status_code=404, detail="No verified inference run exists yet")
    return _json(files[0])


@app.get("/api/showcase")
def showcase(limit: int = Query(3, ge=1, le=5)):
    files = _run_files()
    if not files:
        return {"run_id": None, "samples": []}
    run_path = files[0]
    run = _json(run_path)
    samples = []
    ranked_assets = sorted(
        run.get("assets", []),
        key=lambda item: (
            item.get("counts", {}).get("high_confidence_event_frames", 0),
            item.get("counts", {}).get("review_events", 0),
        ),
        reverse=True,
    )[:limit]
    for asset in ranked_assets:
        asset_id = asset["asset_id"]
        asset_dir = run_path.parent / asset_id
        evidence_dir = asset_dir / (asset.get("outputs", {}).get("review_event_frame_directory") or "review_event_frames")
        evidence_files = sorted(evidence_dir.glob("*.jpg")) if evidence_dir.is_dir() else []
        max_score = 0.0
        event_types: set[str] = set()
        event_path = asset_dir / (asset.get("outputs", {}).get("review_events") or "review_events.jsonl")
        if event_path.is_file():
            for line in event_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    max_score = max(max_score, float(event.get("evidence_score", 0)))
                    if event.get("event_type"):
                        event_types.add(str(event["event_type"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        video = asset.get("video", {})
        counts = asset.get("counts", {})
        source = asset.get("source", {})
        source_preview = SHOWCASE_INPUTS / f"{asset_id}_source_preview.mp4"
        samples.append({
            "run_id": run["run_id"], "asset_id": asset_id,
            "title": source.get("filename", asset_id),
            "source": {
                "url": source.get("source_url"), "filename": source.get("filename"),
                "bytes": source.get("bytes"), "sha256": source.get("sha256"),
                "local_file_available": source_preview.is_file(),
                "preview_url": f"/api/showcase/{quote(run['run_id'])}/{quote(asset_id)}/source-preview" if source_preview.is_file() else None,
                "preview_meaning": "Browser-compatible transcode of the hash-verified original; no annotations." if source_preview.is_file() else None,
            },
            "annotated_video_url": f"/api/showcase/{quote(run['run_id'])}/{quote(asset_id)}/annotated",
            "annotated_video_browser_optimized": (SHOWCASE_OUTPUTS / f"{asset_id}_annotated_preview.mp4").is_file(),
            "events_url": f"/api/showcase/{quote(run['run_id'])}/{quote(asset_id)}/events",
            "evidence_frames": [
                {"filename": frame.name, "url": f"/api/showcase/{quote(run['run_id'])}/{quote(asset_id)}/evidence/{quote(frame.name)}"}
                for frame in evidence_files[:6]
            ],
            "metrics": {
                "width": video.get("width"), "height": video.get("height"), "fps": video.get("fps"),
                "frames": counts.get("frames", 0), "people": counts.get("people", 0),
                "objects": counts.get("objects", 0), "review_events": counts.get("review_events", 0),
                "high_confidence_event_frames": counts.get("high_confidence_event_frames", 0),
                "max_evidence_score": max_score, "event_types": sorted(event_types),
                "elapsed_sec": asset.get("elapsed_sec"), "reasoning_engine": asset.get("reasoning_engine"),
            },
        })
    return {"run_id": run["run_id"], "completed_at_utc": run.get("completed_at_utc"), "samples": samples}


@app.get("/api/showcase/{run_id}/{asset_id}/annotated")
def showcase_video(run_id: str, asset_id: str):
    asset_dir, asset = _verified_asset(run_id, asset_id)
    browser_preview = SHOWCASE_OUTPUTS / f"{asset_id}_annotated_preview.mp4"
    path = browser_preview if browser_preview.is_file() else asset_dir / asset["outputs"]["annotated_video"]
    return FileResponse(path, media_type="video/mp4", filename=path.name, content_disposition_type="inline")


@app.get("/api/showcase/{run_id}/{asset_id}/source-preview")
def showcase_source_preview(run_id: str, asset_id: str):
    _verified_asset(run_id, asset_id)
    path = SHOWCASE_INPUTS / f"{asset_id}_source_preview.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Local source preview not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name, content_disposition_type="inline")


@app.get("/api/showcase/{run_id}/{asset_id}/events")
def showcase_events(run_id: str, asset_id: str):
    asset_dir, asset = _verified_asset(run_id, asset_id)
    path = asset_dir / asset["outputs"]["review_events"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Event log not found")
    return FileResponse(path, media_type="application/x-ndjson", filename=path.name)


@app.get("/api/showcase/{run_id}/{asset_id}/evidence/{filename}")
def showcase_evidence(run_id: str, asset_id: str, filename: str):
    if Path(filename).name != filename or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid evidence filename")
    asset_dir, asset = _verified_asset(run_id, asset_id)
    directory = asset_dir / (asset.get("outputs", {}).get("review_event_frame_directory") or "review_event_frames")
    path = directory / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence frame not found")
    return FileResponse(path)


@app.get("/api/reports/research-paper")
def research_paper_report():
    runs = _run_files()
    if not runs:
        raise HTTPException(status_code=404, detail="No verified run is available for the report")
    source_manifest = runs[0]
    if not RESEARCH_REPORT.is_file() or RESEARCH_REPORT.stat().st_mtime < source_manifest.stat().st_mtime:
        result = subprocess.run(
            [
                sys.executable, str(BASE_DIR / "scripts" / "generate_research_report.py"),
                "--run", str(source_manifest.relative_to(BASE_DIR)),
                "--output", str(RESEARCH_REPORT.relative_to(BASE_DIR)),
            ],
            cwd=BASE_DIR, capture_output=True, text=True,
        )
        if result.returncode != 0 or not RESEARCH_REPORT.is_file():
            raise HTTPException(status_code=500, detail="Research report generation failed")
    return FileResponse(
        RESEARCH_REPORT, media_type="application/pdf",
        filename="NETRA_4D_Technical_Research_Report.pdf",
    )


@app.get("/api/reports/operations")
def operations_report():
    runs = _run_files()
    if not runs:
        raise HTTPException(status_code=404, detail="No verified run is available for the report")
    output = FINAL_OUTPUTS / "NETRA_4D_Operational_Report.pdf"
    if output.is_file():
        return FileResponse(output, media_type="application/pdf", filename=output.name)
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "generate_operational_report.py"), "--run", str(runs[0]), "--output", str(output)],
        cwd=BASE_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0 or not output.is_file():
        raise HTTPException(status_code=500, detail="Operational report generation failed")
    return FileResponse(output, media_type="application/pdf", filename=output.name)


@app.get("/api/preloaded-demo")
def preloaded_demo_assets():
    """Expose user-provided demo evidence without copying it into frontend assets."""
    if not PRELOADED_DEMO.is_dir():
        return {"video": None, "images": [], "logs": []}
    images = sorted(PRELOADED_DEMO.glob("*.jpg"))
    videos = sorted(PRELOADED_DEMO.glob("*.mp4"))
    browser_video = next((item for item in videos if item.stem.endswith("_browser")), videos[0] if videos else None)
    logs = sorted(PRELOADED_DEMO.glob("*_features.json"))
    return {
        "video": {"name": browser_video.name, "url": f"/api/preloaded-demo/media/{quote(browser_video.name)}"} if browser_video else None,
        "images": [{"name": item.name, "url": f"/api/preloaded-demo/media/{quote(item.name)}"} for item in images],
        "logs": [{"name": item.name, "url": f"/api/preloaded-demo/log/{quote(item.name)}"} for item in logs],
    }


@app.get("/api/preloaded-demo/media/{filename}")
def preloaded_demo_media(filename: str):
    if Path(filename).name != filename or Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".mp4"}:
        raise HTTPException(status_code=400, detail="Invalid preloaded demo media")
    path = PRELOADED_DEMO / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Preloaded demo media not found")
    return FileResponse(path, media_type="video/mp4" if path.suffix.lower() == ".mp4" else "image/jpeg")


@app.get("/api/preloaded-demo/replay-frame")
def preloaded_replay_frame(timestamp: float = Query(0.0, ge=0.0, le=66.64)):
    """JPEG evidence frame extracted from the exact video used by the replay player."""
    source = PRELOADED_DEMO / "video_01_overlay.mp4"
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Preloaded replay video not found")
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok:
            raise HTTPException(status_code=404, detail="Replay frame could not be decoded")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise HTTPException(status_code=500, detail="Replay frame could not be encoded")
        return Response(content=encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
    finally:
        capture.release()


@app.get("/api/preloaded-demo/log/{filename}")
def preloaded_demo_log(filename: str):
    if Path(filename).name != filename or not filename.endswith("_features.json"):
        raise HTTPException(status_code=400, detail="Invalid preloaded demo log")
    path = PRELOADED_DEMO / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Preloaded demo log not found")
    data = _json(path)
    features = data.get("features", [])
    findings = []
    for item in features:
        head, hands, torso = item.get("head", {}), item.get("hands", {}), item.get("torso", {})
        abnormal = head.get("direction") not in {None, "FRONT"} or hands.get("left_state") not in {None, "NORMAL"} or hands.get("right_state") not in {None, "NORMAL"} or hands.get("left_below_desk") or hands.get("right_below_desk") or torso.get("angle_toward_neighbor") not in {None, 0}
        if abnormal:
            findings.append({"frame": item.get("frame_idx"), "timestamp": item.get("timestamp"), "track_id": item.get("track_id"), "seat_id": item.get("seat_id"), "head_direction": head.get("direction"), "left_hand": hands.get("left_state"), "right_hand": hands.get("right_state"), "below_desk": bool(hands.get("left_below_desk") or hands.get("right_below_desk"))})
        if len(findings) >= 24:
            break
    return {"source": data.get("source"), "total_frames": data.get("total_frames"), "duration_sec": data.get("duration_sec"), "findings": findings}


@app.post("/api/events", response_model=EventRecord)
def ingest_event(event: EventRecord):
    return repository.upsert(event)


@app.get("/api/events", response_model=list[EventRecord])
def list_events(review_status: str | None = None, event_type: str | None = None, video_id: str | None = None, candidate_id: int | None = None, workstation_id: str | None = None, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    return repository.list(review_status=review_status, event_type=event_type, video_id=video_id, candidate_id=candidate_id, workstation_id=workstation_id, limit=limit, offset=offset)


@app.get("/api/events/{event_id}", response_model=EventRecord)
def get_event(event_id: str):
    return _event_or_404(event_id)


@app.patch("/api/events/{event_id}/review", response_model=EventRecord)
def review_event(event_id: str, decision: ReviewDecision):
    result = repository.review(event_id, decision)
    if result is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return result


@app.post("/api/events/{event_id}/specialist-resolution", response_model=EventRecord)
def specialist_resolution(event_id: str, resolution: SpecialistResolution):
    result = repository.resolve_specialist(event_id, resolution)
    if result is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return result


@app.get("/api/candidates/{candidate_id}/events", response_model=list[EventRecord])
def candidate_timeline(candidate_id: int, limit: int = Query(100, ge=1, le=500)):
    return repository.list(candidate_id=candidate_id, limit=limit)


@app.get("/api/workstations/{workstation_id}/events", response_model=list[EventRecord])
def workstation_timeline(workstation_id: str, limit: int = Query(100, ge=1, le=500)):
    return repository.list(workstation_id=workstation_id, limit=limit)


@app.get("/api/events/{event_id}/keyframe")
def event_keyframe(event_id: str):
    return _media(_event_or_404(event_id), "keyframe_path")


@app.get("/api/events/{event_id}/clip")
def event_clip(event_id: str):
    return _media(_event_or_404(event_id), "clip_path")


@app.websocket("/ws/events")
async def event_stream(websocket: WebSocket):
    await websocket.accept()
    seen: dict[str, str] = {}
    try:
        while True:
            for event in reversed(repository.list(limit=200)):
                version = event.updated_at_utc.isoformat()
                if seen.get(event.event_id) != version:
                    await websocket.send_json(event.model_dump(mode="json"))
                    seen[event.event_id] = version
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


LIVE_ROOT = FINAL_OUTPUTS / "live_sessions"
UPLOAD_ROOT = LIVE_ROOT / "uploads"
live_processes: dict[str, subprocess.Popen] = {}


def _write_session_metadata(publish_dir: Path, metadata: dict[str, Any]) -> None:
    (publish_dir / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _record_monitoring_action(publish_dir: Path, action: str, operator: str) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "operator": operator.strip() or "INVIGILATOR",
    }
    with (publish_dir / "monitoring_control.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")


def _arm_live_session(source: str, source_kind: str, browser_settings: dict[str, float] | None = None) -> dict[str, Any]:
    session_id = f"live_{uuid4().hex[:12]}"
    publish_dir = LIVE_ROOT / session_id
    publish_dir.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "session_id": session_id, "source_kind": source_kind, "source": source,
        "status": "ARMED", "pid": None,
        "stream_url": f"/api/live/sessions/{session_id}/stream.mjpg",
        "telemetry_url": f"/ws/live/{session_id}",
        "processor_log": str((publish_dir / "processor.log").relative_to(BASE_DIR)),
        "control_log_url": f"/api/live/sessions/{session_id}/controls",
    }
    if browser_settings is not None:
        browser_input = publish_dir / "browser_input"
        browser_input.mkdir()
        metadata["browser_frame_url"] = f"/api/live/sessions/{session_id}/browser-frame"
        metadata["browser_settings"] = browser_settings
    _write_session_metadata(publish_dir, metadata)
    _record_monitoring_action(publish_dir, "ARMED", "INVIGILATOR")
    return metadata


def _launch_live_process(session_id: str, operator: str) -> dict[str, Any]:
    publish_dir = LIVE_ROOT / session_id
    metadata_path = publish_dir / "session.json"
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Live session not found")
    metadata = _json(metadata_path)
    current_status = metadata.get("status")
    if current_status == "PAUSED":
        process = live_processes.get(session_id)
        if process is None or process.poll() is not None:
            raise HTTPException(status_code=409, detail="Paused processor is no longer available; arm a new source")
        control_file = publish_dir / "monitoring.enabled"
        control_file.touch(exist_ok=True)
        metadata.update({
            "status": "RUNNING",
            "monitoring_resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "monitoring_resumed_by": operator.strip() or "INVIGILATOR",
        })
        _write_session_metadata(publish_dir, metadata)
        _record_monitoring_action(publish_dir, "MONITORING_RESUMED", operator)
        return metadata
    if current_status != "ARMED":
        raise HTTPException(status_code=409, detail=f"Session is {current_status}; arm a new source before starting")
    source = str(metadata["source"])
    command = [
        sys.executable, str(BASE_DIR / "scripts" / "run_live_webcam.py"),
        "--source", source, "--camera-id", session_id, "--device", "auto",
        "--live-publish-dir", str(publish_dir.relative_to(BASE_DIR)),
        "--output-dir", str((FINAL_OUTPUTS / "live_runs").relative_to(BASE_DIR)),
        "--auto-calibration-warmup", "1.0",
    ]
    browser_settings = metadata.get("browser_settings")
    if browser_settings is not None:
        browser_input = publish_dir / "browser_input"
        command.extend([
            "--browser-frame-dir", str(browser_input.relative_to(BASE_DIR)),
            "--browser-width", str(int(browser_settings["width"])),
            "--browser-height", str(int(browser_settings["height"])),
            "--browser-fps", str(float(browser_settings["fps"])),
        ])
    is_continuous_source = metadata["source_kind"] in {"BROWSER_WEBCAM", "WEBCAM", "RTSP"}
    # Continuous browser/RTSP sessions publish the latest annotated frame only.
    # Avoid writing a second full video while the operator is monitoring live.
    if not is_continuous_source:
        command.append("--record-stream")
    if is_continuous_source:
        control_file = publish_dir / "monitoring.enabled"
        control_file.touch(exist_ok=True)
        command.extend(["--monitoring-control-file", str(control_file.relative_to(BASE_DIR))])
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process_log_path = publish_dir / "processor.log"
    with process_log_path.open("a", encoding="utf-8") as process_log:
        process = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            creationflags=creationflags,
            stdout=process_log,
            stderr=subprocess.STDOUT,
        )
    live_processes[session_id] = process
    metadata.update({
        "status": "RUNNING", "pid": process.pid,
        "monitoring_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitoring_started_by": operator.strip() or "INVIGILATOR",
    })
    if is_continuous_source:
        metadata["monitoring_control_file"] = str((publish_dir / "monitoring.enabled").relative_to(BASE_DIR))
    _write_session_metadata(publish_dir, metadata)
    _record_monitoring_action(publish_dir, "MONITORING_STARTED", operator)
    return metadata


@app.post("/api/live/sessions/webcam")
def start_webcam(source: str = Query("0", description="Webcam index or rtsp:// MediaMTX URL")):
    if not (source.isdigit() or source.lower().startswith("rtsp://")):
        raise HTTPException(status_code=400, detail="Source must be a webcam index or RTSP URL")
    return _arm_live_session(source, "RTSP" if source.lower().startswith("rtsp://") else "WEBCAM")


@app.post("/api/live/sessions/browser")
def start_browser_webcam(
    width: int = Query(640, ge=160, le=3840),
    height: int = Query(480, ge=120, le=2160),
    fps: float = Query(5.0, ge=1.0, le=15.0),
):
    return _arm_live_session("browser:getUserMedia", "BROWSER_WEBCAM", {
        "width": float(width), "height": float(height), "fps": float(fps),
    })


@app.post("/api/live/sessions/{session_id}/browser-frame", status_code=202)
async def ingest_browser_frame(session_id: str, request: Request):
    directory = LIVE_ROOT / session_id
    metadata_path = directory / "session.json"
    if not metadata_path.is_file() or _json(metadata_path).get("source_kind") != "BROWSER_WEBCAM":
        raise HTTPException(status_code=404, detail="Browser webcam session not found")
    process = live_processes.get(session_id)
    if _json(metadata_path).get("status") != "RUNNING" or process is None or process.poll() is not None:
        raise HTTPException(status_code=409, detail="Browser webcam processor is not running")
    payload = await request.body()
    if not payload or len(payload) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Browser frame must be between 1 byte and 4 MiB")
    if not payload.startswith(b"\xff\xd8"):
        raise HTTPException(status_code=415, detail="Browser frame must be JPEG")
    input_dir = directory / "browser_input"
    temporary = input_dir / "latest.jpg.tmp"
    temporary.write_bytes(payload)
    temporary.replace(input_dir / "latest.jpg")
    return {"session_id": session_id, "accepted_bytes": len(payload)}


@app.post("/api/live/sessions/upload")
async def start_uploaded_video(request: Request, filename: str = Query("uploaded_video.mp4")):
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported video extension")
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_ROOT / f"{uuid4().hex}{suffix}"
    size = 0
    with destination.open("wb") as output:
        async for chunk in request.stream():
            size += len(chunk)
            if size > 2 * 1024 * 1024 * 1024:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds 2 GiB demo limit")
            output.write(chunk)
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty")
    return {**_arm_live_session(str(destination), "UPLOADED_VIDEO"), "uploaded_bytes": size, "original_filename": Path(filename).name}


@app.get("/api/live/sessions")
def list_live_sessions():
    sessions = []
    for metadata_path in sorted(LIVE_ROOT.glob("live_*/session.json"), reverse=True) if LIVE_ROOT.exists() else []:
        metadata = _json(metadata_path)
        process = live_processes.get(metadata["session_id"])
        if process:
            # A continuous source keeps its capture worker alive while the
            # invigilator pauses inference. Do not let the worker heartbeat
            # overwrite the durable PAUSED gate state on every list request.
            if process.poll() is None:
                if metadata.get("status") not in {"PAUSED", "STOPPED"}:
                    metadata["status"] = "RUNNING"
            else:
                metadata["status"] = "ENDED"
        metadata["telemetry_available"] = (metadata_path.parent / "latest.json").is_file()
        sessions.append(metadata)
    return sessions


@app.post("/api/live/sessions/{session_id}/start")
def start_monitoring_session(session_id: str, operator: str = Query("INVIGILATOR", max_length=80)):
    """Begin inference only after an invigilator explicitly authorizes monitoring."""
    return _launch_live_process(session_id, operator)


@app.get("/api/live/sessions/{session_id}/controls")
def monitoring_control_log(session_id: str):
    path = LIVE_ROOT / session_id / "monitoring_control.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Monitoring control log not found")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.post("/api/live/sessions/{session_id}/stop")
def stop_live_session(session_id: str, operator: str = Query("INVIGILATOR", max_length=80)):
    publish_dir = LIVE_ROOT / session_id
    metadata_path = publish_dir / "session.json"
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Live session not found")
    metadata = _json(metadata_path)
    is_continuous_source = metadata.get("source_kind") in {"BROWSER_WEBCAM", "WEBCAM", "RTSP"}
    if is_continuous_source:
        (publish_dir / "monitoring.enabled").unlink(missing_ok=True)
        status, action = "PAUSED", "MONITORING_PAUSED"
    else:
        process = live_processes.get(session_id)
        if process and process.poll() is None:
            process.terminate()
        status, action = "STOPPED", "MONITORING_STOPPED"
    metadata.update({
        "status": status,
        "monitoring_stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitoring_stopped_by": operator.strip() or "INVIGILATOR",
    })
    _write_session_metadata(publish_dir, metadata)
    _record_monitoring_action(publish_dir, action, operator)
    return {"session_id": session_id, "status": status}


@app.get("/api/live/sessions/{session_id}/stream.mjpg")
def live_mjpeg(session_id: str):
    directory = LIVE_ROOT / session_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Live session not found")

    def frames():
        last_mtime = -1
        while True:
            image = directory / "latest.jpg"
            if image.is_file() and image.stat().st_mtime_ns != last_mtime:
                last_mtime = image.stat().st_mtime_ns
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + image.read_bytes() + b"\r\n"
            process = live_processes.get(session_id)
            if process and process.poll() is not None:
                break
            time.sleep(0.03)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/live/{session_id}")
async def live_telemetry(websocket: WebSocket, session_id: str):
    directory = LIVE_ROOT / session_id
    if not directory.is_dir():
        await websocket.close(code=4404)
        return
    await websocket.accept()
    last_mtime = -1
    try:
        while True:
            latest = directory / "latest.json"
            if latest.is_file() and latest.stat().st_mtime_ns != last_mtime:
                last_mtime = latest.stat().st_mtime_ns
                await websocket.send_text(latest.read_text(encoding="utf-8"))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return


FRONTEND_DIR = BASE_DIR / "Frontend_Updated" / "dist"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

"""NETRA-4D event, review, media, and verified-run API."""

from __future__ import annotations

import asyncio
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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from drishti.event_repository import EventRepository
from drishti.events import EventRecord, ReviewDecision, SpecialistResolution

BASE_DIR = Path(__file__).resolve().parents[3]
MANIFEST_PATH = BASE_DIR / "drishti4d" / "configs" / "datasets.json"
VERIFIED_OUTPUTS = BASE_DIR / "outputs" / "verified"
FINAL_OUTPUTS = BASE_DIR / "OUTPUT"
SHOWCASE_INPUTS = FINAL_OUTPUTS / "showcase_inputs"
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
    path = asset_dir / asset["outputs"]["annotated_video"]
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


def _start_live_process(source: str, source_kind: str) -> dict[str, Any]:
    session_id = f"live_{uuid4().hex[:12]}"
    publish_dir = LIVE_ROOT / session_id
    publish_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, str(BASE_DIR / "scripts" / "run_live_webcam.py"),
        "--source", source, "--camera-id", session_id, "--device", "auto",
        "--live-publish-dir", str(publish_dir.relative_to(BASE_DIR)),
        "--output-dir", str((FINAL_OUTPUTS / "live_runs").relative_to(BASE_DIR)),
        "--auto-calibration-warmup", "1.0",
        "--record-stream",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=BASE_DIR, creationflags=creationflags)
    live_processes[session_id] = process
    metadata = {
        "session_id": session_id, "source_kind": source_kind, "source": source,
        "status": "INITIALIZING", "pid": process.pid,
        "stream_url": f"/api/live/sessions/{session_id}/stream.mjpg",
        "telemetry_url": f"/ws/live/{session_id}",
    }
    (publish_dir / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@app.post("/api/live/sessions/webcam")
def start_webcam(source: str = Query("0", description="Webcam index or rtsp:// MediaMTX URL")):
    if not (source.isdigit() or source.lower().startswith("rtsp://")):
        raise HTTPException(status_code=400, detail="Source must be a webcam index or RTSP URL")
    return _start_live_process(source, "RTSP" if source.lower().startswith("rtsp://") else "WEBCAM")


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
    return {**_start_live_process(str(destination), "UPLOADED_VIDEO"), "uploaded_bytes": size, "original_filename": Path(filename).name}


@app.get("/api/live/sessions")
def list_live_sessions():
    sessions = []
    for metadata_path in sorted(LIVE_ROOT.glob("live_*/session.json"), reverse=True) if LIVE_ROOT.exists() else []:
        metadata = _json(metadata_path)
        process = live_processes.get(metadata["session_id"])
        metadata["status"] = "RUNNING" if process and process.poll() is None else "ENDED" if process else "RECOVERED"
        metadata["telemetry_available"] = (metadata_path.parent / "latest.json").is_file()
        sessions.append(metadata)
    return sessions


@app.post("/api/live/sessions/{session_id}/stop")
def stop_live_session(session_id: str):
    process = live_processes.get(session_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Live session not found in this API process")
    if process.poll() is None:
        process.terminate()
    return {"session_id": session_id, "status": "STOPPING"}


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

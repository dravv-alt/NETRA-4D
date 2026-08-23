from pathlib import Path

from fastapi.testclient import TestClient

from drishti.api import app as api_module
from drishti.event_repository import EventRepository
from drishti.events import EvidenceKind, EventLifecycle, EventRecord, ReviewDecision, ReviewStatus


def test_api_functions_use_review_repository(monkeypatch):
    path = Path("outputs/test_api_events.sqlite")
    path.unlink(missing_ok=True)
    try:
        repository = EventRepository(path)
        monkeypatch.setattr(api_module, "repository", repository)
        record = EventRecord(
            event_id="api-event-1", run_id="run", camera_id="cam", video_id="video",
            event_type="TEST_REVIEW", label="TEST", lifecycle=EventLifecycle.CLOSED,
            evidence_kind=EvidenceKind.BEHAVIORAL_PROXY, evidence_score=0.5,
            candidate_ids=[4], started_at_sec=0, ended_at_sec=1, duration_sec=1,
        )
        assert api_module.ingest_event(record).event_id == "api-event-1"
        assert api_module.candidate_timeline(4, limit=100)[0].event_id == "api-event-1"
        reviewed = api_module.review_event(
            "api-event-1",
            ReviewDecision(status=ReviewStatus.FALSE_POSITIVE, reviewed_by="human"),
        )
        assert reviewed.review_status == ReviewStatus.FALSE_POSITIVE
    finally:
        path.unlink(missing_ok=True)


def test_http_and_websocket_event_contract(monkeypatch):
    path = Path("outputs/test_api_http.sqlite")
    path.unlink(missing_ok=True)
    try:
        repository = EventRepository(path)
        monkeypatch.setattr(api_module, "repository", repository)
        record = EventRecord(
            event_id="api-event-http", run_id="run", camera_id="cam", video_id="video",
            event_type="PHONE_USE_REVIEW", label="PHONE USE REVIEW", lifecycle=EventLifecycle.CLOSED,
            evidence_kind=EvidenceKind.MODEL_PLUS_GEOMETRY, evidence_score=0.7,
            candidate_ids=[9], workstation_ids=["W9"], started_at_sec=1, ended_at_sec=2, duration_sec=1,
        )
        repository.upsert(record)
        client = TestClient(api_module.app)
        response = client.get("/api/events", params={"candidate_id": 9})
        assert response.status_code == 200
        assert response.json()[0]["event_id"] == "api-event-http"
        workstation = client.get("/api/workstations/W9/events")
        assert workstation.status_code == 200
        assert workstation.json()[0]["event_id"] == "api-event-http"
        resolution = client.post("/api/events/api-event-http/specialist-resolution", json={
            "provider": "MULTI_HMR_SMPLX", "supports_event": True, "confidence": .8,
            "measurements": {"torso_yaw_deg": 39}, "model_provenance": {"checkpoint": "real"},
            "explanation": "Torso direction supports the target geometry.",
        })
        assert resolution.status_code == 200
        assert resolution.json()["specialist_status"] == "RESOLVED"
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["event_id"] == "api-event-http"
    finally:
        path.unlink(missing_ok=True)

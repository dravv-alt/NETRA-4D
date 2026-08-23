from drishti.event_repository import EventRepository
from drishti.events import EvidenceKind, EventLifecycle, EventRecord, ReviewDecision, ReviewStatus, SpecialistResolution, SpecialistStatus


def event() -> EventRecord:
    return EventRecord(
        event_id="event-0001",
        run_id="run-1",
        camera_id="CAM-1",
        video_id="VIDEO_01",
        event_type="PHONE_USE_REVIEW",
        label="PHONE USE REVIEW",
        lifecycle=EventLifecycle.CLOSED,
        evidence_kind=EvidenceKind.MODEL_PLUS_GEOMETRY,
        evidence_score=0.8,
        candidate_ids=[7],
        workstation_ids=["W7"],
        started_at_sec=2.0,
        ended_at_sec=3.0,
        duration_sec=1.0,
    )


def test_event_repository_review_audit():
    path = Path("outputs/test_event_repository.sqlite")
    path.unlink(missing_ok=True)
    try:
        repository = EventRepository(path)
        repository.upsert(event())
        assert repository.counts() == {"PENDING": 1, "TOTAL": 1}
        assert repository.list(candidate_id=7)[0].event_id == "event-0001"
        assert repository.list(workstation_id="W7")[0].event_id == "event-0001"
        reviewed = repository.review(
            "event-0001",
            ReviewDecision(status=ReviewStatus.CONFIRMED, reviewed_by="reviewer", comment="visible"),
        )
        assert reviewed is not None
        assert reviewed.review_status == ReviewStatus.CONFIRMED
        assert repository.counts() == {"CONFIRMED": 1, "TOTAL": 1}
        repository.upsert(event())
        assert repository.get("event-0001").review_status == ReviewStatus.CONFIRMED
        assert repository.delete_pending_run("run-1") == 0
    finally:
        path.unlink(missing_ok=True)


def test_closed_event_requires_end_time():
    payload = event().model_dump()
    payload["ended_at_sec"] = None
    try:
        EventRecord.model_validate(payload)
    except ValueError as exc:
        assert "ended_at_sec" in str(exc)
    else:
        raise AssertionError("CLOSED event without end time should fail")


def test_specialist_resolution_updates_evidence_and_uncertainty():
    path = Path("outputs/test_specialist_repository.sqlite")
    path.unlink(missing_ok=True)
    try:
        repository = EventRepository(path)
        record = event()
        record.uncertainty = .7
        repository.upsert(record)
        resolved = repository.resolve_specialist(record.event_id, SpecialistResolution(
            provider="MULTI_HMR_SMPLX", supports_event=False, confidence=.8,
            measurements={"torso_yaw_deg": 3.0}, model_provenance={"checkpoint": "real"},
            explanation="Torso remained forward.",
        ))
        assert resolved.specialist_status == SpecialistStatus.RESOLVED
        assert resolved.evidence_score < .8
        assert resolved.uncertainty <= .2
    finally:
        path.unlink(missing_ok=True)
from pathlib import Path

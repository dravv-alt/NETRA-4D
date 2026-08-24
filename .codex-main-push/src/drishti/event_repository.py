"""SQLite persistence for review events and human decisions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .events import EventRecord, ReviewDecision, SpecialistResolution, SpecialistStatus


class EventRepository:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    lifecycle TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    started_at_sec REAL NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_created ON review_events(created_at_utc, event_id);
                CREATE INDEX IF NOT EXISTS idx_events_video ON review_events(video_id, started_at_sec);
                CREATE INDEX IF NOT EXISTS idx_events_type ON review_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_review ON review_events(review_status);
                CREATE TABLE IF NOT EXISTS review_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    previous_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    reviewed_by TEXT NOT NULL,
                    comment TEXT,
                    reviewed_at_utc TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES review_events(event_id)
                );
                CREATE TABLE IF NOT EXISTS specialist_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    supports_event INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    resolved_at_utc TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES review_events(event_id)
                );
            """)

    @staticmethod
    def _payload(row: sqlite3.Row) -> EventRecord:
        return EventRecord.model_validate_json(row["payload_json"])

    def upsert(self, event: EventRecord) -> EventRecord:
        existing = self.get(event.event_id)
        if existing is not None and existing.review_status.value != "PENDING":
            event.review_status = existing.review_status
            event.review_comment = existing.review_comment
            event.reviewed_by = existing.reviewed_by
            event.reviewed_at_utc = existing.reviewed_at_utc
        event.updated_at_utc = datetime.now(timezone.utc)
        payload = event.model_dump_json()
        with self._connection() as connection:
            connection.execute("""
                INSERT INTO review_events
                    (event_id, run_id, camera_id, video_id, event_type, lifecycle,
                     review_status, started_at_sec, created_at_utc, updated_at_utc, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    lifecycle=excluded.lifecycle,
                    review_status=excluded.review_status,
                    updated_at_utc=excluded.updated_at_utc,
                    payload_json=excluded.payload_json
            """, (
                event.event_id, event.run_id, event.camera_id, event.video_id,
                event.event_type, event.lifecycle.value, event.review_status.value,
                event.started_at_sec, event.created_at_utc.isoformat(),
                event.updated_at_utc.isoformat(), payload,
            ))
        return event

    def delete_pending_run(self, run_id: str) -> int:
        """Remove regenerable pending imports without deleting human decisions."""
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM review_events WHERE run_id = ? AND review_status = 'PENDING'",
                (run_id,),
            )
            return int(cursor.rowcount)

    def get(self, event_id: str) -> EventRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM review_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._payload(row) if row else None

    def list(
        self, *, review_status: str | None = None, event_type: str | None = None,
        video_id: str | None = None, candidate_id: int | None = None,
        workstation_id: str | None = None,
        limit: int = 100, offset: int = 0,
    ) -> list[EventRecord]:
        clauses, params = [], []
        for column, value in (("review_status", review_status), ("event_type", event_type), ("video_id", video_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        query = "SELECT payload_json FROM review_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        bounded_limit, bounded_offset = min(max(limit, 1), 500), max(offset, 0)
        query += " ORDER BY created_at_utc DESC, event_id DESC"
        if candidate_id is None and workstation_id is None:
            query += " LIMIT ? OFFSET ?"
            params.extend([bounded_limit, bounded_offset])
        with self._connection() as connection:
            records = [self._payload(row) for row in connection.execute(query, params).fetchall()]
        if candidate_id is not None:
            records = [record for record in records if candidate_id in record.candidate_ids]
        if workstation_id is not None:
            records = [record for record in records if workstation_id in record.workstation_ids]
        if candidate_id is not None or workstation_id is not None:
            records = records[bounded_offset:bounded_offset + bounded_limit]
        return records

    def review(self, event_id: str, decision: ReviewDecision) -> EventRecord | None:
        event = self.get(event_id)
        if event is None:
            return None
        previous = event.review_status.value
        reviewed_at = datetime.now(timezone.utc)
        event.review_status = decision.status
        event.review_comment = decision.comment
        event.reviewed_by = decision.reviewed_by
        event.reviewed_at_utc = reviewed_at
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO review_audit (event_id, previous_status, new_status, reviewed_by, comment, reviewed_at_utc) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, previous, decision.status.value, decision.reviewed_by, decision.comment, reviewed_at.isoformat()),
            )
        return self.upsert(event)

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT review_status, COUNT(*) AS count FROM review_events GROUP BY review_status"
            ).fetchall()
        result = {row["review_status"]: int(row["count"]) for row in rows}
        result["TOTAL"] = sum(result.values())
        return result

    def resolve_specialist(self, event_id: str, resolution: SpecialistResolution) -> EventRecord | None:
        event = self.get(event_id)
        if event is None:
            return None
        confidence = resolution.confidence
        if resolution.supports_event:
            event.evidence_score = min(1.0, event.evidence_score + (1.0 - event.evidence_score) * 0.35 * confidence)
        else:
            event.evidence_score = max(0.0, event.evidence_score * (1.0 - 0.50 * confidence))
        event.uncertainty = min(event.uncertainty if event.uncertainty is not None else 1.0, 1.0 - confidence)
        event.specialist_status = SpecialistStatus.RESOLVED
        event.details["specialist_resolution"] = resolution.model_dump(mode="json")
        event.reasoning_provenance["specialist_model"] = resolution.model_provenance
        resolved_at = datetime.now(timezone.utc)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO specialist_audit (event_id, provider, supports_event, confidence, payload_json, resolved_at_utc) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, resolution.provider, int(resolution.supports_event), confidence,
                 resolution.model_dump_json(), resolved_at.isoformat()),
            )
        return self.upsert(event)

"""Stable backend-facing event contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventLifecycle(str, Enum):
    OPEN = "OPEN"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"


class EvidenceKind(str, Enum):
    MODEL_DETECTION = "MODEL_DETECTION"
    MODEL_PLUS_GEOMETRY = "MODEL_PLUS_GEOMETRY"
    BEHAVIORAL_PROXY = "BEHAVIORAL_PROXY"
    MODEL_PLUS_GEOMETRY_PROXY = "MODEL_PLUS_GEOMETRY_PROXY"


class SpecialistStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    REQUESTED = "REQUESTED"
    RESOLVED = "RESOLVED"


class EventRecord(BaseModel):
    schema_version: str = "1.0.0"
    event_id: str = Field(min_length=8, max_length=128)
    run_id: str
    camera_id: str
    video_id: str
    event_type: str
    label: str
    lifecycle: EventLifecycle
    evidence_kind: EvidenceKind
    evidence_score: float = Field(ge=0.0, le=1.0)
    candidate_ids: list[int] = Field(min_length=1)
    workstation_ids: list[str] = Field(default_factory=list)
    started_at_sec: float = Field(ge=0.0)
    ended_at_sec: float | None = Field(default=None, ge=0.0)
    started_at_utc: datetime | None = None
    ended_at_utc: datetime | None = None
    duration_sec: float = Field(default=0.0, ge=0.0)
    source_signals: list[str] = Field(default_factory=list)
    keyframe_path: str | None = None
    clip_path: str | None = None
    model_provenance: dict[str, Any] = Field(default_factory=dict)
    reasoning_provenance: dict[str, Any] = Field(default_factory=dict)
    review_priority: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    specialist_status: SpecialistStatus = SpecialistStatus.NOT_REQUIRED
    specialist_request: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_comment: str | None = None
    reviewed_by: str | None = None
    reviewed_at_utc: datetime | None = None
    created_at_utc: datetime = Field(default_factory=utc_now)
    updated_at_utc: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_lifecycle(self):
        if self.ended_at_sec is not None and self.ended_at_sec < self.started_at_sec:
            raise ValueError("ended_at_sec cannot precede started_at_sec")
        if self.lifecycle == EventLifecycle.CLOSED and self.ended_at_sec is None:
            raise ValueError("CLOSED events require ended_at_sec")
        if self.ended_at_utc is not None and self.started_at_utc is not None and self.ended_at_utc < self.started_at_utc:
            raise ValueError("ended_at_utc cannot precede started_at_utc")
        return self


class ReviewDecision(BaseModel):
    status: ReviewStatus
    reviewed_by: str = Field(min_length=1, max_length=120)
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def reject_pending(self):
        if self.status == ReviewStatus.PENDING:
            raise ValueError("A review decision cannot set status back to PENDING")
        return self


class SpecialistResolution(BaseModel):
    provider: str = Field(pattern="^(MULTI_HMR_SMPLX|RTMPOSE|HUMAN_GEOMETRY_REVIEW)$")
    supports_event: bool
    confidence: float = Field(ge=0.0, le=1.0)
    measurements: dict[str, Any] = Field(default_factory=dict)
    model_provenance: dict[str, Any] = Field(default_factory=dict)
    explanation: str = Field(min_length=1, max_length=2000)

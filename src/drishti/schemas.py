"""Typed records emitted directly by verified perception models."""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field


class QualityEnum(str, Enum):
    GOOD = "GOOD"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


class ObjectClassEnum(str, Enum):
    MOBILE_PHONE = "MOBILE_PHONE"
    PAPER_CHIT = "PAPER_CHIT"
    MOUSE = "MOUSE"
    MONITOR = "MONITOR"
    KEYBOARD = "KEYBOARD"
    UNCERTAIN_OBJECT = "UNCERTAIN_OBJECT"


class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float


class ComponentQuality(BaseModel):
    head: QualityEnum = QualityEnum.UNKNOWN
    torso: QualityEnum = QualityEnum.UNKNOWN
    hands: QualityEnum = QualityEnum.UNKNOWN
    track: QualityEnum = QualityEnum.UNKNOWN


class PersonObservation(BaseModel):
    schema_version: str = "2.0.0"
    provider: str
    camera_id: str
    video_id: str
    frame_index: int
    timestamp_sec: float
    entity_local_id: int
    bbox_xyxy_px: Tuple[float, float, float, float]
    bbox_xyxy_norm: Tuple[float, float, float, float]
    detection_confidence: float
    keypoints_2d: Dict[str, Keypoint] = Field(default_factory=dict)
    quality: ComponentQuality = Field(default_factory=ComponentQuality)
    seat_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObjectObservation(BaseModel):
    schema_version: str = "2.0.0"
    camera_id: str
    video_id: str
    frame_index: int
    timestamp_sec: float
    object_class: ObjectClassEnum
    confidence: float
    bbox_xyxy_px: Tuple[float, float, float, float]
    bbox_xyxy_norm: Tuple[float, float, float, float]
    associated_entity_id: Optional[int] = None
    associated_seat_id: Optional[str] = None
    association_confidence: float = 0.0
    is_static_equipment: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

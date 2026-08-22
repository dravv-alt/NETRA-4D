"""
Typed Data Schemas for Drishti-4D / NETRA-4D Forensic Engine.
Provides decoupling between perception models (YOLO26, RTMPose, Multi-HMR)
and downstream behavioural risk reasoning.
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field
import time


class RiskStateEnum(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    REVIEW = "REVIEW"
    UNKNOWN = "UNKNOWN"


class QualityEnum(str, Enum):
    GOOD = "GOOD"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


class BehaviourEnum(str, Enum):
    NORMAL = "NORMAL"
    REPEATED_SIDEWARD_GLANCE = "REPEATED_SIDEWARD_GLANCE"
    SUSTAINED_EXCESSIVE_HEAD_TURN = "SUSTAINED_EXCESSIVE_HEAD_TURN"
    TORSO_ROTATION_TOWARD_NEIGHBOUR = "TORSO_ROTATION_TOWARD_NEIGHBOUR"
    UNUSUAL_HAND_ACTIVITY = "UNUSUAL_HAND_ACTIVITY"
    POSSIBLE_NONVERBAL_INTERACTION = "POSSIBLE_NONVERBAL_INTERACTION"
    POSSIBLE_OBJECT_INTERACTION = "POSSIBLE_OBJECT_INTERACTION"
    SEAT_EXCHANGE = "SEAT_EXCHANGE"
    UNKNOWN = "UNKNOWN"


class ObjectClassEnum(str, Enum):
    MOBILE_PHONE = "MOBILE_PHONE"
    PAPER_CHIT = "PAPER_CHIT"
    MOUSE = "MOUSE"
    MONITOR = "MONITOR"
    KEYBOARD = "KEYBOARD"
    OTHER_PROHIBITED_OBJECT = "OTHER_PROHIBITED_OBJECT"
    UNCERTAIN_OBJECT = "UNCERTAIN_OBJECT"


class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float
    z: Optional[float] = None  # 3D z-depth if from Multi-HMR / 3D


class ComponentQuality(BaseModel):
    head: QualityEnum = QualityEnum.GOOD
    torso: QualityEnum = QualityEnum.GOOD
    hands: QualityEnum = QualityEnum.GOOD
    track: QualityEnum = QualityEnum.GOOD
    camera: QualityEnum = QualityEnum.GOOD


class PersonObservation(BaseModel):
    """Unified observation record emitted by any perception provider (YOLO26, RTMPose, Multi-HMR)."""
    schema_version: str = "1.0.0"
    provider: str  # "yolo26" | "rtmpose" | "multihmr" | "replay"
    camera_id: str
    video_id: str
    frame_index: int
    timestamp_sec: float
    entity_local_id: int
    bbox_xyxy_px: Tuple[float, float, float, float]
    bbox_xyxy_norm: Tuple[float, float, float, float]
    detection_confidence: float
    keypoints_2d: Optional[Dict[str, Keypoint]] = None
    joints_3d: Optional[Dict[str, List[float]]] = None  # name -> [x, y, z]
    rotation_vectors: Optional[Dict[str, List[float]]] = None  # name -> [rx, ry, rz]
    body_parameters: Optional[Dict[str, Any]] = None  # SMPL-X parameters or external artifact ref
    quality: ComponentQuality = Field(default_factory=ComponentQuality)
    seat_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObjectObservation(BaseModel):
    """Observation record for detected objects."""
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


class TrackState(BaseModel):
    """Persistent track state across frames."""
    track_id: int
    seat_id: Optional[str] = None
    first_seen_sec: float
    last_seen_sec: float
    track_quality: float = 1.0
    visible_keypoint_ratio: float = 1.0
    history_size: int = 0
    status: str = "ACTIVE"  # "ACTIVE" | "LOST" | "RECOVERED"
    consecutive_seat_hits: int = 0


class BehaviourEvidence(BaseModel):
    """Atomic evidence signal extracted from temporal feature analysis."""
    evidence_id: str
    camera_id: str
    timestamp_sec: float
    start_sec: float
    end_sec: float
    entity_id: int
    seat_id: Optional[str] = None
    evidence_type: str  # e.g., "LAP_REGION_ASSOCIATION", "TORSO_ROTATION", "EXCESSIVE_HEAD_TURN"
    target_type: Optional[str] = None  # "NEIGHBOUR_SEAT" | "OWN_MONITOR" | "BOARD" | "LAP"
    target_id: Optional[str] = None  # "A02", "MONITOR", "PHONE_IN_LAP"
    strength: float = 0.0  # 0.0 .. 1.0
    quality: float = 1.0   # 0.0 .. 1.0
    direction: Optional[str] = None  # "LEFT" | "RIGHT" | "DOWN"
    reason_codes: List[str] = Field(default_factory=list)
    source_provider: str = "unified"


class RiskSnapshot(BaseModel):
    """Per-entity risk state snapshot at a given timestamp."""
    camera_id: str
    timestamp_sec: float
    entity_id: int
    seat_id: Optional[str] = None
    raw_evidence_score: float
    counter_evidence_score: float
    persistence_weight: float
    quality_factor: float
    risk_score: float  # calibrated final risk [0.0 .. 1.0]
    state: RiskStateEnum = RiskStateEnum.NORMAL
    active_evidence_ids: List[str] = Field(default_factory=list)
    active_reasons: List[str] = Field(default_factory=list)


class ReviewEvent(BaseModel):
    """Prioritized forensic event emitted for human invigilator review."""
    event_id: str
    camera_id: str
    video_id: str
    entity_id: int
    seat_id: Optional[str] = None
    start_sec: float
    end_sec: float
    behaviour: BehaviourEnum
    state: RiskStateEnum = RiskStateEnum.REVIEW
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    peak_risk: float
    mean_quality: float
    evidence_signals: List[str] = Field(default_factory=list)
    counter_evidence_signals: List[str] = Field(default_factory=list)
    summary: str
    clip_path: Optional[str] = None
    contact_sheet_path: Optional[str] = None
    human_status: str = "UNREVIEWED"  # "UNREVIEWED" | "DISMISSED" | "CONFIRMED_RELEVANT"
    created_at: float = Field(default_factory=time.time)


class SeatProfile(BaseModel):
    """Configuration for an individual examination seat."""
    seat_id: str
    polygon: List[List[float]]  # Normalized polygon [[x, y], ...]
    monitor_polygon: Optional[List[List[float]]] = None
    keyboard_region: Optional[List[List[float]]] = None
    mouse_region: Optional[List[List[float]]] = None
    lap_region: Optional[List[List[float]]] = None
    neighbors: Dict[str, Optional[str]] = Field(default_factory=dict)  # {"left": "A01", "right": "A03"}


class CameraProfile(BaseModel):
    """Multi-camera calibration and zone mapping."""
    camera_id: str
    resolution: Tuple[int, int] = (1280, 720)
    seats: Dict[str, SeatProfile] = Field(default_factory=dict)
    zones: Dict[str, Optional[List[List[float]]]] = Field(default_factory=dict)
    blind_spots: List[List[List[float]]] = Field(default_factory=list)


class CameraHealth(BaseModel):
    """Real-time camera feed health metrics."""
    camera_id: str
    timestamp_sec: float
    is_online: bool = True
    fps: float = 0.0
    frame_drop_rate: float = 0.0
    brightness_score: float = 1.0  # 0 (too dark) .. 1 (good)
    blur_score: float = 1.0        # higher is sharper
    quality_status: QualityEnum = QualityEnum.GOOD


class BenchmarkMetrics(BaseModel):
    """Benchmark evaluation metrics against Ground Truth."""
    pipeline_name: str
    precision: float
    recall: float
    f1_score: float
    macro_f1: float
    temporal_iou: float
    false_positives_per_min: float
    mean_inference_latency_ms: float
    effective_fps: float
    peak_vram_gb: float
    coverage_rate: float
    unknown_rate: float

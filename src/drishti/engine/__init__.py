"""Unified Temporal Behavioral Engine for NETRA-4D / DrishtiAI.

This package provides the unified, model-agnostic, environment-adaptive
engine adhering to docs/ARCHITECTURE.md.
"""

from .observation import (
    HandLandmarksObservation,
    NormalizedKeypoint,
    ObservationQuality,
    RTMPoseAdapter,
    VisibilityState,
    VisionObservation,
    YOLO26Adapter,
    DetectedObjectObservation,
)
from .candidate_state import (
    CandidateIdentity,
    CandidateState,
    CandidateStateManager,
    IdentityLockStatus,
)
from .temporal_buffer import (
    CandidateTemporalBuffer,
    KeypointTrajectoryPoint,
    TemporalBufferManager,
)
from .smoothing import (
    DynamicMADFilter,
    KeypointSmoother,
    SmoothedSignal,
    TimeWeightedEMA,
)
from .features import (
    BiomechanicalFeatureExtractor,
    CandidateBiomechanicalFeatures,
    DetailedHandFeatures,
    GazeDirection,
    HandDynamicsFeatures,
    HeadFeatures,
    TorsoFeatures,
)
from .primitive_events import (
    PrimitiveEvent,
    PrimitiveEventDetector,
    PrimitiveEventType,
)
from .temporal_patterns import (
    AggregatedTemporalPattern,
    PersistenceDetector,
    RepetitionDetector,
    TemporalPatternType,
)
from .zones import (
    NormalizedPolygonZone,
    SemanticZoneEngine,
    SemanticZoneType,
    box_to_polygon,
)
from .hypotheses import (
    CompetingHypothesisEngine,
    EvidenceEvaluationResult,
    HypothesisDistribution,
    HypothesisKey,
)
from .state_machine import (
    CandidateBehaviouralState,
    CandidateStateMachine,
    StateTransitionRecord,
)
from .escalation import (
    EscalationGate,
    EscalationStage,
    SpecialistResolutionResult,
)

__all__ = [
    "HandLandmarksObservation",
    "NormalizedKeypoint",
    "ObservationQuality",
    "RTMPoseAdapter",
    "VisibilityState",
    "VisionObservation",
    "YOLO26Adapter",
    "DetectedObjectObservation",
    "CandidateIdentity",
    "CandidateState",
    "CandidateStateManager",
    "IdentityLockStatus",
    "CandidateTemporalBuffer",
    "KeypointTrajectoryPoint",
    "TemporalBufferManager",
    "DynamicMADFilter",
    "KeypointSmoother",
    "SmoothedSignal",
    "TimeWeightedEMA",
    "BiomechanicalFeatureExtractor",
    "CandidateBiomechanicalFeatures",
    "DetailedHandFeatures",
    "GazeDirection",
    "HandDynamicsFeatures",
    "HeadFeatures",
    "TorsoFeatures",
    "PrimitiveEvent",
    "PrimitiveEventDetector",
    "PrimitiveEventType",
    "AggregatedTemporalPattern",
    "PersistenceDetector",
    "RepetitionDetector",
    "TemporalPatternType",
    "NormalizedPolygonZone",
    "SemanticZoneEngine",
    "SemanticZoneType",
    "box_to_polygon",
    "CompetingHypothesisEngine",
    "EvidenceEvaluationResult",
    "HypothesisDistribution",
    "HypothesisKey",
    "CandidateBehaviouralState",
    "CandidateStateMachine",
    "StateTransitionRecord",
    "EscalationGate",
    "EscalationStage",
    "SpecialistResolutionResult",
]

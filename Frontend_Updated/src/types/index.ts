export type OperationalState = 'normal' | 'observed' | 'review' | 'escalated';

export type EventSeverity = 'observe' | 'review' | 'escalated';

export type BehaviorCategory = 
  | 'possible_object_interaction' 
  | 'repeated_sideward_attention' 
  | 'peer_interaction' 
  | 'unusual_hand_movement' 
  | 'desk_anomaly';

export type AIProvider = 'fusion' | 'yolo_pose' | 'rtmpose' | 'multihmr_3d';

export interface SeatNodeData {
  id: string; // e.g. "A12"
  candidateId: string; // e.g. "CAND-018"
  candidateName: string;
  cameraIds: string[]; // ["CAM-03"]
  gridPosition: { row: string; col: number };
  spatialPosition?: {
    x: number; // 0 to 100 percentage of room width
    y: number; // 0 to 100 percentage of room height
    facing: 'North' | 'South' | 'East' | 'West';
    zoneId?: string;
    zoneName?: string;
    neighbors?: string[];
  };
  state: OperationalState;
  telemetry: {
    trackContinuity: 'GOOD' | 'MODERATE' | 'POOR';
    headEvidence: 'GOOD' | 'MODERATE' | 'DEGRADED';
    handsEvidence: 'GOOD' | 'MODERATE' | 'OCCLUDED' | 'UNKNOWN';
    objectEvidence: { label: string; confidence: number } | null;
    lastEventSummary: string;
    lastEventTimeAgo: string;
  };
}

export interface RoomLandmark {
  id: string;
  name: string;
  type: 'podium' | 'board' | 'door' | 'bookshelf' | 'walkway';
  x: number;
  y: number;
  width: number;
  height: number;
  facing?: 'North' | 'South' | 'East' | 'West';
}

export interface CameraPlacement {
  id: string;
  name: string;
  x: number;
  y: number;
  directionDeg: number;
  fovAngle: number;
  coverageSeatIds: string[];
}

export interface SpatialLayoutConfig {
  layoutId: string;
  name: string;
  sourceType: 'default_grid' | 'blueprint_cad' | 'cctv_reverse_calibrated';
  roomDimensions: { widthMeters: number; lengthMeters: number };
  landmarks: RoomLandmark[];
  cameras: CameraPlacement[];
  terminals: SeatNodeData[];
}

export interface ReverseCalibrationResult {
  success: boolean;
  cameraId: string;
  sourceFrameUrl?: string;
  homographyGridPoints: [number, number][][];
  detectedTerminalCount: number;
  terminals: SeatNodeData[];
  inferredZones: { id: string; name: string; orientation: 'North' | 'South' | 'East' | 'West'; terminalIds: string[] }[];
}

export interface EvidenceComponent {
  id: string;
  stepNumber: string; // "01", "02", etc.
  title: string;
  category: 'object' | 'hand' | 'pose' | 'temporal' | 'contextual' | 'counter';
  confidence: number; // 0.0 - 1.0
  durationSec: number;
  description: string;
  isCounterEvidence?: boolean;
}

export interface CounterEvidenceFactor {
  id: string;
  text: string;
  type: 'suppression' | 'neutral' | 'escalation';
}

export interface TriagedAIEvent {
  id: string; // e.g. "EV-2038"
  timestamp: string; // e.g. "15:42:18"
  videoTimestampSec: number; // e.g. 14.8
  seatId: string;
  candidateId: string;
  candidateName: string;
  cameraId: string;
  cameraName: string;
  behavior: BehaviorCategory;
  behaviorDisplayTitle: string;
  severity: EventSeverity;
  aiConfidence: number; // Raw model score (e.g. 0.74)
  reviewPriority: 'ROUTINE' | 'ELEVATED' | 'CRITICAL'; // Context-weighted priority
  persistenceSec: number;
  videoClipUrl?: string;
  evidenceStack: EvidenceComponent[];
  counterEvidence: {
    factorsFound: string[];
    riskSuppressed: boolean;
    suppressionReason?: string;
  };
  context: {
    seatId: string;
    neighborSeatId: string;
    boardDirection: 'FRONT' | 'LEFT' | 'RIGHT';
    invigilatorProximity: 'NEARBY' | 'DISTANT' | 'ABSENT';
  };
  humanDecision?: {
    status: 'pending' | 'marked_normal' | 'needs_follow_up' | 'confirmed_violation' | 'escalated';
    reviewerId?: string;
    notes?: string;
    decidedAt?: string;
  };
}

export interface Keypoint2D {
  x: number;
  y: number;
  score: number;
  name: string;
}

export interface BoundingBox2D {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  confidence: number;
  trackId?: number;
  seatId?: string;
}

export interface FrameTelemetry {
  frameIndex: number;
  timestampSec: number;
  persons: {
    trackId: number;
    seatId: string;
    candidateName: string;
    box: BoundingBox2D;
    keypoints: Keypoint2D[];
    gazeRay?: { originX: number; originY: number; targetX: number; targetY: number };
    headYawDeg: number;
    wristDistanceToObject?: number;
  }[];
  objects: BoundingBox2D[];
  activeSignals: {
    seatId: string;
    label: string;
    score: number;
    severity: EventSeverity;
  }[];
}

export interface CameraFeedInfo {
  id: string;
  name: string;
  location: string;
  status: 'ONLINE' | 'DEGRADED' | 'OFFLINE';
  resolution: string;
  fps: number;
  trackingQuality: 'GOOD' | 'MODERATE' | 'POOR';
  poseQuality: 'GOOD' | 'MODERATE' | 'POOR';
  objectVisibility: 'GOOD' | 'MODERATE' | 'POOR';
  assignedSeats: string[];
}

export interface ProviderBenchmarkRow {
  providerName: string;
  role: string;
  videosTested: string;
  coveragePct: string;
  eventF1: string;
  fpPerMin: string;
  fps: string;
  vramGb: string;
  status: 'PRIMARY' | 'REFINEMENT' | 'SPECIALIST' | 'INTEGRATED';
}

export interface PerVideoBenchmark {
  videoId: string;
  filename: string;
  resolution: string;
  fps: number;
  durationSec: number;
  notes: string;
  yoloMetrics: { frames: number; tracks: number; fps: number; vram: string };
  rtmposeMetrics: { frames: number; tracks: number; fps: number; vram: string };
  multiHmrMetrics: { frames: number; tracks: number; fps: number; vram: string };
  predictedEvents: number;
  groundTruthEvents: string;
  overlap: string;
  falseAlerts: number;
}

export interface SystemHealthData {
  status: 'OPTIMAL' | 'DEGRADED' | 'WARNING';
  camerasOnline: number;
  camerasTotal: number;
  t4GpuUtilPct: number;
  t4GpuTempC: number;
  vramUsedGb: number;
  vramTotalGb: number;
  pipelineLatencyMs: number;
  processingQueueSize: number;
  lastEventTimeAgo: string;
  providers: {
    yoloPose: { status: 'HEALTHY' | 'DEGRADED' | 'OFFLINE'; fps: number; detail: string };
    rtmpose: { status: 'HEALTHY' | 'DEGRADED' | 'AVAILABLE' | 'OFFLINE'; fps: number; detail: string };
    objectDetector: { status: 'HEALTHY' | 'DEGRADED' | 'OFFLINE'; fps: number; detail: string };
    multiHmr: { status: 'AVAILABLE' | 'ON_DEMAND' | 'OFFLINE'; fps: number; detail: string };
  };
}

import { SeatNodeData, TriagedAIEvent, CameraFeedInfo, SystemHealthData, FrameTelemetry } from '../types';

export const SEATS_DATA: SeatNodeData[] = [
  // ROW A (NORTH WALL LAB DESK - 10 TERMINALS FACING NORTH / LAPTOPS AT TOP)
  { id: 'A01', candidateId: '1', candidateName: 'Candidate #1', cameraIds: ['CAM-01'], gridPosition: { row: 'A', col: 1 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Focusing on laptop screen', lastEventTimeAgo: '4m ago' } },
  { id: 'A02', candidateId: '2', candidateName: 'Candidate #2', cameraIds: ['CAM-01'], gridPosition: { row: 'A', col: 2 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Writing code on laptop', lastEventTimeAgo: '8m ago' } },
  { id: 'A03', candidateId: '3', candidateName: 'Candidate #3', cameraIds: ['CAM-01'], gridPosition: { row: 'A', col: 3 }, state: 'observed', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Wall timer glance (Suppressed)', lastEventTimeAgo: '3m ago' } },
  { id: 'A04', candidateId: '4', candidateName: 'Candidate #4', cameraIds: ['CAM-01'], gridPosition: { row: 'A', col: 4 }, state: 'observed', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'MODERATE', objectEvidence: { label: 'MOUSE', confidence: 0.88 }, lastEventSummary: 'Peripheral mouse verified (Hard-negative)', lastEventTimeAgo: '2m ago' } },
  { id: 'A05', candidateId: '5', candidateName: 'Candidate #5', cameraIds: ['CAM-01', 'CAM-02'], gridPosition: { row: 'A', col: 5 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Normal laptop posture', lastEventTimeAgo: '15m ago' } },
  { id: 'A06', candidateId: '6', candidateName: 'Candidate #6', cameraIds: ['CAM-02'], gridPosition: { row: 'A', col: 6 }, state: 'review', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'MODERATE', objectEvidence: { label: 'PHONE', confidence: 0.74 }, lastEventSummary: 'Phone candidate detected below desk (lap proximity)', lastEventTimeAgo: '1m ago' } },
  { id: 'A07', candidateId: '7', candidateName: 'Candidate #7', cameraIds: ['CAM-02'], gridPosition: { row: 'A', col: 7 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'OCCLUDED', objectEvidence: null, lastEventSummary: 'Laptop keyboard interaction', lastEventTimeAgo: '5m ago' } },
  { id: 'A08', candidateId: '8', candidateName: 'Candidate #8', cameraIds: ['CAM-02'], gridPosition: { row: 'A', col: 8 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Normal laptop posture', lastEventTimeAgo: '18m ago' } },
  { id: 'A09', candidateId: '9', candidateName: 'Candidate #9', cameraIds: ['CAM-02'], gridPosition: { row: 'A', col: 9 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Continuous screen focus', lastEventTimeAgo: '11m ago' } },
  { id: 'A10', candidateId: '10', candidateName: 'Candidate #10', cameraIds: ['CAM-02'], gridPosition: { row: 'A', col: 10 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Normal exam posture', lastEventTimeAgo: '20m ago' } },

  // ROW B (SOUTH WALL LAB DESK - 10 TERMINALS FACING SOUTH / LAPTOPS AT BOTTOM)
  { id: 'B01', candidateId: '11', candidateName: 'Candidate #11', cameraIds: ['CAM-03'], gridPosition: { row: 'B', col: 1 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Focusing on laptop screen', lastEventTimeAgo: '9m ago' } },
  { id: 'B02', candidateId: '12', candidateName: 'Candidate #12', cameraIds: ['CAM-03'], gridPosition: { row: 'B', col: 2 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Normal posture', lastEventTimeAgo: '14m ago' } },
  { id: 'B03', candidateId: '13', candidateName: 'Candidate #13', cameraIds: ['CAM-03'], gridPosition: { row: 'B', col: 3 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Continuous typing', lastEventTimeAgo: '22m ago' } },
  { id: 'B04', candidateId: '14', candidateName: 'Candidate #14', cameraIds: ['CAM-03'], gridPosition: { row: 'B', col: 4 }, state: 'review', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'MODERATE', objectEvidence: null, lastEventSummary: 'Repeated head rotation backward toward aisle (+38° yaw)', lastEventTimeAgo: '30s ago' } },
  { id: 'B05', candidateId: '15', candidateName: 'Candidate #15', cameraIds: ['CAM-03', 'CAM-04'], gridPosition: { row: 'B', col: 5 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Normal laptop posture', lastEventTimeAgo: '7m ago' } },
  { id: 'B06', candidateId: '16', candidateName: 'Candidate #16', cameraIds: ['CAM-04'], gridPosition: { row: 'B', col: 6 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Normal posture', lastEventTimeAgo: '16m ago' } },
  { id: 'B07', candidateId: '17', candidateName: 'Candidate #17', cameraIds: ['CAM-04'], gridPosition: { row: 'B', col: 7 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Active terminal session', lastEventTimeAgo: '13m ago' } },
  { id: 'B08', candidateId: '18', candidateName: 'Candidate #18', cameraIds: ['CAM-04'], gridPosition: { row: 'B', col: 8 }, state: 'escalated', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'MODERATE', objectEvidence: null, lastEventSummary: 'Direct peer interaction with B09 (Paper exchange suspected)', lastEventTimeAgo: '15s ago' } },
  { id: 'B09', candidateId: '19', candidateName: 'Candidate #19', cameraIds: ['CAM-04'], gridPosition: { row: 'B', col: 9 }, state: 'review', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'MODERATE', objectEvidence: null, lastEventSummary: 'Mutual orientation with B08', lastEventTimeAgo: '15s ago' } },
  { id: 'B10', candidateId: '20', candidateName: 'Candidate #20', cameraIds: ['CAM-04'], gridPosition: { row: 'B', col: 10 }, state: 'normal', telemetry: { trackContinuity: 'GOOD', headEvidence: 'GOOD', handsEvidence: 'GOOD', objectEvidence: null, lastEventSummary: 'Normal laptop focus', lastEventTimeAgo: '25m ago' } },
];

export const INITIAL_EVENTS: TriagedAIEvent[] = [
  {
    id: 'EV-2038',
    timestamp: '15:42:18',
    videoTimestampSec: 14.8,
    seatId: 'A06',
    candidateId: '6',
    candidateName: 'Candidate #6',
    cameraId: 'CAM-02',
    cameraName: 'Computer Lab North Camera',
    behavior: 'possible_object_interaction',
    behaviorDisplayTitle: 'Possible Object Interaction (Phone in Lap)',
    severity: 'review',
    aiConfidence: 0.74,
    reviewPriority: 'ELEVATED',
    persistenceSec: 6.2,
    evidenceStack: [
      { id: 'ev-1', stepNumber: '01', title: 'OBJECT EVIDENCE', category: 'object', confidence: 0.74, durationSec: 5.1, description: 'Handheld mobile device silhouette detected in lap beneath laptop desk.' },
      { id: 'ev-2', stepNumber: '02', title: 'HAND INTERACTION', category: 'hand', confidence: 0.81, durationSec: 6.2, description: 'Right hand removed from laptop keyboard, resting 0.18m inside lap.' },
      { id: 'ev-3', stepNumber: '03', title: 'BODY ORIENTATION', category: 'pose', confidence: 0.68, durationSec: 5.8, description: 'Head pitched -28° downward away from laptop screen.' },
      { id: 'ev-4', stepNumber: '04', title: 'TEMPORAL PERSISTENCE', category: 'temporal', confidence: 0.85, durationSec: 6.2, description: 'Sustained across 14 consecutive observation frames (12 FPS).' },
    ],
    counterEvidence: {
      factorsFound: [
        'Laptop screen active with IDE interface',
        'Invigilator currently at opposite end of central walkway',
      ],
      riskSuppressed: false,
    },
    context: {
      seatId: 'A06',
      neighborSeatId: 'A07',
      boardDirection: 'FRONT',
      invigilatorProximity: 'DISTANT',
    },
    humanDecision: {
      status: 'pending',
    }
  },
  {
    id: 'EV-2037',
    timestamp: '15:41:02',
    videoTimestampSec: 32.4,
    seatId: 'B04',
    candidateId: '14',
    candidateName: 'Candidate #14',
    cameraId: 'CAM-03',
    cameraName: 'Computer Lab South Camera',
    behavior: 'repeated_sideward_attention',
    behaviorDisplayTitle: 'Repeated Turn Back Toward Central Aisle (+38° Yaw)',
    severity: 'review',
    aiConfidence: 0.81,
    reviewPriority: 'ELEVATED',
    persistenceSec: 4.8,
    evidenceStack: [
      { id: 'ev-21', stepNumber: '01', title: 'HEAD DIRECTION ANOMALY', category: 'pose', confidence: 0.81, durationSec: 4.8, description: 'Head yaw exceeded threshold (+38°) turned away from laptop toward neighboring terminal B05.' },
      { id: 'ev-22', stepNumber: '02', title: 'REPETITIVE PATTERN', category: 'temporal', confidence: 0.89, durationSec: 120.0, description: '4 distinct glances detected in a 2-minute rolling temporal window.' },
      { id: 'ev-23', stepNumber: '03', title: 'TORSO CO-ALIGNMENT', category: 'pose', confidence: 0.62, durationSec: 4.1, description: 'Torso vector shifted +14° toward back center aisle.' },
    ],
    counterEvidence: {
      factorsFound: [
        'Invigilator not in line of sight',
        'Typing on laptop paused during glance interval',
      ],
      riskSuppressed: false,
    },
    context: {
      seatId: 'B04',
      neighborSeatId: 'B05',
      boardDirection: 'LEFT',
      invigilatorProximity: 'DISTANT',
    },
    humanDecision: {
      status: 'pending',
    }
  },
  {
    id: 'EV-2036',
    timestamp: '15:39:44',
    videoTimestampSec: 68.0,
    seatId: 'A03',
    candidateId: '3',
    candidateName: 'Candidate #3',
    cameraId: 'CAM-01',
    cameraName: 'Computer Lab North Camera',
    behavior: 'repeated_sideward_attention',
    behaviorDisplayTitle: 'Gaze Upward / Wall Timer Glance (Suppressed)',
    severity: 'observe',
    aiConfidence: 0.76,
    reviewPriority: 'ROUTINE',
    persistenceSec: 2.4,
    evidenceStack: [
      { id: 'ev-31', stepNumber: '01', title: 'HEAD PITCH ELEVATION', category: 'pose', confidence: 0.76, durationSec: 2.4, description: 'Gaze vector elevated +22° above laptop display.' },
      { id: 'ev-32', stepNumber: '02', title: 'COUNTER-EVIDENCE MATCH', category: 'counter', confidence: 0.94, durationSec: 2.4, description: 'Gaze cone directly intersects wall timer polygon above north desk.', isCounterEvidence: true },
    ],
    counterEvidence: {
      factorsFound: [
        'Official exam countdown timer located directly in line of sight (Vector match: 94%)',
        'Normal laptop screen focus resumed in 2.4s',
      ],
      riskSuppressed: true,
      suppressionReason: 'Gaze aligned with official wall timer polygon; operational risk suppressed.',
    },
    context: {
      seatId: 'A03',
      neighborSeatId: 'A04',
      boardDirection: 'FRONT',
      invigilatorProximity: 'NEARBY',
    },
    humanDecision: {
      status: 'marked_normal',
      reviewerId: 'Invigilator-02',
      notes: 'Checked exam instructions on board. Valid routine reference.',
      decidedAt: '15:40:12',
    }
  },
  {
    id: 'EV-2035',
    timestamp: '15:36:10',
    videoTimestampSec: 94.2,
    seatId: 'A04',
    candidateId: '4',
    candidateName: 'Candidate #4',
    cameraId: 'CAM-01',
    cameraName: 'Computer Lab North Camera',
    behavior: 'possible_object_interaction',
    behaviorDisplayTitle: 'Hard-Negative Object Discrimination (Mouse vs Phone)',
    severity: 'observe',
    aiConfidence: 0.44,
    reviewPriority: 'ROUTINE',
    persistenceSec: 3.1,
    evidenceStack: [
      { id: 'ev-41', stepNumber: '01', title: 'HAND PROXIMITY', category: 'hand', confidence: 0.69, durationSec: 3.1, description: 'Hand resting on desk perimeter near laptop peripheral.' },
      { id: 'ev-42', stepNumber: '02', title: 'OBJECT CLASSIFICATION', category: 'object', confidence: 0.88, durationSec: 3.1, description: 'Wired optical mouse confirmed by secondary detector (Phone prob: 0.12).' },
    ],
    counterEvidence: {
      factorsFound: [
        'Registered lab laptop peripheral (Mouse ID: M-04)',
        'Active screen cursor telemetry corroborated on laptop',
      ],
      riskSuppressed: true,
      suppressionReason: 'Hard-negative object discrimination confirmed wired optical mouse.',
    },
    context: {
      seatId: 'A04',
      neighborSeatId: 'A05',
      boardDirection: 'FRONT',
      invigilatorProximity: 'ABSENT',
    },
    humanDecision: {
      status: 'marked_normal',
      reviewerId: 'System Auto-Triage',
      notes: 'Mouse hard-negative filter verified.',
      decidedAt: '15:36:20',
    }
  },
  {
    id: 'EV-2034',
    timestamp: '15:31:55',
    videoTimestampSec: 118.0,
    seatId: 'B08',
    candidateId: '18',
    candidateName: 'Candidate #18',
    cameraId: 'CAM-04',
    cameraName: 'Computer Lab South Camera',
    behavior: 'peer_interaction',
    behaviorDisplayTitle: 'Coordinated Peer Interaction (Paper Exchange)',
    severity: 'escalated',
    aiConfidence: 0.88,
    reviewPriority: 'CRITICAL',
    persistenceSec: 8.4,
    evidenceStack: [
      { id: 'ev-51', stepNumber: '01', title: 'MUTUAL BODY ROTATION', category: 'pose', confidence: 0.88, durationSec: 8.4, description: 'Candidate #18 turned +44° while Candidate #19 turned -41° creating reciprocal interaction.' },
      { id: 'ev-52', stepNumber: '02', title: 'HAND PROXIMITY CROSSING', category: 'hand', confidence: 0.84, durationSec: 4.2, description: 'Right wrist reached across desk gap into neighboring terminal B09.' },
      { id: 'ev-53', stepNumber: '03', title: 'OBJECT DISPLACEMENT', category: 'object', confidence: 0.72, durationSec: 3.0, description: 'Small white sheet tracked across laptops.' },
    ],
    counterEvidence: {
      factorsFound: [
        'No invigilator present in central aisle (Distance: 12.5m)',
        'Movement paused upon invigilator footsteps in walkway',
      ],
      riskSuppressed: false,
    },
    context: {
      seatId: 'B08',
      neighborSeatId: 'B09',
      boardDirection: 'FRONT',
      invigilatorProximity: 'ABSENT',
    },
    humanDecision: {
      status: 'confirmed_violation',
      reviewerId: 'Chief Invigilator Dr. K. Rao',
      notes: 'Confirmed unauthorized paper passing between Candidate #18 and Candidate #19. Recorded in Case #C-024.',
      decidedAt: '15:34:00',
    }
  },
];

export const CAMERA_FEEDS: CameraFeedInfo[] = [
  { id: 'CAM-01', name: 'CAM-01 (Lab North Wall Left)', location: 'North Ceiling Left', status: 'ONLINE', resolution: '1920×1080', fps: 18.2, trackingQuality: 'GOOD', poseQuality: 'GOOD', objectVisibility: 'GOOD', assignedSeats: ['A01', 'A02', 'A03', 'A04', 'A05'] },
  { id: 'CAM-02', name: 'CAM-02 (Lab North Wall Right)', location: 'North Ceiling Right', status: 'ONLINE', resolution: '1920×1080', fps: 18.0, trackingQuality: 'GOOD', poseQuality: 'GOOD', objectVisibility: 'GOOD', assignedSeats: ['A06', 'A07', 'A08', 'A09', 'A10'] },
  { id: 'CAM-03', name: 'CAM-03 (Lab South Wall Left)', location: 'South Ceiling Left', status: 'ONLINE', resolution: '1920×1080', fps: 17.8, trackingQuality: 'GOOD', poseQuality: 'GOOD', objectVisibility: 'GOOD', assignedSeats: ['B01', 'B02', 'B03', 'B04', 'B05'] },
  { id: 'CAM-04', name: 'CAM-04 (Lab South Wall Right)', location: 'South Ceiling Right', status: 'ONLINE', resolution: '1920×1080', fps: 18.4, trackingQuality: 'GOOD', poseQuality: 'GOOD', objectVisibility: 'GOOD', assignedSeats: ['B06', 'B07', 'B08', 'B09', 'B10'] },
];

export const SYSTEM_HEALTH: SystemHealthData = {
  status: 'OPTIMAL',
  camerasOnline: 4,
  camerasTotal: 4,
  t4GpuUtilPct: 62,
  t4GpuTempC: 54,
  vramUsedGb: 5.9,
  vramTotalGb: 16.0,
  pipelineLatencyMs: 32.4,
  processingQueueSize: 1,
  lastEventTimeAgo: '15s ago',
  providers: {
    yoloPose: { status: 'HEALTHY', fps: 18.3, detail: 'YOLOv8-Pose Primary Ingestion' },
    rtmpose: { status: 'HEALTHY', fps: 8.6, detail: 'RTMPose Refinement Active' },
    objectDetector: { status: 'HEALTHY', fps: 14.2, detail: 'Mobile & Device Detector' },
    multiHmr: { status: 'AVAILABLE', fps: 0, detail: 'Multi-HMR 3D Joint Mesh' },
  },
};

export const MOCK_TELEMETRY_FRAMES: FrameTelemetry[] = [
  {
    frameIndex: 1042,
    timestampSec: 14.8,
    persons: [
      {
        trackId: 6,
        seatId: 'A06',
        candidateName: 'Candidate #6',
        box: { x: 380, y: 160, w: 260, h: 380, label: 'PERSON', confidence: 0.94, trackId: 6, seatId: 'A06' },
        keypoints: [
          { name: 'nose', x: 490, y: 220, score: 0.94 },
          { name: 'left_eye', x: 480, y: 215, score: 0.92 },
          { name: 'right_eye', x: 500, y: 215, score: 0.91 },
          { name: 'left_ear', x: 465, y: 225, score: 0.88 },
          { name: 'right_ear', x: 515, y: 225, score: 0.87 },
          { name: 'left_shoulder', x: 440, y: 290, score: 0.96 },
          { name: 'right_shoulder', x: 540, y: 290, score: 0.95 },
          { name: 'left_elbow', x: 415, y: 380, score: 0.89 },
          { name: 'right_elbow', x: 565, y: 375, score: 0.88 },
          { name: 'left_wrist', x: 435, y: 460, score: 0.92 },
          { name: 'right_wrist', x: 545, y: 450, score: 0.90 },
          { name: 'left_hip', x: 455, y: 480, score: 0.93 },
          { name: 'right_hip', x: 525, y: 480, score: 0.91 },
        ],
        headYawDeg: -28,
        gazeRay: { originX: 490, originY: 220, targetX: 430, targetY: 460 },
      },
      {
        trackId: 7,
        seatId: 'A07',
        candidateName: 'Candidate #7',
        box: { x: 740, y: 160, w: 250, h: 380, label: 'PERSON', confidence: 0.96, trackId: 7, seatId: 'A07' },
        keypoints: [
          { name: 'nose', x: 865, y: 220, score: 0.95 },
          { name: 'left_shoulder', x: 815, y: 290, score: 0.97 },
          { name: 'right_shoulder', x: 915, y: 290, score: 0.96 },
          { name: 'left_elbow', x: 790, y: 380, score: 0.90 },
          { name: 'right_elbow', x: 940, y: 380, score: 0.91 },
          { name: 'left_wrist', x: 810, y: 460, score: 0.92 },
          { name: 'right_wrist', x: 920, y: 460, score: 0.93 },
          { name: 'left_hip', x: 830, y: 480, score: 0.94 },
          { name: 'right_hip', x: 900, y: 480, score: 0.92 },
        ],
        headYawDeg: 2,
      }
    ],
    objects: [
      {
        label: 'PHONE CANDIDATE (0.74)',
        confidence: 0.74,
        x: 480,
        y: 435,
        w: 48,
        h: 70,
        seatId: 'A06',
      }
    ],
    activeSignals: [
      { seatId: 'A06', label: 'Possible Object Interaction', score: 0.74, severity: 'review' }
    ]
  }
];

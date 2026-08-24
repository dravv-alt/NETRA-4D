export interface ValidatedVideoRun {
  assetId: string;
  filename: string;
  resolution: string;
  sourceFps: number;
  durationSec: number;
  frames: number;
  decodedFrames: number;
  decodeFailures: number;
  personObservations: number;
  objectObservations: number;
  privacyMasks: number;
  gazeProxyObservations: number;
  eventCandidateTrackIds: number;
  eventTypes: string[];
  reviewEvents: number;
  evidenceStills: number;
  temporalDurationSec: number;
  maxEventDurationSec: number;
  inferenceRuntimeSec: number;
  calibration: string;
}

export const VALIDATED_RUN = {
  runId: 'netra4d_final_20260823',
  videos: 5,
  frames: 15346,
  sourceDurationSec: 856.75,
  reviewEvents: 29,
  evidenceStills: 9,
  eventCandidateTrackIds: 19,
  privacyMasks: 39481,
  gazeProxyObservations: 12337,
  phoneStillsAtStrictThreshold: 0,
  device: 'Tesla P100-PCIE-16GB',
  torch: '2.7.1+cu118',
  python: '3.12.13',
};

export const VALIDATED_VIDEO_RUNS: ValidatedVideoRun[] = [
  {
    assetId: 'VIDEO_01',
    filename: 'Seat No. 12 was seen taking a piece of paper from the desk.mkv',
    resolution: '1280×720', sourceFps: 25, durationSec: 88.4,
    frames: 2210, decodedFrames: 2198, decodeFailures: 12,
    personObservations: 15506, objectObservations: 27549, privacyMasks: 15506, gazeProxyObservations: 3935,
    eventCandidateTrackIds: 10,
    eventTypes: ['Low-hand / possible-chit review', 'Neighbour-screen overwatch review', 'Other-screen look review', 'Phone retrieve/use/conceal review'],
    reviewEvents: 13, evidenceStills: 4, temporalDurationSec: 13.84, maxEventDurationSec: 2.76,
    inferenceRuntimeSec: 247.9, calibration: 'AUTOMATIC_MAP_ACTIVE',
  },
  {
    assetId: 'VIDEO_02', filename: '04.CCTV Candidate Talking.mkv',
    resolution: '640×480', sourceFps: 8, durationSec: 143.12,
    frames: 1145, decodedFrames: 1145, decodeFailures: 0,
    personObservations: 5147, objectObservations: 6338, privacyMasks: 5147, gazeProxyObservations: 1560,
    eventCandidateTrackIds: 3,
    eventTypes: ['Low-hand / possible-chit review', 'Neighbour-screen overwatch review', 'Phone retrieve/use/conceal review'],
    reviewEvents: 7, evidenceStills: 3, temporalDurationSec: 9, maxEventDurationSec: 2.75,
    inferenceRuntimeSec: 90.9, calibration: 'AUTOMATIC_MAP_ACTIVE',
  },
  {
    assetId: 'VIDEO_03', filename: '01.Candidate was found using a mobile phone in the examination hall..mkv',
    resolution: '1280×720', sourceFps: 25, durationSec: 131.28,
    frames: 3282, decodedFrames: 3282, decodeFailures: 0,
    personObservations: 4674, objectObservations: 21801, privacyMasks: 4674, gazeProxyObservations: 725,
    eventCandidateTrackIds: 5,
    eventTypes: ['Low-hand / possible-chit review', 'Phone retrieve/use/conceal review'],
    reviewEvents: 6, evidenceStills: 1, temporalDurationSec: 8.32, maxEventDurationSec: 3.96,
    inferenceRuntimeSec: 199.8, calibration: 'AUTOMATIC_MAP_ACTIVE',
  },
  {
    assetId: 'VIDEO_04', filename: '02.Candidate was found using a mobile phone in the examination hall..mkv',
    resolution: '1280×720', sourceFps: 25, durationSec: 212.16,
    frames: 5304, decodedFrames: 5304, decodeFailures: 0,
    personObservations: 10231, objectObservations: 30006, privacyMasks: 10231, gazeProxyObservations: 3465,
    eventCandidateTrackIds: 0, eventTypes: [], reviewEvents: 0, evidenceStills: 0,
    temporalDurationSec: 0, maxEventDurationSec: 0, inferenceRuntimeSec: 339.5, calibration: 'AUTOMATIC_MAP_ACTIVE',
  },
  {
    assetId: 'VIDEO_05', filename: '03.CCTV Mobile Usage.mkv',
    resolution: '1280×720', sourceFps: 12.083, durationSec: 281.79,
    frames: 3405, decodedFrames: 3405, decodeFailures: 0,
    personObservations: 3923, objectObservations: 40404, privacyMasks: 3923, gazeProxyObservations: 2652,
    eventCandidateTrackIds: 1,
    eventTypes: ['Low-hand / possible-chit review', 'Phone retrieve/use/conceal review'],
    reviewEvents: 3, evidenceStills: 1, temporalDurationSec: 146.48, maxEventDurationSec: 143.75,
    inferenceRuntimeSec: 208, calibration: 'AUTOMATIC_MAP_ACTIVE',
  },
];

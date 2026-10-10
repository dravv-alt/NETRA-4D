# Video 3 head-direction diagnostic

## Current baseline

The frozen run is `video03_review_verified_20261003`. Its stored direction
labels were evaluated on 1,853 IoU-matched Student samples. Accuracy is 5.40%,
macro precision 30.02%, macro recall 27.91%, and macro F1 4.54%. This is a
head-direction proxy evaluation, not eye-gaze or gaze-target evaluation.

The original baseline artifact was not edited.

## Actual implementation traced

`YOLO26Adapter.COCO_KEYPOINT_NAMES` uses standard COCO-17 order: nose (0),
left eye (1), right eye (2), left ear (3), right ear (4), then shoulders and
the remaining body joints. The head proxy uses only nose, left eye, and right
eye. Shoulders and ears are not used for direction.

Keypoints are normalized by image width and height. Image x increases right;
image y increases downward. There is no x/y swap in the adapter or feature
extractor. `raw_yaw = (nose_x - eye_mid_x)/(horizontal_eye_span/2)` and
`raw_pitch = (nose_y - eye_mid_y)/horizontal_eye_span`; facial points are
smoothed with tau=0.12 s and yaw has a second EMA with tau=0.15 s.

Classification is ordered: RIGHT if yaw > 0.40; LEFT if yaw < -0.40; otherwise
DOWN if pitch > 0.85; UP if pitch < -0.40; otherwise FORWARD. Thus horizontal
classification takes precedence over vertical classification. Positive yaw is
defined as image-right, not a calibrated subject-facing direction.

## Convention transforms

All metrics, including each per-class metric and complete confusion matrix, are
in `outputs/analysis/video03_head_direction_diagnostic_20261004_v2/transformation_metrics.json`.
Rows are CVAT truth and columns are transformed predictions.

| Transform | Accuracy | Macro F1 | Interpretation |
|---|---:|---:|---|
| A baseline | 5.40% | 4.54% | Stored run |
| B left/right inversion | 4.32% | 3.16% | Does not fix it |
| C up/down inversion | 4.26% | 3.44% | Does not fix it |
| D both inversions | 3.18% | 2.06% | Does not fix it |
| E yaw sign inversion | 4.32% | 3.16% | Same mapping as B |
| F pitch sign inversion | 4.26% | 3.44% | Same mapping as C |
| G representable 180-degree yaw | 4.32% | 3.16% | Same mapping as B; front/back cannot be represented by this label set |
| H hypothetical x/y semantic swap | 42.15% | 16.27% | Diagnostic only; not a valid production correction |

H is informative because 722 of 743 CVAT DOWN labels were stored as RIGHT.
It does not prove axes are swapped in code: code inspection shows they are not.
It instead flags a likely label/reference-direction semantic mismatch and an
inadequate 2D proxy.

## Threshold sensitivity

`threshold_sensitivity_video03_development_calibration.json` contains complete
metrics and confusion matrices for each sweep. It is explicitly **VIDEO03
DEVELOPMENT CALIBRATION ONLY**, based on diagnostic replay intermediates, and
must not be used as independent performance or production threshold selection.

| yaw / down / up thresholds | Accuracy | Macro F1 |
|---|---:|---:|
| 0.20 / 0.60 / -0.20 | 3.40% | 2.35% |
| 0.40 / 0.85 / -0.40 (current) | 5.34% | 4.12% |
| 0.60 / 1.10 / -0.60 | 5.88% | 4.65% |
| 0.80 / 1.30 / -0.80 | 6.80% | 5.35% |

## Raw values and visual inspection

The 1,853 raw/intermediate records are in
`outputs/analysis/video03_head_direction_intermediates_20261004/head_direction_intermediates.jsonl`.
They include raw COCO keypoints, the derived vector, yaw, pitch, stored
prediction, CVAT label, and confidence. The file also records that a simplified
state replay differed from stored direction in 724 samples; therefore its
derived continuous values are diagnostic, not an authoritative historical
trace.

47 original-frame visualizations are in
`outputs/analysis/video03_head_direction_diagnostic_20261004_v2/`. They cover
CVAT FRONT predicted RIGHT/DOWN, CVAT DOWN predicted RIGHT, and CVAT LEFT/RIGHT.
They show the person pose, nose/eye geometry vector, stored direction, CVAT
direction, and proxy values.

## Evidence-based failure classification

- `COORDINATE_CONVENTION_ERROR`: not established in x/y code; semantic/reference mismatch is plausible.
- `KEYPOINT_ORDER_ERROR`: no evidence; implementation follows the intended COCO ordering.
- `AXIS_SIGN_ERROR`: simple sign inversions worsen results.
- `REFERENCE_DIRECTION_ERROR`: plausible; image-right is treated as semantic RIGHT without camera/subject calibration.
- `THRESHOLD_ERROR`: contributory at most; broad sweeps remain unusable.
- `POSE_ESTIMATION_ERROR`: strongly plausible; small eye spans and low eye confidence make the ratio unstable.
- `HEAD_VS_BODY_ORIENTATION_MISMATCH`: plausible; this is nose/eye 2D geometry, not head pose or body orientation.
- `GENUINE_PROXY_FAILURE`: strongly supported by the poor baseline and unsuccessful valid inversions.
- `INSUFFICIENT_LABEL_SUPPORT`: confirmed for RIGHT and UP (six examples each), so their class metrics are not stable.

## Proposed fix

Do not retrain yet and do not relabel the baseline automatically. First preserve
facial intermediate values during a controlled rerun, define whether CVAT means
image-facing or subject-facing directions, and inspect the saved representatives.
Then replace this proxy only if an explicitly calibrated pose/head-orientation
method demonstrates improvement on a separate held-out session.

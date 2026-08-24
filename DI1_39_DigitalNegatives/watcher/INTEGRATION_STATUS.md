# DigitalNegatives Watcher integration status

This directory is merged as an **experimental analysis subsystem**. It is not
the authority for the verified annotated videos already produced by NETRA-4D.

## Accepted source functionality

- Real Ultralytics YOLO object detection.
- Real Ultralytics YOLO pose inference.
- Frame-quality measurement and optional enhancement.
- Temporal keypoint smoothing.
- Person tracking and camera-specific seat association.
- Explainable heuristic feature, fusion, rule, risk, alert, and SQLite layers.
- File, webcam, HTTP, and RTSP source abstractions.

## Cleanup performed during merge

- Generated SQLite databases, model weights, bytecode, historical logs, debug
  frames, and snapshots are ignored. They remain local but are not accepted as
  new verified evidence.
- Chair detections can no longer become human tracks.
- A generic `book` is no longer treated as a paper chit or suspicious object.
- Object-hand fusion defaults to exact raw `cell phone` detections only.
- The phone rule now requires confidence `>= 0.70`.
- The fixed seat polygons are explicitly labelled as calibration for the
  incoming sample camera only.

## Important boundary

The Watcher risk score is a transparent heuristic attention-ranking signal. It
is **not** a trained malpractice classifier, accuracy result, or accusation.
Its inherited databases and alert screenshots are historical development
artifacts and have not been reproduced against the verified five-video run.

Before this subsystem can become part of the primary production path, each
camera needs reviewed seat calibration and the temporal rules need annotated
ground truth plus precision/recall evaluation.


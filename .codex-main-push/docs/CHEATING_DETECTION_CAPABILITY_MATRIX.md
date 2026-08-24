# Cheating-detection capability matrix

Last audited: 2026-08-22

This file separates actual model evidence from behavioral review proxies. A
proxy may select a short review window, but it must never be rendered or
reported as a confirmed cheating incident.

| Requested capability | Current real evidence | Status | Required next implementation |
|---|---|---|---|
| Phone present | Exact YOLO COCO `cell phone` box and confidence | Working; the verified five-video run had zero boxes at the 0.70 evidence threshold | Add lower-threshold temporal proposals, then require persistence and hand/person association before review |
| Phone use | Phone box associated to a visible candidate or wrist | Experimental proxy | Validate thresholds on labelled phone-use clips and retain raw evidence for every decision |
| Chit/paper use | Wrist below desk/lap | Proxy only, not a chit detector | Train or integrate a paper/chit detector using reviewed bounding-box labels; combine it with hand association and temporal persistence |
| Two-candidate interaction | Independent pose/head/hand features per track | Not implemented as a pairwise event | Require two stable tracks, seat adjacency, mutual head direction, distance, and temporal overlap; label result `interaction review` until validated |
| Looking at another screen | 2-D head-direction gaze proxy | Not target-aware | Calibrate screen/desk regions per camera and intersect a confidence-bearing gaze ray with another candidate's screen region over time |
| 3-D corroboration | Official Multi-HMR wrapper and supplied SMPL-X asset | Integration code only; no completed 3-D output | Run only on flagged windows at low sampling rate and preserve official checkpoint/model provenance |

## Required adaptive cascade

```text
original video
  -> YOLO26n object + YOLO26 pose + ByteTrack at 4-8 FPS
  -> cheap temporal proposals per stable candidate ID
  -> event windows with pre/post-roll and raw JSONL evidence
  -> Multi-HMR + SMPL-X only on ambiguous flagged windows at 0.5-1 FPS
  -> privacy blur during rendering only
  -> candidate ID, gaze proxy, evidence labels, event clip and key frame
```

Inference always runs on the original frame. Face blur is a render-only step,
and gaze/candidate annotations are drawn after the blur. Long videos should be
encoded as H.264 annotated videos plus short event clips rather than repeatedly
running heavy 3-D inference over every frame.

## Calibration rule

`DI1_39_DigitalNegatives/watcher/configs/seats.yaml` contains fixed polygons
for one 1280x720 sample camera. Those coordinates must not be reused for the
five dataset videos. Each camera needs its own visually verified seat, desk,
and screen regions before interaction or screen-overwatch results can be
claimed.

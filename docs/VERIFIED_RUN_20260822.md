# Verified real-model run: 20260822T114113Z

This is a measured CPU verification run over all five user-supplied Drive
videos. It is not the pending Colab T4 run and makes no GPU claim.

- Result: `outputs/verified/20260822T114113Z/run.json`
- Result SHA-256: `19ff4c82af521720d71bbe026034645c4878f9a6d85c86a658ae5ea2fa6792fc`
- Runtime: Python 3.12.6, PyTorch 2.13.0+cpu, `cuda_available=false`
- Pose model: `yolo26n-pose.pt`, SHA-256
  `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`
- Object model: `yolo11n.pt`, SHA-256
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`

| Video | Sample | People | Objects | Combined CPU inference |
|---|---:|---:|---:|---:|
| VIDEO_01 | frame 1105 at 44.200 s | 7 | 23 | 4186.6 ms (includes warm-up) |
| VIDEO_02 | frame 572 at 71.500 s | 6 | 6 | 144.6 ms |
| VIDEO_03 | frame 1641 at 65.640 s | 3 | 6 | 96.4 ms |
| VIDEO_04 | frame 2652 at 106.080 s | 2 | 6 | 262.4 ms |
| VIDEO_05 | frame 1702 at 140.855 s | 1 | 18 | 117.2 ms |

Totals: five videos, five sampled frames, 19 person detections, and 59
non-person object detections. `VIDEO_05` included one raw COCO `cell phone`
detection at confidence 0.269. That is a model candidate, not a confirmed phone
and not a malpractice verdict.

Each source file was downloaded in chunks into an OS temporary directory,
SHA-256 hashed, decoded, processed, and deleted before the next source. The
repository retained only JSON provenance and annotated preview images.

# Video 3 adaptive preprocessing ablation

## Decision

> **Adaptive preprocessing is beneficial as a conditional phone-detection specialist, but is not suitable as a universal preprocessing stage due to degraded person recall and increased runtime.**

## Controlled comparison

Both runs processed the same 2,200 frames (`0–2199`) of Video 3 at the same
IoU threshold (`0.5`), with the same registry models and phone confidence.
The frozen raw-frame baseline remains unchanged.

| Metric | Raw baseline | Adaptive preprocessing | Difference |
| --- | ---: | ---: | ---: |
| Phone recall | 94.60% (473/500) | 95.80% (479/500) | +1.20 pp |
| Phone mean matched IoU | 0.9138 | 0.9037 | -0.0101 |
| Person recall | 70.15% (2950/4205) | 67.85% (2853/4205) | -2.31 pp |
| Runtime | 752.70 s | 897.36 s | +19.2% |
| Processed FPS | 2.92 | 2.45 | -16.1% |

Runs:

- Raw baseline: `outputs/unified_pipeline/video03_review_verified_20261003`
- Preprocessed comparison: `outputs/unified_pipeline/video03_review_preprocessed_fast_gpu_20261004_run2`
- Comparison configuration: `configs/preprocessing_video03_comparison_fast.yaml`

## What was active

The comparison used adaptive quality assessment and the low-latency bilateral
denoiser. All 2,200 frames were enhanced; bilateral denoising was applied on
every frame. CLAHE, gamma correction, sharpening, and white balance were not
triggered. Enhancement was applied only to inference frames; privacy masking
and evidence rendering continued to use the original decoded frames.

## Scope and limitations

This is a single-video development diagnostic, not a held-out generalization
result. Precision, F1, mAP, and false-positive-per-hour are not reported:
unannotated phone proposals remain `UNKNOWN`, not negative examples. The
result does not establish behavioural-event accuracy or a cheating verdict.


# Adaptive specialist escalation

The baseline remains continuous. This controller only decides when existing
specialists may run; it does not emit a behavioural label or a cheating
probability.

## Policies

Policies live in `configs/specialist_escalation.yaml` and are recorded in each
experimental run manifest.

- `baseline_only`: no specialist execution.
- `specialists_every_frame`: explicit cost/reference policy.
- `conditional_escalation`: sustained quality degradation only.

## Conditional state machine

```text
BASELINE
  -- quality >= trigger for N consecutive frames --> SPECIALIST
SPECIALIST
  -- window complete or sustained recovery --> COOLDOWN
COOLDOWN
  -- cooldown elapsed --> BASELINE
```

`min_specialist_interval_frames` bounds specialist sampling within the active
window. `recovery_threshold`, `min_recovery_frames`, and `cooldown_frames`
provide hysteresis, preventing per-frame oscillation.

The current quality signal is the maximum of existing baseline uncertainty and
one minus baseline person-detection confidence. It is system-quality evidence,
not a behavioural score.

## Artifacts

`specialist_escalation.jsonl` records every observed track/frame, including:

- baseline quality evidence;
- consecutive low-quality and recovery counts;
- controller state and transitions;
- specialist types invoked; and
- reproducible policy provenance.

When enabled, RTMPose and hand observations are attached under
`specialist_evidence` in `temporal_evidence.jsonl`, allowing the existing
experimental evidence-fusion output to consume them without changing its
schema or semantics.

## Isolated LSTM boundary

`drishti.experimental.temporal_lstm` is offline-only. It converts the existing
temporal-evidence JSONL into 32/48/64-frame contiguous per-track windows and
predicts the next **observable evidence vector**. It has no behavioural target,
is not trained by default, and is never imported by the canonical pipeline.
The next valid comparison is persistence baseline versus LSTM on a held-out
video/session for next-observation and short-gap reconstruction metrics.

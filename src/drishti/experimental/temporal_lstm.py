"""Isolated LSTM boundary for observable temporal-evidence experiments.

This module is intentionally offline-only. It models the next observation in a
per-track evidence sequence; it has no cheating, suspicion, or behavioural
class target and is not imported by the canonical review pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "head_left", "head_right", "head_forward", "head_unknown",
    "hand_observed", "phone_observed", "neighbour_observed",
    "object_relation_count", "occlusion_unknown", "specialist_active",
    "rtmpose_keypoints_observed", "hands_specialist_observed",
)


@dataclass(frozen=True)
class ObservableEvidenceSample:
    frame_index: int
    timestamp_sec: float
    track_id: int | str
    candidate_id: str
    values: np.ndarray


def evidence_sample_from_record(record: dict[str, Any]) -> ObservableEvidenceSample:
    """Convert existing ``temporal_evidence.jsonl`` schema into fixed features."""
    evidence = record.get("evidence", {})
    direction = evidence.get("pose_evidence", {}).get("head_direction_proxy", "UNKNOWN")
    hand = evidence.get("hand_evidence", {})
    phone = evidence.get("phone_evidence", {})
    neighbour = evidence.get("neighbour_evidence", {})
    obj = evidence.get("object_evidence", {})
    specialist = evidence.get("specialist_evidence", {})
    values = np.asarray([
        float(direction == "LEFT"), float(direction == "RIGHT"),
        float(direction in ("FRONT", "FORWARD", "FORWARD_OR_UNKNOWN")),
        float(direction not in ("LEFT", "RIGHT", "FRONT", "FORWARD", "FORWARD_OR_UNKNOWN")),
        float(bool(hand.get("observed"))), float(bool(phone.get("associated_phone"))),
        float(bool(neighbour.get("near_neighbour"))), min(1.0, float(obj.get("relations", 0)) / 4.0),
        float(evidence.get("occlusion", {}).get("status", "UNKNOWN") == "UNKNOWN"),
        float(bool(specialist.get("escalation", {}).get("invoke_specialist"))),
        float(bool(specialist.get("rtmpose_keypoints_observed"))),
        float(bool(specialist.get("hands_observed"))),
    ], dtype=np.float32)
    return ObservableEvidenceSample(
        frame_index=int(record["frame_index"]), timestamp_sec=float(record["timestamp_sec"]),
        track_id=record["track_id"], candidate_id=str(record["candidate_id"]), values=values,
    )


def load_evidence_samples(path: str | Path) -> list[ObservableEvidenceSample]:
    with Path(path).open(encoding="utf-8") as handle:
        return [evidence_sample_from_record(json.loads(line)) for line in handle if line.strip()]


def build_next_observation_windows(samples: Iterable[ObservableEvidenceSample], window_size: int = 32):
    """Create only contiguous per-track windows; gaps are never silently bridged."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    grouped: dict[tuple[str, str], list[ObservableEvidenceSample]] = {}
    for sample in samples:
        grouped.setdefault((str(sample.track_id), sample.candidate_id), []).append(sample)
    windows = []
    for key, rows in grouped.items():
        rows.sort(key=lambda item: item.frame_index)
        for end in range(window_size, len(rows)):
            context = rows[end - window_size:end]
            target = rows[end]
            if any(right.frame_index != left.frame_index + 1 for left, right in zip(context, context[1:])):
                continue
            if target.frame_index != context[-1].frame_index + 1:
                continue
            windows.append((key, np.stack([item.values for item in context]), target.values, target.frame_index))
    return windows


class TemporalEvidenceLSTM(nn.Module):
    """Predicts the next observable evidence vector, not a behaviour label."""

    def __init__(self, input_size: int = len(FEATURE_NAMES), hidden_size: int = 32, layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden_size, input_size), nn.Sigmoid())

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        return self.output(hidden[-1])


def smoke_evidence_data_flow(path: str | Path, window_size: int = 32, device: str = "cpu") -> dict[str, Any]:
    """Validate existing evidence can flow through an untrained LSTM safely."""
    samples = load_evidence_samples(path)
    windows = build_next_observation_windows(samples, window_size=window_size)
    result = {
        "status": "PENDING_TRAINING_AND_HELDOUT_VALIDATION",
        "semantics": "next observable evidence prediction only; not behavioural classification",
        "feature_names": list(FEATURE_NAMES), "window_size": window_size,
        "samples": len(samples), "contiguous_windows": len(windows), "device": device,
    }
    if not windows:
        result["data_flow_smoke"] = "NOT_RUN_NO_CONTIGUOUS_WINDOWS"
        return result
    model = TemporalEvidenceLSTM().to(device).eval()
    _, context, _, target_frame = windows[0]
    with torch.inference_mode():
        prediction = model(torch.from_numpy(context).unsqueeze(0).to(device)).cpu().numpy()[0]
    result.update({
        "data_flow_smoke": "PASS_UNTRAINED_FORWARD_ONLY",
        "example_target_frame": target_frame,
        "prediction_shape": list(prediction.shape),
        "training_performed": False,
    })
    return result

"""Sustained, evidence-only specialist escalation.

The controller schedules expensive specialists from observable system-quality
signals. It does not assign behavioural labels or alter the baseline inference
path. A single low-quality frame can never cause an escalation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import yaml


class SpecialistPolicyMode(str, Enum):
    BASELINE_ONLY = "baseline_only"
    EVERY_FRAME = "specialists_every_frame"
    CONDITIONAL = "conditional_escalation"


class EscalationState(str, Enum):
    BASELINE = "BASELINE"
    SPECIALIST = "SPECIALIST"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class SpecialistEscalationConfig:
    mode: SpecialistPolicyMode = SpecialistPolicyMode.CONDITIONAL
    trigger_threshold: float = 0.55
    recovery_threshold: float = 0.35
    min_consecutive_low_quality_frames: int = 3
    specialist_window_frames: int = 12
    cooldown_frames: int = 20
    min_specialist_interval_frames: int = 2
    min_recovery_frames: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery_threshold <= self.trigger_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= recovery <= trigger <= 1")
        for field_name in (
            "min_consecutive_low_quality_frames", "specialist_window_frames",
            "cooldown_frames", "min_specialist_interval_frames", "min_recovery_frames",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(frozen=True)
class EscalationDecision:
    frame_index: int
    timestamp_sec: float
    state: EscalationState
    invoke_specialist: bool
    quality_signal: float
    consecutive_low_quality_frames: int
    recovery_frames: int
    cooldown_remaining_frames: int
    transition: str | None
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


class AdaptiveSpecialistEscalationController:
    """Per-track hysteretic state machine for specialist scheduling."""

    def __init__(self, config: SpecialistEscalationConfig | None = None) -> None:
        self.config = config or SpecialistEscalationConfig()
        self.state = EscalationState.BASELINE
        self.consecutive_low = 0
        self.recovery_frames = 0
        self.specialist_until_frame = -1
        self.cooldown_until_frame = -1
        self.last_specialist_frame = -10**9

    def step(self, frame_index: int, timestamp_sec: float, quality_signal: float) -> EscalationDecision:
        quality = min(1.0, max(0.0, float(quality_signal)))
        transition = None

        if self.config.mode == SpecialistPolicyMode.BASELINE_ONLY:
            return self._decision(frame_index, timestamp_sec, quality, False, None, "policy_baseline_only")

        if self.config.mode == SpecialistPolicyMode.EVERY_FRAME:
            self.state = EscalationState.SPECIALIST
            self.last_specialist_frame = frame_index
            return self._decision(frame_index, timestamp_sec, quality, True, None, "policy_specialists_every_frame")

        if self.state == EscalationState.COOLDOWN:
            if frame_index < self.cooldown_until_frame:
                return self._decision(frame_index, timestamp_sec, quality, False, None, "cooldown_active")
            self.state = EscalationState.BASELINE
            self.consecutive_low = 0
            self.recovery_frames = 0
            transition = "COOLDOWN_TO_BASELINE"

        if self.state == EscalationState.BASELINE:
            self.consecutive_low = self.consecutive_low + 1 if quality >= self.config.trigger_threshold else 0
            if self.consecutive_low >= self.config.min_consecutive_low_quality_frames:
                self.state = EscalationState.SPECIALIST
                self.specialist_until_frame = frame_index + self.config.specialist_window_frames - 1
                self.recovery_frames = 0
                transition = "BASELINE_TO_SPECIALIST"
            else:
                return self._decision(frame_index, timestamp_sec, quality, False, transition, "awaiting_sustained_trigger")

        # SPECIALIST state. Recovery needs both the configured number of
        # healthy frames and the minimum specialist window, preventing jitter.
        if quality <= self.config.recovery_threshold:
            self.recovery_frames += 1
        else:
            self.recovery_frames = 0
        elapsed_window = frame_index >= self.specialist_until_frame
        recovered = self.recovery_frames >= self.config.min_recovery_frames
        if elapsed_window and recovered:
            self.state = EscalationState.COOLDOWN
            self.cooldown_until_frame = frame_index + self.config.cooldown_frames
            return self._decision(frame_index, timestamp_sec, quality, False, "SPECIALIST_TO_COOLDOWN", "sustained_recovery_after_window")
        if elapsed_window and not recovered:
            self.state = EscalationState.COOLDOWN
            self.cooldown_until_frame = frame_index + self.config.cooldown_frames
            return self._decision(frame_index, timestamp_sec, quality, False, "SPECIALIST_TO_COOLDOWN", "specialist_window_complete")
        invoke = frame_index - self.last_specialist_frame >= self.config.min_specialist_interval_frames
        if invoke:
            self.last_specialist_frame = frame_index
        return self._decision(frame_index, timestamp_sec, quality, invoke, transition, "sustained_quality_degradation")

    def _decision(self, frame_index: int, timestamp_sec: float, quality: float, invoke: bool,
                  transition: str | None, reason: str) -> EscalationDecision:
        cooldown = max(0, self.cooldown_until_frame - frame_index) if self.state == EscalationState.COOLDOWN else 0
        return EscalationDecision(
            frame_index=frame_index, timestamp_sec=timestamp_sec, state=self.state,
            invoke_specialist=invoke, quality_signal=quality,
            consecutive_low_quality_frames=self.consecutive_low,
            recovery_frames=self.recovery_frames,
            cooldown_remaining_frames=cooldown, transition=transition, reason=reason,
        )


def load_specialist_escalation_config(path: str | Path, policy_name: str) -> SpecialistEscalationConfig:
    """Load one named policy without silently substituting another policy."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    policies = document.get("policies", {})
    if policy_name not in policies:
        raise KeyError(f"Specialist policy {policy_name!r} not found in {source}")
    payload = dict(policies[policy_name])
    payload["mode"] = SpecialistPolicyMode(payload.get("mode", policy_name))
    return SpecialistEscalationConfig(**payload)

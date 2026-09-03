"""Signal smoothing and statistical outlier filtering.

Conforms to docs/ARCHITECTURE.md Layer 3: Temporal Dynamics.
Eliminates single-frame keypoint jitter and detector noise without introducing
lag. Uses continuous-time Exponential Moving Average (EMA) parameterized by
time-constant tau, and Median Absolute Deviation (MAD) for dynamic outlier rejection.
Maintains dual raw and smoothed values for transparent auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
from collections import deque
import numpy as np


@dataclass
class SmoothedSignal:
    """Holds raw, smoothed, and derivative values of a scalar signal."""
    raw_value: float
    smoothed_value: float
    velocity: float  # Rate of change: d(value)/dt
    timestamp_sec: float
    is_outlier: bool = False


class TimeWeightedEMA:
    """Continuous-time Exponential Moving Average.
    
    alpha(dt) = 1 - exp(-dt / tau), where tau is the characteristic time constant.
    Works seamlessly across fluctuating frame rates (FPS-independent).
    """

    def __init__(self, time_constant_sec: float = 0.15):
        self.time_constant_sec = max(1e-4, time_constant_sec)
        self.last_timestamp: Optional[float] = None
        self.current_smoothed: Optional[float] = None
        self.current_velocity: float = 0.0

    def update(self, raw_value: float, timestamp_sec: float) -> SmoothedSignal:
        if self.last_timestamp is None or self.current_smoothed is None:
            self.last_timestamp = timestamp_sec
            self.current_smoothed = raw_value
            self.current_velocity = 0.0
            return SmoothedSignal(
                raw_value=raw_value,
                smoothed_value=raw_value,
                velocity=0.0,
                timestamp_sec=timestamp_sec,
                is_outlier=False
            )

        dt = max(1e-4, timestamp_sec - self.last_timestamp)
        # Continuous time smoothing factor
        alpha = float(1.0 - np.exp(-dt / self.time_constant_sec))
        new_smoothed = (1.0 - alpha) * self.current_smoothed + alpha * raw_value

        # Instantaneous rate of change
        self.current_velocity = (new_smoothed - self.current_smoothed) / dt
        self.current_smoothed = new_smoothed
        self.last_timestamp = timestamp_sec

        return SmoothedSignal(
            raw_value=raw_value,
            smoothed_value=new_smoothed,
            velocity=self.current_velocity,
            timestamp_sec=timestamp_sec,
            is_outlier=False
        )

    def reset(self) -> None:
        self.last_timestamp = None
        self.current_smoothed = None
        self.current_velocity = 0.0


class DynamicMADFilter:
    """Outlier detector using running Median Absolute Deviation (MAD).
    
    Identifies measurement spikes without hardcoded thresholds by measuring
    deviation relative to the candidate's running baseline variance.
    """

    def __init__(self, window_size: int = 15, z_threshold: float = 3.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history: Deque[float] = deque(maxlen=window_size)

    def is_outlier(self, value: float) -> bool:
        if len(self.history) < 5:
            self.history.append(value)
            return False

        arr = np.array(self.history)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))

        # Consistent estimator of normal standard deviation from MAD
        sigma = 1.4826 * mad
        if sigma < 1e-4:
            self.history.append(value)
            return False

        z_score = abs(value - median) / sigma
        if z_score > self.z_threshold:
            # Outlier detected; do not corrupt running distribution
            return True

        self.history.append(value)
        return False

    def reset(self) -> None:
        self.history.clear()


class KeypointSmoother:
    """Manages continuous EMA smoothing and outlier filtering for multi-keypoint sets."""

    def __init__(self, time_constant_sec: float = 0.12):
        self.time_constant_sec = time_constant_sec
        self.filters_x: dict[str, TimeWeightedEMA] = {}
        self.filters_y: dict[str, TimeWeightedEMA] = {}
        self.mad_x: dict[str, DynamicMADFilter] = {}
        self.mad_y: dict[str, DynamicMADFilter] = {}

    def smooth_point(
        self,
        name: str,
        x: float,
        y: float,
        timestamp_sec: float
    ) -> Tuple[float, float, float, bool]:
        """Returns (smoothed_x, smoothed_y, speed, is_outlier)."""
        if name not in self.filters_x:
            self.filters_x[name] = TimeWeightedEMA(self.time_constant_sec)
            self.filters_y[name] = TimeWeightedEMA(self.time_constant_sec)
            self.mad_x[name] = DynamicMADFilter()
            self.mad_y[name] = DynamicMADFilter()

        outlier_x = self.mad_x[name].is_outlier(x)
        outlier_y = self.mad_y[name].is_outlier(y)
        is_outlier = outlier_x or outlier_y

        # If flagged as outlier measurement spike, use previous smoothed state
        if is_outlier and self.filters_x[name].current_smoothed is not None:
            sx = self.filters_x[name].current_smoothed
            sy = self.filters_y[name].current_smoothed
            speed = float(np.hypot(self.filters_x[name].current_velocity, self.filters_y[name].current_velocity))
            return sx, sy, speed, True

        sig_x = self.filters_x[name].update(x, timestamp_sec)
        sig_y = self.filters_y[name].update(y, timestamp_sec)
        speed = float(np.hypot(sig_x.velocity, sig_y.velocity))

        return sig_x.smoothed_value, sig_y.smoothed_value, speed, False

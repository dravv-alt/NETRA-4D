"""Frame Quality Assessment Engine for NETRA-4D Preprocessing Pipeline.

Calculates luminance, RMS contrast, noise variance, focus/blur metrics,
and LAB color cast to produce a typed QualityReport with defect flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from drishti.preprocessing.config import PreprocessingConfig, QualityGateConfig

logger = logging.getLogger(__name__)


class QualityFlag(str, Enum):
    """Specific frame quality degradation flags."""
    CLEAR = "CLEAR"
    DARK = "DARK"
    BRIGHT = "BRIGHT"
    LOW_CONTRAST = "LOW_CONTRAST"
    NOISY = "NOISY"
    BLURRY = "BLURRY"
    COLOR_CAST = "COLOR_CAST"
    HIGH_MOTION = "HIGH_MOTION"


@dataclass
class QualityReport:
    """Detailed quality assessment report for a single video frame."""
    brightness: float             # Mean luminance (0.0 - 255.0)
    contrast: float               # RMS contrast / luminance standard deviation (0.0 - 128.0)
    noise_level: float            # Estimated high-frequency noise standard deviation
    blur_score: float             # Focus measure (Tenengrad / Laplacian variance, higher is sharper)
    color_cast_score: float       # LAB chroma deviation distance from neutral gray
    overall_score: float          # Composite quality score normalized to [0.0, 1.0]
    flags: List[QualityFlag] = field(default_factory=list)
    needs_enhancement: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    assessment_time_ms: float = 0.0

    @property
    def is_dark(self) -> bool:
        return QualityFlag.DARK in self.flags

    @property
    def is_bright(self) -> bool:
        return QualityFlag.BRIGHT in self.flags

    @property
    def is_low_contrast(self) -> bool:
        return QualityFlag.LOW_CONTRAST in self.flags

    @property
    def is_noisy(self) -> bool:
        return QualityFlag.NOISY in self.flags

    @property
    def is_blurry(self) -> bool:
        return QualityFlag.BLURRY in self.flags

    @property
    def has_color_cast(self) -> bool:
        return QualityFlag.COLOR_CAST in self.flags

    @property
    def is_high_motion(self) -> bool:
        return QualityFlag.HIGH_MOTION in self.flags

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality report to JSON-serializable dictionary."""
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "noise_level": self.noise_level,
            "blur_score": self.blur_score,
            "color_cast_score": self.color_cast_score,
            "overall_score": self.overall_score,
            "flags": [f.value if hasattr(f, "value") else str(f) for f in self.flags],
            "needs_enhancement": self.needs_enhancement,
            "details": dict(self.details),
            "assessment_time_ms": self.assessment_time_ms,
        }


class FrameQualityAssessor:
    """Evaluates video frame quality and flags degradations."""

    def __init__(self, config: Optional[Union[PreprocessingConfig, QualityGateConfig]] = None):
        if isinstance(config, PreprocessingConfig):
            self.config = config.quality_gate
            self.full_config = config
        elif isinstance(config, QualityGateConfig):
            self.config = config
            self.full_config = None
        else:
            self.config = QualityGateConfig()
            self.full_config = None
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset historical temporal tracking state."""
        self._prev_gray = None

    def assess(self, frame_bgr: np.ndarray) -> QualityReport:
        """Analyze frame and produce a comprehensive QualityReport.
        
        Args:
            frame_bgr: BGR image as uint8 numpy array (H, W, 3).

        Returns:
            QualityReport with computed metrics and defect flags.
        """
        start_time = time.perf_counter()

        if frame_bgr is None or frame_bgr.size == 0 or len(frame_bgr.shape) < 2:
            return QualityReport(
                brightness=0.0,
                contrast=0.0,
                noise_level=0.0,
                blur_score=0.0,
                color_cast_score=0.0,
                overall_score=0.0,
                flags=[QualityFlag.DARK, QualityFlag.LOW_CONTRAST],
                needs_enhancement=True,
                details={"error": "Empty or invalid frame"},
                assessment_time_ms=0.0,
            )

        h, w = frame_bgr.shape[:2]

        # Fast downsampling for metric calculation if enabled
        eval_frame = frame_bgr
        if self.config.downsample_assessment and max(h, w) > self.config.assessment_max_dim:
            scale = self.config.assessment_max_dim / float(max(h, w))
            new_w = max(16, int(round(w * scale)))
            new_h = max(16, int(round(h * scale)))
            eval_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to Grayscale for luminance / structure metrics
        if len(eval_frame.shape) == 3 and eval_frame.shape[2] == 3:
            gray = cv2.cvtColor(eval_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = eval_frame if len(eval_frame.shape) == 2 else eval_frame[:, :, 0]

        # 1. Brightness: Mean luminance
        brightness = float(np.mean(gray))

        # 2. Contrast: RMS Contrast (standard deviation of luminance)
        contrast = float(np.std(gray))

        # 3. Noise Level: Median Absolute Deviation (MAD) of Laplacian
        # Robust estimator: sigma = MAD / 0.6745
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        abs_lap = np.abs(laplacian)
        median_abs = float(np.median(abs_lap))
        noise_level = float(median_abs / 0.6745) if median_abs > 0 else 0.0

        # 4. Blur / Focus score: Variance of Laplacian (Tenengrad focus measure)
        blur_score = float(np.var(laplacian))

        # 5. Color Cast: LAB Chroma imbalance distance
        if len(eval_frame.shape) == 3 and eval_frame.shape[2] == 3:
            lab = cv2.cvtColor(eval_frame, cv2.COLOR_BGR2LAB)
            # In OpenCV LAB, L is [0, 255], a is [0, 255] (centered at 128), b is [0, 255] (centered at 128)
            mean_a = float(np.mean(lab[:, :, 1]))
            mean_b = float(np.mean(lab[:, :, 2]))
            color_cast_score = float(math.sqrt((mean_a - 128.0) ** 2 + (mean_b - 128.0) ** 2))
        else:
            mean_a, mean_b = 128.0, 128.0
            color_cast_score = 0.0

        # 6. Motion estimation via temporal frame differencing
        motion_score = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            motion_score = float(np.mean(diff))
        self._prev_gray = gray.copy()

        # Evaluate Defect Flags
        flags: List[QualityFlag] = []

        # Brightness flags
        low_b_thresh = self.full_config.gamma.low_brightness_threshold if self.full_config else self.config.min_brightness
        high_b_thresh = self.full_config.gamma.high_brightness_threshold if self.full_config else self.config.max_brightness
        if brightness < low_b_thresh:
            flags.append(QualityFlag.DARK)
        elif brightness > high_b_thresh:
            flags.append(QualityFlag.BRIGHT)

        # Contrast flag
        contrast_thresh = self.full_config.clahe.trigger_contrast_threshold if self.full_config else self.config.min_contrast
        if contrast < contrast_thresh:
            flags.append(QualityFlag.LOW_CONTRAST)

        # Noise flag
        noise_thresh = self.full_config.denoise.trigger_noise_threshold if self.full_config else 12.0
        if noise_level > noise_thresh:
            flags.append(QualityFlag.NOISY)

        # Blur flag
        blur_thresh = self.full_config.sharpen.trigger_blur_threshold if self.full_config else self.config.min_focus_score
        if blur_score < blur_thresh:
            flags.append(QualityFlag.BLURRY)

        # Color cast flag
        cast_thresh = self.full_config.white_balance.trigger_cast_threshold if self.full_config else 14.0
        if color_cast_score > cast_thresh:
            flags.append(QualityFlag.COLOR_CAST)

        # High motion flag
        motion_thresh = getattr(self.full_config, "motion_threshold", 25.0) if self.full_config else 25.0
        if motion_score > motion_thresh:
            flags.append(QualityFlag.HIGH_MOTION)

        if not flags:
            flags.append(QualityFlag.CLEAR)

        # Composite overall score calculation (0.0 to 1.0, 1.0 being optimal)
        # Optimal brightness is ~110-150. Symmetrical Gaussian or smooth bell penalty.
        b_score = math.exp(-0.5 * ((brightness - 125.0) / 45.0) ** 2)

        # Contrast score: sigmoid-like saturation above 45
        c_score = min(1.0, contrast / 45.0)

        # Noise score: 1.0 when noise < 6, decreasing to 0.0 at noise > 25
        n_score = max(0.0, min(1.0, 1.0 - (noise_level - 6.0) / 20.0))

        # Focus score: logarithmic ramp from 0 to 1
        f_score = min(1.0, math.log1p(blur_score) / math.log1p(300.0))

        # Color score: 1.0 when cast < 5, 0.0 when cast > 25
        w_score = max(0.0, min(1.0, 1.0 - (color_cast_score - 5.0) / 20.0))

        # Weighted composite overall quality
        overall_score = float(
            0.25 * b_score +
            0.25 * c_score +
            0.15 * n_score +
            0.20 * f_score +
            0.15 * w_score
        )
        overall_score = max(0.0, min(1.0, round(overall_score, 4)))

        # Needs enhancement decision:
        # If bypass threshold is enabled and overall_score >= bypass_threshold and no severe flags
        bypass_thresh = self.config.bypass_threshold
        needs_enhancement = (overall_score < bypass_thresh) or (QualityFlag.CLEAR not in flags)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        details = {
            "brightness_subscore": round(b_score, 3),
            "contrast_subscore": round(c_score, 3),
            "noise_subscore": round(n_score, 3),
            "focus_subscore": round(f_score, 3),
            "color_subscore": round(w_score, 3),
            "motion_score": round(motion_score, 2),
            "mean_lab_a": round(mean_a, 2),
            "mean_lab_b": round(mean_b, 2),
            "frame_orig_size": (w, h),
            "downsampled": bool(eval_frame.shape[:2] != (h, w)),
        }

        return QualityReport(
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            noise_level=round(noise_level, 2),
            blur_score=round(blur_score, 2),
            color_cast_score=round(color_cast_score, 2),
            overall_score=overall_score,
            flags=flags,
            needs_enhancement=needs_enhancement,
            details=details,
            assessment_time_ms=round(elapsed_ms, 3),
        )

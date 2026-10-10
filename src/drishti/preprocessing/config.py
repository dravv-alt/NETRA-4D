"""Configuration schemas and loaders for NETRA-4D Deep Preprocessing Pipeline.

Provides strongly typed configuration classes for quality assessment and
the 5-stage adaptive enhancement pipeline, with support for YAML serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DenoiseConfig:
    """Configuration for Stage 1: Noise reduction."""
    enabled: bool = True
    method: str = "fast_nl_means"  # "fast_nl_means", "bilateral", "gaussian"
    # Bilateral filter parameters (fast, edge-preserving)
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    # Fast NL-Means parameters (heavy, high-fidelity for low-light offline)
    nlmeans_h: float = 7.0
    nlmeans_h_color: float = 7.0
    nlmeans_template_window: int = 7
    nlmeans_search_window: int = 21
    # Threshold in Laplacian noise estimate above which denoising triggers
    trigger_noise_threshold: float = 12.0


@dataclass
class CLAHEConfig:
    """Configuration for Stage 2: Contrast-Limited Adaptive Histogram Equalization."""
    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)
    # RMS contrast threshold below which CLAHE triggers
    trigger_contrast_threshold: float = 40.0


@dataclass
class GammaConfig:
    """Configuration for Stage 3: Adaptive Gamma Correction."""
    enabled: bool = True
    target_brightness: float = 125.0
    min_gamma: float = 0.5
    max_gamma: float = 2.2
    # Thresholds for triggering gamma adjustment
    low_brightness_threshold: float = 75.0   # triggers brightening (gamma < 1.0)
    high_brightness_threshold: float = 185.0  # triggers darkening (gamma > 1.0)


@dataclass
class SharpenConfig:
    """Configuration for Stage 4: High-Frequency Detail Restoration."""
    enabled: bool = True
    amount: float = 1.0
    radius: float = 1.0
    threshold: int = 3
    # Tenengrad/Laplacian focus score below which sharpening triggers
    trigger_blur_threshold: float = 110.0


@dataclass
class WhiteBalanceConfig:
    """Configuration for Stage 5: Color Cast Neutralization."""
    enabled: bool = True
    method: str = "gray_world"  # "gray_world", "shades_of_gray", "white_patch"
    p_norm: int = 6  # for shades_of_gray
    # Chroma distance threshold in LAB color space
    trigger_cast_threshold: float = 14.0


@dataclass
class QualityGateConfig:
    """Configuration for Quality Assessment Gate & Bypass logic."""
    enabled: bool = True
    # Bypass threshold: if overall_score >= bypass_threshold, skip all enhancement
    bypass_threshold: float = 0.85
    min_brightness: float = 30.0
    max_brightness: float = 220.0
    min_contrast: float = 15.0
    min_focus_score: float = 60.0
    # Fast assessment downsampling
    downsample_assessment: bool = True
    assessment_max_dim: int = 320


@dataclass
class PreprocessingConfig:
    """Top-level configuration for the drishti deep preprocessing pipeline."""
    enabled: bool = True
    mode: str = "adaptive"  # "adaptive", "always", "disabled"
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    gamma: GammaConfig = field(default_factory=GammaConfig)
    sharpen: SharpenConfig = field(default_factory=SharpenConfig)
    white_balance: WhiteBalanceConfig = field(default_factory=WhiteBalanceConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    save_enhanced_frames: bool = False
    save_flagged_frames: bool = True
    save_high_motion_frames: bool = True
    motion_threshold: float = 25.0
    flagged_frames_dir: str = "artifacts/flagged_evidence"
    log_quality_reports: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingConfig:
        """Construct a PreprocessingConfig from a raw dictionary with deep nested parsing."""
        if not data:
            return cls()

        denoise_data = data.get("denoise", {})
        clahe_data = data.get("clahe", {})
        gamma_data = data.get("gamma", {})
        sharpen_data = data.get("sharpen", {})
        wb_data = data.get("white_balance", {})
        qgate_data = data.get("quality_gate", {})

        tile_size = clahe_data.get("tile_grid_size", (8, 8))
        if isinstance(tile_size, list):
            tile_size = tuple(tile_size)

        return cls(
            enabled=data.get("enabled", True),
            mode=data.get("mode", "adaptive"),
            denoise=DenoiseConfig(
                enabled=denoise_data.get("enabled", True),
                method=denoise_data.get("method", "fast_nl_means"),
                bilateral_d=denoise_data.get("bilateral_d", 7),
                bilateral_sigma_color=float(denoise_data.get("bilateral_sigma_color", 50.0)),
                bilateral_sigma_space=float(denoise_data.get("bilateral_sigma_space", 50.0)),
                nlmeans_h=float(denoise_data.get("nlmeans_h", 7.0)),
                nlmeans_h_color=float(denoise_data.get("nlmeans_h_color", 7.0)),
                nlmeans_template_window=denoise_data.get("nlmeans_template_window", 7),
                nlmeans_search_window=denoise_data.get("nlmeans_search_window", 21),
                trigger_noise_threshold=float(denoise_data.get("trigger_noise_threshold", 12.0)),
            ),
            clahe=CLAHEConfig(
                enabled=clahe_data.get("enabled", True),
                clip_limit=float(clahe_data.get("clip_limit", 2.0)),
                tile_grid_size=tile_size,
                trigger_contrast_threshold=float(clahe_data.get("trigger_contrast_threshold", 40.0)),
            ),
            gamma=GammaConfig(
                enabled=gamma_data.get("enabled", True),
                target_brightness=float(gamma_data.get("target_brightness", 125.0)),
                min_gamma=float(gamma_data.get("min_gamma", 0.5)),
                max_gamma=float(gamma_data.get("max_gamma", 2.2)),
                low_brightness_threshold=float(gamma_data.get("low_brightness_threshold", 75.0)),
                high_brightness_threshold=float(gamma_data.get("high_brightness_threshold", 185.0)),
            ),
            sharpen=SharpenConfig(
                enabled=sharpen_data.get("enabled", True),
                amount=float(sharpen_data.get("amount", 1.0)),
                radius=float(sharpen_data.get("radius", 1.0)),
                threshold=int(sharpen_data.get("threshold", 3)),
                trigger_blur_threshold=float(sharpen_data.get("trigger_blur_threshold", 110.0)),
            ),
            white_balance=WhiteBalanceConfig(
                enabled=wb_data.get("enabled", True),
                method=wb_data.get("method", "gray_world"),
                p_norm=int(wb_data.get("p_norm", 6)),
                trigger_cast_threshold=float(wb_data.get("trigger_cast_threshold", 14.0)),
            ),
            quality_gate=QualityGateConfig(
                enabled=qgate_data.get("enabled", True),
                bypass_threshold=float(qgate_data.get("bypass_threshold", 0.85)),
                min_brightness=float(qgate_data.get("min_brightness", 30.0)),
                max_brightness=float(qgate_data.get("max_brightness", 220.0)),
                min_contrast=float(qgate_data.get("min_contrast", 15.0)),
                min_focus_score=float(qgate_data.get("min_focus_score", 60.0)),
                downsample_assessment=bool(qgate_data.get("downsample_assessment", True)),
                assessment_max_dim=int(qgate_data.get("assessment_max_dim", 320)),
            ),
            save_enhanced_frames=bool(data.get("save_enhanced_frames", False)),
            save_flagged_frames=bool(data.get("save_flagged_frames", True)),
            save_high_motion_frames=bool(data.get("save_high_motion_frames", True)),
            motion_threshold=float(data.get("motion_threshold", 25.0)),
            flagged_frames_dir=str(data.get("flagged_frames_dir", "artifacts/flagged_evidence")),
            log_quality_reports=bool(data.get("log_quality_reports", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "denoise": {
                "enabled": self.denoise.enabled,
                "method": self.denoise.method,
                "bilateral_d": self.denoise.bilateral_d,
                "bilateral_sigma_color": self.denoise.bilateral_sigma_color,
                "bilateral_sigma_space": self.denoise.bilateral_sigma_space,
                "nlmeans_h": self.denoise.nlmeans_h,
                "nlmeans_h_color": self.denoise.nlmeans_h_color,
                "nlmeans_template_window": self.denoise.nlmeans_template_window,
                "nlmeans_search_window": self.denoise.nlmeans_search_window,
                "trigger_noise_threshold": self.denoise.trigger_noise_threshold,
            },
            "clahe": {
                "enabled": self.clahe.enabled,
                "clip_limit": self.clahe.clip_limit,
                "tile_grid_size": list(self.clahe.tile_grid_size),
                "trigger_contrast_threshold": self.clahe.trigger_contrast_threshold,
            },
            "gamma": {
                "enabled": self.gamma.enabled,
                "target_brightness": self.gamma.target_brightness,
                "min_gamma": self.gamma.min_gamma,
                "max_gamma": self.gamma.max_gamma,
                "low_brightness_threshold": self.gamma.low_brightness_threshold,
                "high_brightness_threshold": self.gamma.high_brightness_threshold,
            },
            "sharpen": {
                "enabled": self.sharpen.enabled,
                "amount": self.sharpen.amount,
                "radius": self.sharpen.radius,
                "threshold": self.sharpen.threshold,
                "trigger_blur_threshold": self.sharpen.trigger_blur_threshold,
            },
            "white_balance": {
                "enabled": self.white_balance.enabled,
                "method": self.white_balance.method,
                "p_norm": self.white_balance.p_norm,
                "trigger_cast_threshold": self.white_balance.trigger_cast_threshold,
            },
            "quality_gate": {
                "enabled": self.quality_gate.enabled,
                "bypass_threshold": self.quality_gate.bypass_threshold,
                "min_brightness": self.quality_gate.min_brightness,
                "max_brightness": self.quality_gate.max_brightness,
                "min_contrast": self.quality_gate.min_contrast,
                "min_focus_score": self.quality_gate.min_focus_score,
                "downsample_assessment": self.quality_gate.downsample_assessment,
                "assessment_max_dim": self.quality_gate.assessment_max_dim,
            },
            "save_enhanced_frames": self.save_enhanced_frames,
            "save_flagged_frames": self.save_flagged_frames,
            "save_high_motion_frames": self.save_high_motion_frames,
            "motion_threshold": self.motion_threshold,
            "flagged_frames_dir": self.flagged_frames_dir,
            "log_quality_reports": self.log_quality_reports,
        }


def load_preprocessing_config(path: Optional[Union[str, Path]] = None) -> PreprocessingConfig:
    """Load preprocessing configuration from YAML path or default locations.
    
    If path is None, tries:
    1. configs/preprocessing.yaml relative to current working directory
    2. Fallback to PreprocessingConfig() with default values
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.extend([
            Path("configs/preprocessing.yaml"),
            Path(__file__).resolve().parent.parent.parent.parent / "configs" / "preprocessing.yaml",
        ])

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                if isinstance(raw, dict):
                    logger.info("Loaded preprocessing config from %s", candidate)
                    return PreprocessingConfig.from_dict(raw)
            except Exception as e:
                logger.warning("Failed to parse config at %s: %s. Using defaults.", candidate, e)
                break

    logger.info("Using default PreprocessingConfig")
    return PreprocessingConfig()


def save_preprocessing_config(config: PreprocessingConfig, path: Union[str, Path]) -> None:
    """Save preprocessing configuration to a YAML file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    logger.info("Saved preprocessing config to %s", target)

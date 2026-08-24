"""
Video Quality Enhancement for Watcher.

Implements preprocessing to improve frame quality before detection:
- CLAHE for contrast enhancement
- Gamma correction for brightness adjustment
- Denoising for noise reduction
- Quality gate to flag low-quality frames
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QualityFlag(Enum):
    """Frame quality flags."""
    GOOD = "GOOD"
    LOW_LIGHT = "LOW_LIGHT"
    BLURRY = "BLURRY"
    OVEREXPOSED = "OVEREXPOSED"
    LOW_CONTRAST = "LOW_CONTRAST"
    MOTION_BLUR = "MOTION_BLUR"


@dataclass
class QualityInfo:
    """Frame quality assessment."""
    flag: QualityFlag
    score: float  # 0-1, higher is better
    brightness: float
    contrast: float
    blur_score: float
    details: Dict[str, float]


class VideoEnhancer:
    """
    Video frame enhancement pipeline.
    
    Applies a series of enhancements to improve detection performance
    on poor quality video frames.
    """
    
    def __init__(self, config_path: str = "configs/enhancement.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_enhancers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load enhancement configuration."""
        if not self.config_path.exists():
            logger.warning(f"Enhancement config not found: {self.config_path}, using defaults")
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default enhancement configuration."""
        return {
            "enhancement": {
                "enabled": True,
                "clahe": {
                    "enabled": True,
                    "clip_limit": 2.0,
                    "tile_grid_size": [8, 8]
                },
                "gamma_correction": {
                    "enabled": True,
                    "auto_gamma": True,
                    "target_brightness": 128,
                    "min_gamma": 0.5,
                    "max_gamma": 2.0
                },
                "denoising": {
                    "enabled": True,
                    "method": "fast_nl_means",  # "fast_nl_means", "bilateral", "gaussian"
                    "h": 10,
                    "h_color": 10,
                    "template_window": 7,
                    "search_window": 21
                }
            },
            "quality_gate": {
                "enabled": True,
                "min_brightness": 30,
                "max_brightness": 220,
                "min_contrast": 15,
                "max_blur_score": 100,
                "action_on_poor": "flag_only"  # "flag_only", "enhance_aggressive", "skip"
            }
        }
    
    def _setup_enhancers(self):
        """Initialize enhancement components."""
        enh_config = self.config.get("enhancement", {})
        
        # CLAHE
        clahe_config = enh_config.get("clahe", {})
        if clahe_config.get("enabled", True):
            clip_limit = clahe_config.get("clip_limit", 2.0)
            tile_grid = tuple(clahe_config.get("tile_grid_size", [8, 8]))
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        else:
            self.clahe = None
        
        # Gamma correction
        gamma_config = enh_config.get("gamma_correction", {})
        self.gamma_enabled = gamma_config.get("enabled", True)
        self.auto_gamma = gamma_config.get("auto_gamma", True)
        self.target_brightness = gamma_config.get("target_brightness", 128)
        self.min_gamma = gamma_config.get("min_gamma", 0.5)
        self.max_gamma = gamma_config.get("max_gamma", 2.0)
        
        # Denoising
        denoise_config = enh_config.get("denoising", {})
        self.denoise_enabled = denoise_config.get("enabled", True)
        self.denoise_method = denoise_config.get("method", "fast_nl_means")
        self.denoise_h = denoise_config.get("h", 10)
        self.denoise_h_color = denoise_config.get("h_color", 10)
        self.denoise_template = denoise_config.get("template_window", 7)
        self.denoise_search = denoise_config.get("search_window", 21)
        
        # Quality gate
        qgate_config = self.config.get("quality_gate", {})
        self.quality_gate_enabled = qgate_config.get("enabled", True)
        self.min_brightness = qgate_config.get("min_brightness", 30)
        self.max_brightness = qgate_config.get("max_brightness", 220)
        self.min_contrast = qgate_config.get("min_contrast", 15)
        self.max_blur_score = qgate_config.get("max_blur_score", 100)
        self.poor_frame_action = qgate_config.get("action_on_poor", "flag_only")
    
    def enhance(self, frame: np.ndarray) -> Tuple[np.ndarray, QualityInfo]:
        """
        Enhance a frame and assess its quality.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (enhanced_frame, QualityInfo)
        """
        # Assess quality first
        quality = self.assess_quality(frame)
        
        # If enhancement is disabled, return original
        enh_config = self.config.get("enhancement", {})
        if not enh_config.get("enabled", True):
            return frame, quality
        
        enhanced = frame.copy()
        
        # Apply denoising first (on original)
        if self.denoise_enabled:
            enhanced = self._denoise(enhanced)
        
        # Apply CLAHE for contrast enhancement
        if self.clahe is not None:
            enhanced = self._apply_clahe(enhanced)
        
        # Apply gamma correction
        if self.gamma_enabled:
            enhanced = self._apply_gamma(enhanced)
        
        # Re-assess quality after enhancement
        quality = self.assess_quality(enhanced)
        
        return enhanced, quality
    
    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply fast denoising to frame."""
        if not self.denoise_enabled:
            return frame
        if self.denoise_method == "bilateral":
            return cv2.bilateralFilter(frame, 5, 50, 50)
        elif self.denoise_method == "gaussian":
            return cv2.GaussianBlur(frame, (3, 3), 0)
        elif self.denoise_method == "median":
            return cv2.medianBlur(frame, 3)
        # Default lightweight gaussian for speed and noise suppression
        return cv2.GaussianBlur(frame, (3, 3), 0)
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L channel in LAB color space."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction to frame."""
        if self.auto_gamma:
            # Calculate current brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness > 0:
                # Calculate gamma to reach target brightness
                gamma = np.log(self.target_brightness / 255.0) / np.log(mean_brightness / 255.0)
                gamma = np.clip(gamma, self.min_gamma, self.max_gamma)
            else:
                gamma = 1.0
        else:
            gamma = 1.0
        
        if abs(gamma - 1.0) < 0.05:
            return frame
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        
        return cv2.LUT(frame, table)
    
    def assess_quality(self, frame: np.ndarray) -> QualityInfo:
        """
        Assess frame quality.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            QualityInfo with flag and metrics
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness (mean pixel value)
        brightness = float(np.mean(gray))
        
        # Contrast (std deviation)
        contrast = float(np.std(gray))
        
        # Blur score (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Determine quality flag
        flag = QualityFlag.GOOD
        details = {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score
        }
        
        if brightness < self.min_brightness:
            flag = QualityFlag.LOW_LIGHT
        elif brightness > self.max_brightness:
            flag = QualityFlag.OVEREXPOSED
        elif contrast < self.min_contrast:
            flag = QualityFlag.LOW_CONTRAST
        elif blur_score > self.max_blur_score:
            flag = QualityFlag.MOTION_BLUR
        
        # Overall quality score (0-1)
        # Normalize metrics
        brightness_score = 1.0 - abs(brightness - 128) / 128.0
        contrast_score = min(contrast / 50.0, 1.0)
        blur_score_norm = min(blur_score / 500.0, 1.0)  # Lower is better for blur
        blur_score_norm = 1.0 - blur_score_norm
        
        score = (brightness_score + contrast_score + blur_score_norm) / 3.0
        score = np.clip(score, 0.0, 1.0)
        
        return QualityInfo(
            flag=flag,
            score=score,
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
            details=details
        )
    
    def should_process(self, quality: QualityInfo) -> bool:
        """
        Determine if frame should be processed based on quality.
        
        Args:
            quality: QualityInfo from assess_quality
            
        Returns:
            True if frame should be processed
        """
        if not self.quality_gate_enabled:
            return True
        
        if self.poor_frame_action == "skip":
            return quality.flag == QualityFlag.GOOD
        
        # For "flag_only" and "enhance_aggressive", always process
        return True


def create_enhancer(config_path: str = "configs/enhancement.yaml") -> VideoEnhancer:
    """Factory function to create VideoEnhancer from config."""
    return VideoEnhancer(config_path)
"""Unified Preprocessing Pipeline for NETRA-4D Vision Subsystem.

Combines FrameQualityAssessor and AdaptiveFrameEnhancer into a single,
high-throughput pipeline with intelligent quality-gated bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from drishti.preprocessing.config import PreprocessingConfig, load_preprocessing_config
from drishti.preprocessing.enhancer import AdaptiveFrameEnhancer, EnhancementResult
from drishti.preprocessing.quality import FrameQualityAssessor, QualityFlag, QualityReport

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStats:
    """Cumulative performance and throughput statistics."""
    frames_processed: int = 0
    frames_enhanced: int = 0
    frames_bypassed: int = 0
    total_assessment_time_ms: float = 0.0
    total_enhancement_time_ms: float = 0.0
    stage_counts: Dict[str, int] = field(default_factory=lambda: {
        "denoise": 0,
        "clahe": 0,
        "gamma": 0,
        "sharpen": 0,
        "white_balance": 0,
    })

    @property
    def avg_assessment_ms(self) -> float:
        return (self.total_assessment_time_ms / self.frames_processed) if self.frames_processed > 0 else 0.0

    @property
    def avg_enhancement_ms(self) -> float:
        return (self.total_enhancement_time_ms / self.frames_enhanced) if self.frames_enhanced > 0 else 0.0

    @property
    def bypass_rate(self) -> float:
        return (self.frames_bypassed / self.frames_processed) if self.frames_processed > 0 else 0.0


class PreprocessingPipeline:
    """Orchestrator for deep image preprocessing and quality assessment.
    
    Acts as an intelligent filter placed between video decode and YOLO inference.
    If a frame has sufficient quality, it is bypassed with zero modification.
    If quality degradations (noise, poor contrast, extreme lighting, blur, color cast)
    are detected, targeted adaptive enhancement stages are applied.
    """

    def __init__(
        self,
        config: Optional[Union[PreprocessingConfig, str, Path]] = None,
    ):
        if isinstance(config, (str, Path)):
            self.config = load_preprocessing_config(config)
        elif isinstance(config, PreprocessingConfig):
            self.config = config
        else:
            self.config = load_preprocessing_config()

        self.assessor = FrameQualityAssessor(self.config)
        self.enhancer = AdaptiveFrameEnhancer(self.config)
        self.stats = PreprocessingStats()
        self.last_enhancement_result: Optional[EnhancementResult] = None
        self.last_quality_report: Optional[QualityReport] = None

    def reset(self) -> None:
        """Reset temporal state across stream boundaries."""
        self.assessor.reset()
        self.stats = PreprocessingStats()
        self.last_enhancement_result = None
        self.last_quality_report = None

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        is_flagged_event: bool = False,
        flag_reason: str = "",
    ) -> Tuple[np.ndarray, QualityReport]:
        """Assess quality and conditionally enhance frame before YOLO inference.
        
        Args:
            frame_bgr: Raw input BGR frame from cv2.VideoCapture or RTSP stream.
            frame_idx: Frame sequence index.
            is_flagged_event: Whether caller has identified a high-priority security/cheating event.
            flag_reason: Description of the flagging event (e.g. 'phone_detected', 'chit_paper').

        Returns:
            Tuple of:
                - enhanced_bgr: Enhanced BGR frame (or original if bypassed).
                - quality_report: Frame quality metrics and flags.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            dummy_report = self.assessor.assess(frame_bgr)
            self.last_quality_report = dummy_report
            self.last_enhancement_result = None
            return frame_bgr, dummy_report

        # 1. Pipeline disabled bypass
        if not self.config.enabled or self.config.mode == "disabled":
            dummy_report = QualityReport(
                brightness=128.0,
                contrast=50.0,
                noise_level=0.0,
                blur_score=200.0,
                color_cast_score=0.0,
                overall_score=1.0,
                flags=[QualityFlag.CLEAR],
                needs_enhancement=False,
            )
            self.last_quality_report = dummy_report
            self.last_enhancement_result = None
            return frame_bgr, dummy_report

        # 2. Quality Assessment
        t_assess_start = time.perf_counter()
        report = self.assessor.assess(frame_bgr)
        t_assess_ms = (time.perf_counter() - t_assess_start) * 1000.0
        self.last_quality_report = report

        self.stats.frames_processed += 1
        self.stats.total_assessment_time_ms += t_assess_ms

        # 3. Intelligent Bypass Gate
        # If frame is high quality and needs no enhancement, bypass immediately
        can_bypass = (
            self.config.quality_gate.enabled and
            self.config.mode == "adaptive" and
            not report.needs_enhancement and
            report.overall_score >= self.config.quality_gate.bypass_threshold
        )

        if can_bypass:
            self.stats.frames_bypassed += 1
            self.last_enhancement_result = EnhancementResult(
                frame=frame_bgr,
                stages_applied=[],
                stage_latencies_ms={},
                total_time_ms=0.0,
            )
            if self.config.log_quality_reports and frame_idx % 100 == 0:
                logger.info(
                    "[Preprocessing] Frame %d bypassed (score=%.2f, flags=%s)",
                    frame_idx, report.overall_score, [f.value for f in report.flags],
                )

            # Check if caller requested flagged frame saving on bypass
            if is_flagged_event and self.config.save_flagged_frames:
                self.save_flagged_event_frame(
                    orig_bgr=frame_bgr,
                    enh_bgr=frame_bgr,
                    frame_idx=frame_idx,
                    reason=flag_reason or "FLAGGED_EVENT",
                    metadata={"overall_score": report.overall_score, "flags": [f.value for f in report.flags]}
                )

            return frame_bgr, report

        # 4. Adaptive Enhancement
        t_enh_start = time.perf_counter()
        enh_result = self.enhancer.enhance(frame_bgr, quality_report=report)
        t_enh_ms = (time.perf_counter() - t_enh_start) * 1000.0

        self.last_enhancement_result = enh_result
        self.stats.frames_enhanced += 1
        self.stats.total_enhancement_time_ms += t_enh_ms

        for stage in enh_result.stages_applied:
            if stage in self.stats.stage_counts:
                self.stats.stage_counts[stage] += 1

        if self.config.log_quality_reports and (frame_idx % 100 == 0 or len(enh_result.stages_applied) > 0):
            logger.info(
                "[Preprocessing] Frame %d enhanced in %.1fms (score=%.2f, stages=%s)",
                frame_idx,
                t_enh_ms,
                report.overall_score,
                enh_result.stages_applied,
            )

        # 5. Diagnostic frame saving
        if self.config.save_enhanced_frames and frame_idx % 30 == 0:
            self._save_diagnostic_comparison(frame_bgr, enh_result.frame, frame_idx, report)

        # 6. High motion frame saving
        if report.is_high_motion and self.config.save_high_motion_frames:
            m_score = report.details.get("motion_score", 0.0)
            self.save_flagged_event_frame(
                orig_bgr=frame_bgr,
                enh_bgr=enh_result.frame,
                frame_idx=frame_idx,
                reason=f"HIGH_MOTION_{m_score:.1f}",
                metadata={"motion_score": m_score, "overall_score": report.overall_score, "flags": [f.value for f in report.flags]}
            )

        # 7. Flagged event frame saving
        if is_flagged_event and self.config.save_flagged_frames:
            self.save_flagged_event_frame(
                orig_bgr=frame_bgr,
                enh_bgr=enh_result.frame,
                frame_idx=frame_idx,
                reason=flag_reason or "FLAGGED_EVENT",
                metadata={"overall_score": report.overall_score, "flags": [f.value for f in report.flags]}
            )

        return enh_result.frame, report

    def enhance_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Lightweight multi-stage enhancement for person, hand, and object crop regions.
        
        Applies fast edge-preserving denoising, CLAHE on LAB L-channel, and detail
        sharpening specifically calibrated for small crops (128x128 - 256x256).
        Significantly reduces false positives (shadow noise) and false negatives
        (low-contrast slips/phones).
        """
        if crop_bgr is None or crop_bgr.size == 0 or not self.config.enabled:
            return crop_bgr

        if len(crop_bgr.shape) != 3 or crop_bgr.shape[2] != 3:
            return crop_bgr

        enhanced = crop_bgr

        # 1. Fast Edge-Preserving Denoising (bilateral with tight spatial window for crops)
        # Suppresses sensor noise grain that causes false edge detections
        if self.config.denoise.enabled:
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35.0, sigmaSpace=35.0)

        # 2. Local Contrast Boost (CLAHE in LAB color space)
        # Recovers fine text, folds, or edges of chits/phones in dim lighting
        if self.config.clahe.enabled:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Use smaller grid (4x4) for small crop resolution
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l_enh = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)

        # 3. Light Unsharp Masking
        # Sharpens edge boundaries of paper chits and phones
        if self.config.sharpen.enabled:
            blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0, sigmaY=1.0)
            enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

        return enhanced

    def save_flagged_event_frame(
        self,
        orig_bgr: np.ndarray,
        enh_bgr: Optional[np.ndarray],
        frame_idx: int,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save a high-resolution evidence comparison frame for flagged incidents or high motion.
        
        Args:
            orig_bgr: Original unmodified surveillance frame.
            enh_bgr: Enhanced frame used for inference.
            frame_idx: Frame index in the video stream.
            reason: Short description of the triggering event (e.g. 'phone_detected', 'HIGH_MOTION').
            metadata: Optional dictionary of associated metrics/event details to save into companion JSON.

        Returns:
            Path to the saved evidence image, or None on failure.
        """
        if orig_bgr is None or orig_bgr.size == 0:
            return None
        try:
            out_dir = Path(self.config.flagged_frames_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            enh_img = enh_bgr if (enh_bgr is not None and enh_bgr.size > 0) else orig_bgr
            h, w = orig_bgr.shape[:2]
            scale = min(1.0, 960.0 / w)
            new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
            o_resized = cv2.resize(orig_bgr, (new_w, new_h))
            e_resized = cv2.resize(enh_img, (new_w, new_h))

            # Header banner
            banner_h = 36
            banner = np.zeros((banner_h, new_w * 2, 3), dtype=np.uint8)
            clean_reason = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in reason)
            banner_text = f"NETRA-4D FLAGGED EVIDENCE | Frame {frame_idx:06d} | Event: {clean_reason}"
            cv2.putText(banner, banner_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Frame label overlays
            cv2.putText(o_resized, "RAW SURVEILLANCE FEED", (15, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(e_resized, "ADAPTIVE ENHANCED (YOLO)", (15, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            comp = np.vstack([banner, np.hstack([o_resized, e_resized])])
            target_path = out_dir / f"flagged_frame_{frame_idx:06d}_{clean_reason}.jpg"
            cv2.imwrite(str(target_path), comp)

            if metadata:
                json_path = out_dir / f"flagged_frame_{frame_idx:06d}_{clean_reason}.json"
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump({
                        "frame_idx": frame_idx,
                        "reason": reason,
                        "saved_image": str(target_path),
                        "timestamp": time.time(),
                        **metadata,
                    }, jf, indent=2)

            return target_path
        except Exception as e:
            logger.debug("Failed to save flagged event frame: %s", e)
            return None

    def _save_diagnostic_comparison(
        self,
        orig_bgr: np.ndarray,
        enh_bgr: np.ndarray,
        frame_idx: int,
        report: QualityReport,
    ) -> None:
        """Save side-by-side diagnostic image."""
        try:
            out_dir = Path("artifacts/preprocessing_debug")
            out_dir.mkdir(parents=True, exist_ok=True)
            # Resize if very large
            h, w = orig_bgr.shape[:2]
            scale = min(1.0, 640.0 / w)
            new_w, new_h = int(w * scale), int(h * scale)
            o_resized = cv2.resize(orig_bgr, (new_w, new_h))
            e_resized = cv2.resize(enh_bgr, (new_w, new_h))

            # Put labels
            cv2.putText(o_resized, f"ORIGINAL (score: {report.overall_score:.2f})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            stages_str = "+".join(self.last_enhancement_result.stages_applied) if self.last_enhancement_result else "none"
            cv2.putText(e_resized, f"ENHANCED [{stages_str}]", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            comp = np.hstack([o_resized, e_resized])
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), comp)
        except Exception as e:
            logger.debug("Failed to save debug comparison frame: %s", e)

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of cumulative execution statistics."""
        return {
            "frames_processed": self.stats.frames_processed,
            "frames_enhanced": self.stats.frames_enhanced,
            "frames_bypassed": self.stats.frames_bypassed,
            "bypass_rate": round(self.stats.bypass_rate, 3),
            "avg_assessment_ms": round(self.stats.avg_assessment_ms, 2),
            "avg_enhancement_ms": round(self.stats.avg_enhancement_ms, 2),
            "stage_counts": dict(self.stats.stage_counts),
        }

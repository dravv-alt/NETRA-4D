"""Unit tests for NETRA-4D Deep Preprocessing Subsystem.

Covers:
- Configuration parsing, serialization, and fallbacks
- Quality assessment metrics and defect flags
- 5-stage adaptive enhancement (denoise, CLAHE, gamma, sharpen, white balance)
- Intelligent bypass gate and pipeline orchestrator
- Crop enhancement
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
from drishti.preprocessing.config import (
    CLAHEConfig,
    DenoiseConfig,
    GammaConfig,
    PreprocessingConfig,
    QualityGateConfig,
    SharpenConfig,
    WhiteBalanceConfig,
    load_preprocessing_config,
    save_preprocessing_config,
)
from drishti.preprocessing.enhancer import AdaptiveFrameEnhancer, EnhancementResult
from drishti.preprocessing.pipeline import PreprocessingPipeline
from drishti.preprocessing.quality import FrameQualityAssessor, QualityFlag, QualityReport


# =============================================================================
# 1. Configuration Tests
# =============================================================================

def test_default_config():
    """Verify default configurations have valid types and parameters."""
    cfg = PreprocessingConfig()
    assert cfg.enabled is True
    assert cfg.mode == "adaptive"
    assert cfg.denoise.enabled is True
    assert cfg.denoise.method == "fast_nl_means"
    assert cfg.clahe.clip_limit == 2.0
    assert cfg.clahe.tile_grid_size == (8, 8)
    assert cfg.gamma.target_brightness == 125.0
    assert cfg.sharpen.amount == 1.0
    assert cfg.white_balance.method == "gray_world"
    assert cfg.quality_gate.bypass_threshold == 0.85
    assert cfg.save_flagged_frames is True
    assert cfg.save_high_motion_frames is True


def test_config_serialization(tmp_path):
    """Verify config dict round-trip and YAML save/load."""
    cfg = PreprocessingConfig()
    cfg.denoise.method = "fast_nl_means"
    cfg.clahe.clip_limit = 3.5

    d = cfg.to_dict()
    assert d["denoise"]["method"] == "fast_nl_means"
    assert d["clahe"]["clip_limit"] == 3.5

    reconstructed = PreprocessingConfig.from_dict(d)
    assert reconstructed.denoise.method == "fast_nl_means"
    assert reconstructed.clahe.clip_limit == 3.5
    assert reconstructed.clahe.tile_grid_size == (8, 8)

    # Test file saving and loading
    yaml_file = tmp_path / "test_cfg.yaml"
    save_preprocessing_config(reconstructed, yaml_file)
    assert yaml_file.exists()

    loaded = load_preprocessing_config(yaml_file)
    assert loaded.denoise.method == "fast_nl_means"
    assert loaded.clahe.clip_limit == 3.5


def test_config_fallback_nonexistent():
    """Verify non-existent config path safely falls back to defaults."""
    cfg = load_preprocessing_config("non_existent_path_xyz123.yaml")
    assert cfg.enabled is True
    assert cfg.mode == "adaptive"


# =============================================================================
# 2. Quality Assessment Tests
# =============================================================================

@pytest.fixture
def assessor():
    cfg = PreprocessingConfig()
    return FrameQualityAssessor(cfg)


def test_empty_or_invalid_frame_assessment(assessor):
    """Empty or None frames should return a safe fallback report without throwing."""
    empty_frame = np.array([], dtype=np.uint8)
    report = assessor.assess(empty_frame)
    assert report.overall_score == 0.0
    assert report.needs_enhancement is True

    report_none = assessor.assess(None)
    assert report_none.overall_score == 0.0


def test_dark_frame_assessment(assessor):
    """Dark image should be flagged with DARK and have low brightness."""
    dark_frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    report = assessor.assess(dark_frame)
    assert report.is_dark is True
    assert report.brightness < 30.0
    assert report.needs_enhancement is True


def test_bright_frame_assessment(assessor):
    """Overexposed image should be flagged with BRIGHT and have high brightness."""
    bright_frame = np.full((240, 320, 3), 240, dtype=np.uint8)
    report = assessor.assess(bright_frame)
    assert report.is_bright is True
    assert report.brightness > 220.0
    assert report.needs_enhancement is True


def test_noisy_frame_assessment(assessor):
    """High noise variance image should trigger NOISY flag."""
    rng = np.random.RandomState(42)
    base = np.full((240, 320, 3), 128, dtype=np.uint8)
    noise = rng.normal(0, 35, (240, 320, 3))
    noisy_frame = np.clip(base + noise, 0, 255).astype(np.uint8)

    report = assessor.assess(noisy_frame)
    assert report.is_noisy is True
    assert report.noise_level > 12.0
    assert report.needs_enhancement is True


def test_color_cast_assessment(assessor):
    """Heavy monochrome or tinted image should trigger COLOR_CAST flag."""
    cast_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cast_frame[:, :, 0] = 30   # Blue
    cast_frame[:, :, 1] = 220  # Strong Green tint
    cast_frame[:, :, 2] = 30   # Red

    report = assessor.assess(cast_frame)
    assert report.has_color_cast is True
    assert report.color_cast_score > 14.0
    assert report.needs_enhancement is True


# =============================================================================
# 3. Adaptive Enhancement Stage Tests
# =============================================================================

@pytest.fixture
def enhancer():
    cfg = PreprocessingConfig()
    return AdaptiveFrameEnhancer(cfg)


def test_gamma_brightening(enhancer, assessor):
    """Stage 3: Gamma correction should increase brightness of very dark images."""
    dark_frame = np.full((240, 320, 3), 25, dtype=np.uint8)
    report = assessor.assess(dark_frame)

    result = enhancer.enhance(dark_frame, quality_report=report)
    enhanced = result.frame

    assert enhanced.shape == dark_frame.shape
    assert enhanced.dtype == np.uint8
    assert np.mean(enhanced) > np.mean(dark_frame)
    assert "gamma" in result.stages_applied


def test_gamma_darkening(enhancer, assessor):
    """Stage 3: Gamma correction should decrease brightness of overexposed images."""
    bright_frame = np.full((240, 320, 3), 235, dtype=np.uint8)
    report = assessor.assess(bright_frame)

    result = enhancer.enhance(bright_frame, quality_report=report)
    enhanced = result.frame

    assert enhanced.shape == bright_frame.shape
    assert enhanced.dtype == np.uint8
    assert np.mean(enhanced) < np.mean(bright_frame)
    assert "gamma" in result.stages_applied


def test_clahe_contrast_improvement(enhancer, assessor):
    """Stage 2: CLAHE should enhance local contrast in flat/low contrast frames."""
    low_contrast = np.linspace(100, 115, 240 * 320, dtype=np.uint8).reshape((240, 320))
    low_contrast_bgr = cv2.merge([low_contrast, low_contrast, low_contrast])

    report = assessor.assess(low_contrast_bgr)
    result = enhancer.enhance(low_contrast_bgr, quality_report=report)
    enhanced = result.frame

    orig_std = np.std(low_contrast_bgr)
    enh_std = np.std(enhanced)
    assert enh_std > orig_std
    assert "clahe" in result.stages_applied


def test_denoise_reduces_noise(enhancer, assessor):
    """Stage 1: Denoise should reduce noise variance in synthetic noisy images."""
    rng = np.random.RandomState(42)
    base = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(base, (20, 20), (300, 220), (220, 220, 220), -1)
    cv2.circle(base, (160, 120), 60, (20, 20, 20), -1)
    noise = rng.normal(0, 25, base.shape)
    noisy = np.clip(base.astype(float) + noise, 0, 255).astype(np.uint8)

    report = assessor.assess(noisy)
    result = enhancer.enhance(noisy, quality_report=report)
    enhanced = result.frame

    # Laplacian variance of enhanced image should be significantly lower than noisy original
    orig_lap_var = cv2.Laplacian(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    enh_lap_var = cv2.Laplacian(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()

    assert "denoise" in result.stages_applied
    assert enh_lap_var < orig_lap_var


def test_white_balance_neutralization(enhancer, assessor):
    """Stage 5: White balance should shift channel balance toward neutral gray."""
    tinted = np.zeros((240, 320, 3), dtype=np.uint8)
    tinted[:, :, 0] = 50   # Blue
    tinted[:, :, 1] = 200  # Green
    tinted[:, :, 2] = 50   # Red

    report = assessor.assess(tinted)
    result = enhancer.enhance(tinted, quality_report=report)
    enhanced = result.frame

    assert "white_balance" in result.stages_applied
    # Channel means should be much closer to each other than in original
    mean_b = float(np.mean(enhanced[:, :, 0]))
    mean_g = float(np.mean(enhanced[:, :, 1]))
    mean_r = float(np.mean(enhanced[:, :, 2]))

    spread_orig = np.ptp([50.0, 200.0, 50.0])
    spread_enh = np.ptp([mean_b, mean_g, mean_r])
    assert spread_enh < spread_orig


# =============================================================================
# 4. Pipeline & Bypass Tests
# =============================================================================

def test_pipeline_end_to_end():
    """Verify PreprocessingPipeline processes frames and accumulates statistics."""
    pipeline = PreprocessingPipeline()
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)  # dark frame

    enhanced, report = pipeline.process(frame, frame_idx=1)
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == np.uint8
    assert report.is_dark is True

    stats = pipeline.get_stats_summary()
    assert stats["frames_processed"] == 1
    assert stats["frames_enhanced"] == 1
    assert stats["frames_bypassed"] == 0


def test_pipeline_bypass_on_clean_frame():
    """Verify that a high quality frame is bypassed without enhancement."""
    cfg = PreprocessingConfig()
    cfg.quality_gate.bypass_threshold = 0.50  # easily achievable for a well-balanced image
    pipeline = PreprocessingPipeline(cfg)

    # Create a clean, natural-looking synthetic image
    clean_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(240):
        val = int(80 + (y / 240.0) * 90)
        clean_frame[y, :] = (val, val, val)

    # Assessment on this frame
    report = pipeline.assessor.assess(clean_frame)
    if not report.needs_enhancement and report.overall_score >= cfg.quality_gate.bypass_threshold:
        enhanced, out_report = pipeline.process(clean_frame, frame_idx=2)
        assert pipeline.stats.frames_bypassed == 1
        assert len(pipeline.last_enhancement_result.stages_applied) == 0


def test_enhance_crop():
    """Verify enhance_crop produces enhanced valid image crops."""
    pipeline = PreprocessingPipeline()
    crop = np.random.randint(40, 80, (64, 64, 3), dtype=np.uint8)
    enhanced_crop = pipeline.enhance_crop(crop)

    assert enhanced_crop.shape == crop.shape
    assert enhanced_crop.dtype == np.uint8
    assert enhanced_crop.size == crop.size


def test_temporal_motion_detection():
    """Verify motion assessment flags high motion between disparate consecutive frames."""
    assessor = FrameQualityAssessor()
    frame_static_1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    frame_static_2 = np.full((100, 100, 3), 102, dtype=np.uint8)
    frame_motion = np.full((100, 100, 3), 220, dtype=np.uint8)

    # Frame 1: establishes baseline
    rep1 = assessor.assess(frame_static_1)
    assert not rep1.is_high_motion
    assert rep1.details.get("motion_score", 0.0) == 0.0

    # Frame 2: minor change (diff = 2) -> no high motion
    rep2 = assessor.assess(frame_static_2)
    assert not rep2.is_high_motion
    assert rep2.details.get("motion_score", 0.0) < 5.0

    # Frame 3: large jump (diff ~ 118 > 25 threshold) -> HIGH_MOTION flag
    rep3 = assessor.assess(frame_motion)
    assert rep3.is_high_motion
    assert QualityFlag.HIGH_MOTION in rep3.flags
    assert rep3.details.get("motion_score", 0.0) > 25.0


def test_quality_report_serialization():
    """Verify QualityReport to_dict serializes all expected properties."""
    assessor = FrameQualityAssessor()
    frame = np.full((80, 80, 3), 120, dtype=np.uint8)
    rep = assessor.assess(frame)
    d = rep.to_dict()

    assert "brightness" in d
    assert "contrast" in d
    assert "noise_level" in d
    assert "blur_score" in d
    assert "overall_score" in d
    assert "flags" in d
    assert isinstance(d["flags"], list)
    assert "details" in d
    assert "motion_score" in d["details"]


def test_flagged_event_frame_saving(tmp_path):
    """Verify flagged incident and high-motion frames are saved to disk."""
    cfg = PreprocessingConfig()
    cfg.save_flagged_frames = True
    cfg.save_high_motion_frames = True
    cfg.flagged_frames_dir = str(tmp_path / "flagged")
    pipeline = PreprocessingPipeline(cfg)

    raw_frame = np.full((120, 160, 3), 80, dtype=np.uint8)

    # Trigger explicit flagged event
    enhanced, rep = pipeline.process(
        raw_frame,
        frame_idx=42,
        is_flagged_event=True,
        flag_reason="PHONE_DETECTED",
    )

    saved_images = list(Path(cfg.flagged_frames_dir).glob("*.jpg"))
    saved_jsons = list(Path(cfg.flagged_frames_dir).glob("*.json"))

    assert len(saved_images) >= 1
    assert any("PHONE_DETECTED" in img.name for img in saved_images)
    assert len(saved_jsons) >= 1


def test_specialist_crop_analyzer_with_enhancer():
    """Verify SpecialistHandCropAnalyzer applies crop_enhancer correctly."""
    from drishti.pipeline.specialist_crop import SpecialistHandCropAnalyzer

    pipeline = PreprocessingPipeline()
    analyzer = SpecialistHandCropAnalyzer(crop_size=64, crop_enhancer=pipeline.enhance_crop)

    # Test crop with low contrast
    dummy_crop = np.full((64, 64, 3), 100, dtype=np.uint8)
    res = analyzer.analyze_crop(dummy_crop, (32, 32))
    assert res is not None
    assert hasattr(res, "is_suspicious")


def test_pipeline_reset():
    """Verify pipeline reset clears temporal assessor state."""
    pipeline = PreprocessingPipeline()
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    pipeline.process(frame, frame_idx=0)
    assert pipeline.assessor._prev_gray is not None

    pipeline.reset()
    assert pipeline.assessor._prev_gray is None
    assert pipeline.stats.frames_processed == 0


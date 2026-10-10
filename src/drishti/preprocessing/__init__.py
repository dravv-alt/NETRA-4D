"""NETRA-4D Deep Preprocessing Subsystem.

Provides end-to-end frame quality assessment and 5-stage adaptive enhancement
to optimize low-quality CCTV video streams for downstream YOLO detection.
"""

from __future__ import annotations

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
from drishti.preprocessing.enhancer import (
    AdaptiveFrameEnhancer,
    EnhancementResult,
)
from drishti.preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessingStats,
)
from drishti.preprocessing.quality import (
    FrameQualityAssessor,
    QualityFlag,
    QualityReport,
)

__all__ = [
    "PreprocessingConfig",
    "DenoiseConfig",
    "CLAHEConfig",
    "GammaConfig",
    "SharpenConfig",
    "WhiteBalanceConfig",
    "QualityGateConfig",
    "load_preprocessing_config",
    "save_preprocessing_config",
    "FrameQualityAssessor",
    "QualityFlag",
    "QualityReport",
    "AdaptiveFrameEnhancer",
    "EnhancementResult",
    "PreprocessingPipeline",
    "PreprocessingStats",
]

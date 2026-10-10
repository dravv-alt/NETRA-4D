from pathlib import Path
import pytest

from drishti.release.models import ModelRegistry
from drishti.release.pipeline import CanonicalInferenceService


def test_phone_candidate_cannot_enter_default_release_pipeline():
    fixture = Path(__file__).resolve().parent/'fixtures/model_registry'
    registry = ModelRegistry(fixture/'phone_registry.yaml',fixture)
    with pytest.raises(PermissionError):
        CanonicalInferenceService(registry,device='cpu',phone_role='phone_candidate')

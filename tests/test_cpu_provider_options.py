from __future__ import annotations

from drishti.providers.object_provider import ObjectDisambiguationProvider
from drishti.providers.yolo26_provider import YOLO26PoseProvider


class _PoseModel:
    def __init__(self): self.kwargs = None
    def predict(self, _frame, **kwargs): self.kwargs = kwargs; return []


class _ObjectModel:
    def __init__(self): self.kwargs = None
    def predict(self, _frame, **kwargs): self.kwargs = kwargs; return []


def test_pose_provider_does_not_force_cpu_device():
    provider = YOLO26PoseProvider(device="cpu")
    provider.model = _PoseModel()
    provider.process_frame(0, 0.0, __import__("numpy").zeros((8, 8, 3), dtype="uint8"))
    assert "device" not in provider.model.kwargs


def test_object_provider_does_not_force_cpu_device():
    provider = ObjectDisambiguationProvider(device="cpu")
    model = _ObjectModel()
    assert provider._predict_raw(model, __import__("numpy").zeros((8, 8, 3), dtype="uint8"), .25) == []
    assert "device" not in model.kwargs

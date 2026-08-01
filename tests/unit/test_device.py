"""Tests for explicit CPU/CUDA device selection."""

import pytest
import torch

from people_flow.detection.yolo_detector import validate_device
from people_flow.errors import ModelInitializationError


def test_cpu_device_is_accepted() -> None:
    """CPU should initialize without probing or changing model selection."""

    assert validate_device("cpu") == "cpu"


def test_unavailable_cuda_is_reported_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable requested CUDA device must produce a clear error."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ModelInitializationError, match="is false"):
        validate_device("cuda:0")


def test_unknown_device_is_rejected() -> None:
    """Unsupported accelerators should not be reinterpreted as CPU."""

    with pytest.raises(ModelInitializationError, match="Unsupported device"):
        validate_device("magic")

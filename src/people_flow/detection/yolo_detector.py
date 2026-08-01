"""Ultralytics YOLO detector and tracking adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from people_flow.datasets.video_source import Frame
from people_flow.errors import ModelInitializationError


def validate_device(device: str) -> str:
    """Validate CPU/CUDA availability without silently changing the requested device."""

    normalized = device.strip().lower()
    if normalized == "cpu":
        return normalized
    if normalized == "cuda" or normalized.startswith("cuda:") or normalized.isdigit():
        try:
            import torch
        except ImportError as exc:
            raise ModelInitializationError(
                f"CUDA device '{device}' was requested but PyTorch is unavailable"
            ) from exc
        if not torch.cuda.is_available():
            raise ModelInitializationError(
                f"CUDA device '{device}' was requested but torch.cuda.is_available() is false"
            )
        return normalized
    raise ModelInitializationError(
        f"Unsupported device '{device}'. Use 'cpu', 'cuda', 'cuda:N', or a CUDA index."
    )


class YoloDetector:
    """Load exactly the requested YOLO model and expose persistent tracking inference."""

    def __init__(self, model_path: Path, *, device: str) -> None:
        self.model_path = model_path.expanduser().resolve()
        self.device = validate_device(device)
        if not self.model_path.is_file():
            raise ModelInitializationError(f"Model file does not exist: {self.model_path}")
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self._model: Any = YOLO(str(self.model_path))
        except Exception as exc:
            raise ModelInitializationError(
                f"Unable to initialize requested model {self.model_path}: {exc}"
            ) from exc

    def track(
        self,
        frame: Frame,
        *,
        tracker: str,
        classes: Sequence[int],
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> Any:
        """Track people in one frame while preserving model state across calls."""

        try:
            return self._model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                classes=list(classes),
                conf=confidence,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise ModelInitializationError(
                f"Tracking inference failed for model {self.model_path}: {exc}"
            ) from exc

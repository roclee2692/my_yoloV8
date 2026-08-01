"""Deterministic runtime-metric calculation for video runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Required runtime metrics emitted for every completed pipeline run."""

    model: str
    tracker: str
    device: str
    input_frames: int
    processed_frames: int
    video_fps: float
    processing_fps: float
    average_latency_ms: float
    p95_latency_ms: float
    peak_gpu_memory_mb: float
    total_runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return asdict(self)


def percentile_95(values: list[float]) -> float:
    """Return the nearest-rank 95th percentile, or zero for an empty collection."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def reset_peak_gpu_memory(device: str) -> None:
    """Reset CUDA peak memory accounting only for an explicitly requested CUDA device."""

    if device == "cpu":
        return
    import torch

    torch.cuda.reset_peak_memory_stats()


def read_peak_gpu_memory_mb(device: str) -> float:
    """Return peak CUDA memory or 0.0 when the run explicitly used CPU."""

    if device == "cpu":
        return 0.0
    import torch

    return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)


def build_runtime_metrics(
    *,
    model: str,
    tracker: str,
    device: str,
    input_frames: int,
    processed_frames: int,
    video_fps: float,
    latencies_ms: list[float],
    peak_gpu_memory_mb: float,
    total_runtime_seconds: float,
) -> RuntimeMetrics:
    """Build metrics with safe zero-frame behavior."""

    processing_fps = processed_frames / total_runtime_seconds if total_runtime_seconds > 0 else 0.0
    average_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    return RuntimeMetrics(
        model=model,
        tracker=tracker,
        device=device,
        input_frames=input_frames,
        processed_frames=processed_frames,
        video_fps=video_fps,
        processing_fps=processing_fps,
        average_latency_ms=average_latency_ms,
        p95_latency_ms=percentile_95(latencies_ms),
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        total_runtime_seconds=total_runtime_seconds,
    )

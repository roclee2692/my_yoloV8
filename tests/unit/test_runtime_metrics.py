"""Tests for required runtime metrics and safe empty behavior."""

from people_flow.evaluation.runtime_metrics import build_runtime_metrics, percentile_95


def test_percentile_95_uses_nearest_rank() -> None:
    """The p95 calculation should be deterministic for small runs."""

    assert percentile_95([1.0, 4.0, 2.0, 3.0]) == 4.0
    assert percentile_95([]) == 0.0


def test_runtime_metrics_handle_zero_runtime() -> None:
    """Empty or zero-duration runs must not divide by zero."""

    metrics = build_runtime_metrics(
        model="yolo26n.pt",
        tracker="bytetrack.yaml",
        device="cpu",
        input_frames=0,
        processed_frames=0,
        video_fps=25.0,
        latencies_ms=[],
        peak_gpu_memory_mb=0.0,
        total_runtime_seconds=0.0,
    )

    assert metrics.processing_fps == 0.0
    assert metrics.average_latency_ms == 0.0
    assert metrics.p95_latency_ms == 0.0

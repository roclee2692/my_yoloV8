"""Shared model-free test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def report_run(tmp_path: Path) -> Path:
    """Create one complete structured run without raw media dependencies."""

    run_dir = tmp_path / "structured_run"
    run_dir.mkdir()
    summary = {
        "roi_name": "entrance_area",
        "total_enter": 5,
        "total_exit": 1,
        "maximum_occupancy": 2,
        "average_occupancy": 0.744,
        "peak_time": 200.0,
        "peak_frame_id": 11,
        "average_dwell_seconds": 1.27,
        "median_dwell_seconds": 0.77,
        "processed_frames": 250,
        "processing_fps": 17.98,
        "tracks_with_dwell": 4,
        "inside_at_end": 2,
    }
    runtime = {
        "model": "yolo26n.pt",
        "tracker": "botsort.yaml",
        "device": "0",
        "input_frames": 250,
        "processed_frames": 250,
        "video_fps": 50.0,
        "processing_fps": 17.98,
        "average_latency_ms": 52.7,
        "p95_latency_ms": 65.4,
        "peak_gpu_memory_mb": 85.3,
        "total_runtime_seconds": 13.9,
    }
    evaluation = {
        "ground_truth_available": False,
        "reason": "No associated MOT Ground Truth.",
        "ground_truth_policy": None,
        "tracking_iou_threshold": 0.5,
        "ground_truth_enter": None,
        "ground_truth_exit": None,
        "ground_truth_total": None,
        "predicted_enter": 0,
        "predicted_exit": 1,
        "predicted_total": 1,
        "id_switches": None,
        "enter_absolute_error": None,
        "exit_absolute_error": None,
        "total_count_absolute_error": None,
        "count_mae": None,
        "count_mape": None,
        "occupancy_mae": None,
    }
    for name, payload in (
        ("summary.json", summary),
        ("runtime_metrics.json", runtime),
        ("evaluation.json", evaluation),
    ):
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "annotated.mp4").write_bytes(b"must-not-be-read")
    (run_dir / "tracks.csv").write_text("must,not,be,read\n", encoding="utf-8")
    return run_dir

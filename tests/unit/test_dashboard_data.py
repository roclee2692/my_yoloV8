"""Tests for source-backed Dashboard artifact loading."""

from __future__ import annotations

import csv
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from people_flow.dashboard.data import (
    build_results_archive,
    discover_completed_runs,
    load_run_dashboard,
)
from people_flow.errors import DashboardError


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_completed_run(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "annotated.mp4").write_bytes(b"dashboard-video")
    (path / "run.log").write_text("completed\n", encoding="utf-8")
    (path / "run_config.yaml").write_text("model: yolo26n.pt\n", encoding="utf-8")
    _write_csv(
        path / "tracks.csv",
        ("frame_id", "track_id"),
        [{"frame_id": 1, "track_id": 7}],
    )
    _write_csv(
        path / "events.csv",
        ("frame_id", "event_type"),
        [
            {"frame_id": 2, "event_type": "enter"},
            {"frame_id": 3, "event_type": "exit"},
        ],
    )
    _write_csv(
        path / "occupancy.csv",
        ("frame_id", "timestamp_ms", "occupancy"),
        [
            {"frame_id": 1, "timestamp_ms": 0.0, "occupancy": 0},
            {"frame_id": 2, "timestamp_ms": 100.0, "occupancy": 1},
            {"frame_id": 3, "timestamp_ms": 200.0, "occupancy": 0},
        ],
    )
    _write_csv(
        path / "dwell_times.csv",
        ("track_id", "total_dwell_seconds"),
        [
            {"track_id": 7, "total_dwell_seconds": 0.5},
            {"track_id": 8, "total_dwell_seconds": 4.0},
        ],
    )
    (path / "runtime_metrics.json").write_text(
        json.dumps({"processed_frames": 3, "processing_fps": 12.5}),
        encoding="utf-8",
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "processed_frames": 3,
                "maximum_occupancy": 1,
                "peak_time": 100.0,
            }
        ),
        encoding="utf-8",
    )
    (path / "evaluation.json").write_text(
        json.dumps(
            {
                "ground_truth_available": False,
                "reason": "No associated MOT Ground Truth.",
            }
        ),
        encoding="utf-8",
    )


def test_load_run_dashboard_reconciles_metrics_and_series(tmp_path: Path) -> None:
    """The Dashboard should derive KPIs only from validated persisted artifacts."""

    run_dir = tmp_path / "runs" / "demo"
    _create_completed_run(run_dir)

    data = load_run_dashboard(run_dir)

    assert data.processed_frames == 3
    assert data.current_occupancy == 0
    assert data.total_enter == 1
    assert data.total_exit == 1
    assert data.maximum_occupancy == 1
    assert data.peak_time_ms == 100.0
    assert data.processing_fps == 12.5
    assert data.ground_truth_available is False
    assert data.ground_truth_message == "No associated MOT Ground Truth."
    assert data.flow_series()[-1] == {
        "frame_id": 3,
        "timestamp_seconds": 0.2,
        "occupancy": 0,
        "cumulative_enter": 1,
        "cumulative_exit": 1,
    }
    assert data.dwell_histogram() == {
        "<1s": 1,
        "1–3s": 0,
        "3–5s": 1,
        "5–10s": 0,
        "≥10s": 0,
    }


def test_discovery_and_archive_require_completed_source_artifacts(tmp_path: Path) -> None:
    """Incomplete folders stay hidden and ZIPs contain only the selected run."""

    completed = tmp_path / "runs" / "completed"
    _create_completed_run(completed)
    incomplete = tmp_path / "runs" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "runtime_metrics.json").write_text("{}", encoding="utf-8")

    assert discover_completed_runs(tmp_path / "runs") == (completed.resolve(),)
    archive = build_results_archive(load_run_dashboard(completed))
    with zipfile.ZipFile(BytesIO(archive)) as zipped:
        assert "runtime_metrics.json" in zipped.namelist()
        assert "../runtime_metrics.json" not in zipped.namelist()


def test_load_run_dashboard_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    """Contradictory runtime and occupancy evidence must not be rendered."""

    run_dir = tmp_path / "run"
    _create_completed_run(run_dir)
    payload = {"processed_frames": 2, "processing_fps": 1.0}
    (run_dir / "runtime_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DashboardError, match="Occupancy row count"):
        load_run_dashboard(run_dir)

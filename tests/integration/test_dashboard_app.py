"""Model-free Streamlit render test for the Phase 10 Dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_render_fixture(root: Path) -> None:
    run_dir = root / "runs" / "render_fixture"
    run_dir.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(run_dir / "annotated.mp4"),
        cv2.VideoWriter.fourcc(*"mp4v"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    writer.write(np.full((24, 32, 3), 80, dtype=np.uint8))
    writer.release()
    (run_dir / "run.log").write_text("completed\n", encoding="utf-8")
    (run_dir / "run_config.yaml").write_text("model: yolo26n.pt\n", encoding="utf-8")
    _write_csv(run_dir / "tracks.csv", ("frame_id", "track_id"), [])
    _write_csv(
        run_dir / "events.csv",
        ("frame_id", "event_type"),
        [{"frame_id": 1, "event_type": "enter"}],
    )
    _write_csv(
        run_dir / "occupancy.csv",
        ("frame_id", "timestamp_ms", "occupancy"),
        [{"frame_id": 1, "timestamp_ms": 0.0, "occupancy": 1}],
    )
    _write_csv(
        run_dir / "dwell_times.csv",
        ("track_id", "total_dwell_seconds"),
        [{"track_id": 1, "total_dwell_seconds": 0.5}],
    )
    (run_dir / "runtime_metrics.json").write_text(
        json.dumps({"processed_frames": 1, "processing_fps": 25.0}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"processed_frames": 1, "maximum_occupancy": 1, "peak_time": 0.0}),
        encoding="utf-8",
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps({"ground_truth_available": False}),
        encoding="utf-8",
    )


def test_dashboard_renders_persisted_results_and_gt_warning(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """A local app render should expose source-backed KPIs and the no-GT caveat."""

    _create_render_fixture(tmp_path)
    monkeypatch.setenv("PEOPLE_FLOW_PROJECT_ROOT", str(tmp_path))  # type: ignore[attr-defined]
    app_path = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    app.selectbox[0].select("yolov8n.pt").run()

    assert not app.exception
    assert app.title[0].value == "基于 YOLO26 与多目标跟踪的人流分析系统"
    assert any("Ground Truth" in warning.value for warning in app.warning)
    assert [metric.label for metric in app.metric] == [
        "当前 ROI 人数",
        "计数线进入",
        "计数线离开",
        "峰值人数",
        "峰值时刻",
        "处理 FPS",
    ]
    assert [metric.value for metric in app.metric[:4]] == ["1", "1", "0", "1"]

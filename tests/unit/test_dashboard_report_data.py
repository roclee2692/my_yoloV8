"""Dashboard integration with validated Phase 11 report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from people_flow.dashboard.data import load_run_dashboard
from people_flow.errors import DashboardError
from people_flow.reporting.service import generate_report


def _complete_dashboard_artifacts(run_dir: Path) -> None:
    (run_dir / "annotated.mp4").write_bytes(b"video")
    (run_dir / "tracks.csv").write_text("frame_id,track_id\n", encoding="utf-8")
    (run_dir / "run_config.yaml").write_text("model: yolo26n.pt\n", encoding="utf-8")
    (run_dir / "run.log").write_text("complete\n", encoding="utf-8")


def test_dashboard_loads_report_and_generation_metadata(report_run: Path) -> None:
    """The UI should expose only a report accompanied by safety metadata."""

    _complete_dashboard_artifacts(report_run)
    generate_report(report_run)

    data = load_run_dashboard(report_run)

    assert data.report_markdown is not None
    assert data.report_markdown.startswith("# 固定摄像头人流分析报告")
    assert data.report_metadata is not None
    assert data.report_metadata["llm_used"] is False
    assert data.report_metadata["raw_video_read"] is False


def test_dashboard_rejects_unpaired_report_file(report_run: Path) -> None:
    """A hand-written report without provenance metadata must not be trusted."""

    _complete_dashboard_artifacts(report_run)
    (report_run / "report.md").write_text("untrusted", encoding="utf-8")

    with pytest.raises(DashboardError, match="must exist together"):
        load_run_dashboard(report_run)


def test_dashboard_rejects_report_after_evidence_changes(report_run: Path) -> None:
    """A report fingerprint must continue to match its three source JSON objects."""

    _complete_dashboard_artifacts(report_run)
    generate_report(report_run)
    summary_path = report_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["total_enter"] = 6
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(DashboardError, match="fingerprint does not match"):
        load_run_dashboard(report_run)

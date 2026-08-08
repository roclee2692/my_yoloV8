"""Tests for deterministic Markdown metrics and mandatory caveats."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from people_flow.reporting.evidence import load_report_evidence
from people_flow.reporting.narrative import deterministic_narrative
from people_flow.reporting.renderer import (
    NO_GROUND_TRUTH_NOTICE,
    ReportProvenance,
    render_markdown_report,
)


def test_renderer_inserts_exact_metrics_and_no_gt_notice(report_run: Path) -> None:
    """Numeric claims must come from evidence and unknown errors remain unavailable."""

    evidence = load_report_evidence(report_run)
    report = render_markdown_report(
        evidence,
        deterministic_narrative(evidence),
        ReportProvenance(
            provider="deterministic",
            model=None,
            generated_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
            evidence_fingerprint=evidence.fingerprint,
        ),
    )

    assert report.startswith("# 固定摄像头人流分析报告\n\n## Executive Summary")
    assert NO_GROUND_TRUTH_NOTICE in report
    assert "| 模型 | yolo26n.pt |" in report
    assert "| 处理 FPS | 17.980 |" in report
    assert "| 峰值人数 | 2 |" in report
    assert "| 平均停留时间 | 1.270 秒 |" in report
    assert "不能将这些空值解释为零误差" in report
    assert "evidence_sha256=" in report

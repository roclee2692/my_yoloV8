"""Tests for the bounded Phase 11 evidence contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from people_flow.errors import ReportGenerationError
from people_flow.reporting.evidence import load_report_evidence


def test_report_evidence_loads_only_three_reconciled_objects(report_run: Path) -> None:
    """A complete run should produce stable, bounded prompt evidence."""

    evidence = load_report_evidence(report_run)

    assert set(evidence.prompt_payload()) == {"summary", "runtime_metrics", "evaluation"}
    assert evidence.runtime_metrics.processed_frames == 250
    assert evidence.evaluation.ground_truth_available is False
    assert len(evidence.fingerprint) == 64
    assert evidence.fingerprint == load_report_evidence(report_run).fingerprint


def test_report_evidence_rejects_false_zero_accuracy(report_run: Path) -> None:
    """Unavailable Ground Truth metrics cannot be silently written as zero."""

    path = report_run / "evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["count_mae"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReportGenerationError, match="must be null"):
        load_report_evidence(report_run)


def test_report_evidence_rejects_cross_file_frame_mismatch(report_run: Path) -> None:
    """Contradictory summaries must fail before a prompt can be built."""

    path = report_run / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["processed_frames"] = 249
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReportGenerationError, match="frame counts do not match"):
        load_report_evidence(report_run)


def test_report_evidence_requires_all_three_json_files(tmp_path: Path) -> None:
    """Raw video or CSV files cannot substitute for structured evidence."""

    tmp_path.joinpath("annotated.mp4").write_bytes(b"video")
    with pytest.raises(ReportGenerationError, match="missing"):
        load_report_evidence(tmp_path)

"""End-to-end model-free tests for Phase 11 report outputs."""

from __future__ import annotations

import json
from pathlib import Path

from people_flow.reporting.cli import main
from people_flow.reporting.renderer import NO_GROUND_TRUTH_NOTICE


def test_deterministic_report_cli_writes_honest_outputs(report_run: Path) -> None:
    """Offline CI should produce an explicitly non-LLM report and metadata."""

    assert main(["--run-dir", str(report_run)]) == 0

    report_path = report_run / "report.md"
    metadata_path = report_run / "report_metadata.json"
    assert report_path.is_file()
    assert metadata_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert NO_GROUND_TRUTH_NOTICE in report
    assert metadata["provider"] == "deterministic"
    assert metadata["llm_used"] is False
    assert metadata["raw_video_read"] is False
    assert metadata["input_files"] == [
        "summary.json",
        "runtime_metrics.json",
        "evaluation.json",
    ]
    assert main(["--run-dir", str(report_run)]) == 2
    assert main(["--run-dir", str(report_run), "--overwrite"]) == 0


def test_openai_provider_requires_explicit_connection_settings(report_run: Path) -> None:
    """The CLI cannot silently select a remote model or endpoint."""

    assert (
        main(
            [
                "--run-dir",
                str(report_run),
                "--provider",
                "openai-compatible",
            ]
        )
        == 2
    )

"""Integration of Phase 11 with the installed people-flow command."""

from __future__ import annotations

from pathlib import Path

from people_flow.cli import main


def test_people_flow_report_subcommand(report_run: Path) -> None:
    """The primary CLI should expose the same model-free report service."""

    report = report_run / "custom_report.md"
    metadata = report_run / "custom_report_metadata.json"

    exit_code = main(
        [
            "report",
            "--run-dir",
            str(report_run),
            "--output",
            str(report),
            "--metadata-output",
            str(metadata),
        ]
    )

    assert exit_code == 0
    assert report.is_file()
    assert metadata.is_file()

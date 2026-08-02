"""Tests for model-free CLI behavior through Phase 3."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from people_flow.config import RunConfig


def test_validate_config_command_succeeds() -> None:
    """The CLI should validate the checked-in base configuration."""

    from people_flow.cli import main

    assert main(["validate-config", "--config", "configs/base.yaml"]) == 0


def test_validate_config_command_reports_missing_file() -> None:
    """The CLI should return a stable non-zero code for missing configuration."""

    from people_flow.cli import main

    assert main(["validate-config", "--config", "missing.yaml"]) == 2


def test_run_command_rejects_nonperson_classes() -> None:
    """The CLI should reject classes other than COCO person before loading a model."""

    from people_flow.cli import main

    assert (
        main(
            [
                "run",
                "--source",
                "missing.mp4",
                "--classes",
                "1",
                "--output-dir",
                "runs/test",
            ]
        )
        == 2
    )


def test_run_command_builds_directional_counting_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI line coordinates should enable typed counting without model initialization."""

    from people_flow.cli import main

    captured: list[RunConfig] = []

    def fake_run(config: RunConfig) -> SimpleNamespace:
        captured.append(config)
        return SimpleNamespace(output_dir=Path("runs/counting-test"))

    monkeypatch.setattr("people_flow.cli.run_pipeline", fake_run)
    exit_code = main(
        [
            "run",
            "--source",
            "input.mp4",
            "--output-dir",
            "runs/counting-test",
            "--counting-line",
            "0",
            "12",
            "31",
            "12",
            "--enter-side",
            "negative",
            "--min-track-age",
            "3",
            "--min-displacement-pixels",
            "8",
            "--cooldown-frames",
            "10",
            "--max-track-gap-frames",
            "4",
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    counting = captured[0].counting
    assert counting.enabled is True
    assert counting.line is not None
    assert counting.line.p1 == (0.0, 12.0)
    assert counting.line.p2 == (31.0, 12.0)
    assert counting.line.enter_side == "negative"
    assert counting.min_track_age == 3


def test_run_command_builds_polygon_roi_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated CLI vertices should enable a validated polygon ROI."""

    from people_flow.cli import main

    captured: list[RunConfig] = []

    def fake_run(config: RunConfig) -> SimpleNamespace:
        captured.append(config)
        return SimpleNamespace(output_dir=Path("runs/roi-test"))

    monkeypatch.setattr("people_flow.cli.run_pipeline", fake_run)
    exit_code = main(
        [
            "run",
            "--source",
            "input.mp4",
            "--output-dir",
            "runs/roi-test",
            "--roi-name",
            "entrance_area",
            "--roi-point",
            "0",
            "0",
            "--roi-point",
            "20",
            "0",
            "--roi-point",
            "20",
            "20",
            "--roi-point",
            "0",
            "20",
            "--roi-max-track-gap-frames",
            "4",
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    roi = captured[0].roi
    assert roi.enabled is True
    assert roi.name == "entrance_area"
    assert roi.points == ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))
    assert roi.max_track_gap_frames == 4

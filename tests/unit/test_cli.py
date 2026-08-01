"""Tests for model-free CLI behavior through Phase 3."""


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

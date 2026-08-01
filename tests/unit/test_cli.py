"""Tests for the model-free Phase 2 CLI."""

def test_validate_config_command_succeeds() -> None:
    """The CLI should validate the checked-in base configuration."""

    from people_flow.cli import main

    assert main(["validate-config", "--config", "configs/base.yaml"]) == 0


def test_validate_config_command_reports_missing_file() -> None:
    """The CLI should return a stable non-zero code for missing configuration."""

    from people_flow.cli import main

    assert main(["validate-config", "--config", "missing.yaml"]) == 2

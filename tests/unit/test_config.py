"""Tests for the implemented Phase 2 configuration layer."""

from pathlib import Path

import pytest

from people_flow.config import CountingSettings, load_config, resolve_path
from people_flow.errors import ConfigurationError


def test_load_base_config_uses_pathlib() -> None:
    """The repository base config should validate into typed Path values."""

    config = load_config(Path("configs/base.yaml"))

    assert config.project_name == "people-flow-analytics"
    assert config.seed == 42
    assert isinstance(config.paths.data_dir, Path)
    assert config.runtime.device == "cpu"


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    """Undocumented YAML keys should fail schema validation."""

    config_path = tmp_path / "unknown.yaml"
    config_path.write_text(
        """schema_version: 1
project_name: test
seed: 42
paths:
  data_dir: data
  runs_dir: runs
  weights_dir: weights
runtime:
  device: cpu
  workers: 0
  log_level: INFO
unexpected: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="schema validation failed"):
        load_config(config_path)


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    """A missing configuration should produce a clear project exception."""

    missing_path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(missing_path)


def test_resolve_path_uses_explicit_base_directory(tmp_path: Path) -> None:
    """Relative paths should be resolved without machine-specific constants."""

    resolved = resolve_path(Path("data/samples"), base_dir=tmp_path)

    assert resolved == (tmp_path / "data" / "samples").resolve()


def test_counting_requires_line_when_enabled() -> None:
    """Schema validation should prevent an enabled counter without geometry."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="counting.line is required"):
        CountingSettings(enabled=True)

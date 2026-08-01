"""Typed configuration loading and path helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from people_flow.errors import ConfigurationError

TrackerName = Literal["bytetrack.yaml", "botsort.yaml"]


class StrictConfigModel(BaseModel):
    """Base model that rejects undocumented configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathSettings(StrictConfigModel):
    """Repository-relative data, output, and weight locations."""

    data_dir: Path
    runs_dir: Path
    weights_dir: Path


class RuntimeSettings(StrictConfigModel):
    """Runtime settings that do not initialize a model or device."""

    device: str = Field(min_length=1)
    workers: int = Field(ge=0, le=64)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppConfig(StrictConfigModel):
    """Validated Phase 2 application configuration."""

    schema_version: Literal[1]
    project_name: str = Field(min_length=1)
    seed: int = Field(ge=0, le=4_294_967_295)
    paths: PathSettings
    runtime: RuntimeSettings

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, value: str) -> str:
        """Reject names containing only whitespace."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("project_name must not be blank")
        return normalized


class RunConfig(StrictConfigModel):
    """Validated arguments for a Phase 3 video tracking run."""

    source: Path
    model: str = Field(min_length=1)
    tracker: TrackerName
    classes: tuple[int, ...] = (0,)
    output_dir: Path
    weights_dir: Path = Path("weights")
    device: str = Field(default="cpu", min_length=1)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=32, le=4096)
    overwrite: bool = False

    @field_validator("classes")
    @classmethod
    def validate_person_only(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Keep the Phase 3 pipeline strictly limited to the COCO person class."""

        if value != (0,):
            raise ValueError("People Flow only supports COCO person class 0")
        return value


def resolve_path(value: Path, *, base_dir: Path) -> Path:
    """Resolve a user-provided path against an explicit base directory."""

    candidate = value.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir.expanduser().resolve() / candidate).resolve()


def load_config(config_path: Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to a YAML file. Relative paths are resolved from the current directory.

    Raises:
        ConfigurationError: If the file is missing, unreadable, malformed, or violates the schema.
    """

    resolved_path = resolve_path(config_path, base_dir=Path.cwd())
    if not resolved_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {resolved_path}")

    try:
        raw_data: Any = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration: {resolved_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration: {resolved_path}") from exc

    if not isinstance(raw_data, Mapping):
        raise ConfigurationError(f"Configuration root must be a mapping: {resolved_path}")

    try:
        return AppConfig.model_validate(dict(raw_data))
    except ValidationError as exc:
        message = f"Configuration schema validation failed: {resolved_path}\n{exc}"
        raise ConfigurationError(message) from exc

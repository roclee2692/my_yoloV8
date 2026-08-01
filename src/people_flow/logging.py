"""Logging configuration for CLI and future pipeline modules."""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(*, level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure deterministic console logging and an optional UTF-8 log file."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        resolved_log_file = log_file.expanduser().resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(resolved_log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("people_flow")

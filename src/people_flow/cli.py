"""Command-line entry point for the Phase 2 scaffold."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from people_flow import __version__
from people_flow.config import load_config
from people_flow.errors import ConfigurationError
from people_flow.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without initializing models or hardware."""

    parser = argparse.ArgumentParser(
        prog="people-flow",
        description="People Flow Analytics command line interface",
    )
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="validate a Phase 2 base configuration",
    )
    validate_parser.add_argument("--config", required=True, type=Path)
    subparsers.add_parser("version", help="show the package version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logger = configure_logging()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "version":
        logger.info("people-flow-analytics %s", __version__)
        return 0

    config_path = cast(Path, args.config)
    try:
        config = load_config(config_path)
    except ConfigurationError as exc:
        logger.error("Configuration validation failed: %s", exc)
        return 2

    logger.info(
        "Configuration is valid: project=%s schema_version=%s",
        config.project_name,
        config.schema_version,
    )
    return 0


def entrypoint() -> None:
    """Console-script wrapper that preserves the CLI exit code."""

    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()

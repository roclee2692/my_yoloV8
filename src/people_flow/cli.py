"""Command-line entry point for the Phase 2 scaffold."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from people_flow import __version__
from people_flow.config import RunConfig, TrackerName, load_config
from people_flow.errors import ConfigurationError, PeopleFlowError
from people_flow.logging import configure_logging
from people_flow.pipeline import run_pipeline


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

    run_parser = subparsers.add_parser("run", help="run person detection and tracking")
    run_parser.add_argument("--source", required=True, type=Path)
    run_parser.add_argument("--model", default="yolo26n.pt")
    run_parser.add_argument(
        "--tracker",
        choices=("bytetrack.yaml", "botsort.yaml"),
        default="bytetrack.yaml",
    )
    run_parser.add_argument("--classes", nargs="+", type=int, default=[0])
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--confidence", type=float, default=0.25)
    run_parser.add_argument("--iou", type=float, default=0.7)
    run_parser.add_argument("--imgsz", type=int, default=640)
    run_parser.add_argument("--overwrite", action="store_true")
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

    if args.command == "run":
        try:
            run_config = RunConfig(
                source=cast(Path, args.source),
                model=cast(str, args.model),
                tracker=cast(TrackerName, args.tracker),
                classes=tuple(cast(list[int], args.classes)),
                output_dir=cast(Path, args.output_dir),
                weights_dir=cast(Path, args.weights_dir),
                device=cast(str, args.device),
                confidence=cast(float, args.confidence),
                iou=cast(float, args.iou),
                imgsz=cast(int, args.imgsz),
                overwrite=cast(bool, args.overwrite),
            )
            result = run_pipeline(run_config)
        except (ValidationError, PeopleFlowError) as exc:
            logger.error("Run failed: %s", exc)
            return 2
        logger.info("Artifacts written to %s", result.output_dir)
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

"""Command-line interface for controlled Phase 8 experiment suites."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from people_flow.errors import PeopleFlowError
from people_flow.evaluation.experiment_runner import load_experiment_suite, run_experiment_suite
from people_flow.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline-suite argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="base directory for paths declared in the suite YAML",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a validated experiment suite and return a process exit code."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logger = configure_logging(level="INFO")
    try:
        config = load_experiment_suite(cast(Path, args.config))
        result = run_experiment_suite(
            config,
            base_dir=cast(Path, args.project_root),
            overwrite=cast(bool, args.overwrite),
        )
    except (ValidationError, PeopleFlowError) as exc:
        logger.error("Baseline suite failed: %s", exc)
        return 2
    logger.info(
        "Baseline suite complete: experiments=%s comparison=%s",
        len(result.rows),
        result.comparison_csv,
    )
    return 0

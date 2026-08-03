"""Assess whether current baseline evidence permits Phase 9 training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from people_flow.errors import PeopleFlowError
from people_flow.evaluation.training_readiness import assess_training_readiness
from people_flow.logging import configure_logging
from people_flow.outputs.json_writer import write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the training-readiness CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("runs/experiments/comparison.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/training/phase9_readiness.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write a structured decision and return zero for a valid deferred outcome."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logger = configure_logging(level="INFO")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        logger.error("Readiness output already exists: %s. Use --overwrite.", output)
        return 2
    try:
        decision = assess_training_readiness(args.comparison)
        write_json(output, decision.as_dict())
    except PeopleFlowError as exc:
        logger.error("Training-readiness assessment failed: %s", exc)
        return 2
    logger.info(
        "Training-readiness decision: %s training_allowed=%s output=%s",
        decision.decision,
        decision.training_allowed,
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

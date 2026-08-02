"""Validate a locally downloaded MOT17 split and write an integrity summary."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from people_flow.datasets.mot_reader import validate_mot17_split
from people_flow.errors import PeopleFlowError
from people_flow.logging import configure_logging
from people_flow.outputs.json_writer import write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the local MOT17 validation CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/MOT17"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Validate selected sequences without claiming the official split is complete",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Validation JSON path (default: data/interim/mot17_<split>_validation.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a split without downloading, moving, or modifying raw data."""

    args = build_parser().parse_args(argv)
    logger = configure_logging()
    output = (
        args.output
        if args.output is not None
        else Path("data/interim") / f"mot17_{args.split}_validation.json"
    )
    try:
        summary = validate_mot17_split(args.root, args.split, allow_subset=args.allow_subset)
        written_path = write_json(output, summary)
    except (PeopleFlowError, OSError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    logger.info(
        "Validated MOT17 %s: %d sequences, %d frames; summary=%s",
        args.split,
        summary["sequence_count"],
        summary["total_frames"],
        written_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

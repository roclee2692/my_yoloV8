"""Convert a validated MOTChallenge image sequence to a frame-complete MP4."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from people_flow.datasets.mot_converter import convert_mot_sequence_to_video
from people_flow.errors import PeopleFlowError
from people_flow.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the MOT sequence conversion CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="MOT sequence directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination .mp4 path")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path (default: output name with .summary.json)",
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC codec (default: mp4v)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run conversion and report an explicit nonzero exit code on failure."""

    args = build_parser().parse_args(argv)
    logger = configure_logging()
    try:
        result = convert_mot_sequence_to_video(
            args.sequence,
            args.output,
            summary_path=args.summary,
            codec=args.codec,
            overwrite=args.overwrite,
        )
    except (PeopleFlowError, OSError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    logger.info(
        "Converted %s: %d input frames -> %d verified MP4 frames at %.3f FPS",
        result.sequence_name,
        result.input_frames,
        result.output_frames,
        result.frame_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

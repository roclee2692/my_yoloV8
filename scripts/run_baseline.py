"""Convenience wrapper for one Phase 3 baseline run."""

from __future__ import annotations

import argparse
from pathlib import Path

from people_flow.cli import main as cli_main


def build_parser() -> argparse.ArgumentParser:
    """Build a small wrapper around the canonical people-flow CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/samples/People-counting-compressed.mp4"),
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument(
        "--tracker",
        choices=("bytetrack.yaml", "botsort.yaml"),
        default="bytetrack.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    """Delegate to the canonical CLI without duplicating pipeline logic."""

    args = build_parser().parse_args()
    argv = [
        "run",
        "--source",
        str(args.source),
        "--model",
        args.model,
        "--tracker",
        args.tracker,
        "--classes",
        "0",
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.overwrite:
        argv.append("--overwrite")
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for structured Phase 11 report generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from people_flow.errors import PeopleFlowError
from people_flow.logging import configure_logging
from people_flow.reporting.service import ProviderName, generate_report


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the shared report-generation arguments to a parser."""

    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--provider",
        choices=("deterministic", "openai-compatible"),
        default="deterministic",
        help="deterministic is an explicitly non-LLM offline fallback",
    )
    parser.add_argument("--endpoint", help="explicit chat-completions HTTP(S) endpoint")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="PEOPLE_FLOW_LLM_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone report CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_report_arguments(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    """Generate one report from already parsed arguments."""

    logger = configure_logging(level="INFO")
    try:
        result = generate_report(
            cast(Path, args.run_dir),
            output=cast(Path | None, args.output),
            metadata_output=cast(Path | None, args.metadata_output),
            provider=cast(ProviderName, args.provider),
            endpoint=cast(str | None, args.endpoint),
            model=cast(str | None, args.model),
            api_key_env=cast(str, args.api_key_env),
            timeout_seconds=cast(float, args.timeout_seconds),
            overwrite=cast(bool, args.overwrite),
        )
    except PeopleFlowError as exc:
        logger.error("Report generation failed: %s", exc)
        return 2
    logger.info(
        "Report written: %s provider=%s llm_used=%s metadata=%s",
        result.report_path,
        result.provider,
        result.llm_used,
        result.metadata_path,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a stable process exit code."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())

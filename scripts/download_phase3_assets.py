"""Download verified official model and sample assets for Phase 3."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from people_flow.assets import OFFICIAL_MODELS, download_official_sample, resolve_model_path
from people_flow.errors import AssetDownloadError


def build_parser() -> argparse.ArgumentParser:
    """Build the asset-download command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(OFFICIAL_MODELS), default="yolo26n.pt")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--samples-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    return parser


def main() -> int:
    """Download requested assets and return a stable exit code."""

    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)
    try:
        if not args.skip_model:
            model_path = resolve_model_path(args.model, weights_dir=args.weights_dir)
            logger.info("Verified model: %s", model_path)
        if not args.skip_sample:
            sample_path = download_official_sample(samples_dir=args.samples_dir)
            logger.info("Verified sample: %s", sample_path)
    except AssetDownloadError as exc:
        logger.error("Asset download failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

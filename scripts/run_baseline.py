"""Baseline experiment command placeholder."""

from __future__ import annotations

import logging


def main() -> int:
    """Return a clear error until the Phase 3 pipeline is implemented."""

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).error("run_baseline.py requires the Phase 3 pipeline")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

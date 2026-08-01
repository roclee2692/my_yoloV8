"""Experiment evaluation command placeholder."""

from __future__ import annotations

import logging


def main() -> int:
    """Return a clear error until ground-truth evaluation is implemented."""

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).error("evaluate_experiment.py is not available until Phase 7")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

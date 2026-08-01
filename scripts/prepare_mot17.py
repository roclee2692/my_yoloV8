"""Phase 4 MOT17 preparation command placeholder."""

from __future__ import annotations

import logging


def main() -> int:
    """Return a clear error until MOT17 support is implemented."""

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).error("prepare_mot17.py is not available until Phase 4")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

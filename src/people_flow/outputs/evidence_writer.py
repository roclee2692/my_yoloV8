"""Evidence-frame image output with explicit OpenCV failure handling."""

from __future__ import annotations

from pathlib import Path

import cv2

from people_flow.datasets.video_source import Frame
from people_flow.errors import OutputError


def write_evidence_frame(path: Path, frame: Frame) -> Path:
    """Write one annotated JPEG and fail if OpenCV cannot encode it."""

    resolved = path.expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(resolved), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    except (OSError, cv2.error) as exc:
        raise OutputError(f"Unable to write evidence frame: {resolved}") from exc
    if not written:
        raise OutputError(f"OpenCV could not encode evidence frame: {resolved}")
    return resolved

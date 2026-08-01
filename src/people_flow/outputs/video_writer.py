"""Validated annotated MP4 writer."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2

from people_flow.datasets.video_source import Frame
from people_flow.errors import OutputError


class AnnotatedVideoWriter:
    """Write frames using the source video's dimensions and FPS."""

    def __init__(self, path: Path, *, fps: float, width: int, height: int) -> None:
        self.path = path.expanduser().resolve()
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> AnnotatedVideoWriter:
        """Create an MP4V output stream and validate codec availability."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened():
            writer.release()
            raise OutputError(f"OpenCV could not create annotated video: {self.path}")
        self._writer = writer
        return self

    def write(self, frame: Frame) -> None:
        """Write one frame after validating its dimensions."""

        if self._writer is None:
            raise OutputError("AnnotatedVideoWriter must be used inside a with block")
        height, width = frame.shape[:2]
        if width != self.width or height != self.height:
            raise OutputError(
                f"Frame dimensions changed: expected {self.width}x{self.height}, "
                f"got {width}x{height}"
            )
        self._writer.write(frame)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the native encoder."""

        if self._writer is not None:
            self._writer.release()
        self._writer = None

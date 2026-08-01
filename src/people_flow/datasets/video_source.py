"""Validated OpenCV video-source abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from people_flow.errors import VideoSourceError

Frame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Source video properties required by the output pipeline."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """A decoded frame with deterministic one-based indexing and timing."""

    frame_id: int
    timestamp_ms: float
    image: Frame


class VideoSource:
    """Context-managed iterator over a local video file."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0
        self.metadata: VideoMetadata | None = None

    def __enter__(self) -> VideoSource:
        """Open the video and validate the metadata required for deterministic output."""

        if not self.path.is_file():
            raise VideoSourceError(f"Video source does not exist: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"OpenCV could not open video source: {self.path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise VideoSourceError(
                f"Invalid video metadata for {self.path}: fps={fps}, size={width}x{height}"
            )

        self._capture = capture
        self._frame_id = 0
        self.metadata = VideoMetadata(self.path, fps, width, height, frame_count)
        return self

    def __iter__(self) -> VideoSource:
        """Return the source iterator after it has been opened."""

        if self._capture is None or self.metadata is None:
            raise VideoSourceError("VideoSource must be used inside a with block")
        return self

    def __next__(self) -> VideoFrame:
        """Decode the next frame or stop cleanly at the end of the stream."""

        if self._capture is None or self.metadata is None:
            raise VideoSourceError("VideoSource must be used inside a with block")
        ok, frame = self._capture.read()
        if not ok:
            raise StopIteration
        self._frame_id += 1
        timestamp_ms = (self._frame_id - 1) * 1000.0 / self.metadata.fps
        return VideoFrame(self._frame_id, timestamp_ms, cast(Frame, frame))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the native video handle."""

        if self._capture is not None:
            self._capture.release()
        self._capture = None

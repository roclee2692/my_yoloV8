"""Tests for local video validation and deterministic timestamps."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from people_flow.datasets.video_source import VideoSource
from people_flow.errors import VideoSourceError


def _write_test_video(path: Path, *, fps: float = 12.0, frames: int = 3) -> None:
    """Create a tiny local MP4 fixture without external data."""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (32, 24))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def test_video_source_reads_all_frames_with_source_fps(tmp_path: Path) -> None:
    """Frame count, dimensions, FPS, and timestamps should remain stable."""

    path = tmp_path / "short.mp4"
    _write_test_video(path)

    with VideoSource(path) as source:
        metadata = source.metadata
        frames = list(source)

    assert metadata is not None
    assert metadata.fps == pytest.approx(12.0, abs=0.1)
    assert (metadata.width, metadata.height) == (32, 24)
    assert len(frames) == 3
    assert frames[1].timestamp_ms == pytest.approx(1000.0 / 12.0, abs=0.5)


def test_video_source_reports_missing_file(tmp_path: Path) -> None:
    """A missing input video should produce a project-specific error."""

    with (
        pytest.raises(VideoSourceError, match="does not exist"),
        VideoSource(tmp_path / "missing.mp4"),
    ):
        pass

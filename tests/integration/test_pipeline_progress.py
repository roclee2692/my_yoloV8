"""Integration coverage for the core progress callback used by the Dashboard."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from people_flow.config import RunConfig
from people_flow.datasets.video_source import Frame
from people_flow.pipeline import run_pipeline
from people_flow.tracking.track_record import TrackRecord


class EmptyTracker:
    """Model-free tracker that exercises the empty-detection path."""

    def process(self, frame: Frame, *, frame_id: int, timestamp_ms: float) -> list[TrackRecord]:
        """Return no detections for every frame."""

        return []


def test_pipeline_reports_initial_and_per_frame_progress(tmp_path: Path) -> None:
    """The callback should receive a monotonic update ending at the source frame count."""

    source = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"mp4v"), 5.0, (32, 24))
    assert writer.isOpened()
    for value in (20, 40):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    model = tmp_path / "fake.pt"
    model.write_bytes(b"model-free-test")
    progress: list[tuple[int, int]] = []

    run_pipeline(
        RunConfig(
            source=source,
            model=str(model),
            tracker="bytetrack.yaml",
            output_dir=tmp_path / "run",
            device="cpu",
        ),
        tracker_runner=EmptyTracker(),
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )

    assert progress == [(0, 2), (1, 2), (2, 2)]

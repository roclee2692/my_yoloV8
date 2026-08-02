"""Model-free integration tests for tracking and counting artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from people_flow.config import CountingLineSettings, CountingSettings, RunConfig
from people_flow.datasets.video_source import Frame
from people_flow.pipeline import run_pipeline
from people_flow.tracking.track_record import TrackRecord


class FakeTracker:
    """Stable model-free tracker used to validate orchestration and outputs."""

    def process(self, frame: Frame, *, frame_id: int, timestamp_ms: float) -> list[TrackRecord]:
        """Return one stable ID in the first two frames and no target in the last frame."""

        if frame_id >= 3:
            return []
        return [
            TrackRecord.from_bbox(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                track_id=7,
                class_id=0,
                confidence=0.9,
                bbox=(4.0 + frame_id, 3.0, 18.0 + frame_id, 20.0),
            )
        ]


def _create_video(path: Path) -> None:
    """Create three frames at a non-default FPS to test metadata preservation."""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 7.0, (32, 24))
    assert writer.isOpened()
    for value in (20, 40, 60):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()


@pytest.mark.slow
def test_short_video_pipeline(tmp_path: Path) -> None:
    """The pipeline should handle stable IDs and an empty-detection frame."""

    source = tmp_path / "input.mp4"
    model = tmp_path / "fake.pt"
    output = tmp_path / "run"
    _create_video(source)
    model.write_bytes(b"model-free-test")

    result = run_pipeline(
        RunConfig(
            source=source,
            model=str(model),
            tracker="bytetrack.yaml",
            classes=(0,),
            output_dir=output,
            weights_dir=tmp_path / "weights",
            device="cpu",
        ),
        tracker_runner=FakeTracker(),
    )

    expected = {
        "annotated.mp4",
        "tracks.csv",
        "run_config.yaml",
        "runtime_metrics.json",
        "run.log",
    }
    assert {path.name for path in output.iterdir()} == expected
    with result.tracks_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["track_id"] for row in rows] == ["7", "7"]

    metrics = json.loads(result.runtime_metrics.read_text(encoding="utf-8"))
    assert metrics["input_frames"] == 3
    assert metrics["processed_frames"] == 3
    assert metrics["video_fps"] == pytest.approx(7.0, abs=0.1)
    assert metrics["peak_gpu_memory_mb"] == 0.0

    capture = cv2.VideoCapture(str(result.annotated_video))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(7.0, abs=0.1)
    capture.release()


class CrossingFakeTracker:
    """Produce one stable ID crossing a horizontal line on frame two."""

    def process(self, frame: Frame, *, frame_id: int, timestamp_ms: float) -> list[TrackRecord]:
        """Return a negative-side then positive-side foot point."""

        if frame_id > 2:
            return []
        foot_y = 8.0 if frame_id == 1 else 16.0
        return [
            TrackRecord.from_bbox(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                track_id=11,
                class_id=0,
                confidence=0.95,
                bbox=(8.0, 2.0, 14.0, foot_y),
            )
        ]


@pytest.mark.slow
def test_short_video_pipeline_writes_line_event_and_evidence(tmp_path: Path) -> None:
    """Enabled counting should add portable event evidence without changing tracking."""

    source = tmp_path / "input.mp4"
    model = tmp_path / "fake.pt"
    output = tmp_path / "counting_run"
    _create_video(source)
    model.write_bytes(b"model-free-test")

    result = run_pipeline(
        RunConfig(
            source=source,
            model=str(model),
            tracker="bytetrack.yaml",
            classes=(0,),
            output_dir=output,
            weights_dir=tmp_path / "weights",
            device="cpu",
            counting=CountingSettings(
                enabled=True,
                line=CountingLineSettings(
                    p1=(0.0, 12.0),
                    p2=(31.0, 12.0),
                    enter_side="positive",
                ),
                min_track_age=2,
                min_displacement_pixels=5.0,
                cooldown_frames=0,
                max_track_gap_frames=3,
            ),
        ),
        tracker_runner=CrossingFakeTracker(),
    )

    assert result.events_csv is not None
    assert result.evidence_dir is not None
    assert result.total_enter == 1
    assert result.total_exit == 0
    assert result.events_csv.is_file()
    with result.events_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "event_000001"
    assert rows[0]["track_id"] == "11"
    assert rows[0]["event_type"] == "enter"
    assert rows[0]["frame_id"] == "2"
    assert rows[0]["previous_side"] == "negative"
    assert rows[0]["current_side"] == "positive"
    evidence_path = output / Path(rows[0]["evidence_frame_path"])
    assert evidence_path.is_file()
    assert evidence_path.parent == result.evidence_dir
    evidence = cv2.imread(str(evidence_path))
    assert evidence is not None
    assert evidence.shape[:2] == (24, 32)

"""Tests for conversion of tracker output into person records."""

from __future__ import annotations

from typing import Any

import numpy as np

from people_flow.tracking.tracker_runner import TrackerRunner


class FakeTensor:
    """Small tensor facade supporting the methods used by TrackerRunner."""

    def __init__(self, value: list[Any]) -> None:
        self.value = value

    def int(self) -> FakeTensor:
        """Return self for integer conversion."""

        return self

    def float(self) -> FakeTensor:
        """Return self for float conversion."""

        return self

    def cpu(self) -> FakeTensor:
        """Return self for CPU conversion."""

        return self

    def tolist(self) -> list[Any]:
        """Return the stored Python values."""

        return self.value


class FakeBoxes:
    """Ultralytics-compatible boxes fixture."""

    id = FakeTensor([12, 99])
    cls = FakeTensor([0, 2])
    conf = FakeTensor([0.9, 0.7])
    xyxy = FakeTensor([[1.0, 2.0, 11.0, 22.0], [3.0, 4.0, 13.0, 24.0]])


class FakeResult:
    """Result fixture with two classes."""

    boxes = FakeBoxes()


class FakeDetector:
    """Detector fixture that returns a stable ID and a non-person distraction."""

    def track(self, *_args: Any, **_kwargs: Any) -> list[FakeResult]:
        """Return deterministic results."""

        return [FakeResult()]


def test_tracker_runner_keeps_only_person_records() -> None:
    """The runner must enforce class 0 even if a backend returns another class."""

    runner = TrackerRunner(
        FakeDetector(),
        tracker="bytetrack.yaml",
        classes=(0,),
        confidence=0.25,
        iou=0.7,
        imgsz=640,
    )

    records = runner.process(np.zeros((32, 32, 3), dtype=np.uint8), frame_id=1, timestamp_ms=0)

    assert len(records) == 1
    assert records[0].track_id == 12
    assert records[0].class_id == 0

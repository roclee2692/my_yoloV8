"""Tests for MOT adaptation, strict Track CSV input, and shared evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from people_flow.config import CountingLineSettings, CountingSettings, RoiSettings
from people_flow.datasets.mot_ground_truth import GroundTruthPolicy, MotGroundTruth
from people_flow.datasets.mot_reader import MotGroundTruthRecord, MotSequenceInfo
from people_flow.errors import EvaluationError
from people_flow.evaluation.trajectory_evaluator import evaluate_mot_prediction
from people_flow.evaluation.trajectory_io import read_track_records_csv
from people_flow.outputs.csv_writer import TrackCsvWriter
from people_flow.tracking.track_record import TrackRecord


def _ground_truth(tmp_path: Path) -> MotGroundTruth:
    info = MotSequenceInfo(
        sequence_dir=tmp_path,
        name="MOT17-99-SDP",
        image_dir=tmp_path / "img1",
        frame_rate=10.0,
        sequence_length=2,
        image_width=100,
        image_height=50,
        image_extension=".jpg",
    )
    records = (
        MotGroundTruthRecord(1, 1, 4.0, 1.0, 2.0, 4.0, 1.0, 1, 1.0),
        MotGroundTruthRecord(2, 1, 4.0, 11.0, 2.0, 4.0, 1.0, 1, 1.0),
        MotGroundTruthRecord(1, 2, 20.0, 1.0, 2.0, 4.0, 1.0, 3, 1.0),
        MotGroundTruthRecord(1, 3, 30.0, 1.0, 2.0, 4.0, 0.0, 1, 1.0),
    )
    return MotGroundTruth(info, records, {1: records[0:1], 2: records[1:2]})


def _predicted_records() -> tuple[TrackRecord, ...]:
    return (
        TrackRecord.from_bbox(
            frame_id=1,
            timestamp_ms=0.0,
            track_id=10,
            class_id=0,
            confidence=0.9,
            bbox=(4.0, 1.0, 6.0, 5.0),
        ),
        TrackRecord.from_bbox(
            frame_id=2,
            timestamp_ms=100.0,
            track_id=10,
            class_id=0,
            confidence=0.9,
            bbox=(4.0, 1.0, 6.0, 5.0),
        ),
    )


def test_mot_and_prediction_use_same_line_and_roi_counters(tmp_path: Path) -> None:
    """A GT-only crossing should become count and occupancy error through shared logic."""

    evaluation = evaluate_mot_prediction(
        _ground_truth(tmp_path),
        _predicted_records(),
        counting=CountingSettings(
            enabled=True,
            line=CountingLineSettings(p1=(0.0, 10.0), p2=(20.0, 10.0)),
            min_track_age=2,
            min_displacement_pixels=5.0,
            cooldown_frames=0,
            max_track_gap_frames=2,
        ),
        roi=RoiSettings(
            enabled=True,
            name="inside",
            points=((0.0, 10.0), (20.0, 10.0), (20.0, 20.0), (0.0, 20.0)),
            max_track_gap_frames=2,
        ),
        ground_truth_policy=GroundTruthPolicy(),
    )

    assert evaluation.ground_truth_track_records == 2
    assert len(evaluation.ground_truth.events) == 1
    assert evaluation.ground_truth.events[0].event_type == "enter"
    assert evaluation.predicted.events == ()
    assert evaluation.ground_truth.occupancy_values == (0, 1)
    assert evaluation.predicted.occupancy_values == (0, 0)
    assert evaluation.metrics.enter_absolute_error == 1
    assert evaluation.metrics.exit_absolute_error == 0
    assert evaluation.metrics.total_count_absolute_error == 1
    assert evaluation.metrics.count_mae == 0.5
    assert evaluation.metrics.count_mape == 100.0
    assert evaluation.metrics.occupancy_mae == 0.5


def test_track_csv_round_trip_and_derived_field_validation(tmp_path: Path) -> None:
    """Canonical Track CSV should round-trip and reject tampered derived geometry."""

    path = tmp_path / "tracks.csv"
    records = list(_predicted_records())
    with TrackCsvWriter(path) as writer:
        writer.write(records)
    assert read_track_records_csv(path) == tuple(records)

    tampered = path.read_text(encoding="utf-8").replace(",5.0,5.0\n", ",999.0,5.0\n", 1)
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(EvaluationError, match="Derived field"):
        read_track_records_csv(path)

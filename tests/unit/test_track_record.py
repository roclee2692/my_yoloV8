"""Tests for trajectory coordinate derivation."""

from people_flow.tracking.track_record import TrackRecord


def test_track_record_derives_center_and_foot_points() -> None:
    """The foot point should be the bounding box bottom center."""

    record = TrackRecord.from_bbox(
        frame_id=3,
        timestamp_ms=80.0,
        track_id=9,
        class_id=0,
        confidence=0.8,
        bbox=(10.0, 20.0, 30.0, 60.0),
    )

    assert (record.center_x, record.center_y) == (20.0, 40.0)
    assert (record.foot_x, record.foot_y) == (20.0, 60.0)

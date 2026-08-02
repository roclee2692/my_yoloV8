"""Tests for polygon occupancy, Track transitions, and dwell-time aggregation."""

import pytest

from people_flow.counting.geometry import PolygonRegion
from people_flow.counting.roi_counter import RoiCounter
from people_flow.errors import CountingError
from people_flow.tracking.track_record import TrackRecord


def _record(frame_id: int, foot_x: float, foot_y: float, *, track_id: int = 7) -> TrackRecord:
    return TrackRecord.from_bbox(
        frame_id=frame_id,
        timestamp_ms=(frame_id - 1) * 100.0,
        track_id=track_id,
        class_id=0,
        confidence=0.9,
        bbox=(foot_x - 1.0, foot_y - 4.0, foot_x + 1.0, foot_y),
    )


def _counter(*, max_track_gap_frames: int = 2) -> RoiCounter:
    return RoiCounter(
        PolygonRegion("entrance", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))),
        max_track_gap_frames=max_track_gap_frames,
    )


def test_observed_membership_reports_enter_exit_and_occupancy() -> None:
    """Foot-point transitions should produce deterministic per-frame membership."""

    counter = _counter()
    first = counter.process_frame([_record(1, -1.0, 5.0)], frame_id=1, timestamp_ms=0.0)
    second = counter.process_frame([_record(2, 5.0, 5.0)], frame_id=2, timestamp_ms=100.0)
    third = counter.process_frame([_record(3, 11.0, 5.0)], frame_id=3, timestamp_ms=200.0)

    assert first.occupancy == 0
    assert second.track_ids == (7,)
    assert second.entered_track_ids == (7,)
    assert third.exited_track_ids == (7,)
    assert counter.total_enter == 1
    assert counter.total_exit == 1


def test_boundary_foot_point_is_inside() -> None:
    """ROI borders should be included so exact-edge detections do not flicker outside."""

    snapshot = _counter().process_frame([_record(1, 0.0, 5.0)], frame_id=1, timestamp_ms=0.0)

    assert snapshot.track_ids == (7,)
    assert snapshot.entered_track_ids == (7,)


def test_dwell_accumulates_multiple_visits_and_keeps_first_and_last_times() -> None:
    """Repeated visits should sum intervals while preserving audit timestamps."""

    counter = _counter()
    positions = ((5.0, 5.0), (11.0, 5.0), (5.0, 5.0), (11.0, 5.0))
    for frame_id, position in enumerate(positions, start=1):
        counter.process_frame(
            [_record(frame_id, *position)],
            frame_id=frame_id,
            timestamp_ms=(frame_id - 1) * 100.0,
        )

    record = counter.dwell_records()[0]
    assert record.first_entry_frame == 1
    assert record.first_entry_timestamp_ms == 0.0
    assert record.last_seen_frame == 4
    assert record.last_seen_timestamp_ms == 300.0
    assert record.entry_count == 2
    assert record.exit_count == 2
    assert record.total_dwell_ms == 200.0
    assert record.total_dwell_seconds == pytest.approx(0.2)
    assert record.is_inside_at_end is False


def test_short_loss_preserves_active_visit_without_duplicate_enter() -> None:
    """A recovered ID inside the gap tolerance should continue one dwell interval."""

    counter = _counter(max_track_gap_frames=2)
    counter.process_frame([_record(1, 5.0, 5.0)], frame_id=1, timestamp_ms=0.0)
    counter.process_frame([], frame_id=2, timestamp_ms=100.0)
    snapshot = counter.process_frame([_record(3, 5.0, 5.0)], frame_id=3, timestamp_ms=200.0)

    assert snapshot.entered_track_ids == ()
    record = counter.dwell_records()[0]
    assert record.entry_count == 1
    assert record.total_dwell_ms == 200.0


def test_long_loss_closes_at_last_evidence_and_reentry_starts_new_visit() -> None:
    """A stale ID should not accumulate unseen time or remain logically inside."""

    counter = _counter(max_track_gap_frames=1)
    counter.process_frame([_record(1, 5.0, 5.0)], frame_id=1, timestamp_ms=0.0)
    counter.process_frame([], frame_id=2, timestamp_ms=100.0)
    counter.process_frame([], frame_id=3, timestamp_ms=200.0)
    snapshot = counter.process_frame([_record(4, 5.0, 5.0)], frame_id=4, timestamp_ms=300.0)

    assert snapshot.entered_track_ids == (7,)
    record = counter.dwell_records()[0]
    assert record.entry_count == 2
    assert record.total_dwell_ms == 0.0
    assert counter.total_exit == 0


def test_summary_reports_peak_average_and_zero_safe_dwell() -> None:
    """Aggregate output should use all processed frames and avoid empty-stat errors."""

    counter = _counter()
    counter.process_frame([], frame_id=1, timestamp_ms=0.0)
    counter.process_frame(
        [_record(2, 5.0, 5.0), _record(2, 6.0, 5.0, track_id=8)],
        frame_id=2,
        timestamp_ms=100.0,
    )
    counter.process_frame([_record(3, 5.0, 5.0)], frame_id=3, timestamp_ms=200.0)

    summary = counter.build_summary(processing_fps=25.0)
    assert summary.total_enter == 2
    assert summary.total_exit == 0
    assert summary.maximum_occupancy == 2
    assert summary.average_occupancy == pytest.approx(1.0)
    assert summary.peak_time == 100.0
    assert summary.peak_frame_id == 2
    assert summary.average_dwell_seconds == pytest.approx(0.05)
    assert summary.median_dwell_seconds == pytest.approx(0.05)
    assert summary.processed_frames == 3
    assert summary.processing_fps == 25.0


def test_duplicate_track_id_and_nonincreasing_frame_are_rejected() -> None:
    """Invalid tracker batches should fail explicitly instead of corrupting state."""

    counter = _counter()
    duplicate = [_record(1, 5.0, 5.0), _record(1, 6.0, 5.0)]
    with pytest.raises(CountingError, match="appears more than once"):
        counter.process_frame(duplicate, frame_id=1, timestamp_ms=0.0)

    fresh = _counter()
    fresh.process_frame([], frame_id=1, timestamp_ms=0.0)
    with pytest.raises(CountingError, match="must increase"):
        fresh.process_frame([], frame_id=1, timestamp_ms=0.0)

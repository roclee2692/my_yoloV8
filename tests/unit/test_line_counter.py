"""Tests for Track-ID crossing thresholds, recovery, cooldown, and deduplication."""

from people_flow.counting.geometry import DirectedLine, LineSide
from people_flow.counting.line_counter import LineCounter
from people_flow.tracking.track_record import TrackRecord


def _record(frame_id: int, foot_y: float, *, track_id: int = 7, foot_x: float = 5.0) -> TrackRecord:
    return TrackRecord.from_bbox(
        frame_id=frame_id,
        timestamp_ms=frame_id * 100.0,
        track_id=track_id,
        class_id=0,
        confidence=0.9,
        bbox=(foot_x - 1.0, foot_y - 4.0, foot_x + 1.0, foot_y),
    )


def _counter(
    *,
    enter_side: LineSide = LineSide.POSITIVE,
    min_track_age: int = 2,
    min_displacement_pixels: float = 0.0,
    cooldown_frames: int = 0,
    max_track_gap_frames: int = 30,
) -> LineCounter:
    return LineCounter(
        DirectedLine((0.0, 0.0), (10.0, 0.0), enter_side=enter_side),
        min_track_age=min_track_age,
        min_displacement_pixels=min_displacement_pixels,
        cooldown_frames=cooldown_frames,
        max_track_gap_frames=max_track_gap_frames,
    )


def test_negative_to_positive_is_enter_by_default() -> None:
    """A stable Track ID moving to the configured side emits one enter."""

    counter = _counter(min_displacement_pixels=5.0)

    assert counter.process_frame([_record(1, -5.0)], frame_id=1) == []
    events = counter.process_frame([_record(2, 5.0)], frame_id=2)

    assert len(events) == 1
    assert events[0].event_id == "event_000001"
    assert events[0].event_type == "enter"
    assert events[0].previous_side == "negative"
    assert events[0].current_side == "positive"
    assert counter.total_enter == 1
    assert counter.total_exit == 0


def test_swapping_enter_side_swaps_event_direction() -> None:
    """The same physical movement becomes exit when positive is not enter."""

    counter = _counter(enter_side=LineSide.NEGATIVE)
    counter.process_frame([_record(1, -5.0)], frame_id=1)
    events = counter.process_frame([_record(2, 5.0)], frame_id=2)

    assert [event.event_type for event in events] == ["exit"]


def test_same_track_same_direction_is_never_counted_twice() -> None:
    """One ID may produce one enter and one exit, but not another enter."""

    counter = _counter()
    event_types: list[str] = []
    for frame_id, foot_y in enumerate((-5.0, 5.0, -5.0, 5.0), start=1):
        events = counter.process_frame([_record(frame_id, foot_y)], frame_id=frame_id)
        event_types.extend(event.event_type for event in events)

    assert event_types == ["enter", "exit"]
    assert counter.total_enter == 1
    assert counter.total_exit == 1


def test_pending_crossing_waits_for_age_and_displacement() -> None:
    """A real crossing remains pending until both anti-jitter thresholds pass."""

    counter = _counter(min_track_age=3, min_displacement_pixels=10.0)

    assert counter.process_frame([_record(1, -2.0)], frame_id=1) == []
    assert counter.process_frame([_record(2, 2.0)], frame_id=2) == []
    events = counter.process_frame([_record(3, 12.0)], frame_id=3)

    assert [event.event_type for event in events] == ["enter"]
    assert events[0].frame_id == 3


def test_cooldown_suppresses_bounce_then_accepts_stable_opposite_side() -> None:
    """An opposite crossing can be accepted after the configured cooldown."""

    counter = _counter(cooldown_frames=5)
    counter.process_frame([_record(1, -5.0)], frame_id=1)
    assert [event.event_type for event in counter.process_frame([_record(2, 5.0)], frame_id=2)] == [
        "enter"
    ]
    assert counter.process_frame([_record(3, -5.0)], frame_id=3) == []
    for frame_id in (4, 5, 6):
        assert counter.process_frame([_record(frame_id, -5.0)], frame_id=frame_id) == []
    events = counter.process_frame([_record(7, -5.0)], frame_id=7)

    assert [event.event_type for event in events] == ["exit"]


def test_short_track_loss_preserves_crossing_state() -> None:
    """The same ID may cross after a temporary gap within max_track_gap_frames."""

    counter = _counter(max_track_gap_frames=3)
    counter.process_frame([_record(1, -5.0)], frame_id=1)
    counter.process_frame([], frame_id=2)
    counter.process_frame([], frame_id=3)
    events = counter.process_frame([_record(4, 5.0)], frame_id=4)

    assert [event.event_type for event in events] == ["enter"]


def test_long_track_loss_does_not_infer_a_crossing() -> None:
    """After a long gap the next position starts fresh instead of joining a jump."""

    counter = _counter(max_track_gap_frames=2)
    counter.process_frame([_record(1, -5.0)], frame_id=1)
    counter.process_frame([], frame_id=2)
    counter.process_frame([], frame_id=3)
    counter.process_frame([], frame_id=4)

    assert counter.process_frame([_record(5, 5.0)], frame_id=5) == []


def test_crossing_outside_endpoints_changes_side_without_event() -> None:
    """Moving across the infinite extension outside p1-p2 must not count."""

    counter = _counter()
    counter.process_frame([_record(1, -5.0, foot_x=20.0)], frame_id=1)
    assert counter.process_frame([_record(2, 5.0, foot_x=20.0)], frame_id=2) == []
    assert counter.total_enter == 0

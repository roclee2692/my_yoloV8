"""Tests for Ground Truth count and occupancy error metrics."""

import pytest

from people_flow.counting.line_counter import LineCrossingEvent
from people_flow.errors import EvaluationError
from people_flow.evaluation.counting_metrics import compute_counting_metrics


def _event(event_id: str, event_type: str) -> LineCrossingEvent:
    return LineCrossingEvent(
        event_id=event_id,
        track_id=1,
        event_type=event_type,  # type: ignore[arg-type]
        frame_id=2,
        timestamp_ms=100.0,
        previous_side="negative",
        current_side="positive",
        foot_x=5.0,
        foot_y=5.0,
        confidence=1.0,
    )


def test_counting_metrics_compare_directions_total_and_occupancy() -> None:
    """All required errors should be calculated from event and frame evidence."""

    metrics = compute_counting_metrics(
        [_event("gt1", "enter"), _event("gt2", "exit")],
        [_event("pred1", "enter"), _event("pred2", "enter")],
        ground_truth_occupancy=[0, 2, 1],
        predicted_occupancy=[1, 1, 1],
    )

    assert metrics.enter_absolute_error == 1
    assert metrics.exit_absolute_error == 1
    assert metrics.total_count_absolute_error == 0
    assert metrics.count_mae == 1.0
    assert metrics.count_mape == 0.0
    assert metrics.occupancy_mae == pytest.approx(2.0 / 3.0)


def test_count_mape_is_none_when_ground_truth_total_is_zero() -> None:
    """Zero Ground Truth must return JSON-safe None instead of dividing by zero."""

    metrics = compute_counting_metrics([], [_event("pred1", "enter")])

    assert metrics.ground_truth_total == 0
    assert metrics.predicted_total == 1
    assert metrics.total_count_absolute_error == 1
    assert metrics.count_mape is None
    assert metrics.occupancy_mae is None


def test_occupancy_length_mismatch_is_rejected() -> None:
    """Occupancy MAE must compare exactly the same frame interval."""

    with pytest.raises(EvaluationError, match="lengths differ"):
        compute_counting_metrics(
            [],
            [],
            ground_truth_occupancy=[0, 1],
            predicted_occupancy=[0],
        )

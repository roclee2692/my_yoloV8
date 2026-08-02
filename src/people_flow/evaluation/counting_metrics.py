"""Ground Truth counting and occupancy error metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from people_flow.counting.line_counter import LineCrossingEvent
from people_flow.errors import EvaluationError


@dataclass(frozen=True, slots=True)
class CountingEvaluationMetrics:
    """Count errors and optional per-frame occupancy error."""

    ground_truth_enter: int
    ground_truth_exit: int
    ground_truth_total: int
    predicted_enter: int
    predicted_exit: int
    predicted_total: int
    enter_absolute_error: int
    exit_absolute_error: int
    total_count_absolute_error: int
    count_mae: float
    count_mape: float | None
    occupancy_mae: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metrics mapping."""

        return asdict(self)


def compute_counting_metrics(
    ground_truth_events: Sequence[LineCrossingEvent],
    predicted_events: Sequence[LineCrossingEvent],
    *,
    ground_truth_occupancy: Sequence[int] | None = None,
    predicted_occupancy: Sequence[int] | None = None,
) -> CountingEvaluationMetrics:
    """Compare events with zero-safe total-count MAPE and occupancy MAE."""

    ground_truth_enter = _event_count(ground_truth_events, "enter")
    ground_truth_exit = _event_count(ground_truth_events, "exit")
    predicted_enter = _event_count(predicted_events, "enter")
    predicted_exit = _event_count(predicted_events, "exit")
    ground_truth_total = ground_truth_enter + ground_truth_exit
    predicted_total = predicted_enter + predicted_exit
    enter_error = abs(predicted_enter - ground_truth_enter)
    exit_error = abs(predicted_exit - ground_truth_exit)
    total_error = abs(predicted_total - ground_truth_total)
    occupancy_mae = _occupancy_mae(ground_truth_occupancy, predicted_occupancy)
    return CountingEvaluationMetrics(
        ground_truth_enter=ground_truth_enter,
        ground_truth_exit=ground_truth_exit,
        ground_truth_total=ground_truth_total,
        predicted_enter=predicted_enter,
        predicted_exit=predicted_exit,
        predicted_total=predicted_total,
        enter_absolute_error=enter_error,
        exit_absolute_error=exit_error,
        total_count_absolute_error=total_error,
        count_mae=(enter_error + exit_error) / 2.0,
        count_mape=(total_error / ground_truth_total * 100.0 if ground_truth_total else None),
        occupancy_mae=occupancy_mae,
    )


def _event_count(events: Sequence[LineCrossingEvent], event_type: str) -> int:
    return sum(event.event_type == event_type for event in events)


def _occupancy_mae(
    ground_truth: Sequence[int] | None,
    predicted: Sequence[int] | None,
) -> float | None:
    if ground_truth is None and predicted is None:
        return None
    if ground_truth is None or predicted is None:
        raise EvaluationError("Both Ground Truth and predicted occupancy are required")
    if len(ground_truth) != len(predicted):
        raise EvaluationError(
            "Ground Truth and predicted occupancy lengths differ: "
            f"{len(ground_truth)} != {len(predicted)}"
        )
    if not ground_truth:
        return 0.0
    return sum(
        abs(prediction - truth) for truth, prediction in zip(ground_truth, predicted, strict=False)
    ) / len(ground_truth)

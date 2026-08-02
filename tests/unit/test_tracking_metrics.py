"""Tests for deterministic Phase 8 ID-switch measurement."""

from __future__ import annotations

import pytest

from people_flow.errors import EvaluationError
from people_flow.evaluation.tracking_metrics import bbox_iou, compute_tracking_metrics
from people_flow.tracking.track_record import TrackRecord


def _record(frame_id: int, track_id: int, x1: float = 0.0) -> TrackRecord:
    return TrackRecord.from_bbox(
        frame_id=frame_id,
        timestamp_ms=(frame_id - 1) * 100.0,
        track_id=track_id,
        class_id=0,
        confidence=1.0,
        bbox=(x1, 0.0, x1 + 10.0, 10.0),
    )


def test_bbox_iou_handles_overlap_and_disjoint_boxes() -> None:
    assert bbox_iou(_record(1, 1), _record(1, 2)) == 1.0
    assert bbox_iou(_record(1, 1), _record(1, 2, 20.0)) == 0.0
    assert bbox_iou(_record(1, 1), _record(1, 2, 5.0)) == pytest.approx(1.0 / 3.0)


def test_tracking_metrics_count_changed_prediction_identity() -> None:
    """A GT identity moving from predicted ID 10 to 11 is one ID switch."""

    metrics = compute_tracking_metrics(
        [_record(1, 1), _record(2, 1)],
        [_record(1, 10), _record(2, 11)],
        frame_count=2,
        iou_threshold=0.5,
    )

    assert metrics.matching_method == "continuity_first_greedy_iou"
    assert metrics.matched_detections == 2
    assert metrics.id_switches == 1
    assert metrics.unmatched_ground_truth_detections == 0
    assert metrics.unmatched_predicted_detections == 0
    assert metrics.mean_matched_iou == 1.0


def test_tracking_metrics_retain_valid_existing_association() -> None:
    """Continuity wins while the prior pairing remains above the IoU threshold."""

    metrics = compute_tracking_metrics(
        [_record(1, 1), _record(2, 1)],
        [_record(1, 10), _record(2, 10, 1.0), _record(2, 11)],
        frame_count=2,
        iou_threshold=0.5,
    )

    assert metrics.id_switches == 0
    assert metrics.matched_detections == 2
    assert metrics.unmatched_predicted_detections == 1


def test_tracking_metrics_reject_duplicate_ids_per_frame() -> None:
    with pytest.raises(EvaluationError, match="Duplicate prediction Track ID"):
        compute_tracking_metrics(
            [_record(1, 1)],
            [_record(1, 10), _record(1, 10)],
            frame_count=1,
        )

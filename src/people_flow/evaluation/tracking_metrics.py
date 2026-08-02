"""Transparent IoU matching and ID-switch metrics for MOT trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from people_flow.errors import EvaluationError
from people_flow.tracking.track_record import TrackRecord


@dataclass(frozen=True, slots=True)
class TrackingEvaluationMetrics:
    """Frame-level matching evidence and identity switches."""

    matching_method: str
    match_iou_threshold: float
    ground_truth_detections: int
    predicted_detections: int
    matched_detections: int
    unmatched_ground_truth_detections: int
    unmatched_predicted_detections: int
    id_switches: int
    mean_matched_iou: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metrics mapping."""

        return asdict(self)


def compute_tracking_metrics(
    ground_truth_records: Sequence[TrackRecord],
    predicted_records: Sequence[TrackRecord],
    *,
    frame_count: int,
    iou_threshold: float = 0.5,
) -> TrackingEvaluationMetrics:
    """Count ID switches using continuity-first one-to-one IoU matching.

    Existing valid GT-to-prediction associations are retained before remaining
    candidates are greedily matched by descending IoU. This deterministic method is
    reported explicitly and is not presented as the official MOTChallenge score.
    """

    if frame_count <= 0:
        raise EvaluationError("tracking frame_count must be positive")
    if not 0.0 < iou_threshold <= 1.0:
        raise EvaluationError("tracking IoU threshold must be in (0, 1]")
    truth_by_frame = _group_by_frame(ground_truth_records, frame_count=frame_count, label="GT")
    prediction_by_frame = _group_by_frame(
        predicted_records,
        frame_count=frame_count,
        label="prediction",
    )
    previous_prediction: dict[int, int] = {}
    matched_ious: list[float] = []
    id_switches = 0

    for frame_id in range(1, frame_count + 1):
        truth = sorted(truth_by_frame.get(frame_id, ()), key=lambda record: record.track_id)
        predictions = sorted(
            prediction_by_frame.get(frame_id, ()), key=lambda record: record.track_id
        )
        truth_by_id = {record.track_id: record for record in truth}
        prediction_by_id = {record.track_id: record for record in predictions}
        matches: dict[int, tuple[int, float]] = {}
        used_predictions: set[int] = set()

        for truth_id in sorted(truth_by_id):
            prior_id = previous_prediction.get(truth_id)
            if prior_id is None or prior_id not in prediction_by_id:
                continue
            overlap = bbox_iou(truth_by_id[truth_id], prediction_by_id[prior_id])
            if overlap >= iou_threshold and prior_id not in used_predictions:
                matches[truth_id] = (prior_id, overlap)
                used_predictions.add(prior_id)

        candidates: list[tuple[float, int, int]] = []
        for truth_id, truth_record in truth_by_id.items():
            if truth_id in matches:
                continue
            for prediction_id, prediction_record in prediction_by_id.items():
                if prediction_id in used_predictions:
                    continue
                overlap = bbox_iou(truth_record, prediction_record)
                if overlap >= iou_threshold:
                    candidates.append((-overlap, truth_id, prediction_id))
        for negative_iou, truth_id, prediction_id in sorted(candidates):
            if truth_id in matches or prediction_id in used_predictions:
                continue
            overlap = -negative_iou
            matches[truth_id] = (prediction_id, overlap)
            used_predictions.add(prediction_id)

        for truth_id, (prediction_id, overlap) in matches.items():
            prior_id = previous_prediction.get(truth_id)
            if prior_id is not None and prior_id != prediction_id:
                id_switches += 1
            previous_prediction[truth_id] = prediction_id
            matched_ious.append(overlap)

    matched_count = len(matched_ious)
    return TrackingEvaluationMetrics(
        matching_method="continuity_first_greedy_iou",
        match_iou_threshold=iou_threshold,
        ground_truth_detections=len(ground_truth_records),
        predicted_detections=len(predicted_records),
        matched_detections=matched_count,
        unmatched_ground_truth_detections=len(ground_truth_records) - matched_count,
        unmatched_predicted_detections=len(predicted_records) - matched_count,
        id_switches=id_switches,
        mean_matched_iou=(sum(matched_ious) / matched_count if matched_count else None),
    )


def bbox_iou(first: TrackRecord, second: TrackRecord) -> float:
    """Return intersection-over-union for two TrackRecord boxes."""

    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _group_by_frame(
    records: Sequence[TrackRecord],
    *,
    frame_count: int,
    label: str,
) -> dict[int, tuple[TrackRecord, ...]]:
    grouped: dict[int, list[TrackRecord]] = {}
    seen: set[tuple[int, int]] = set()
    for record in records:
        if not 1 <= record.frame_id <= frame_count:
            raise EvaluationError(
                f"{label} frame {record.frame_id} is outside tracking range 1..{frame_count}"
            )
        key = (record.frame_id, record.track_id)
        if key in seen:
            raise EvaluationError(
                f"Duplicate {label} Track ID {record.track_id} in frame {record.frame_id}"
            )
        seen.add(key)
        grouped.setdefault(record.frame_id, []).append(record)
    return {frame_id: tuple(frame_records) for frame_id, frame_records in grouped.items()}

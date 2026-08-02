"""Frame-indexed access to unfiltered MOT16/17 ground-truth records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from people_flow.datasets.mot_reader import MotGroundTruthRecord, MotSequence, MotSequenceInfo
from people_flow.errors import EvaluationError
from people_flow.tracking.track_record import TrackRecord


@dataclass(frozen=True, slots=True)
class MotGroundTruth:
    """Ground-truth records grouped for deterministic frame lookup."""

    sequence: MotSequenceInfo
    records: tuple[MotGroundTruthRecord, ...]
    records_by_frame: dict[int, tuple[MotGroundTruthRecord, ...]]

    def for_frame(self, frame_id: int) -> tuple[MotGroundTruthRecord, ...]:
        """Return all annotations for a frame, or an empty tuple when none exist."""

        return self.records_by_frame.get(frame_id, ())

    @property
    def track_ids(self) -> frozenset[int]:
        """Return every trajectory ID present in the source annotation."""

        return frozenset(record.track_id for record in self.records)


def load_mot_ground_truth(sequence_dir: Path) -> MotGroundTruth:
    """Load one sequence GT without applying pedestrian or confidence filters."""

    sequence = MotSequence(sequence_dir)
    records = sequence.read_ground_truth(required=True)
    grouped: dict[int, list[MotGroundTruthRecord]] = {}
    for record in records:
        grouped.setdefault(record.frame_id, []).append(record)
    records_by_frame = {
        frame_id: tuple(frame_records) for frame_id, frame_records in sorted(grouped.items())
    }
    return MotGroundTruth(sequence.info, records, records_by_frame)


@dataclass(frozen=True, slots=True)
class GroundTruthPolicy:
    """Explicit MOT annotation filtering before shared trajectory analysis."""

    person_class_ids: tuple[int, ...] = (1,)
    require_marked: bool = True
    min_visibility: float = 0.0

    def __post_init__(self) -> None:
        if not self.person_class_ids or any(class_id <= 0 for class_id in self.person_class_ids):
            raise EvaluationError("Ground Truth person_class_ids must contain positive IDs")
        if not 0.0 <= self.min_visibility <= 1.0:
            raise EvaluationError("Ground Truth min_visibility must be between 0 and 1")


def ground_truth_track_records(
    ground_truth: MotGroundTruth,
    *,
    policy: GroundTruthPolicy | None = None,
) -> tuple[TrackRecord, ...]:
    """Filter MOT pedestrian annotations and normalize them to COCO person Tracks."""

    effective_policy = policy or GroundTruthPolicy()
    records: list[TrackRecord] = []
    for annotation in ground_truth.records:
        if annotation.class_id not in effective_policy.person_class_ids:
            continue
        if effective_policy.require_marked and annotation.confidence <= 0.0:
            continue
        if annotation.visibility < effective_policy.min_visibility:
            continue
        timestamp_ms = (annotation.frame_id - 1) * 1000.0 / ground_truth.sequence.frame_rate
        records.append(
            TrackRecord.from_bbox(
                frame_id=annotation.frame_id,
                timestamp_ms=timestamp_ms,
                track_id=annotation.track_id,
                class_id=0,
                confidence=annotation.confidence,
                bbox=annotation.bbox_xyxy,
            )
        )
    return tuple(sorted(records, key=lambda record: (record.frame_id, record.track_id)))

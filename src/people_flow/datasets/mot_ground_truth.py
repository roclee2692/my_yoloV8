"""Frame-indexed access to unfiltered MOT16/17 ground-truth records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from people_flow.datasets.mot_reader import MotGroundTruthRecord, MotSequence, MotSequenceInfo


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

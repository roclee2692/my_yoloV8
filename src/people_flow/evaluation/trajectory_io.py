"""Strict CSV input for persisted predicted Track records."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from people_flow.errors import EvaluationError
from people_flow.outputs.csv_writer import TRACK_FIELDS
from people_flow.tracking.track_record import TrackRecord


def read_track_records_csv(path: Path) -> tuple[TrackRecord, ...]:
    """Read the canonical tracks.csv schema with explicit numeric validation."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationError(f"Predicted tracks CSV does not exist: {resolved}")
    records: list[TrackRecord] = []
    seen_per_frame: set[tuple[int, int]] = set()
    previous_frame = 0
    try:
        with resolved.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != TRACK_FIELDS:
                raise EvaluationError(
                    f"Predicted tracks CSV header does not match required schema: {resolved}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    frame_id = _parse_int(row["frame_id"], "frame_id")
                    track_id = _parse_int(row["track_id"], "track_id")
                    class_id = _parse_int(row["class_id"], "class_id")
                    timestamp_ms = _parse_float(row["timestamp_ms"], "timestamp_ms")
                    confidence = _parse_float(row["confidence"], "confidence")
                    bbox = tuple(
                        _parse_float(row[field], field) for field in ("x1", "y1", "x2", "y2")
                    )
                    supplied = {
                        field: _parse_float(row[field], field)
                        for field in ("center_x", "center_y", "foot_x", "foot_y")
                    }
                except (KeyError, ValueError) as exc:
                    raise EvaluationError(
                        f"Invalid predicted Track row at {resolved}:{line_number}: {exc}"
                    ) from exc
                if frame_id <= 0 or track_id <= 0:
                    raise EvaluationError(
                        f"frame_id and track_id must be positive at {resolved}:{line_number}"
                    )
                if frame_id < previous_frame:
                    raise EvaluationError(
                        f"Predicted Track frames are not ordered at {resolved}:{line_number}"
                    )
                previous_frame = frame_id
                if class_id != 0:
                    raise EvaluationError(
                        f"Predicted Track class must be person class 0 at {resolved}:{line_number}"
                    )
                if not 0.0 <= confidence <= 1.0 or timestamp_ms < 0.0:
                    raise EvaluationError(
                        f"Invalid confidence or timestamp at {resolved}:{line_number}"
                    )
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    raise EvaluationError(f"Invalid bbox at {resolved}:{line_number}")
                record = TrackRecord.from_bbox(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    track_id=track_id,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
                for field, expected in (
                    ("center_x", record.center_x),
                    ("center_y", record.center_y),
                    ("foot_x", record.foot_x),
                    ("foot_y", record.foot_y),
                ):
                    if not math.isclose(supplied[field], expected, rel_tol=1e-6, abs_tol=1e-6):
                        raise EvaluationError(
                            f"Derived field {field} disagrees with bbox at {resolved}:{line_number}"
                        )
                key = (frame_id, track_id)
                if key in seen_per_frame:
                    raise EvaluationError(
                        f"Duplicate Track ID {track_id} in frame {frame_id}: {resolved}"
                    )
                seen_per_frame.add(key)
                records.append(record)
    except OSError as exc:
        raise EvaluationError(f"Unable to read predicted tracks CSV: {resolved}") from exc
    return tuple(records)


def _parse_float(value: str, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _parse_int(value: str, field: str) -> int:
    parsed = _parse_float(value, field)
    if not parsed.is_integer():
        raise ValueError(f"{field} must be integral")
    return int(parsed)

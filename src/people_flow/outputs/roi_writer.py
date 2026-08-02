"""CSV outputs for per-frame ROI occupancy and per-Track dwell times."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from people_flow.counting.dwell_time import DwellTimeRecord
from people_flow.counting.roi_counter import RoiFrameSnapshot
from people_flow.errors import OutputError

OCCUPANCY_FIELDS = [
    "frame_id",
    "timestamp_ms",
    "roi_name",
    "occupancy",
    "track_ids",
    "entered_track_ids",
    "exited_track_ids",
]

DWELL_FIELDS = [
    "track_id",
    "roi_name",
    "first_entry_frame",
    "first_entry_timestamp_ms",
    "last_seen_frame",
    "last_seen_timestamp_ms",
    "entry_count",
    "exit_count",
    "total_dwell_ms",
    "total_dwell_seconds",
    "is_inside_at_end",
]


class OccupancyCsvWriter:
    """Stream one auditable ROI membership snapshot per processed frame."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._stream: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> OccupancyCsvWriter:
        """Open a UTF-8 CSV and immediately persist its header."""

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._stream, fieldnames=OCCUPANCY_FIELDS)
            self._writer.writeheader()
        except OSError as exc:
            raise OutputError(f"Unable to create occupancy CSV: {self.path}") from exc
        return self

    def write(self, snapshot: RoiFrameSnapshot) -> None:
        """Append one snapshot with JSON arrays in the Track-ID columns."""

        if self._writer is None:
            raise OutputError("OccupancyCsvWriter must be used inside a with block")
        row: dict[str, Any] = snapshot.as_dict()
        for field in ("track_ids", "entered_track_ids", "exited_track_ids"):
            row[field] = json.dumps(row[field], separators=(",", ":"))
        self._writer.writerow(row)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush and close the output stream."""

        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._writer = None


def write_dwell_times_csv(path: Path, records: Sequence[DwellTimeRecord]) -> Path:
    """Write final dwell records, including a header for an empty result."""

    resolved = path.expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DWELL_FIELDS)
            writer.writeheader()
            writer.writerows(record.as_row() for record in records)
    except OSError as exc:
        raise OutputError(f"Unable to write dwell-times CSV: {resolved}") from exc
    return resolved

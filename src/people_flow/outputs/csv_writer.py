"""Streaming CSV writer for per-frame track records."""

from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from people_flow.errors import OutputError
from people_flow.tracking.track_record import TrackRecord

TRACK_FIELDS = [
    "frame_id",
    "timestamp_ms",
    "track_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "foot_x",
    "foot_y",
]


class TrackCsvWriter:
    """Context-managed writer that always emits the required CSV header."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._stream: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> TrackCsvWriter:
        """Open a UTF-8 CSV and write its stable schema."""

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._stream, fieldnames=TRACK_FIELDS)
            self._writer.writeheader()
        except OSError as exc:
            raise OutputError(f"Unable to create tracks CSV: {self.path}") from exc
        return self

    def write(self, records: list[TrackRecord]) -> None:
        """Append zero or more track records."""

        if self._writer is None:
            raise OutputError("TrackCsvWriter must be used inside a with block")
        rows: list[dict[str, Any]] = [record.as_row() for record in records]
        self._writer.writerows(rows)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush and close the CSV stream."""

        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._writer = None

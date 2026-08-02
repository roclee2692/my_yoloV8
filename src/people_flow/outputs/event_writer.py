"""Streaming CSV writer for accepted line-crossing events."""

from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from people_flow.counting.line_counter import LineCrossingEvent
from people_flow.errors import OutputError

EVENT_FIELDS = [
    "event_id",
    "track_id",
    "event_type",
    "frame_id",
    "timestamp_ms",
    "previous_side",
    "current_side",
    "foot_x",
    "foot_y",
    "confidence",
    "evidence_frame_path",
]


class EventCsvWriter:
    """Context-managed writer that emits the stable Phase 5 event schema."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._stream: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> EventCsvWriter:
        """Open a UTF-8 CSV and write its header even when no events occur."""

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._stream, fieldnames=EVENT_FIELDS)
            self._writer.writeheader()
        except OSError as exc:
            raise OutputError(f"Unable to create events CSV: {self.path}") from exc
        return self

    def write(self, events: list[LineCrossingEvent]) -> None:
        """Append zero or more accepted events."""

        if self._writer is None:
            raise OutputError("EventCsvWriter must be used inside a with block")
        rows: list[dict[str, Any]] = [event.as_row() for event in events]
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

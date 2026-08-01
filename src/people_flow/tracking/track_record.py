"""Typed per-frame person trajectory records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """One tracked person observation in one source frame."""

    frame_id: int
    timestamp_ms: float
    track_id: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    foot_x: float
    foot_y: float

    @classmethod
    def from_bbox(
        cls,
        *,
        frame_id: int,
        timestamp_ms: float,
        track_id: int,
        class_id: int,
        confidence: float,
        bbox: tuple[float, float, float, float],
    ) -> TrackRecord:
        """Create a record and derive center and bottom-center foot points."""

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return cls(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            track_id=track_id,
            class_id=class_id,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            center_x=center_x,
            center_y=center_y,
            foot_x=center_x,
            foot_y=y2,
        )

    def as_row(self) -> dict[str, Any]:
        """Return a CSV/JSON-compatible mapping in declared field order."""

        return asdict(self)

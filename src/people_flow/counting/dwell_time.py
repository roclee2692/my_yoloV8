"""Dwell-time records and deterministic ROI summary calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Any


@dataclass(frozen=True, slots=True)
class DwellTimeRecord:
    """Final per-Track dwell evidence for one ROI."""

    track_id: int
    roi_name: str
    first_entry_frame: int
    first_entry_timestamp_ms: float
    last_seen_frame: int
    last_seen_timestamp_ms: float
    entry_count: int
    exit_count: int
    total_dwell_ms: float
    total_dwell_seconds: float
    is_inside_at_end: bool

    def as_row(self) -> dict[str, Any]:
        """Return the stable CSV row mapping."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoiSummary:
    """Aggregate ROI occupancy and dwell measurements for one run."""

    roi_name: str
    total_enter: int
    total_exit: int
    maximum_occupancy: int
    average_occupancy: float
    peak_time: float | None
    peak_frame_id: int | None
    average_dwell_seconds: float
    median_dwell_seconds: float
    processed_frames: int
    processing_fps: float
    tracks_with_dwell: int
    inside_at_end: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary mapping."""

        return asdict(self)


def build_roi_summary(
    *,
    roi_name: str,
    total_enter: int,
    total_exit: int,
    maximum_occupancy: int,
    occupancy_sum: int,
    peak_time: float | None,
    peak_frame_id: int | None,
    dwell_records: Sequence[DwellTimeRecord],
    processed_frames: int,
    processing_fps: float,
) -> RoiSummary:
    """Build zero-safe aggregate occupancy and dwell statistics."""

    average_occupancy = occupancy_sum / processed_frames if processed_frames else 0.0
    dwell_seconds = [record.total_dwell_seconds for record in dwell_records]
    return RoiSummary(
        roi_name=roi_name,
        total_enter=total_enter,
        total_exit=total_exit,
        maximum_occupancy=maximum_occupancy,
        average_occupancy=average_occupancy,
        peak_time=peak_time,
        peak_frame_id=peak_frame_id,
        average_dwell_seconds=fmean(dwell_seconds) if dwell_seconds else 0.0,
        median_dwell_seconds=median(dwell_seconds) if dwell_seconds else 0.0,
        processed_frames=processed_frames,
        processing_fps=processing_fps,
        tracks_with_dwell=len(dwell_records),
        inside_at_end=sum(record.is_inside_at_end for record in dwell_records),
    )
